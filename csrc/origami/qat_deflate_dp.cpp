// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from /home/td/dpucomp/kvtc/qat_dp_codec.cpp for Origami native CPU lossless.

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cfloat>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <unistd.h>

extern "C" {
#include <qat/cpa.h>
#include <qat/cpa_dc.h>
#include <qat/cpa_dc_dp.h>
#include <qat/icp_sal_poll.h>
#include <qat/icp_sal_user.h>
#include <qat/qae_mem.h>
}

extern "C" CpaStatus qaeMemInit(void);
extern "C" void qaeMemDestroy(void);

namespace {

namespace py = pybind11;

constexpr uint32_t kAlignment = 64;
constexpr uint32_t kMaxQaeAllocBytes = 64U * 1024U * 1024U;
constexpr uint32_t kMinDcDestBytes = 2048U;
constexpr uint32_t kProjectWorkerCap = 4U;

void check_cpu_u8(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, name, " must be uint8");
}

void check_cpu_i32(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
}

void throw_qat(CpaStatus status, const char* what) {
  throw std::runtime_error(
      std::string(what) + " failed with QAT status " + std::to_string(status));
}

void require_qat(CpaStatus status, const char* what) {
  if (status != CPA_STATUS_SUCCESS) {
    throw_qat(status, what);
  }
}

CpaPhysicalAddr virt_to_phys(void* ptr) {
  return static_cast<CpaPhysicalAddr>(qaeVirtToPhysNUMA(ptr));
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#else
  std::this_thread::yield();
#endif
}

std::vector<int> parse_cpu_set_spec(const char* value) {
  std::vector<int> cpus;
  if (value == nullptr || *value == '\0') {
    return cpus;
  }
  std::string spec(value);
  size_t pos = 0;
  while (pos < spec.size()) {
    const size_t comma = spec.find(',', pos);
    const std::string token = spec.substr(
        pos,
        comma == std::string::npos ? std::string::npos : comma - pos);
    if (!token.empty()) {
      const size_t dash = token.find('-');
      if (dash == std::string::npos) {
        cpus.push_back(std::stoi(token));
      } else {
        const int start = std::stoi(token.substr(0, dash));
        const int end = std::stoi(token.substr(dash + 1));
        TORCH_CHECK(end >= start, "invalid KVTC_QAT_CPUSET range");
        for (int cpu = start; cpu <= end; ++cpu) {
          cpus.push_back(cpu);
        }
      }
    }
    if (comma == std::string::npos) {
      break;
    }
    pos = comma + 1;
  }
  return cpus;
}

const std::vector<int>& qat_worker_cpus_global() {
  static const std::vector<int> cpus =
      parse_cpu_set_spec(std::getenv("KVTC_QAT_CPUSET"));
  return cpus;
}

const std::vector<int>& qat_worker_cpus_for_node(int node) {
  static const std::vector<int> node0 =
      parse_cpu_set_spec(std::getenv("KVTC_QAT_CPUSET_NODE0"));
  static const std::vector<int> node1 =
      parse_cpu_set_spec(std::getenv("KVTC_QAT_CPUSET_NODE1"));
  if (node == 0 && !node0.empty()) {
    return node0;
  }
  if (node == 1 && !node1.empty()) {
    return node1;
  }
  return qat_worker_cpus_global();
}

void pin_qat_worker_thread(uint32_t worker, int node) {
  const auto& cpus = qat_worker_cpus_for_node(node);
  if (cpus.empty()) {
    return;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpus[worker % cpus.size()], &set);
  const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (rc != 0) {
    throw std::runtime_error(
        "pthread_setaffinity_np failed with errno " + std::to_string(rc));
  }
}

struct DpSgl {
  CpaPhysBufferList* list = nullptr;
  std::vector<uint8_t*> segments;

  DpSgl() = default;
  DpSgl(const DpSgl&) = delete;
  DpSgl& operator=(const DpSgl&) = delete;

  ~DpSgl() {
    reset();
  }

  void reset() {
    for (uint8_t* segment : segments) {
      if (segment != nullptr) {
        qaeMemFreeNUMA(reinterpret_cast<void**>(&segment));
      }
    }
    segments.clear();
    if (list != nullptr) {
      qaeMemFreeNUMA(reinterpret_cast<void**>(&list));
      list = nullptr;
    }
  }

  void allocate(int node, uint32_t total_len) {
    reset();
    uint32_t count = (total_len + kMaxQaeAllocBytes - 1U) / kMaxQaeAllocBytes;
    if (count == 0) {
      count = 1;
    }
    const size_t list_size = sizeof(CpaPhysBufferList) +
        static_cast<size_t>(count) * sizeof(CpaPhysFlatBuffer);
    list = static_cast<CpaPhysBufferList*>(
        qaeMemAllocNUMA(list_size, node, kAlignment));
    if (list == nullptr) {
      throw std::runtime_error("failed to allocate QAT DP SGL list");
    }
    std::memset(list, 0, list_size);
    list->numBuffers = count;
    segments.resize(count, nullptr);

    uint32_t remaining = total_len;
    for (uint32_t idx = 0; idx < count; ++idx) {
      uint32_t len = remaining > kMaxQaeAllocBytes ? kMaxQaeAllocBytes : remaining;
      if (len == 0) {
        len = kAlignment;
      }
      const uint32_t capacity = static_cast<uint32_t>(align_up(len, kAlignment));
      segments[idx] = static_cast<uint8_t*>(
          qaeMemAllocNUMA(capacity, node, kAlignment));
      if (segments[idx] == nullptr) {
        throw std::runtime_error("failed to allocate QAT DP SGL segment");
      }
      list->flatBuffers[idx].dataLenInBytes = len;
      list->flatBuffers[idx].bufferPhysAddr = virt_to_phys(segments[idx]);
      remaining = remaining > len ? remaining - len : 0;
    }
  }

  void copy_from_host(const uint8_t* src, uint32_t len) {
    uint32_t copied = 0;
    for (uint32_t idx = 0; idx < segments.size() && copied < len; ++idx) {
      uint32_t seg_len = list->flatBuffers[idx].dataLenInBytes;
      if (seg_len > len - copied) {
        seg_len = len - copied;
      }
      std::memcpy(segments[idx], src + copied, seg_len);
      copied += seg_len;
    }
  }

  void copy_to_host(uint8_t* dst, uint32_t len) const {
    uint32_t copied = 0;
    for (uint32_t idx = 0; idx < segments.size() && copied < len; ++idx) {
      uint32_t seg_len = list->flatBuffers[idx].dataLenInBytes;
      if (seg_len > len - copied) {
        seg_len = len - copied;
      }
      std::memcpy(dst + copied, segments[idx], seg_len);
      copied += seg_len;
    }
  }
};

struct DmaChunk {
  uint32_t input_len = 0;
  uint32_t output_len = 0;
  uint64_t input_offset = 0;
  uint64_t output_offset = 0;
  uint8_t* data = nullptr;
  uint32_t data_capacity = 0;
  std::unique_ptr<DpSgl> sgl;

  DmaChunk() = default;
  DmaChunk(const DmaChunk&) = delete;
  DmaChunk& operator=(const DmaChunk&) = delete;

  ~DmaChunk() {
    if (data != nullptr) {
      qaeMemFreeNUMA(reinterpret_cast<void**>(&data));
    }
  }
};

struct QatCtx {
  CpaInstanceHandle instance = nullptr;
  CpaDcSessionHandle session_static = nullptr;
  CpaDcSessionHandle session_dynamic = nullptr;
  int node = 0;
  bool is_polled = false;
};

struct Slot {
  CpaDcDpOpData* op = nullptr;
  uint8_t* dst = nullptr;
  uint32_t dst_capacity = 0;
  std::unique_ptr<DpSgl> dst_sgl;
  std::atomic<int> done{1};
  uint32_t expected_output = 0;
  uint64_t chunk_index = 0;
  bool failed = false;

  Slot() = default;
  Slot(const Slot&) = delete;
  Slot& operator=(const Slot&) = delete;

  ~Slot() {
    if (op != nullptr) {
      qaeMemFreeNUMA(reinterpret_cast<void**>(&op));
    }
    if (dst != nullptr) {
      qaeMemFreeNUMA(reinterpret_cast<void**>(&dst));
    }
  }
};

struct PreparedChunk {
  uint8_t* data = nullptr;
  uint32_t data_capacity = 0;
  uint32_t input_len = 0;
  uint32_t output_len = 0;
  uint64_t output_offset = 0;

  PreparedChunk() = default;
  PreparedChunk(const PreparedChunk&) = delete;
  PreparedChunk& operator=(const PreparedChunk&) = delete;

  PreparedChunk(PreparedChunk&& other) noexcept
      : data(other.data),
        data_capacity(other.data_capacity),
        input_len(other.input_len),
        output_len(other.output_len),
        output_offset(other.output_offset) {
    other.data = nullptr;
    other.data_capacity = 0;
    other.input_len = 0;
    other.output_len = 0;
    other.output_offset = 0;
  }

  PreparedChunk& operator=(PreparedChunk&& other) noexcept {
    if (this != &other) {
      reset();
      data = other.data;
      data_capacity = other.data_capacity;
      input_len = other.input_len;
      output_len = other.output_len;
      output_offset = other.output_offset;
      other.data = nullptr;
      other.data_capacity = 0;
      other.input_len = 0;
      other.output_len = 0;
      other.output_offset = 0;
    }
    return *this;
  }

  ~PreparedChunk() {
    reset();
  }

  void reset() {
    if (data != nullptr) {
      qaeMemFreeNUMA(reinterpret_cast<void**>(&data));
      data = nullptr;
    }
    data_capacity = 0;
  }
};

struct QatPreparedPayload {
  std::vector<PreparedChunk> chunks;
  int64_t output_bytes = 0;
  int64_t chunk_bytes = 0;
  int64_t compressed_bytes = 0;
  bool dynamic_huffman = true;

  int64_t chunk_count() const {
    return static_cast<int64_t>(chunks.size());
  }
};

void dp_callback(CpaDcDpOpData* op_data) {
  auto* slot = static_cast<Slot*>(op_data->pCallbackTag);
  if (slot == nullptr) {
    return;
  }
  if (op_data->responseStatus != CPA_STATUS_SUCCESS ||
      op_data->results.status != CPA_DC_OK ||
      (slot->expected_output > 0 && op_data->results.produced != slot->expected_output)) {
    slot->failed = true;
  }
  slot->done.store(1, std::memory_order_release);
}

CpaDcSessionHandle setup_session(CpaInstanceHandle instance, int node, CpaDcHuffType huff_type) {
  CpaDcSessionSetupData sd{};
  sd.compLevel = CPA_DC_L1;
  sd.compType = CPA_DC_DEFLATE;
  sd.huffType = huff_type;
  sd.autoSelectBestHuffmanTree = CPA_DC_ASB_DISABLED;
  sd.sessDirection = CPA_DC_DIR_COMBINED;
  sd.sessState = CPA_DC_STATELESS;
  sd.windowSize = CPA_DC_WINSIZE_32K;
  sd.minMatch = CPA_DC_MIN_3_BYTE_MATCH;
  sd.checksum = CPA_DC_NONE;

  Cpa32U session_size = 0;
  require_qat(cpaDcDpGetSessionSize(instance, &sd, &session_size),
              "cpaDcDpGetSessionSize");
  auto session = static_cast<CpaDcSessionHandle>(
      qaeMemAllocNUMA(session_size, node, kAlignment));
  if (session == nullptr) {
    throw std::runtime_error("failed to allocate QAT DP session");
  }
  try {
    require_qat(cpaDcDpInitSession(instance, session, &sd), "cpaDcDpInitSession");
  } catch (...) {
    qaeMemFreeNUMA(reinterpret_cast<void**>(&session));
    throw;
  }
  return session;
}

class QatDpState {
 public:
  QatDpState() {
    require_qat(qaeMemInit(), "qaeMemInit");
    try {
      require_qat(icp_sal_userStartMultiProcess("SSL", CPA_FALSE),
                  "icp_sal_userStartMultiProcess");
      Cpa16U n_instances = 0;
      require_qat(cpaDcGetNumInstances(&n_instances), "cpaDcGetNumInstances");
      if (n_instances == 0) {
        throw std::runtime_error("QAT data compression has no instances");
      }
      std::vector<CpaInstanceHandle> handles(n_instances);
      require_qat(cpaDcGetInstances(n_instances, handles.data()), "cpaDcGetInstances");
      for (Cpa16U idx = 0; idx < n_instances; ++idx) {
        CpaInstanceInfo2 info{};
        if (cpaDcInstanceGetInfo2(handles[idx], &info) != CPA_STATUS_SUCCESS ||
            info.isOffloaded != CPA_TRUE) {
          continue;
        }
        CpaDcInstanceCapabilities cap{};
        if (cpaDcQueryCapabilities(handles[idx], &cap) != CPA_STATUS_SUCCESS ||
            !cap.statelessDeflateCompression ||
            !cap.statelessDeflateDecompression) {
          continue;
        }
        require_qat(cpaDcSetAddressTranslation(
                        handles[idx],
                        reinterpret_cast<CpaVirtualToPhysical>(qaeVirtToPhysNUMA)),
                    "cpaDcSetAddressTranslation");
        require_qat(cpaDcStartInstance(handles[idx], 0, nullptr), "cpaDcStartInstance");
        auto ctx = std::make_unique<QatCtx>();
        ctx->instance = handles[idx];
        ctx->node = static_cast<int>(info.nodeAffinity);
        ctx->is_polled = (info.isPolled == CPA_TRUE);
        ctx->session_static = setup_session(handles[idx], ctx->node, CPA_DC_HT_STATIC);
        ctx->session_dynamic = setup_session(handles[idx], ctx->node, CPA_DC_HT_FULL_DYNAMIC);
        require_qat(cpaDcDpRegCbFunc(handles[idx], dp_callback), "cpaDcDpRegCbFunc");
        contexts_.push_back(std::move(ctx));
      }
      if (contexts_.empty()) {
        throw std::runtime_error("QAT data compression has no usable offloaded instances");
      }
    } catch (...) {
      cleanup();
      qaeMemDestroy();
      throw;
    }
  }

  QatDpState(const QatDpState&) = delete;
  QatDpState& operator=(const QatDpState&) = delete;

  ~QatDpState() {
    cleanup();
    qaeMemDestroy();
  }

  size_t size(size_t max_instances) const {
    if (max_instances > 0 && max_instances < contexts_.size()) {
      return max_instances;
    }
    return contexts_.size();
  }

  QatCtx& ctx(size_t idx) {
    return *contexts_.at(idx);
  }

  std::vector<int64_t> nodes(size_t max_instances) const {
    const size_t n = size(max_instances);
    std::vector<int64_t> out;
    out.reserve(n);
    for (size_t idx = 0; idx < n; ++idx) {
      out.push_back(static_cast<int64_t>(contexts_.at(idx)->node));
    }
    return out;
  }

 private:
  void cleanup() {
    for (auto& ctx : contexts_) {
      if (ctx->session_static != nullptr) {
        cpaDcDpRemoveSession(ctx->instance, ctx->session_static);
        qaeMemFreeNUMA(reinterpret_cast<void**>(&ctx->session_static));
      }
      if (ctx->session_dynamic != nullptr) {
        cpaDcDpRemoveSession(ctx->instance, ctx->session_dynamic);
        qaeMemFreeNUMA(reinterpret_cast<void**>(&ctx->session_dynamic));
      }
      if (ctx->instance != nullptr) {
        cpaDcStopInstance(ctx->instance);
      }
    }
    contexts_.clear();
    icp_sal_userStop();
  }

  std::vector<std::unique_ptr<QatCtx>> contexts_;
};

QatDpState& qat_state() {
  static QatDpState state;
  return state;
}

uint32_t compressed_bound(QatCtx& ctx, uint32_t input_len, CpaDcHuffType huff_type) {
  Cpa32U bound = 0;
  const CpaStatus status =
      cpaDcDeflateCompressBound(ctx.instance, huff_type, input_len, &bound);
  if (status != CPA_STATUS_SUCCESS) {
    return input_len + 4096U;
  }
  return bound + 128U;
}

void alloc_chunk_input(QatCtx& ctx, DmaChunk& chunk, const uint8_t* src, uint32_t len) {
  chunk.input_len = len;
  if (len >= kMaxQaeAllocBytes) {
    chunk.sgl = std::make_unique<DpSgl>();
    chunk.sgl->allocate(ctx.node, len);
    chunk.sgl->copy_from_host(src, len);
    return;
  }
  chunk.data_capacity = static_cast<uint32_t>(align_up(len, kAlignment));
  chunk.data = static_cast<uint8_t*>(
      qaeMemAllocNUMA(chunk.data_capacity, ctx.node, kAlignment));
  if (chunk.data == nullptr) {
    throw std::runtime_error("failed to allocate QAT DP input chunk");
  }
  std::memcpy(chunk.data, src, len);
}

void alloc_prepared_chunk(
    QatCtx& ctx,
    PreparedChunk& chunk,
    uint32_t input_len,
    uint32_t output_len,
    uint64_t output_offset) {
  TORCH_CHECK(input_len < kMaxQaeAllocBytes,
              "prepared QAT payload only supports chunks below 64 MiB");
  chunk.input_len = input_len;
  chunk.output_len = output_len;
  chunk.output_offset = output_offset;
  chunk.data_capacity = static_cast<uint32_t>(
      align_up(std::max<uint32_t>(input_len, 1U), kAlignment));
  chunk.data = static_cast<uint8_t*>(
      qaeMemAllocNUMA(chunk.data_capacity, ctx.node, kAlignment));
  if (chunk.data == nullptr) {
    throw std::runtime_error("failed to allocate prepared QAT chunk");
  }
}

void read_exact_fd(int fd, uint8_t* dst, uint32_t bytes) {
  uint32_t done = 0;
  while (done < bytes) {
    const ssize_t n = ::read(fd, dst + done, static_cast<size_t>(bytes - done));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("read failed with errno " + std::to_string(errno));
    }
    if (n == 0) {
      throw std::runtime_error("unexpected EOF while reading QAT payload");
    }
    done += static_cast<uint32_t>(n);
  }
}

void pread_exact_fd(int fd, uint8_t* dst, uint32_t bytes, int64_t offset) {
  uint32_t done = 0;
  while (done < bytes) {
    const ssize_t n = ::pread(
        fd,
        dst + done,
        static_cast<size_t>(bytes - done),
        static_cast<off_t>(offset + static_cast<int64_t>(done)));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("pread failed with errno " + std::to_string(errno));
    }
    if (n == 0) {
      throw std::runtime_error("unexpected EOF while reading QAT payload");
    }
    done += static_cast<uint32_t>(n);
  }
}

void alloc_slot(QatCtx& ctx, Slot& slot, uint32_t dst_capacity) {
  slot.op = static_cast<CpaDcDpOpData*>(
      qaeMemAllocNUMA(sizeof(CpaDcDpOpData), ctx.node, kAlignment));
  if (slot.op == nullptr) {
    throw std::runtime_error("failed to allocate QAT DP op");
  }
  std::memset(slot.op, 0, sizeof(CpaDcDpOpData));
  slot.dst_capacity = static_cast<uint32_t>(
      align_up(std::max(dst_capacity, kMinDcDestBytes), kAlignment));
  if (slot.dst_capacity >= kMaxQaeAllocBytes) {
    slot.dst_sgl = std::make_unique<DpSgl>();
    slot.dst_sgl->allocate(ctx.node, slot.dst_capacity);
    return;
  }
  slot.dst = static_cast<uint8_t*>(
      qaeMemAllocNUMA(slot.dst_capacity, ctx.node, kAlignment));
  if (slot.dst == nullptr) {
    throw std::runtime_error("failed to allocate QAT DP destination");
  }
}

void alloc_op_only_slot(QatCtx& ctx, Slot& slot) {
  slot.op = static_cast<CpaDcDpOpData*>(
      qaeMemAllocNUMA(sizeof(CpaDcDpOpData), ctx.node, kAlignment));
  if (slot.op == nullptr) {
    throw std::runtime_error("failed to allocate QAT DP op");
  }
  std::memset(slot.op, 0, sizeof(CpaDcDpOpData));
}

void reset_slot(Slot& slot) {
  slot.failed = false;
  slot.done.store(0, std::memory_order_release);
  std::memset(&slot.op->results, 0, sizeof(slot.op->results));
  slot.op->results.status = CPA_DC_OK;
  slot.op->responseStatus = CPA_STATUS_FAIL;
}

void prepare_compress_slot_to_prepared_chunk(
    QatCtx& ctx,
    Slot& slot,
    const DmaChunk& src_chunk,
    PreparedChunk& dst_chunk,
    CpaDcSessionHandle session) {
  reset_slot(slot);
  slot.expected_output = 0U;
  CpaDcDpOpData* op = slot.op;
  op->bufferLenToCompress = src_chunk.input_len;
  op->bufferLenForData = dst_chunk.data_capacity;
  op->dcInstance = ctx.instance;
  op->pSessionHandle = session;
  if (src_chunk.sgl) {
    op->srcBuffer = virt_to_phys(src_chunk.sgl->list);
    op->srcBufferLen = CPA_DP_BUFLIST;
  } else {
    op->srcBuffer = virt_to_phys(src_chunk.data);
    op->srcBufferLen = src_chunk.input_len;
  }
  op->destBuffer = virt_to_phys(dst_chunk.data);
  op->destBufferLen = dst_chunk.data_capacity;
  op->sessDirection = CPA_DC_DIR_COMPRESS;
  op->compressAndVerify = CPA_TRUE;
  op->compressAndVerifyAndRecover = CPA_FALSE;
  op->thisPhys = virt_to_phys(op);
  op->pCallbackTag = &slot;
  op->pSetupData = nullptr;
}

void prepare_slot(
    QatCtx& ctx,
    Slot& slot,
    const DmaChunk& chunk,
    CpaDcSessionHandle session,
    CpaDcSessionDir direction,
    bool compress) {
  reset_slot(slot);
  slot.expected_output = compress ? 0U : chunk.output_len;
  CpaDcDpOpData* op = slot.op;
  op->bufferLenToCompress = chunk.input_len;
  op->dcInstance = ctx.instance;
  op->pSessionHandle = session;
  if (chunk.sgl) {
    op->srcBuffer = virt_to_phys(chunk.sgl->list);
    op->srcBufferLen = CPA_DP_BUFLIST;
  } else {
    op->srcBuffer = virt_to_phys(chunk.data);
    op->srcBufferLen = chunk.input_len;
  }
  if (slot.dst_sgl) {
    op->bufferLenForData = slot.dst_capacity;
    op->destBuffer = virt_to_phys(slot.dst_sgl->list);
    op->destBufferLen = CPA_DP_BUFLIST;
  } else {
    op->bufferLenForData = slot.dst_capacity;
    op->destBuffer = virt_to_phys(slot.dst);
    op->destBufferLen = slot.dst_capacity;
  }
  op->sessDirection = direction;
  op->compressAndVerify = compress ? CPA_TRUE : CPA_FALSE;
  op->compressAndVerifyAndRecover = CPA_FALSE;
  op->thisPhys = virt_to_phys(op);
  op->pCallbackTag = &slot;
  op->pSetupData = nullptr;
}

void prepare_slot_from_prepared_chunk(
    QatCtx& ctx,
    Slot& slot,
    const PreparedChunk& chunk,
    CpaDcSessionHandle session,
    CpaDcSessionDir direction) {
  reset_slot(slot);
  slot.expected_output = chunk.output_len;
  CpaDcDpOpData* op = slot.op;
  op->bufferLenToCompress = chunk.input_len;
  op->bufferLenForData = slot.dst_capacity;
  op->dcInstance = ctx.instance;
  op->pSessionHandle = session;
  op->srcBuffer = virt_to_phys(chunk.data);
  op->srcBufferLen = chunk.input_len;
  if (slot.dst_sgl) {
    op->destBuffer = virt_to_phys(slot.dst_sgl->list);
    op->destBufferLen = CPA_DP_BUFLIST;
  } else {
    op->destBuffer = virt_to_phys(slot.dst);
    op->destBufferLen = slot.dst_capacity;
  }
  op->sessDirection = direction;
  op->compressAndVerify = CPA_FALSE;
  op->compressAndVerifyAndRecover = CPA_FALSE;
  op->thisPhys = virt_to_phys(op);
  op->pCallbackTag = &slot;
  op->pSetupData = nullptr;
}

void poll_until_done(QatCtx& ctx, Slot& slot) {
  uint32_t idle_spins = 0;
  while (slot.done.load(std::memory_order_acquire) == 0) {
    if (ctx.is_polled) {
      const CpaStatus status = icp_sal_DcPollDpInstance(ctx.instance, 0);
      if (status == CPA_STATUS_SUCCESS) {
        idle_spins = 0;
        continue;
      }
      if (status != CPA_STATUS_RETRY) {
        throw_qat(status, "icp_sal_DcPollDpInstance");
      }
    }
    if (idle_spins++ < 4096U) {
      cpu_relax();
    } else {
      std::this_thread::yield();
      idle_spins = 0;
    }
  }
  if (slot.failed) {
    throw std::runtime_error(
        "QAT DP request failed response=" + std::to_string(slot.op->responseStatus) +
        " result=" + std::to_string(slot.op->results.status) +
        " produced=" + std::to_string(slot.op->results.produced) +
        " expected=" + std::to_string(slot.expected_output));
  }
}

void submit_batch(QatCtx& ctx, std::vector<CpaDcDpOpData*>& batch) {
  CpaStatus status = CPA_STATUS_RETRY;
  while (status == CPA_STATUS_RETRY) {
    if (batch.size() == 1) {
      status = cpaDcDpEnqueueOp(batch[0], CPA_TRUE);
    } else {
      status = cpaDcDpEnqueueOpBatch(
          static_cast<Cpa32U>(batch.size()), batch.data(), CPA_TRUE);
    }
    if (status == CPA_STATUS_RETRY) {
      if (ctx.is_polled) {
        icp_sal_DcPollDpInstance(ctx.instance, 0);
      }
      cpu_relax();
    }
  }
  require_qat(status, "cpaDcDpEnqueueOpBatch");
}

uint32_t worker_count_for(size_t ctx_count, int64_t max_instances, int64_t chunks) {
  size_t count = ctx_count;
  if (max_instances > 0 && static_cast<size_t>(max_instances) < count) {
    count = static_cast<size_t>(max_instances);
  }
  if (chunks > 0 && static_cast<size_t>(chunks) < count) {
    count = static_cast<size_t>(chunks);
  }
  return static_cast<uint32_t>(count == 0 ? 1 : count);
}

uint64_t worker_chunk_index(uint32_t worker_id, uint32_t worker_count, uint64_t ordinal) {
  return static_cast<uint64_t>(worker_id) + ordinal * static_cast<uint64_t>(worker_count);
}

void copy_slot_output_to_vector(const Slot& slot, std::vector<uint8_t>& output, uint32_t bytes) {
  output.resize(bytes);
  if (slot.dst_sgl) {
    slot.dst_sgl->copy_to_host(output.data(), bytes);
  } else {
    std::memcpy(output.data(), slot.dst, bytes);
  }
}

void copy_slot_output_to_tensor(const Slot& slot, uint8_t* output, uint64_t offset, uint32_t bytes) {
  if (slot.dst_sgl) {
    slot.dst_sgl->copy_to_host(output + offset, bytes);
  } else {
    std::memcpy(output + offset, slot.dst, bytes);
  }
}

template <typename scalar_t>
void dequant4_bytes_to_coeffs_typed(
    const uint8_t* packed,
    uint32_t packed_bytes,
    uint64_t output_byte_offset,
    int64_t symbol_count,
    int64_t rank,
    const float* scale,
    const float* offset,
    scalar_t* coeffs) {
  const uint64_t base_symbol = output_byte_offset * 2ULL;
  const uint64_t total_symbols = static_cast<uint64_t>(symbol_count);
  for (uint32_t byte_idx = 0; byte_idx < packed_bytes; ++byte_idx) {
    const uint64_t symbol0 = base_symbol + static_cast<uint64_t>(byte_idx) * 2ULL;
    if (symbol0 >= total_symbols) {
      break;
    }
    const uint8_t byte = packed[byte_idx];
    const int64_t col0 = static_cast<int64_t>(symbol0 % static_cast<uint64_t>(rank));
    const float value0 = static_cast<float>(byte & 0x0FU) * scale[col0] + offset[col0];
    coeffs[symbol0] = static_cast<scalar_t>(value0);
    const uint64_t symbol1 = symbol0 + 1ULL;
    if (symbol1 < total_symbols) {
      const int64_t col1 = static_cast<int64_t>(symbol1 % static_cast<uint64_t>(rank));
      const float value1 = static_cast<float>((byte >> 4) & 0x0FU) * scale[col1] + offset[col1];
      coeffs[symbol1] = static_cast<scalar_t>(value1);
    }
  }
}

void dequant4_slot_output_to_coeffs(
    const Slot& slot,
    const PreparedChunk& chunk,
    int64_t symbol_count,
    int64_t rank,
    const float* scale,
    const float* offset,
    torch::Tensor& coeffs) {
  TORCH_CHECK(!slot.dst_sgl, "fused QAT dequant does not support SGL destination");
  if (coeffs.scalar_type() == torch::kFloat32) {
    dequant4_bytes_to_coeffs_typed<float>(
        slot.dst,
        chunk.output_len,
        chunk.output_offset,
        symbol_count,
        rank,
        scale,
        offset,
        coeffs.data_ptr<float>());
  } else if (coeffs.scalar_type() == torch::kBFloat16) {
    dequant4_bytes_to_coeffs_typed<at::BFloat16>(
        slot.dst,
        chunk.output_len,
        chunk.output_offset,
        symbol_count,
        rank,
        scale,
        offset,
        coeffs.data_ptr<at::BFloat16>());
  } else {
    TORCH_CHECK(false, "coefficients output dtype must be float32 or bfloat16");
  }
}

template <typename scalar_t>
void dequant4_slot_output_to_coeff_tile_typed(
    const Slot& slot,
    const PreparedChunk& chunk,
    int64_t rank,
    const float* scale,
    const float* offset,
    int64_t tile_row_offset,
    scalar_t* coeffs) {
  TORCH_CHECK(rank % 2 == 0, "fused QAT project requires even 4-bit rank");
  const int64_t row_packed_bytes = rank / 2;
  TORCH_CHECK(chunk.output_offset % static_cast<uint64_t>(row_packed_bytes) == 0,
              "fused QAT project requires row-aligned chunk offsets");
  TORCH_CHECK(chunk.output_len % static_cast<uint32_t>(row_packed_bytes) == 0,
              "fused QAT project requires row-aligned chunk lengths");
  const int64_t rows = static_cast<int64_t>(chunk.output_len) / row_packed_bytes;
  dequant4_bytes_to_coeffs_typed<scalar_t>(
      slot.dst,
      chunk.output_len,
      0,
      rows * rank,
      rank,
      scale,
      offset,
      coeffs + tile_row_offset * rank);
}

void dequant4_slot_output_to_coeff_tile(
    const Slot& slot,
    const PreparedChunk& chunk,
    int64_t rank,
    const float* scale,
    const float* offset,
    int64_t tile_row_offset,
    torch::Tensor& coeff_tile) {
  TORCH_CHECK(!slot.dst_sgl, "fused QAT project does not support SGL destination");
  if (coeff_tile.scalar_type() == torch::kFloat32) {
    dequant4_slot_output_to_coeff_tile_typed<float>(
        slot,
        chunk,
        rank,
        scale,
        offset,
        tile_row_offset,
        coeff_tile.data_ptr<float>());
  } else if (coeff_tile.scalar_type() == torch::kBFloat16) {
    dequant4_slot_output_to_coeff_tile_typed<at::BFloat16>(
        slot,
        chunk,
        rank,
        scale,
        offset,
        tile_row_offset,
        coeff_tile.data_ptr<at::BFloat16>());
  } else {
    TORCH_CHECK(false, "coefficient tile dtype must be float32 or bfloat16");
  }
}

void project_coeff_tile_to_matrix(
    torch::Tensor& coeff_tile,
    int64_t tile_rows,
    int64_t tile_start_row,
    const torch::Tensor& basis_t,
    const torch::Tensor& mean,
    torch::Tensor& matrix) {
  if (tile_rows <= 0) {
    return;
  }
  auto coeff_view = coeff_tile.narrow(0, 0, tile_rows);
  auto projected = at::matmul(coeff_view, basis_t);
  projected.add_(mean);
  matrix.narrow(0, tile_start_row, tile_rows).copy_(projected);
}

}  // namespace

std::vector<torch::Tensor> qat_deflate_compress_dp(
    torch::Tensor input,
    int64_t chunk_bytes,
    bool dynamic_huffman,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  check_cpu_u8(input, "input");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= std::numeric_limits<Cpa32U>::max(),
              "chunk_bytes must fit in a QAT request");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");

  auto& state = qat_state();
  const int64_t input_bytes = input.numel();
  const int64_t chunks = (input_bytes + chunk_bytes - 1) / chunk_bytes;
  auto lengths = torch::empty({chunks}, torch::dtype(torch::kInt32));
  auto* lengths_ptr = lengths.data_ptr<int32_t>();
  const auto* input_ptr = input.data_ptr<uint8_t>();
  std::vector<std::vector<uint8_t>> compressed(static_cast<size_t>(chunks));
  if (chunks == 0) {
    return {torch::empty({0}, torch::dtype(torch::kUInt8)), lengths};
  }

  const uint32_t workers = worker_count_for(state.size(max_instances), max_instances, chunks);
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);
  const CpaDcHuffType huff_type =
      dynamic_huffman ? CPA_DC_HT_FULL_DYNAMIC : CPA_DC_HT_STATIC;

  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker]() {
      try {
        QatCtx& ctx = state.ctx(worker);
        pin_qat_worker_thread(worker, ctx.node);
        const CpaDcSessionHandle session =
            dynamic_huffman ? ctx.session_dynamic : ctx.session_static;
        std::vector<std::unique_ptr<DmaChunk>> worker_chunks;
        for (uint64_t ordinal = 0;; ++ordinal) {
          const uint64_t chunk_idx = worker_chunk_index(worker, workers, ordinal);
          if (chunk_idx >= static_cast<uint64_t>(chunks)) {
            break;
          }
          const int64_t off = static_cast<int64_t>(chunk_idx) * chunk_bytes;
          const uint32_t len =
              static_cast<uint32_t>(std::min<int64_t>(chunk_bytes, input_bytes - off));
          auto chunk = std::make_unique<DmaChunk>();
          chunk->input_offset = static_cast<uint64_t>(off);
          alloc_chunk_input(ctx, *chunk, input_ptr + off, len);
          chunk->output_len = compressed_bound(ctx, len, huff_type);
          worker_chunks.push_back(std::move(chunk));
        }

        const uint32_t slot_count = static_cast<uint32_t>(
            std::min<int64_t>(inflight, std::max<int64_t>(1, worker_chunks.size())));
        std::vector<std::unique_ptr<Slot>> slots;
        slots.reserve(slot_count);
        uint32_t max_bound = 0;
        for (const auto& chunk : worker_chunks) {
          max_bound = std::max(max_bound, chunk->output_len);
        }
        for (uint32_t idx = 0; idx < slot_count; ++idx) {
          auto slot = std::make_unique<Slot>();
          alloc_slot(ctx, *slot, max_bound);
          slots.push_back(std::move(slot));
        }

        std::vector<CpaDcDpOpData*> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t next = 0;
        while (true) {
          while (submitted - completed >= slot_count) {
            Slot& slot = *slots[completed % slot_count];
            poll_until_done(ctx, slot);
            const uint32_t produced = slot.op->results.produced;
            lengths_ptr[slot.chunk_index] = static_cast<int32_t>(produced);
            copy_slot_output_to_vector(slot, compressed[slot.chunk_index], produced);
            completed++;
          }
          batch.clear();
          while (batch.size() < static_cast<size_t>(batch_size) &&
                 submitted - completed + batch.size() < slot_count &&
                 next < worker_chunks.size()) {
            Slot& slot = *slots[(submitted + batch.size()) % slot_count];
            slot.chunk_index = worker_chunk_index(worker, workers, next);
            prepare_slot(
                ctx,
                slot,
                *worker_chunks[next],
                session,
                CPA_DC_DIR_COMPRESS,
                true);
            batch.push_back(slot.op);
            next++;
          }
          if (batch.empty()) {
            break;
          }
          submit_batch(ctx, batch);
          submitted += batch.size();
        }
        while (completed < submitted) {
          Slot& slot = *slots[completed % slot_count];
          poll_until_done(ctx, slot);
          const uint32_t produced = slot.op->results.produced;
          lengths_ptr[slot.chunk_index] = static_cast<int32_t>(produced);
          copy_slot_output_to_vector(slot, compressed[slot.chunk_index], produced);
          completed++;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }

  int64_t total = 0;
  for (int64_t idx = 0; idx < chunks; ++idx) {
    total += lengths_ptr[idx];
  }
  auto bytestream = torch::empty({total}, torch::dtype(torch::kUInt8));
  auto* out = bytestream.data_ptr<uint8_t>();
  int64_t cursor = 0;
  for (int64_t idx = 0; idx < chunks; ++idx) {
    const auto& data = compressed[static_cast<size_t>(idx)];
    std::memcpy(out + cursor, data.data(), data.size());
    cursor += static_cast<int64_t>(data.size());
  }
  return {bytestream, lengths};
}

std::tuple<torch::Tensor, torch::Tensor, std::shared_ptr<QatPreparedPayload>>
qat_deflate_compress_prepare_dp(
    torch::Tensor input,
    int64_t chunk_bytes,
    bool dynamic_huffman,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  check_cpu_u8(input, "input");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= std::numeric_limits<Cpa32U>::max(),
              "chunk_bytes must fit in a QAT request");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");

  auto& state = qat_state();
  const int64_t input_bytes = input.numel();
  const int64_t chunks = (input_bytes + chunk_bytes - 1) / chunk_bytes;
  auto lengths = torch::empty({chunks}, torch::dtype(torch::kInt32));
  auto* lengths_ptr = lengths.data_ptr<int32_t>();
  const auto* input_ptr = input.data_ptr<uint8_t>();
  auto payload = std::make_shared<QatPreparedPayload>();
  payload->output_bytes = input_bytes;
  payload->chunk_bytes = chunk_bytes;
  payload->dynamic_huffman = dynamic_huffman;
  payload->chunks.resize(static_cast<size_t>(chunks));
  if (chunks == 0) {
    return std::make_tuple(
        torch::empty({0}, torch::dtype(torch::kUInt8)),
        lengths,
        payload);
  }

  const uint32_t workers = worker_count_for(state.size(max_instances), max_instances, chunks);
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);
  const CpaDcHuffType huff_type =
      dynamic_huffman ? CPA_DC_HT_FULL_DYNAMIC : CPA_DC_HT_STATIC;

  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker]() {
      try {
        QatCtx& ctx = state.ctx(worker);
        pin_qat_worker_thread(worker, ctx.node);
        const CpaDcSessionHandle session =
            dynamic_huffman ? ctx.session_dynamic : ctx.session_static;
        std::vector<std::unique_ptr<DmaChunk>> worker_chunks;
        std::vector<uint64_t> chunk_indices;
        for (uint64_t ordinal = 0;; ++ordinal) {
          const uint64_t chunk_idx = worker_chunk_index(worker, workers, ordinal);
          if (chunk_idx >= static_cast<uint64_t>(chunks)) {
            break;
          }
          const int64_t off = static_cast<int64_t>(chunk_idx) * chunk_bytes;
          const uint32_t len =
              static_cast<uint32_t>(std::min<int64_t>(chunk_bytes, input_bytes - off));
          auto chunk = std::make_unique<DmaChunk>();
          chunk->input_offset = static_cast<uint64_t>(off);
          alloc_chunk_input(ctx, *chunk, input_ptr + off, len);

          PreparedChunk& prepared = payload->chunks[static_cast<size_t>(chunk_idx)];
          prepared.output_len = len;
          prepared.output_offset = static_cast<uint64_t>(off);
          const uint32_t bound = compressed_bound(ctx, len, huff_type);
          TORCH_CHECK(bound < kMaxQaeAllocBytes,
                      "prepared compression only supports chunks below 64 MiB");
          prepared.data_capacity = static_cast<uint32_t>(
              align_up(std::max<uint32_t>(bound, kMinDcDestBytes), kAlignment));
          prepared.data = static_cast<uint8_t*>(
              qaeMemAllocNUMA(prepared.data_capacity, ctx.node, kAlignment));
          if (prepared.data == nullptr) {
            throw std::runtime_error("failed to allocate prepared compressed chunk");
          }
          worker_chunks.push_back(std::move(chunk));
          chunk_indices.push_back(chunk_idx);
        }

        const uint32_t slot_count = static_cast<uint32_t>(
            std::min<int64_t>(inflight, std::max<int64_t>(1, worker_chunks.size())));
        std::vector<std::unique_ptr<Slot>> slots;
        slots.reserve(slot_count);
        for (uint32_t idx = 0; idx < slot_count; ++idx) {
          auto slot = std::make_unique<Slot>();
          alloc_op_only_slot(ctx, *slot);
          slots.push_back(std::move(slot));
        }

        std::vector<CpaDcDpOpData*> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t next = 0;
        while (true) {
          while (submitted - completed >= slot_count) {
            Slot& slot = *slots[completed % slot_count];
            poll_until_done(ctx, slot);
            const uint32_t produced = slot.op->results.produced;
            PreparedChunk& prepared = payload->chunks[static_cast<size_t>(slot.chunk_index)];
            prepared.input_len = produced;
            lengths_ptr[slot.chunk_index] = static_cast<int32_t>(produced);
            completed++;
          }
          batch.clear();
          while (batch.size() < static_cast<size_t>(batch_size) &&
                 submitted - completed + batch.size() < slot_count &&
                 next < worker_chunks.size()) {
            Slot& slot = *slots[(submitted + batch.size()) % slot_count];
            slot.chunk_index = chunk_indices[next];
            prepare_compress_slot_to_prepared_chunk(
                ctx,
                slot,
                *worker_chunks[next],
                payload->chunks[static_cast<size_t>(slot.chunk_index)],
                session);
            batch.push_back(slot.op);
            next++;
          }
          if (batch.empty()) {
            break;
          }
          submit_batch(ctx, batch);
          submitted += batch.size();
        }
        while (completed < submitted) {
          Slot& slot = *slots[completed % slot_count];
          poll_until_done(ctx, slot);
          const uint32_t produced = slot.op->results.produced;
          PreparedChunk& prepared = payload->chunks[static_cast<size_t>(slot.chunk_index)];
          prepared.input_len = produced;
          lengths_ptr[slot.chunk_index] = static_cast<int32_t>(produced);
          completed++;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }

  int64_t total = 0;
  for (int64_t idx = 0; idx < chunks; ++idx) {
    total += lengths_ptr[idx];
  }
  payload->compressed_bytes = total;
  auto bytestream = torch::empty({total}, torch::dtype(torch::kUInt8));
  auto* out = bytestream.data_ptr<uint8_t>();
  int64_t cursor = 0;
  for (int64_t idx = 0; idx < chunks; ++idx) {
    const PreparedChunk& chunk = payload->chunks[static_cast<size_t>(idx)];
    std::memcpy(out + cursor, chunk.data, chunk.input_len);
    cursor += static_cast<int64_t>(chunk.input_len);
  }
  return std::make_tuple(bytestream, lengths, payload);
}

torch::Tensor qat_deflate_decompress_dp(
    torch::Tensor bytestream,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes,
    bool dynamic_huffman,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  check_cpu_u8(bytestream, "bytestream");
  check_cpu_i32(lengths, "lengths");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be one-dimensional");
  TORCH_CHECK(output_bytes >= 0, "output_bytes must be non-negative");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= std::numeric_limits<Cpa32U>::max(),
              "chunk_bytes must fit in a QAT request");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");

  const int64_t chunks = lengths.numel();
  TORCH_CHECK(chunks == (output_bytes + chunk_bytes - 1) / chunk_bytes,
              "length count must match output_bytes and chunk_bytes");
  std::vector<int64_t> offsets(static_cast<size_t>(chunks + 1), 0);
  const auto* lengths_ptr = lengths.data_ptr<int32_t>();
  for (int64_t idx = 0; idx < chunks; ++idx) {
    TORCH_CHECK(lengths_ptr[idx] >= 0, "lengths must be non-negative");
    offsets[idx + 1] = offsets[idx] + lengths_ptr[idx];
  }
  TORCH_CHECK(offsets.back() == bytestream.numel(), "lengths must sum to bytestream size");

  if (chunks == 0) {
    return torch::empty({output_bytes}, torch::dtype(torch::kUInt8));
  }
  auto& state = qat_state();
  const uint32_t workers = worker_count_for(state.size(max_instances), max_instances, chunks);
  auto output = torch::empty({output_bytes}, torch::dtype(torch::kUInt8));
  const auto* input_ptr = bytestream.data_ptr<uint8_t>();
  auto* output_ptr = output.data_ptr<uint8_t>();
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);

  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker]() {
      try {
        QatCtx& ctx = state.ctx(worker);
        pin_qat_worker_thread(worker, ctx.node);
        const CpaDcSessionHandle session =
            dynamic_huffman ? ctx.session_dynamic : ctx.session_static;
        std::vector<std::unique_ptr<DmaChunk>> worker_chunks;
        for (uint64_t ordinal = 0;; ++ordinal) {
          const uint64_t chunk_idx = worker_chunk_index(worker, workers, ordinal);
          if (chunk_idx >= static_cast<uint64_t>(chunks)) {
            break;
          }
          const uint32_t in_len = static_cast<uint32_t>(lengths_ptr[chunk_idx]);
          const int64_t out_off = static_cast<int64_t>(chunk_idx) * chunk_bytes;
          const uint32_t out_len =
              static_cast<uint32_t>(std::min<int64_t>(chunk_bytes, output_bytes - out_off));
          auto chunk = std::make_unique<DmaChunk>();
          chunk->input_offset = static_cast<uint64_t>(offsets[chunk_idx]);
          chunk->output_offset = static_cast<uint64_t>(out_off);
          chunk->output_len = out_len;
          alloc_chunk_input(ctx, *chunk, input_ptr + offsets[chunk_idx], in_len);
          worker_chunks.push_back(std::move(chunk));
        }

        const uint32_t slot_count = static_cast<uint32_t>(
            std::min<int64_t>(inflight, std::max<int64_t>(1, worker_chunks.size())));
        std::vector<std::unique_ptr<Slot>> slots;
        slots.reserve(slot_count);
        const uint32_t dst_capacity =
            static_cast<uint32_t>(std::min<int64_t>(chunk_bytes, output_bytes));
        for (uint32_t idx = 0; idx < slot_count; ++idx) {
          auto slot = std::make_unique<Slot>();
          alloc_slot(ctx, *slot, dst_capacity);
          slots.push_back(std::move(slot));
        }

        std::vector<CpaDcDpOpData*> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t next = 0;
        while (true) {
          while (submitted - completed >= slot_count) {
            Slot& slot = *slots[completed % slot_count];
            poll_until_done(ctx, slot);
            const auto& chunk = *worker_chunks[slot.chunk_index];
            copy_slot_output_to_tensor(slot, output_ptr, chunk.output_offset, chunk.output_len);
            completed++;
          }
          batch.clear();
          while (batch.size() < static_cast<size_t>(batch_size) &&
                 submitted - completed + batch.size() < slot_count &&
                 next < worker_chunks.size()) {
            Slot& slot = *slots[(submitted + batch.size()) % slot_count];
            slot.chunk_index = next;
            prepare_slot(
                ctx,
                slot,
                *worker_chunks[next],
                session,
                CPA_DC_DIR_DECOMPRESS,
                false);
            batch.push_back(slot.op);
            next++;
          }
          if (batch.empty()) {
            break;
          }
          submit_batch(ctx, batch);
          submitted += batch.size();
        }
        while (completed < submitted) {
          Slot& slot = *slots[completed % slot_count];
          poll_until_done(ctx, slot);
          const auto& chunk = *worker_chunks[slot.chunk_index];
          copy_slot_output_to_tensor(slot, output_ptr, chunk.output_offset, chunk.output_len);
          completed++;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return output;
}

std::shared_ptr<QatPreparedPayload> qat_deflate_prepare_dp(
    torch::Tensor bytestream,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes,
    bool dynamic_huffman,
    int64_t max_instances) {
  check_cpu_u8(bytestream, "bytestream");
  check_cpu_i32(lengths, "lengths");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be one-dimensional");
  TORCH_CHECK(output_bytes >= 0, "output_bytes must be non-negative");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= std::numeric_limits<Cpa32U>::max(),
              "chunk_bytes must fit in a QAT request");
  const int64_t chunks = lengths.numel();
  TORCH_CHECK(chunks == (output_bytes + chunk_bytes - 1) / chunk_bytes,
              "length count must match output_bytes and chunk_bytes");
  auto& state = qat_state();
  const uint32_t workers = worker_count_for(state.size(max_instances), max_instances, chunks);
  const auto* input_ptr = bytestream.data_ptr<uint8_t>();
  const auto* lengths_ptr = lengths.data_ptr<int32_t>();
  int64_t input_offset = 0;
  auto payload = std::make_shared<QatPreparedPayload>();
  payload->chunks.reserve(static_cast<size_t>(chunks));
  payload->output_bytes = output_bytes;
  payload->chunk_bytes = chunk_bytes;
  payload->dynamic_huffman = dynamic_huffman;
  for (int64_t idx = 0; idx < chunks; ++idx) {
    const int32_t input_len_i32 = lengths_ptr[idx];
    TORCH_CHECK(input_len_i32 >= 0, "lengths must be non-negative");
    const uint32_t input_len = static_cast<uint32_t>(input_len_i32);
    TORCH_CHECK(input_len < kMaxQaeAllocBytes,
                "prepared QAT payload only supports chunks below 64 MiB");
    TORCH_CHECK(input_offset + input_len <= bytestream.numel(),
                "lengths exceed bytestream size");
    const int64_t output_offset = idx * chunk_bytes;
    const uint32_t output_len =
        static_cast<uint32_t>(std::min<int64_t>(chunk_bytes, output_bytes - output_offset));
    QatCtx& ctx = state.ctx(static_cast<size_t>(idx % workers));
    PreparedChunk chunk;
    chunk.input_len = input_len;
    chunk.output_len = output_len;
    chunk.output_offset = static_cast<uint64_t>(output_offset);
    const uint32_t capacity = static_cast<uint32_t>(
        align_up(std::max<uint32_t>(input_len, 1U), kAlignment));
    chunk.data = static_cast<uint8_t*>(
        qaeMemAllocNUMA(capacity, ctx.node, kAlignment));
    if (chunk.data == nullptr) {
      throw std::runtime_error("failed to allocate prepared QAT chunk");
    }
    if (input_len > 0) {
      std::memcpy(chunk.data, input_ptr + input_offset, input_len);
    }
    payload->compressed_bytes += input_len;
    input_offset += input_len;
    payload->chunks.push_back(std::move(chunk));
  }
  TORCH_CHECK(input_offset == bytestream.numel(), "lengths must sum to bytestream size");
  return payload;
}

std::shared_ptr<QatPreparedPayload> qat_deflate_prepare_from_fd_dp(
    int fd,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes,
    bool dynamic_huffman,
    int64_t max_instances,
    int64_t file_offset) {
  check_cpu_i32(lengths, "lengths");
  TORCH_CHECK(fd >= 0, "fd must be non-negative");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be one-dimensional");
  TORCH_CHECK(output_bytes >= 0, "output_bytes must be non-negative");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= std::numeric_limits<Cpa32U>::max(),
              "chunk_bytes must fit in a QAT request");
  const int64_t chunks = lengths.numel();
  TORCH_CHECK(chunks == (output_bytes + chunk_bytes - 1) / chunk_bytes,
              "length count must match output_bytes and chunk_bytes");

  auto& state = qat_state();
  const uint32_t workers = worker_count_for(state.size(max_instances), max_instances, chunks);
  const auto* lengths_ptr = lengths.data_ptr<int32_t>();
  auto payload = std::make_shared<QatPreparedPayload>();
  payload->chunks.reserve(static_cast<size_t>(chunks));
  payload->output_bytes = output_bytes;
  payload->chunk_bytes = chunk_bytes;
  payload->dynamic_huffman = dynamic_huffman;
  int64_t input_offset = 0;
  for (int64_t idx = 0; idx < chunks; ++idx) {
    const int32_t input_len_i32 = lengths_ptr[idx];
    TORCH_CHECK(input_len_i32 >= 0, "lengths must be non-negative");
    const uint32_t input_len = static_cast<uint32_t>(input_len_i32);
    const int64_t output_offset = idx * chunk_bytes;
    const uint32_t output_len =
        static_cast<uint32_t>(std::min<int64_t>(chunk_bytes, output_bytes - output_offset));
    QatCtx& ctx = state.ctx(static_cast<size_t>(idx % workers));
    PreparedChunk chunk;
    alloc_prepared_chunk(
        ctx,
        chunk,
        input_len,
        output_len,
        static_cast<uint64_t>(output_offset));
    if (input_len > 0) {
      if (file_offset >= 0) {
        pread_exact_fd(fd, chunk.data, input_len, file_offset + input_offset);
      } else {
        read_exact_fd(fd, chunk.data, input_len);
      }
    }
    payload->compressed_bytes += input_len;
    input_offset += input_len;
    payload->chunks.push_back(std::move(chunk));
  }
  return payload;
}

std::shared_ptr<QatPreparedPayload> qat_deflate_prepare_from_file_dp(
    std::string path,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes,
    bool dynamic_huffman,
    int64_t max_instances,
    int64_t file_offset) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("open failed with errno " + std::to_string(errno));
  }
  try {
    auto payload = qat_deflate_prepare_from_fd_dp(
        fd,
        lengths,
        output_bytes,
        chunk_bytes,
        dynamic_huffman,
        max_instances,
        file_offset);
    ::close(fd);
    return payload;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

torch::Tensor qat_deflate_decompress_prepared_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");
  if (payload->chunks.empty()) {
    return torch::empty({payload->output_bytes}, torch::dtype(torch::kUInt8));
  }
  auto& state = qat_state();
  const uint32_t workers =
      worker_count_for(state.size(max_instances), max_instances, payload->chunks.size());
  auto output = torch::empty({payload->output_bytes}, torch::dtype(torch::kUInt8));
  auto* output_ptr = output.data_ptr<uint8_t>();
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);

  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker]() {
      try {
        QatCtx& ctx = state.ctx(worker);
        pin_qat_worker_thread(worker, ctx.node);
        const CpaDcSessionHandle session =
            payload->dynamic_huffman ? ctx.session_dynamic : ctx.session_static;
        std::vector<uint64_t> worker_chunks;
        for (uint64_t ordinal = 0;; ++ordinal) {
          const uint64_t chunk_idx = worker_chunk_index(worker, workers, ordinal);
          if (chunk_idx >= payload->chunks.size()) {
            break;
          }
          worker_chunks.push_back(chunk_idx);
        }
        const uint32_t slot_count = static_cast<uint32_t>(
            std::min<int64_t>(inflight, std::max<int64_t>(1, worker_chunks.size())));
        std::vector<std::unique_ptr<Slot>> slots;
        slots.reserve(slot_count);
        const uint32_t dst_capacity = static_cast<uint32_t>(
            std::min<int64_t>(payload->chunk_bytes, payload->output_bytes));
        for (uint32_t idx = 0; idx < slot_count; ++idx) {
          auto slot = std::make_unique<Slot>();
          alloc_slot(ctx, *slot, dst_capacity);
          slots.push_back(std::move(slot));
        }

        std::vector<CpaDcDpOpData*> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t next = 0;
        while (true) {
          while (submitted - completed >= slot_count) {
            Slot& slot = *slots[completed % slot_count];
            poll_until_done(ctx, slot);
            const PreparedChunk& chunk = payload->chunks[slot.chunk_index];
            copy_slot_output_to_tensor(slot, output_ptr, chunk.output_offset, chunk.output_len);
            completed++;
          }
          batch.clear();
          while (batch.size() < static_cast<size_t>(batch_size) &&
                 submitted - completed + batch.size() < slot_count &&
                 next < worker_chunks.size()) {
            Slot& slot = *slots[(submitted + batch.size()) % slot_count];
            slot.chunk_index = worker_chunks[next];
            prepare_slot_from_prepared_chunk(
                ctx,
                slot,
                payload->chunks[slot.chunk_index],
                session,
                CPA_DC_DIR_DECOMPRESS);
            batch.push_back(slot.op);
            next++;
          }
          if (batch.empty()) {
            break;
          }
          submit_batch(ctx, batch);
          submitted += batch.size();
        }
        while (completed < submitted) {
          Slot& slot = *slots[completed % slot_count];
          poll_until_done(ctx, slot);
          const PreparedChunk& chunk = payload->chunks[slot.chunk_index];
          copy_slot_output_to_tensor(slot, output_ptr, chunk.output_offset, chunk.output_len);
          completed++;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return output;
}

torch::Tensor qat_deflate_decompress_dequant4_prepared_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    torch::Tensor scale,
    torch::Tensor offset,
    int64_t token_count,
    int64_t rank,
    std::string dtype_name,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");
  TORCH_CHECK(token_count >= 0 && rank > 0, "token_count/rank are invalid");
  TORCH_CHECK(!scale.is_cuda() && !offset.is_cuda(), "scale/offset must be CPU tensors");
  TORCH_CHECK(scale.is_contiguous() && offset.is_contiguous(), "scale/offset must be contiguous");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
  TORCH_CHECK(offset.scalar_type() == torch::kFloat32, "offset must be float32");
  TORCH_CHECK(scale.numel() == rank && offset.numel() == rank,
              "scale/offset length must match rank");
  const int64_t symbol_count = token_count * rank;
  TORCH_CHECK(payload->output_bytes == (symbol_count + 1) / 2,
              "prepared payload output bytes do not match 4-bit symbol count");
  torch::Dtype dtype = torch::kFloat32;
  if (dtype_name == "bfloat16") {
    dtype = torch::kBFloat16;
  } else if (dtype_name != "float32") {
    TORCH_CHECK(false, "dtype_name must be float32 or bfloat16");
  }
  auto coeffs = torch::empty({token_count, rank}, torch::dtype(dtype));
  if (payload->chunks.empty()) {
    return coeffs;
  }

  auto& state = qat_state();
  const uint32_t workers =
      worker_count_for(state.size(max_instances), max_instances, payload->chunks.size());
  const float* scale_ptr = scale.data_ptr<float>();
  const float* offset_ptr = offset.data_ptr<float>();
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);

  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker]() {
      try {
        QatCtx& ctx = state.ctx(worker);
        pin_qat_worker_thread(worker, ctx.node);
        const CpaDcSessionHandle session =
            payload->dynamic_huffman ? ctx.session_dynamic : ctx.session_static;
        std::vector<uint64_t> worker_chunks;
        for (uint64_t ordinal = 0;; ++ordinal) {
          const uint64_t chunk_idx = worker_chunk_index(worker, workers, ordinal);
          if (chunk_idx >= payload->chunks.size()) {
            break;
          }
          worker_chunks.push_back(chunk_idx);
        }
        const uint32_t slot_count = static_cast<uint32_t>(
            std::min<int64_t>(inflight, std::max<int64_t>(1, worker_chunks.size())));
        std::vector<std::unique_ptr<Slot>> slots;
        slots.reserve(slot_count);
        const uint32_t dst_capacity = static_cast<uint32_t>(
            std::min<int64_t>(payload->chunk_bytes, payload->output_bytes));
        for (uint32_t idx = 0; idx < slot_count; ++idx) {
          auto slot = std::make_unique<Slot>();
          alloc_slot(ctx, *slot, dst_capacity);
          slots.push_back(std::move(slot));
        }

        std::vector<CpaDcDpOpData*> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t next = 0;
        while (true) {
          while (submitted - completed >= slot_count) {
            Slot& slot = *slots[completed % slot_count];
            poll_until_done(ctx, slot);
            const PreparedChunk& chunk = payload->chunks[slot.chunk_index];
            dequant4_slot_output_to_coeffs(
                slot, chunk, symbol_count, rank, scale_ptr, offset_ptr, coeffs);
            completed++;
          }
          batch.clear();
          while (batch.size() < static_cast<size_t>(batch_size) &&
                 submitted - completed + batch.size() < slot_count &&
                 next < worker_chunks.size()) {
            Slot& slot = *slots[(submitted + batch.size()) % slot_count];
            slot.chunk_index = worker_chunks[next];
            prepare_slot_from_prepared_chunk(
                ctx,
                slot,
                payload->chunks[slot.chunk_index],
                session,
                CPA_DC_DIR_DECOMPRESS);
            batch.push_back(slot.op);
            next++;
          }
          if (batch.empty()) {
            break;
          }
          submit_batch(ctx, batch);
          submitted += batch.size();
        }
        while (completed < submitted) {
          Slot& slot = *slots[completed % slot_count];
          poll_until_done(ctx, slot);
          const PreparedChunk& chunk = payload->chunks[slot.chunk_index];
          dequant4_slot_output_to_coeffs(
              slot, chunk, symbol_count, rank, scale_ptr, offset_ptr, coeffs);
          completed++;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return coeffs;
}

torch::Tensor qat_deflate_decompress_project4_prepared_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    torch::Tensor scale,
    torch::Tensor offset,
    torch::Tensor basis_t,
    torch::Tensor mean,
    int64_t token_count,
    int64_t rank,
    std::string dtype_name,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances,
    int64_t tile_tokens) {
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");
  TORCH_CHECK(token_count >= 0 && rank > 0, "token_count/rank are invalid");
  TORCH_CHECK(rank % 2 == 0, "fused QAT project requires even 4-bit rank");
  TORCH_CHECK(tile_tokens > 0, "tile_tokens must be positive");
  TORCH_CHECK(!scale.is_cuda() && !offset.is_cuda(), "scale/offset must be CPU tensors");
  TORCH_CHECK(!basis_t.is_cuda() && !mean.is_cuda(), "basis_t/mean must be CPU tensors");
  TORCH_CHECK(scale.is_contiguous() && offset.is_contiguous(), "scale/offset must be contiguous");
  TORCH_CHECK(basis_t.is_contiguous() && mean.is_contiguous(), "basis_t/mean must be contiguous");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
  TORCH_CHECK(offset.scalar_type() == torch::kFloat32, "offset must be float32");
  TORCH_CHECK(scale.numel() == rank && offset.numel() == rank,
              "scale/offset length must match rank");
  TORCH_CHECK(basis_t.dim() == 2, "basis_t must be a matrix");
  TORCH_CHECK(basis_t.size(0) == rank, "basis_t rank dimension must match rank");
  TORCH_CHECK(mean.dim() == 1 && mean.numel() == basis_t.size(1),
              "mean length must match basis_t features");
  const int64_t symbol_count = token_count * rank;
  TORCH_CHECK(payload->output_bytes == symbol_count / 2,
              "prepared payload output bytes do not match even-rank 4-bit symbol count");

  torch::Dtype dtype = torch::kFloat32;
  if (dtype_name == "bfloat16") {
    dtype = torch::kBFloat16;
  } else if (dtype_name != "float32") {
    TORCH_CHECK(false, "dtype_name must be float32 or bfloat16");
  }
  auto basis_runtime = basis_t.to(dtype).contiguous();
  auto mean_runtime = mean.to(dtype).contiguous();
  const int64_t features = basis_runtime.size(1);
  auto matrix = torch::empty({token_count, features}, torch::dtype(dtype));
  if (payload->chunks.empty()) {
    return matrix;
  }

  const int64_t row_packed_bytes = rank / 2;
  int64_t max_chunk_rows = 1;
  for (const PreparedChunk& chunk : payload->chunks) {
    TORCH_CHECK(chunk.output_offset % static_cast<uint64_t>(row_packed_bytes) == 0,
                "fused QAT project requires row-aligned chunk offsets");
    TORCH_CHECK(chunk.output_len % static_cast<uint32_t>(row_packed_bytes) == 0,
                "fused QAT project requires row-aligned chunk lengths");
    max_chunk_rows = std::max<int64_t>(
        max_chunk_rows,
        static_cast<int64_t>(chunk.output_len / row_packed_bytes));
  }
  const int64_t requested_block_rows = std::max<int64_t>(tile_tokens, max_chunk_rows);
  const int64_t block_rows =
      ((requested_block_rows + max_chunk_rows - 1) / max_chunk_rows) * max_chunk_rows;
  const int64_t block_count = (token_count + block_rows - 1) / block_rows;

  struct ProjectBlock {
    int64_t start_row = 0;
    int64_t rows = 0;
    int64_t remaining_chunks = 0;
    torch::Tensor coeffs;
  };

  std::vector<ProjectBlock> blocks(static_cast<size_t>(block_count));
  for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
    ProjectBlock& block = blocks[static_cast<size_t>(block_idx)];
    block.start_row = block_idx * block_rows;
    block.rows = std::min<int64_t>(block_rows, token_count - block.start_row);
    block.coeffs = torch::empty({block.rows, rank}, torch::dtype(dtype));
  }
  std::vector<int64_t> chunk_block(payload->chunks.size(), 0);
  for (size_t chunk_idx = 0; chunk_idx < payload->chunks.size(); ++chunk_idx) {
    const PreparedChunk& chunk = payload->chunks[chunk_idx];
    const int64_t start_row = static_cast<int64_t>(chunk.output_offset / row_packed_bytes);
    const int64_t rows = static_cast<int64_t>(chunk.output_len / row_packed_bytes);
    TORCH_CHECK(rows > 0, "fused QAT project does not support empty output chunks");
    const int64_t block_idx = start_row / block_rows;
    TORCH_CHECK(block_idx >= 0 && block_idx < block_count, "chunk block index out of range");
    TORCH_CHECK(start_row + rows <= blocks[static_cast<size_t>(block_idx)].start_row +
                    blocks[static_cast<size_t>(block_idx)].rows,
                "fused QAT project requires chunks not to cross project block boundaries");
    chunk_block[chunk_idx] = block_idx;
    blocks[static_cast<size_t>(block_idx)].remaining_chunks++;
  }

  at::NoGradGuard no_grad;
  auto& state = qat_state();
  const uint32_t workers = std::min<uint32_t>(
      worker_count_for(state.size(max_instances), max_instances, payload->chunks.size()),
      kProjectWorkerCap);
  const float* scale_ptr = scale.data_ptr<float>();
  const float* offset_ptr = offset.data_ptr<float>();
  std::mutex ready_mutex;
  std::condition_variable ready_cv;
  std::deque<int64_t> ready_blocks;
  bool qat_workers_done = false;
  std::exception_ptr project_error;
  std::thread project_thread([&]() {
    try {
      while (true) {
        int64_t block_idx = -1;
        {
          std::unique_lock<std::mutex> lock(ready_mutex);
          ready_cv.wait(lock, [&]() {
            return !ready_blocks.empty() || qat_workers_done;
          });
          if (ready_blocks.empty()) {
            if (qat_workers_done) {
              break;
            }
            continue;
          }
          block_idx = ready_blocks.front();
          ready_blocks.pop_front();
        }
        ProjectBlock& block = blocks[static_cast<size_t>(block_idx)];
        project_coeff_tile_to_matrix(
            block.coeffs,
            block.rows,
            block.start_row,
            basis_runtime,
            mean_runtime,
            matrix);
        block.coeffs = torch::Tensor();
      }
    } catch (...) {
      project_error = std::current_exception();
    }
  });

  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);

  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker]() {
      try {
        QatCtx& ctx = state.ctx(worker);
        pin_qat_worker_thread(worker, ctx.node);
        const CpaDcSessionHandle session =
            payload->dynamic_huffman ? ctx.session_dynamic : ctx.session_static;
        std::vector<uint64_t> worker_chunks;
        for (uint64_t ordinal = 0;; ++ordinal) {
          const uint64_t chunk_idx = worker_chunk_index(worker, workers, ordinal);
          if (chunk_idx >= payload->chunks.size()) {
            break;
          }
          worker_chunks.push_back(chunk_idx);
        }
        if (worker_chunks.empty()) {
          return;
        }

        const uint32_t slot_count = static_cast<uint32_t>(
            std::min<int64_t>(inflight, std::max<int64_t>(1, worker_chunks.size())));
        std::vector<std::unique_ptr<Slot>> slots;
        slots.reserve(slot_count);
        const uint32_t dst_capacity = static_cast<uint32_t>(
            std::min<int64_t>(payload->chunk_bytes, payload->output_bytes));
        for (uint32_t idx = 0; idx < slot_count; ++idx) {
          auto slot = std::make_unique<Slot>();
          alloc_slot(ctx, *slot, dst_capacity);
          slots.push_back(std::move(slot));
        }

        std::vector<CpaDcDpOpData*> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t next = 0;
        while (true) {
          while (submitted - completed >= slot_count) {
            Slot& slot = *slots[completed % slot_count];
            poll_until_done(ctx, slot);
            const PreparedChunk& chunk = payload->chunks[slot.chunk_index];
            const int64_t start_row =
                static_cast<int64_t>(chunk.output_offset / row_packed_bytes);
            const int64_t block_idx = chunk_block[slot.chunk_index];
            ProjectBlock& block = blocks[static_cast<size_t>(block_idx)];
            const int64_t row_offset = start_row - block.start_row;
            dequant4_slot_output_to_coeff_tile(
                slot,
                chunk,
                rank,
                scale_ptr,
                offset_ptr,
                row_offset,
                block.coeffs);
            {
              std::lock_guard<std::mutex> lock(ready_mutex);
              block.remaining_chunks--;
              if (block.remaining_chunks == 0) {
                ready_blocks.push_back(block_idx);
                ready_cv.notify_one();
              }
            }
            completed++;
          }
          batch.clear();
          while (batch.size() < static_cast<size_t>(batch_size) &&
                 submitted - completed + batch.size() < slot_count &&
                 next < worker_chunks.size()) {
            Slot& slot = *slots[(submitted + batch.size()) % slot_count];
            slot.chunk_index = worker_chunks[next];
            prepare_slot_from_prepared_chunk(
                ctx,
                slot,
                payload->chunks[slot.chunk_index],
                session,
                CPA_DC_DIR_DECOMPRESS);
            batch.push_back(slot.op);
            next++;
          }
          if (batch.empty()) {
            break;
          }
          submit_batch(ctx, batch);
          submitted += batch.size();
        }
        while (completed < submitted) {
          Slot& slot = *slots[completed % slot_count];
          poll_until_done(ctx, slot);
          const PreparedChunk& chunk = payload->chunks[slot.chunk_index];
          const int64_t start_row =
              static_cast<int64_t>(chunk.output_offset / row_packed_bytes);
          const int64_t block_idx = chunk_block[slot.chunk_index];
          ProjectBlock& block = blocks[static_cast<size_t>(block_idx)];
          const int64_t row_offset = start_row - block.start_row;
          dequant4_slot_output_to_coeff_tile(
              slot,
              chunk,
              rank,
              scale_ptr,
              offset_ptr,
              row_offset,
              block.coeffs);
          {
            std::lock_guard<std::mutex> lock(ready_mutex);
            block.remaining_chunks--;
            if (block.remaining_chunks == 0) {
              ready_blocks.push_back(block_idx);
              ready_cv.notify_one();
            }
          }
          completed++;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  {
    std::lock_guard<std::mutex> lock(ready_mutex);
    qat_workers_done = true;
    ready_cv.notify_all();
  }
  project_thread.join();
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  if (project_error) {
    std::rethrow_exception(project_error);
  }
  return matrix;
}

int64_t qat_deflate_dp_instance_count() {
  return static_cast<int64_t>(qat_state().size(0));
}

std::vector<int64_t> qat_deflate_dp_instance_nodes(int64_t max_instances) {
  return qat_state().nodes(static_cast<size_t>(std::max<int64_t>(0, max_instances)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<QatPreparedPayload, std::shared_ptr<QatPreparedPayload>>(
      m, "QatPreparedPayload")
      .def_property_readonly("chunk_count", &QatPreparedPayload::chunk_count)
      .def_readonly("output_bytes", &QatPreparedPayload::output_bytes)
      .def_readonly("chunk_bytes", &QatPreparedPayload::chunk_bytes)
      .def_readonly("compressed_bytes", &QatPreparedPayload::compressed_bytes)
      .def_readonly("dynamic_huffman", &QatPreparedPayload::dynamic_huffman);
  m.def(
      "qat_deflate_compress_dp",
      &qat_deflate_compress_dp,
      py::arg("input"),
      py::arg("chunk_bytes") = 256 << 10,
      py::arg("dynamic_huffman") = true,
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_compress_prepare_dp",
      &qat_deflate_compress_prepare_dp,
      py::arg("input"),
      py::arg("chunk_bytes") = 256 << 10,
      py::arg("dynamic_huffman") = true,
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress_dp",
      &qat_deflate_decompress_dp,
      py::arg("bytestream"),
      py::arg("lengths"),
      py::arg("output_bytes"),
      py::arg("chunk_bytes") = 256 << 10,
      py::arg("dynamic_huffman") = true,
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_prepare_dp",
      &qat_deflate_prepare_dp,
      py::arg("bytestream"),
      py::arg("lengths"),
      py::arg("output_bytes"),
      py::arg("chunk_bytes") = 256 << 10,
      py::arg("dynamic_huffman") = true,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_prepare_from_fd_dp",
      &qat_deflate_prepare_from_fd_dp,
      py::arg("fd"),
      py::arg("lengths"),
      py::arg("output_bytes"),
      py::arg("chunk_bytes") = 256 << 10,
      py::arg("dynamic_huffman") = true,
      py::arg("max_instances") = 16,
      py::arg("file_offset") = -1,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_prepare_from_file_dp",
      &qat_deflate_prepare_from_file_dp,
      py::arg("path"),
      py::arg("lengths"),
      py::arg("output_bytes"),
      py::arg("chunk_bytes") = 256 << 10,
      py::arg("dynamic_huffman") = true,
      py::arg("max_instances") = 16,
      py::arg("file_offset") = 0,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress_prepared_dp",
      &qat_deflate_decompress_prepared_dp,
      py::arg("payload"),
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress_dequant4_prepared_dp",
      &qat_deflate_decompress_dequant4_prepared_dp,
      py::arg("payload"),
      py::arg("scale"),
      py::arg("offset"),
      py::arg("token_count"),
      py::arg("rank"),
      py::arg("dtype_name") = "float32",
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress_project4_prepared_dp",
      &qat_deflate_decompress_project4_prepared_dp,
      py::arg("payload"),
      py::arg("scale"),
      py::arg("offset"),
      py::arg("basis_t"),
      py::arg("mean"),
      py::arg("token_count"),
      py::arg("rank"),
      py::arg("dtype_name") = "float32",
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::arg("tile_tokens") = 8192,
      py::call_guard<py::gil_scoped_release>());
  m.def("qat_deflate_dp_instance_count", &qat_deflate_dp_instance_count);
  m.def(
      "qat_deflate_dp_instance_nodes",
      &qat_deflate_dp_instance_nodes,
      py::arg("max_instances") = 0);
}
