// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from /home/td/dpucomp/kvtc/qat_dp_codec.cpp for Origami native CPU lossless.

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
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
constexpr char kDpuCompDzMagic[8] = {'D', 'P', 'U', 'C', 'D', 'Z', '1', '\0'};

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

double ns_to_ms(uint64_t ns) {
  return static_cast<double>(ns) / 1.0e6;
}

struct QatDpProfile {
  uint64_t total_ns = 0;
  uint64_t worker_wall_ns_max = 0;
  uint64_t slot_alloc_ns_sum = 0;
  uint64_t enqueue_ns_sum = 0;
  uint64_t poll_wait_ns_sum = 0;
  uint64_t output_copy_ns_sum = 0;
  uint64_t submit_poll_ns_critical = 0;
  int64_t chunks = 0;
  int64_t workers = 0;
  int64_t inflight = 0;
  int64_t batch = 0;
  int64_t compressed_bytes = 0;
  int64_t unpacked_bytes = 0;
  bool persistent_workers = false;
};

struct QatPrepareProfile {
  uint64_t total_ns = 0;
  uint64_t parse_ns = 0;
  uint64_t qat_state_ns = 0;
  uint64_t qae_alloc_ns_sum = 0;
  uint64_t qae_input_copy_ns_sum = 0;
  uint64_t file_read_io_ns = 0;
  uint64_t file_read_sleep_ns = 0;
  uint64_t file_read_window_start_ns = 0;
  int64_t file_read_bytes = 0;
  int64_t qae_buffer_reuse_hits = 0;
  int64_t qae_buffer_reuse_misses = 0;
  int64_t chunks = 0;
  int64_t workers = 0;
  int64_t compressed_bytes = 0;
  int64_t unpacked_bytes = 0;
  int64_t bundle_bytes = 0;
  bool dynamic_huffman = true;
};

struct WorkerProfile {
  uint64_t wall_ns = 0;
  uint64_t slot_alloc_ns = 0;
  uint64_t enqueue_ns = 0;
  uint64_t poll_wait_ns = 0;
  uint64_t output_copy_ns = 0;
};

std::mutex& qat_profile_mutex() {
  static std::mutex mutex;
  return mutex;
}

QatDpProfile& qat_last_profile() {
  static QatDpProfile profile;
  return profile;
}

void set_qat_last_profile(const QatDpProfile& profile) {
  std::lock_guard<std::mutex> lock(qat_profile_mutex());
  qat_last_profile() = profile;
}

QatPrepareProfile& qat_last_prepare_profile() {
  static QatPrepareProfile profile;
  return profile;
}

void set_qat_last_prepare_profile(const QatPrepareProfile& profile) {
  std::lock_guard<std::mutex> lock(qat_profile_mutex());
  qat_last_prepare_profile() = profile;
}

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

void check_cpu_i64(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
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

uint32_t load_u32_le(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) |
      (static_cast<uint32_t>(p[1]) << 8) |
      (static_cast<uint32_t>(p[2]) << 16) |
      (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t load_u64_le(const uint8_t* p) {
  uint64_t value = 0;
  for (int idx = 7; idx >= 0; --idx) {
    value = (value << 8) | static_cast<uint64_t>(p[idx]);
  }
  return value;
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

uint8_t* acquire_prepared_qae_buffer(int node, uint32_t capacity, bool* reused);
void release_prepared_qae_buffer(uint8_t* ptr, uint32_t capacity, int node);

struct PreparedChunk {
  uint8_t* data = nullptr;
  uint32_t data_capacity = 0;
  int node = -1;
  bool reused_from_pool = false;
  uint32_t input_len = 0;
  uint32_t output_len = 0;
  uint64_t output_offset = 0;

  PreparedChunk() = default;
  PreparedChunk(const PreparedChunk&) = delete;
  PreparedChunk& operator=(const PreparedChunk&) = delete;

  PreparedChunk(PreparedChunk&& other) noexcept
      : data(other.data),
        data_capacity(other.data_capacity),
        node(other.node),
        reused_from_pool(other.reused_from_pool),
        input_len(other.input_len),
        output_len(other.output_len),
        output_offset(other.output_offset) {
    other.data = nullptr;
    other.data_capacity = 0;
    other.node = -1;
    other.reused_from_pool = false;
    other.input_len = 0;
    other.output_len = 0;
    other.output_offset = 0;
  }

  PreparedChunk& operator=(PreparedChunk&& other) noexcept {
    if (this != &other) {
      reset();
      data = other.data;
      data_capacity = other.data_capacity;
      node = other.node;
      reused_from_pool = other.reused_from_pool;
      input_len = other.input_len;
      output_len = other.output_len;
      output_offset = other.output_offset;
      other.data = nullptr;
      other.data_capacity = 0;
      other.node = -1;
      other.reused_from_pool = false;
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
      release_prepared_qae_buffer(data, data_capacity, node);
      data = nullptr;
    }
    data_capacity = 0;
    node = -1;
    reused_from_pool = false;
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

uint64_t prepared_qae_pool_max_bytes() {
  const char* value = std::getenv("ORIGAMI_QAT_PREPARED_POOL_MAX_BYTES");
  if (value == nullptr || *value == '\0') {
    return 2ULL * 1024ULL * 1024ULL * 1024ULL;
  }
  try {
    return static_cast<uint64_t>(std::stoull(value));
  } catch (...) {
    return 2ULL * 1024ULL * 1024ULL * 1024ULL;
  }
}

class PreparedQaeBufferPool {
 public:
  PreparedQaeBufferPool() = default;
  PreparedQaeBufferPool(const PreparedQaeBufferPool&) = delete;
  PreparedQaeBufferPool& operator=(const PreparedQaeBufferPool&) = delete;

  ~PreparedQaeBufferPool() {
    for (auto& buffer : buffers_) {
      if (buffer.ptr != nullptr) {
        qaeMemFreeNUMA(reinterpret_cast<void**>(&buffer.ptr));
      }
    }
  }

  uint8_t* acquire(int node, uint32_t capacity, bool* reused) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
      if (it->node == node && it->capacity >= capacity) {
        uint8_t* ptr = it->ptr;
        pooled_bytes_ -= it->capacity;
        buffers_.erase(it);
        if (reused != nullptr) {
          *reused = true;
        }
        return ptr;
      }
    }
    if (reused != nullptr) {
      *reused = false;
    }
    return nullptr;
  }

  void release(uint8_t* ptr, uint32_t capacity, int node) {
    if (ptr == nullptr) {
      return;
    }
    const uint64_t max_bytes = prepared_qae_pool_max_bytes();
    std::lock_guard<std::mutex> lock(mutex_);
    if (max_bytes == 0 || pooled_bytes_ + capacity > max_bytes) {
      qaeMemFreeNUMA(reinterpret_cast<void**>(&ptr));
      return;
    }
    buffers_.push_back(Buffer{ptr, capacity, node});
    pooled_bytes_ += capacity;
  }

 private:
  struct Buffer {
    uint8_t* ptr = nullptr;
    uint32_t capacity = 0;
    int node = -1;
  };

  std::mutex mutex_;
  std::vector<Buffer> buffers_;
  uint64_t pooled_bytes_ = 0;
};

PreparedQaeBufferPool& prepared_qae_buffer_pool() {
  static PreparedQaeBufferPool pool;
  return pool;
}

uint8_t* acquire_prepared_qae_buffer(int node, uint32_t capacity, bool* reused) {
  uint8_t* ptr = prepared_qae_buffer_pool().acquire(node, capacity, reused);
  if (ptr != nullptr) {
    return ptr;
  }
  ptr = static_cast<uint8_t*>(qaeMemAllocNUMA(capacity, node, kAlignment));
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate prepared QAT chunk");
  }
  return ptr;
}

void release_prepared_qae_buffer(uint8_t* ptr, uint32_t capacity, int node) {
  prepared_qae_buffer_pool().release(ptr, capacity, node);
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
  chunk.node = ctx.node;
  chunk.data_capacity = static_cast<uint32_t>(
      align_up(std::max<uint32_t>(input_len, 1U), kAlignment));
  bool reused = false;
  chunk.data = acquire_prepared_qae_buffer(ctx.node, chunk.data_capacity, &reused);
  chunk.reused_from_pool = reused;
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

void bandwidth_sleep(double bandwidth_gbps, QatPrepareProfile& profile) {
  if (profile.file_read_bytes <= 0 || bandwidth_gbps <= 0.0) {
    return;
  }
  const double bytes_per_second = bandwidth_gbps * 1.0e9 / 8.0;
  if (bytes_per_second <= 0.0) {
    return;
  }
  const uint64_t now = now_ns();
  if (profile.file_read_window_start_ns == 0 || now <= profile.file_read_window_start_ns) {
    return;
  }
  const uint64_t target_elapsed_ns = static_cast<uint64_t>(
      (static_cast<double>(profile.file_read_bytes) / bytes_per_second) * 1.0e9);
  const uint64_t elapsed_ns = now - profile.file_read_window_start_ns;
  if (target_elapsed_ns <= elapsed_ns) {
    return;
  }
  const uint64_t sleep_ns = target_elapsed_ns - elapsed_ns;
  const uint64_t start_ns = now_ns();
  std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_ns));
  profile.file_read_sleep_ns += now_ns() - start_ns;
}

void pread_exact_fd_profiled(
    int fd,
    uint8_t* dst,
    uint32_t bytes,
    int64_t offset,
    double bandwidth_gbps,
    QatPrepareProfile& profile) {
  if (profile.file_read_window_start_ns == 0) {
    profile.file_read_window_start_ns = now_ns();
  }
  const uint64_t read_start_ns = now_ns();
  pread_exact_fd(fd, dst, bytes, offset);
  profile.file_read_io_ns += now_ns() - read_start_ns;
  profile.file_read_bytes += bytes;
  bandwidth_sleep(bandwidth_gbps, profile);
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

py::dict qat_deflate_last_profile_dp();
py::dict qat_deflate_last_prepare_profile_dp();

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
  const uint64_t total_start_ns = now_ns();
  QatPrepareProfile profile;
  profile.dynamic_huffman = dynamic_huffman;
  profile.bundle_bytes = bytestream.numel();
  uint64_t parse_start_ns = now_ns();
  check_cpu_u8(bytestream, "bytestream");
  check_cpu_i32(lengths, "lengths");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be one-dimensional");
  TORCH_CHECK(output_bytes >= 0, "output_bytes must be non-negative");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= std::numeric_limits<Cpa32U>::max(),
              "chunk_bytes must fit in a QAT request");
  const int64_t chunks = lengths.numel();
  TORCH_CHECK(chunks == (output_bytes + chunk_bytes - 1) / chunk_bytes,
              "length count must match output_bytes and chunk_bytes");
  profile.parse_ns += now_ns() - parse_start_ns;
  const uint64_t state_start_ns = now_ns();
  auto& state = qat_state();
  profile.qat_state_ns += now_ns() - state_start_ns;
  const uint32_t workers = worker_count_for(state.size(max_instances), max_instances, chunks);
  profile.chunks = chunks;
  profile.workers = workers;
  profile.unpacked_bytes = output_bytes;
  const auto* input_ptr = bytestream.data_ptr<uint8_t>();
  const auto* lengths_ptr = lengths.data_ptr<int32_t>();
  int64_t input_offset = 0;
  auto payload = std::make_shared<QatPreparedPayload>();
  payload->chunks.reserve(static_cast<size_t>(chunks));
  payload->output_bytes = output_bytes;
  payload->chunk_bytes = chunk_bytes;
  payload->dynamic_huffman = dynamic_huffman;
  for (int64_t idx = 0; idx < chunks; ++idx) {
    const uint64_t chunk_parse_start_ns = now_ns();
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
    profile.parse_ns += now_ns() - chunk_parse_start_ns;
    const uint64_t alloc_start_ns = now_ns();
    alloc_prepared_chunk(
        ctx,
        chunk,
        input_len,
        output_len,
        static_cast<uint64_t>(output_offset));
    profile.qae_alloc_ns_sum += now_ns() - alloc_start_ns;
    if (chunk.reused_from_pool) {
      profile.qae_buffer_reuse_hits += 1;
    } else {
      profile.qae_buffer_reuse_misses += 1;
    }
    if (input_len > 0) {
      const uint64_t copy_start_ns = now_ns();
      std::memcpy(chunk.data, input_ptr + input_offset, input_len);
      profile.qae_input_copy_ns_sum += now_ns() - copy_start_ns;
    }
    payload->compressed_bytes += input_len;
    profile.compressed_bytes += input_len;
    input_offset += input_len;
    payload->chunks.push_back(std::move(chunk));
  }
  TORCH_CHECK(input_offset == bytestream.numel(), "lengths must sum to bytestream size");
  profile.total_ns = now_ns() - total_start_ns;
  set_qat_last_prepare_profile(profile);
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

std::shared_ptr<QatPreparedPayload> qat_deflate_prepare_bundle_dp(
    torch::Tensor bundle,
    bool dynamic_huffman,
    int64_t max_instances) {
  const uint64_t total_start_ns = now_ns();
  QatPrepareProfile profile;
  profile.dynamic_huffman = dynamic_huffman;
  const uint64_t parse_start_ns = now_ns();
  check_cpu_u8(bundle, "bundle");
  TORCH_CHECK(bundle.dim() == 1, "bundle must be one-dimensional");
  const int64_t bundle_bytes = bundle.numel();
  profile.bundle_bytes = bundle_bytes;
  TORCH_CHECK(bundle_bytes >= 24, "DPUCDZ1 bundle is too small");
  const auto* data = bundle.data_ptr<uint8_t>();
  TORCH_CHECK(
      std::memcmp(data, kDpuCompDzMagic, sizeof(kDpuCompDzMagic)) == 0,
      "invalid DPUCDZ1 bundle magic");

  const uint32_t chunk_bytes = load_u32_le(data + 8);
  const uint32_t chunk_count = load_u32_le(data + 12);
  const uint64_t raw_bytes = load_u64_le(data + 16);
  TORCH_CHECK(chunk_bytes > 0, "DPUCDZ1 chunk_bytes must be positive");
  TORCH_CHECK(chunk_count > 0, "DPUCDZ1 chunk_count must be positive");
  const uint64_t records_bytes = static_cast<uint64_t>(chunk_count) * 8ULL;
  const uint64_t header_bytes = 24ULL + records_bytes;
  TORCH_CHECK(header_bytes <= static_cast<uint64_t>(bundle_bytes),
              "DPUCDZ1 record table exceeds bundle size");

  struct DzRecord {
    uint32_t raw_len = 0;
    uint32_t comp_len = 0;
    uint64_t comp_offset = 0;
    uint64_t output_offset = 0;
  };

  std::vector<DzRecord> records;
  records.reserve(chunk_count);
  uint64_t raw_sum = 0;
  uint64_t comp_sum = 0;
  uint64_t cursor = 24;
  for (uint32_t idx = 0; idx < chunk_count; ++idx) {
    DzRecord record;
    record.raw_len = load_u32_le(data + cursor);
    record.comp_len = load_u32_le(data + cursor + 4);
    record.comp_offset = comp_sum;
    record.output_offset = raw_sum;
    cursor += 8;
    TORCH_CHECK(record.raw_len > 0 && record.raw_len <= chunk_bytes,
                "invalid DPUCDZ1 raw chunk length");
    TORCH_CHECK(record.comp_len > 0, "invalid DPUCDZ1 compressed chunk length");
    TORCH_CHECK(record.comp_len < kMaxQaeAllocBytes,
                "DPUCDZ1 prepared restore only supports compressed chunks below 64 MiB");
    raw_sum += record.raw_len;
    comp_sum += record.comp_len;
    records.push_back(record);
  }
  TORCH_CHECK(raw_sum == raw_bytes, "DPUCDZ1 raw size mismatch");
  TORCH_CHECK(header_bytes + comp_sum == static_cast<uint64_t>(bundle_bytes),
              "DPUCDZ1 compressed payload size mismatch");
  profile.parse_ns = now_ns() - parse_start_ns;
  profile.chunks = chunk_count;
  profile.compressed_bytes = static_cast<int64_t>(comp_sum);
  profile.unpacked_bytes = static_cast<int64_t>(raw_bytes);

  const uint64_t state_start_ns = now_ns();
  auto& state = qat_state();
  profile.qat_state_ns = now_ns() - state_start_ns;
  const uint32_t workers =
      worker_count_for(state.size(max_instances), max_instances, chunk_count);
  profile.workers = workers;
  auto payload = std::make_shared<QatPreparedPayload>();
  payload->chunks.reserve(chunk_count);
  payload->output_bytes = static_cast<int64_t>(raw_bytes);
  payload->chunk_bytes = static_cast<int64_t>(chunk_bytes);
  payload->compressed_bytes = static_cast<int64_t>(comp_sum);
  payload->dynamic_huffman = dynamic_huffman;
  const uint8_t* compressed_payload = data + header_bytes;
  for (uint32_t idx = 0; idx < chunk_count; ++idx) {
    const DzRecord& record = records[idx];
    QatCtx& ctx = state.ctx(static_cast<size_t>(idx % workers));
    PreparedChunk chunk;
    const uint64_t alloc_start_ns = now_ns();
    alloc_prepared_chunk(
        ctx,
        chunk,
        record.comp_len,
        record.raw_len,
        record.output_offset);
    profile.qae_alloc_ns_sum += now_ns() - alloc_start_ns;
    if (chunk.reused_from_pool) {
      profile.qae_buffer_reuse_hits += 1;
    } else {
      profile.qae_buffer_reuse_misses += 1;
    }
    const uint64_t copy_start_ns = now_ns();
    std::memcpy(
        chunk.data,
        compressed_payload + record.comp_offset,
        record.comp_len);
    profile.qae_input_copy_ns_sum += now_ns() - copy_start_ns;
    payload->chunks.push_back(std::move(chunk));
  }
  profile.total_ns = now_ns() - total_start_ns;
  set_qat_last_prepare_profile(profile);
  return payload;
}

std::shared_ptr<QatPreparedPayload> qat_deflate_prepare_bundle_from_file_dp(
    std::string path,
    bool dynamic_huffman,
    int64_t max_instances,
    double bandwidth_gbps,
    int64_t file_offset) {
  const uint64_t total_start_ns = now_ns();
  QatPrepareProfile profile;
  profile.dynamic_huffman = dynamic_huffman;
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("open failed with errno " + std::to_string(errno));
  }
  try {
    uint8_t header[24];
    pread_exact_fd_profiled(
        fd,
        header,
        static_cast<uint32_t>(sizeof(header)),
        file_offset,
        bandwidth_gbps,
        profile);

    uint64_t parse_start_ns = now_ns();
    TORCH_CHECK(
        std::memcmp(header, kDpuCompDzMagic, sizeof(kDpuCompDzMagic)) == 0,
        "invalid DPUCDZ1 bundle magic");
    const uint32_t chunk_bytes = load_u32_le(header + 8);
    const uint32_t chunk_count = load_u32_le(header + 12);
    const uint64_t raw_bytes = load_u64_le(header + 16);
    TORCH_CHECK(chunk_bytes > 0, "DPUCDZ1 chunk_bytes must be positive");
    TORCH_CHECK(chunk_count > 0, "DPUCDZ1 chunk_count must be positive");
    const uint64_t records_bytes = static_cast<uint64_t>(chunk_count) * 8ULL;
    const uint64_t header_bytes = 24ULL + records_bytes;
    TORCH_CHECK(records_bytes <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                "DPUCDZ1 record table too large");
    profile.parse_ns += now_ns() - parse_start_ns;

    std::vector<uint8_t> record_bytes(static_cast<size_t>(records_bytes));
    pread_exact_fd_profiled(
        fd,
        record_bytes.data(),
        static_cast<uint32_t>(record_bytes.size()),
        file_offset + 24,
        bandwidth_gbps,
        profile);

    struct DzRecord {
      uint32_t raw_len = 0;
      uint32_t comp_len = 0;
      uint64_t comp_offset = 0;
      uint64_t output_offset = 0;
    };

    parse_start_ns = now_ns();
    std::vector<DzRecord> records;
    records.reserve(chunk_count);
    uint64_t raw_sum = 0;
    uint64_t comp_sum = 0;
    uint64_t cursor = 0;
    const uint8_t* records_data = record_bytes.data();
    for (uint32_t idx = 0; idx < chunk_count; ++idx) {
      DzRecord record;
      record.raw_len = load_u32_le(records_data + cursor);
      record.comp_len = load_u32_le(records_data + cursor + 4);
      record.comp_offset = comp_sum;
      record.output_offset = raw_sum;
      cursor += 8;
      TORCH_CHECK(record.raw_len > 0 && record.raw_len <= chunk_bytes,
                  "invalid DPUCDZ1 raw chunk length");
      TORCH_CHECK(record.comp_len > 0, "invalid DPUCDZ1 compressed chunk length");
      TORCH_CHECK(record.comp_len < kMaxQaeAllocBytes,
                  "DPUCDZ1 prepared restore only supports compressed chunks below 64 MiB");
      raw_sum += record.raw_len;
      comp_sum += record.comp_len;
      records.push_back(record);
    }
    TORCH_CHECK(raw_sum == raw_bytes, "DPUCDZ1 raw size mismatch");
    profile.parse_ns += now_ns() - parse_start_ns;
    profile.chunks = chunk_count;
    profile.compressed_bytes = static_cast<int64_t>(comp_sum);
    profile.unpacked_bytes = static_cast<int64_t>(raw_bytes);
    profile.bundle_bytes = static_cast<int64_t>(header_bytes + comp_sum);

    const uint64_t state_start_ns = now_ns();
    auto& state = qat_state();
    profile.qat_state_ns = now_ns() - state_start_ns;
    const uint32_t workers =
        worker_count_for(state.size(max_instances), max_instances, chunk_count);
    profile.workers = workers;
    auto payload = std::make_shared<QatPreparedPayload>();
    payload->chunks.reserve(chunk_count);
    payload->output_bytes = static_cast<int64_t>(raw_bytes);
    payload->chunk_bytes = static_cast<int64_t>(chunk_bytes);
    payload->compressed_bytes = static_cast<int64_t>(comp_sum);
    payload->dynamic_huffman = dynamic_huffman;
    for (uint32_t idx = 0; idx < chunk_count; ++idx) {
      const DzRecord& record = records[idx];
      QatCtx& ctx = state.ctx(static_cast<size_t>(idx % workers));
      PreparedChunk chunk;
      const uint64_t alloc_start_ns = now_ns();
      alloc_prepared_chunk(
          ctx,
          chunk,
          record.comp_len,
          record.raw_len,
          record.output_offset);
      profile.qae_alloc_ns_sum += now_ns() - alloc_start_ns;
      if (chunk.reused_from_pool) {
        profile.qae_buffer_reuse_hits += 1;
      } else {
        profile.qae_buffer_reuse_misses += 1;
      }
      pread_exact_fd_profiled(
          fd,
          chunk.data,
          record.comp_len,
          file_offset + static_cast<int64_t>(header_bytes + record.comp_offset),
          bandwidth_gbps,
          profile);
      payload->chunks.push_back(std::move(chunk));
    }
    ::close(fd);
    profile.total_ns = now_ns() - total_start_ns;
    set_qat_last_prepare_profile(profile);
    return payload;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

struct PersistentRestoreJob {
  std::shared_ptr<QatPreparedPayload> payload;
  uint8_t* output_ptr = nullptr;
  int64_t inflight = 0;
  int64_t batch_size = 0;
  int64_t max_instances = 0;
  const std::vector<int64_t>* chunk_indices = nullptr;
  const std::vector<uint64_t>* output_offsets = nullptr;
  int64_t window_output_bytes = -1;
  int64_t window_compressed_bytes = -1;
  uint32_t active_workers = 0;
  uint32_t total_workers = 0;
  std::vector<WorkerProfile>* profiles = nullptr;
  std::vector<std::exception_ptr>* errors = nullptr;
  uint32_t finished_workers = 0;
};

uint64_t persistent_job_chunk_count(const PersistentRestoreJob& job) {
  if (job.chunk_indices != nullptr) {
    return static_cast<uint64_t>(job.chunk_indices->size());
  }
  return static_cast<uint64_t>(job.payload->chunks.size());
}

uint64_t persistent_job_global_chunk_index(
    const PersistentRestoreJob& job,
    uint64_t local_chunk_idx) {
  if (job.chunk_indices == nullptr) {
    return local_chunk_idx;
  }
  return static_cast<uint64_t>((*job.chunk_indices)[local_chunk_idx]);
}

const PreparedChunk& persistent_job_chunk(
    const PersistentRestoreJob& job,
    uint64_t local_chunk_idx) {
  const uint64_t global_idx =
      persistent_job_global_chunk_index(job, local_chunk_idx);
  TORCH_CHECK(global_idx < job.payload->chunks.size(),
              "prepared window chunk index out of range");
  return job.payload->chunks[global_idx];
}

uint64_t persistent_job_output_offset(
    const PersistentRestoreJob& job,
    uint64_t local_chunk_idx) {
  if (job.output_offsets == nullptr) {
    return persistent_job_chunk(job, local_chunk_idx).output_offset;
  }
  return (*job.output_offsets)[local_chunk_idx];
}

struct PersistentWorkerState {
  std::vector<std::unique_ptr<Slot>> slots;
};

class PersistentPreparedRestoreExecutor {
 public:
  PersistentPreparedRestoreExecutor() = default;
  PersistentPreparedRestoreExecutor(const PersistentPreparedRestoreExecutor&) = delete;
  PersistentPreparedRestoreExecutor& operator=(const PersistentPreparedRestoreExecutor&) = delete;

  ~PersistentPreparedRestoreExecutor() {
    stop();
  }

  QatDpProfile run(
      std::shared_ptr<QatPreparedPayload> payload,
      uint8_t* output_ptr,
      int64_t inflight,
      int64_t batch_size,
      int64_t max_instances,
      const std::vector<int64_t>* chunk_indices = nullptr,
      const std::vector<uint64_t>* output_offsets = nullptr,
      int64_t window_output_bytes = -1,
      int64_t window_compressed_bytes = -1) {
    TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
    const uint64_t chunk_count =
        chunk_indices != nullptr
            ? static_cast<uint64_t>(chunk_indices->size())
            : static_cast<uint64_t>(payload->chunks.size());
    const int64_t effective_output_bytes =
        window_output_bytes >= 0 ? window_output_bytes : payload->output_bytes;
    const int64_t effective_compressed_bytes =
        window_compressed_bytes >= 0
            ? window_compressed_bytes
            : payload->compressed_bytes;
    TORCH_CHECK(output_ptr != nullptr || effective_output_bytes == 0,
                "prepared restore output pointer must not be null");
    auto& state = qat_state();
    const uint32_t total_workers =
        static_cast<uint32_t>(state.size(0));
    const uint32_t active_workers =
        worker_count_for(state.size(max_instances), max_instances, chunk_count);
    ensure_started(total_workers);

    std::vector<WorkerProfile> profiles(total_workers);
    std::vector<std::exception_ptr> errors(total_workers);
    PersistentRestoreJob job;
    job.payload = std::move(payload);
    job.output_ptr = output_ptr;
    job.inflight = inflight;
    job.batch_size = batch_size;
    job.max_instances = max_instances;
    job.chunk_indices = chunk_indices;
    job.output_offsets = output_offsets;
    job.window_output_bytes = effective_output_bytes;
    job.window_compressed_bytes = effective_compressed_bytes;
    job.active_workers = active_workers;
    job.total_workers = total_workers;
    job.profiles = &profiles;
    job.errors = &errors;

    const uint64_t total_start_ns = now_ns();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      idle_cv_.wait(lock, [&]() { return current_job_ == nullptr; });
      current_job_ = &job;
      job_generation_++;
      job_cv_.notify_all();
      done_cv_.wait(lock, [&]() {
        return job.finished_workers >= total_workers;
      });
      current_job_ = nullptr;
      idle_cv_.notify_all();
    }
    for (const auto& error : errors) {
      if (error) {
        std::rethrow_exception(error);
      }
    }

    QatDpProfile profile;
    profile.total_ns = now_ns() - total_start_ns;
    profile.chunks = static_cast<int64_t>(chunk_count);
    profile.workers = static_cast<int64_t>(active_workers);
    profile.inflight = inflight;
    profile.batch = batch_size;
    profile.compressed_bytes = effective_compressed_bytes;
    profile.unpacked_bytes = effective_output_bytes;
    profile.persistent_workers = true;
    for (uint32_t worker = 0; worker < active_workers; ++worker) {
      const WorkerProfile& worker_profile = profiles[worker];
      profile.worker_wall_ns_max =
          std::max(profile.worker_wall_ns_max, worker_profile.wall_ns);
      profile.slot_alloc_ns_sum += worker_profile.slot_alloc_ns;
      profile.enqueue_ns_sum += worker_profile.enqueue_ns;
      profile.poll_wait_ns_sum += worker_profile.poll_wait_ns;
      profile.output_copy_ns_sum += worker_profile.output_copy_ns;
      profile.submit_poll_ns_critical = std::max(
          profile.submit_poll_ns_critical,
          worker_profile.enqueue_ns + worker_profile.poll_wait_ns);
    }
    return profile;
  }

 private:
  void ensure_started(uint32_t worker_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!threads_.empty()) {
      TORCH_CHECK(threads_.size() == worker_count,
                  "QAT worker count changed after persistent executor startup");
      return;
    }
    worker_states_.resize(worker_count);
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
      threads_.emplace_back([this, worker]() { worker_loop(worker); });
    }
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      job_cv_.notify_all();
    }
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    threads_.clear();
    worker_states_.clear();
  }

  void ensure_slots(
      QatCtx& ctx,
      PersistentWorkerState& state,
      uint32_t slot_count,
      uint32_t dst_capacity,
      WorkerProfile& profile) {
    bool reuse = state.slots.size() >= slot_count;
    if (reuse) {
      const uint32_t required_capacity =
          static_cast<uint32_t>(align_up(std::max(dst_capacity, kMinDcDestBytes), kAlignment));
      for (uint32_t idx = 0; idx < slot_count; ++idx) {
        const Slot& slot = *state.slots[idx];
        if (slot.op == nullptr || slot.dst_capacity < required_capacity) {
          reuse = false;
          break;
        }
      }
    }
    if (reuse) {
      return;
    }

    const uint64_t alloc_start_ns = now_ns();
    state.slots.clear();
    state.slots.reserve(slot_count);
    for (uint32_t idx = 0; idx < slot_count; ++idx) {
      auto slot = std::make_unique<Slot>();
      alloc_slot(ctx, *slot, dst_capacity);
      state.slots.push_back(std::move(slot));
    }
    profile.slot_alloc_ns += now_ns() - alloc_start_ns;
  }

  void worker_loop(uint32_t worker) {
    uint64_t seen_generation = 0;
    try {
      QatCtx& ctx = qat_state().ctx(worker);
      pin_qat_worker_thread(worker, ctx.node);
      while (true) {
        PersistentRestoreJob* job = nullptr;
        uint64_t generation = 0;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          job_cv_.wait(lock, [&]() {
            return stopping_ || job_generation_ != seen_generation;
          });
          if (stopping_) {
            return;
          }
          job = current_job_;
          generation = job_generation_;
        }
        if (job != nullptr) {
          if (worker < job->active_workers) {
            run_worker(ctx, worker, *job);
          }
          {
            std::lock_guard<std::mutex> lock(mutex_);
            job->finished_workers++;
            if (job->finished_workers >= job->total_workers) {
              done_cv_.notify_one();
            }
          }
        }
        seen_generation = generation;
      }
    } catch (...) {
      // A worker-level setup failure is surfaced through the next job slot.
      std::lock_guard<std::mutex> lock(mutex_);
      if (current_job_ != nullptr && current_job_->errors != nullptr &&
          worker < current_job_->errors->size()) {
        (*current_job_->errors)[worker] = std::current_exception();
        current_job_->finished_workers++;
        if (current_job_->finished_workers >= current_job_->total_workers) {
          done_cv_.notify_one();
        }
      }
    }
  }

  void run_worker(QatCtx& ctx, uint32_t worker, PersistentRestoreJob& job) {
    WorkerProfile& profile = (*job.profiles)[worker];
    const uint64_t worker_start_ns = now_ns();
    try {
      const CpaDcSessionHandle session =
          job.payload->dynamic_huffman ? ctx.session_dynamic : ctx.session_static;
      std::vector<uint64_t> worker_chunks;
      for (uint64_t ordinal = 0;; ++ordinal) {
        const uint64_t chunk_idx =
            worker_chunk_index(worker, job.active_workers, ordinal);
        if (chunk_idx >= persistent_job_chunk_count(job)) {
          break;
        }
        worker_chunks.push_back(chunk_idx);
      }
      const uint32_t slot_count = static_cast<uint32_t>(
          std::min<int64_t>(job.inflight, std::max<int64_t>(1, worker_chunks.size())));
      const uint32_t dst_capacity = static_cast<uint32_t>(
          std::min<int64_t>(job.payload->chunk_bytes, job.payload->output_bytes));
      ensure_slots(ctx, worker_states_[worker], slot_count, dst_capacity, profile);

      std::vector<CpaDcDpOpData*> batch;
      batch.reserve(static_cast<size_t>(job.batch_size));
      uint64_t submitted = 0;
      uint64_t completed = 0;
      uint64_t next = 0;
      auto& slots = worker_states_[worker].slots;
      while (true) {
        while (submitted - completed >= slot_count) {
          Slot& slot = *slots[completed % slot_count];
          const uint64_t poll_start_ns = now_ns();
          poll_until_done(ctx, slot);
          profile.poll_wait_ns += now_ns() - poll_start_ns;
          const PreparedChunk& chunk = persistent_job_chunk(job, slot.chunk_index);
          const uint64_t copy_start_ns = now_ns();
          copy_slot_output_to_tensor(
              slot,
              job.output_ptr,
              persistent_job_output_offset(job, slot.chunk_index),
              chunk.output_len);
          profile.output_copy_ns += now_ns() - copy_start_ns;
          completed++;
        }
        batch.clear();
        while (batch.size() < static_cast<size_t>(job.batch_size) &&
               submitted - completed + batch.size() < slot_count &&
               next < worker_chunks.size()) {
          Slot& slot = *slots[(submitted + batch.size()) % slot_count];
          slot.chunk_index = worker_chunks[next];
          prepare_slot_from_prepared_chunk(
              ctx,
              slot,
              persistent_job_chunk(job, slot.chunk_index),
              session,
              CPA_DC_DIR_DECOMPRESS);
          batch.push_back(slot.op);
          next++;
        }
        if (batch.empty()) {
          break;
        }
        const uint64_t enqueue_start_ns = now_ns();
        submit_batch(ctx, batch);
        profile.enqueue_ns += now_ns() - enqueue_start_ns;
        submitted += batch.size();
      }
      while (completed < submitted) {
        Slot& slot = *slots[completed % slot_count];
        const uint64_t poll_start_ns = now_ns();
        poll_until_done(ctx, slot);
        profile.poll_wait_ns += now_ns() - poll_start_ns;
        const PreparedChunk& chunk = persistent_job_chunk(job, slot.chunk_index);
        const uint64_t copy_start_ns = now_ns();
        copy_slot_output_to_tensor(
            slot,
            job.output_ptr,
            persistent_job_output_offset(job, slot.chunk_index),
            chunk.output_len);
        profile.output_copy_ns += now_ns() - copy_start_ns;
        completed++;
      }
    } catch (...) {
      (*job.errors)[worker] = std::current_exception();
    }
    profile.wall_ns = now_ns() - worker_start_ns;
  }

  std::mutex mutex_;
  std::condition_variable job_cv_;
  std::condition_variable done_cv_;
  std::condition_variable idle_cv_;
  bool stopping_ = false;
  uint64_t job_generation_ = 0;
  PersistentRestoreJob* current_job_ = nullptr;
  std::vector<std::thread> threads_;
  std::vector<PersistentWorkerState> worker_states_;
};

PersistentPreparedRestoreExecutor& persistent_restore_executor() {
  static PersistentPreparedRestoreExecutor executor;
  return executor;
}

QatDpProfile run_persistent_prepared_decompress(
    std::shared_ptr<QatPreparedPayload> payload,
    uint8_t* output_ptr,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances,
    const std::vector<int64_t>* chunk_indices = nullptr,
    const std::vector<uint64_t>* output_offsets = nullptr,
    int64_t window_output_bytes = -1,
    int64_t window_compressed_bytes = -1) {
  (void)qat_state();
  return persistent_restore_executor().run(
      std::move(payload),
      output_ptr,
      inflight,
      batch_size,
      max_instances,
      chunk_indices,
      output_offsets,
      window_output_bytes,
      window_compressed_bytes);
}

std::vector<int64_t> prepared_window_indices(torch::Tensor indices) {
  check_cpu_i64(indices, "indices");
  TORCH_CHECK(indices.dim() == 1, "indices must be one-dimensional");
  const auto* ptr = indices.data_ptr<int64_t>();
  std::vector<int64_t> out(static_cast<size_t>(indices.numel()));
  for (int64_t idx = 0; idx < indices.numel(); ++idx) {
    out[static_cast<size_t>(idx)] = ptr[idx];
  }
  return out;
}

void prepared_window_offsets(
    const QatPreparedPayload& payload,
    const std::vector<int64_t>& indices,
    std::vector<uint64_t>& offsets,
    int64_t& output_bytes,
    int64_t& compressed_bytes) {
  offsets.clear();
  offsets.reserve(indices.size());
  output_bytes = 0;
  compressed_bytes = 0;
  for (const int64_t index : indices) {
    TORCH_CHECK(index >= 0 &&
                    static_cast<uint64_t>(index) < payload.chunks.size(),
                "prepared window chunk index out of range");
    const PreparedChunk& chunk =
        payload.chunks[static_cast<size_t>(index)];
    offsets.push_back(static_cast<uint64_t>(output_bytes));
    output_bytes += static_cast<int64_t>(chunk.output_len);
    compressed_bytes += static_cast<int64_t>(chunk.input_len);
  }
}

torch::Tensor qat_deflate_decompress_prepared_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  const uint64_t total_start_ns = now_ns();
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");
  if (payload->chunks.empty()) {
    QatDpProfile profile;
    profile.total_ns = now_ns() - total_start_ns;
    profile.chunks = 0;
    profile.workers = 0;
    profile.inflight = inflight;
    profile.batch = batch_size;
    profile.compressed_bytes = payload->compressed_bytes;
    profile.unpacked_bytes = payload->output_bytes;
    set_qat_last_profile(profile);
    return torch::empty({payload->output_bytes}, torch::dtype(torch::kUInt8));
  }
  auto output = torch::empty({payload->output_bytes}, torch::dtype(torch::kUInt8));
  auto* output_ptr = output.data_ptr<uint8_t>();
  QatDpProfile profile = run_persistent_prepared_decompress(
      payload, output_ptr, inflight, batch_size, max_instances);
  profile.total_ns = now_ns() - total_start_ns;
  set_qat_last_profile(profile);
  return output;
}

torch::Tensor qat_deflate_decompress_prepared_into_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    torch::Tensor output,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  const uint64_t total_start_ns = now_ns();
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  check_cpu_u8(output, "output");
  TORCH_CHECK(output.dim() == 1, "output must be one-dimensional");
  TORCH_CHECK(output.numel() == payload->output_bytes,
              "output byte size must match prepared payload");
  QatDpProfile profile = run_persistent_prepared_decompress(
      payload, output.data_ptr<uint8_t>(), inflight, batch_size, max_instances);
  profile.total_ns = now_ns() - total_start_ns;
  set_qat_last_profile(profile);
  return output;
}

py::dict qat_dp_profile_to_dict(const QatDpProfile& profile) {
  const uint64_t thread_gap_ns =
      profile.total_ns > profile.worker_wall_ns_max
          ? profile.total_ns - profile.worker_wall_ns_max
          : 0;
  py::dict out;
  out["total_ms"] = ns_to_ms(profile.total_ns);
  out["worker_wall_ms_max"] = ns_to_ms(profile.worker_wall_ns_max);
  out["thread_launch_join_ms"] = ns_to_ms(thread_gap_ns);
  out["slot_alloc_ms_sum"] = ns_to_ms(profile.slot_alloc_ns_sum);
  out["qat_enqueue_ms_sum"] = ns_to_ms(profile.enqueue_ns_sum);
  out["qat_poll_wait_ms_sum"] = ns_to_ms(profile.poll_wait_ns_sum);
  out["qat_submit_poll_ms_sum"] =
      ns_to_ms(profile.enqueue_ns_sum + profile.poll_wait_ns_sum);
  out["qat_submit_poll_ms_critical"] = ns_to_ms(profile.submit_poll_ns_critical);
  out["qae_output_copy_ms_sum"] = ns_to_ms(profile.output_copy_ns_sum);
  out["chunks"] = profile.chunks;
  out["workers"] = profile.workers;
  out["inflight"] = profile.inflight;
  out["batch"] = profile.batch;
  out["compressed_bytes"] = profile.compressed_bytes;
  out["unpacked_bytes"] = profile.unpacked_bytes;
  out["persistent_workers"] = profile.persistent_workers;
  out["compressed_gbps_qat_critical"] =
      profile.submit_poll_ns_critical > 0
          ? static_cast<double>(profile.compressed_bytes) * 8.0 /
                static_cast<double>(profile.submit_poll_ns_critical)
          : 0.0;
  out["unpacked_gbps_qat_critical"] =
      profile.submit_poll_ns_critical > 0
          ? static_cast<double>(profile.unpacked_bytes) * 8.0 /
                static_cast<double>(profile.submit_poll_ns_critical)
          : 0.0;
  return out;
}

torch::Tensor qat_deflate_decompress_prepared_window_into_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    torch::Tensor indices,
    torch::Tensor output,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  const uint64_t total_start_ns = now_ns();
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");
  check_cpu_u8(output, "output");
  TORCH_CHECK(output.dim() == 1, "output must be one-dimensional");
  std::vector<int64_t> window_indices = prepared_window_indices(indices);
  std::vector<uint64_t> output_offsets;
  int64_t output_bytes = 0;
  int64_t compressed_bytes = 0;
  prepared_window_offsets(
      *payload, window_indices, output_offsets, output_bytes, compressed_bytes);
  TORCH_CHECK(output.numel() == output_bytes,
              "output byte size must match prepared window");
  QatDpProfile profile = run_persistent_prepared_decompress(
      payload,
      output.data_ptr<uint8_t>(),
      inflight,
      batch_size,
      max_instances,
      &window_indices,
      &output_offsets,
      output_bytes,
      compressed_bytes);
  profile.total_ns = now_ns() - total_start_ns;
  set_qat_last_profile(profile);
  return output;
}

py::tuple qat_deflate_decompress_prepared_window_with_profile_into_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    torch::Tensor indices,
    torch::Tensor output,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  const uint64_t total_start_ns = now_ns();
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");
  check_cpu_u8(output, "output");
  TORCH_CHECK(output.dim() == 1, "output must be one-dimensional");
  std::vector<int64_t> window_indices = prepared_window_indices(indices);
  std::vector<uint64_t> output_offsets;
  int64_t output_bytes = 0;
  int64_t compressed_bytes = 0;
  prepared_window_offsets(
      *payload, window_indices, output_offsets, output_bytes, compressed_bytes);
  TORCH_CHECK(output.numel() == output_bytes,
              "output byte size must match prepared window");
  QatDpProfile profile;
  {
    py::gil_scoped_release release;
    profile = run_persistent_prepared_decompress(
        payload,
        output.data_ptr<uint8_t>(),
        inflight,
        batch_size,
        max_instances,
        &window_indices,
        &output_offsets,
        output_bytes,
        compressed_bytes);
  }
  profile.total_ns = now_ns() - total_start_ns;
  set_qat_last_profile(profile);
  return py::make_tuple(output, qat_dp_profile_to_dict(profile));
}

torch::Tensor qat_deflate_decompress_prepared_window_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    torch::Tensor indices,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances) {
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  std::vector<int64_t> window_indices = prepared_window_indices(indices);
  std::vector<uint64_t> output_offsets;
  int64_t output_bytes = 0;
  int64_t compressed_bytes = 0;
  prepared_window_offsets(
      *payload, window_indices, output_offsets, output_bytes, compressed_bytes);
  auto output = torch::empty({output_bytes}, torch::dtype(torch::kUInt8));
  return qat_deflate_decompress_prepared_window_into_dp(
      std::move(payload),
      indices,
      output,
      inflight,
      batch_size,
      max_instances);
}

py::dict qat_deflate_profile_prepared_window_dp(
    std::shared_ptr<QatPreparedPayload> payload,
    int64_t inflight,
    int64_t batch_size,
    int64_t max_instances,
    int64_t loops) {
  TORCH_CHECK(payload != nullptr, "prepared payload must not be null");
  TORCH_CHECK(inflight > 0 && batch_size > 0, "inflight and batch_size must be positive");
  TORCH_CHECK(loops > 0, "loops must be positive");
  auto& state = qat_state();
  const uint32_t workers =
      worker_count_for(state.size(max_instances), max_instances, payload->chunks.size());
  if (payload->chunks.empty()) {
    py::dict out;
    out["api"] = "qat_dp_window_batch_no_copy";
    out["total_ms"] = 0.0;
    out["worker_wall_ms_max"] = 0.0;
    out["slot_alloc_ms_sum"] = 0.0;
    out["qat_enqueue_ms_sum"] = 0.0;
    out["qat_poll_wait_ms_sum"] = 0.0;
    out["chunks"] = 0;
    out["workers"] = 0;
    out["loops"] = loops;
    out["compressed_bytes"] = payload->compressed_bytes;
    out["unpacked_bytes"] = payload->output_bytes;
    out["compressed_gbps_worker"] = 0.0;
    out["unpacked_gbps_worker"] = 0.0;
    return out;
  }

  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);
  std::vector<WorkerProfile> worker_profiles(workers);
  std::mutex barrier_mutex;
  std::condition_variable barrier_cv;
  uint32_t ready_workers = 0;
  bool start_workers = false;

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
        const uint64_t slot_alloc_start_ns = now_ns();
        for (uint32_t idx = 0; idx < slot_count; ++idx) {
          auto slot = std::make_unique<Slot>();
          alloc_slot(ctx, *slot, dst_capacity);
          slots.push_back(std::move(slot));
        }
        worker_profiles[worker].slot_alloc_ns += now_ns() - slot_alloc_start_ns;

        {
          std::unique_lock<std::mutex> lock(barrier_mutex);
          ready_workers++;
          barrier_cv.notify_one();
          barrier_cv.wait(lock, [&]() { return start_workers; });
        }

        const uint64_t worker_start_ns = now_ns();
        std::vector<CpaDcDpOpData*> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        uint64_t submitted = 0;
        uint64_t completed = 0;
        for (int64_t loop = 0; loop < loops; ++loop) {
          uint64_t next = 0;
          while (true) {
            while (submitted - completed >= slot_count) {
              Slot& slot = *slots[completed % slot_count];
              const uint64_t poll_start_ns = now_ns();
              poll_until_done(ctx, slot);
              worker_profiles[worker].poll_wait_ns += now_ns() - poll_start_ns;
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
            const uint64_t enqueue_start_ns = now_ns();
            submit_batch(ctx, batch);
            worker_profiles[worker].enqueue_ns += now_ns() - enqueue_start_ns;
            submitted += batch.size();
          }
        }
        while (completed < submitted) {
          Slot& slot = *slots[completed % slot_count];
          const uint64_t poll_start_ns = now_ns();
          poll_until_done(ctx, slot);
          worker_profiles[worker].poll_wait_ns += now_ns() - poll_start_ns;
          completed++;
        }
        worker_profiles[worker].wall_ns = now_ns() - worker_start_ns;
      } catch (...) {
        errors[worker] = std::current_exception();
        {
          std::lock_guard<std::mutex> lock(barrier_mutex);
          ready_workers++;
          barrier_cv.notify_one();
        }
      }
    });
  }

  {
    std::unique_lock<std::mutex> lock(barrier_mutex);
    barrier_cv.wait(lock, [&]() { return ready_workers >= workers; });
    start_workers = true;
  }
  const uint64_t total_start_ns = now_ns();
  barrier_cv.notify_all();
  for (auto& thread : threads) {
    thread.join();
  }
  const uint64_t total_ns = now_ns() - total_start_ns;
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }

  QatDpProfile profile;
  profile.total_ns = total_ns;
  profile.chunks = static_cast<int64_t>(payload->chunks.size());
  profile.workers = static_cast<int64_t>(workers);
  profile.inflight = inflight;
  profile.batch = batch_size;
  profile.compressed_bytes = payload->compressed_bytes * loops;
  profile.unpacked_bytes = payload->output_bytes * loops;
  for (const WorkerProfile& worker_profile : worker_profiles) {
    profile.worker_wall_ns_max =
        std::max(profile.worker_wall_ns_max, worker_profile.wall_ns);
    profile.slot_alloc_ns_sum += worker_profile.slot_alloc_ns;
    profile.enqueue_ns_sum += worker_profile.enqueue_ns;
    profile.poll_wait_ns_sum += worker_profile.poll_wait_ns;
    profile.submit_poll_ns_critical = std::max(
        profile.submit_poll_ns_critical,
        worker_profile.enqueue_ns + worker_profile.poll_wait_ns);
  }
  set_qat_last_profile(profile);

  py::dict out = qat_deflate_last_profile_dp();
  out["api"] = "qat_dp_window_batch_no_copy";
  out["loops"] = loops;
  out["compressed_gbps_worker"] =
      profile.worker_wall_ns_max > 0
          ? static_cast<double>(profile.compressed_bytes) * 8.0 /
                static_cast<double>(profile.worker_wall_ns_max)
          : 0.0;
  out["unpacked_gbps_worker"] =
      profile.worker_wall_ns_max > 0
          ? static_cast<double>(profile.unpacked_bytes) * 8.0 /
                static_cast<double>(profile.worker_wall_ns_max)
          : 0.0;
  return out;
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

py::dict qat_deflate_last_profile_dp() {
  QatDpProfile profile;
  {
    std::lock_guard<std::mutex> lock(qat_profile_mutex());
    profile = qat_last_profile();
  }
  return qat_dp_profile_to_dict(profile);
}

py::dict qat_deflate_last_prepare_profile_dp() {
  QatPrepareProfile profile;
  {
    std::lock_guard<std::mutex> lock(qat_profile_mutex());
    profile = qat_last_prepare_profile();
  }
  const uint64_t accounted_ns =
      profile.parse_ns + profile.qat_state_ns + profile.qae_alloc_ns_sum +
      profile.qae_input_copy_ns_sum + profile.file_read_io_ns +
      profile.file_read_sleep_ns;
  const uint64_t file_read_ns = profile.file_read_io_ns + profile.file_read_sleep_ns;
  const uint64_t other_ns =
      profile.total_ns > accounted_ns ? profile.total_ns - accounted_ns : 0;
  py::dict out;
  out["prepare_total_ms"] = ns_to_ms(profile.total_ns);
  out["prepare_parse_ms"] = ns_to_ms(profile.parse_ns);
  out["prepare_qat_state_ms"] = ns_to_ms(profile.qat_state_ns);
  out["prepare_qae_alloc_ms_sum"] = ns_to_ms(profile.qae_alloc_ns_sum);
  out["prepare_qae_input_copy_ms_sum"] =
      ns_to_ms(profile.qae_input_copy_ns_sum);
  out["prepare_file_read_ms"] = ns_to_ms(file_read_ns);
  out["prepare_file_read_io_ms"] = ns_to_ms(profile.file_read_io_ns);
  out["prepare_file_read_sleep_ms"] = ns_to_ms(profile.file_read_sleep_ns);
  out["prepare_other_ms"] = ns_to_ms(other_ns);
  out["prepare_chunks"] = profile.chunks;
  out["prepare_workers"] = profile.workers;
  out["prepare_compressed_bytes"] = profile.compressed_bytes;
  out["prepare_unpacked_bytes"] = profile.unpacked_bytes;
  out["prepare_bundle_bytes"] = profile.bundle_bytes;
  out["prepare_file_read_bytes"] = profile.file_read_bytes;
  out["prepare_qae_buffer_reuse_hits"] = profile.qae_buffer_reuse_hits;
  out["prepare_qae_buffer_reuse_misses"] = profile.qae_buffer_reuse_misses;
  out["prepare_dynamic_huffman"] = profile.dynamic_huffman;
  out["prepare_qae_input_copy_gbps"] =
      profile.qae_input_copy_ns_sum > 0
          ? static_cast<double>(profile.compressed_bytes) * 8.0 /
                static_cast<double>(profile.qae_input_copy_ns_sum)
          : 0.0;
  out["prepare_file_read_gbps"] =
      file_read_ns > 0
          ? static_cast<double>(profile.file_read_bytes) * 8.0 /
                static_cast<double>(file_read_ns)
          : 0.0;
  out["prepare_file_read_io_gbps"] =
      profile.file_read_io_ns > 0
          ? static_cast<double>(profile.file_read_bytes) * 8.0 /
                static_cast<double>(profile.file_read_io_ns)
          : 0.0;
  out["prepare_qae_allocs_per_chunk"] =
      profile.chunks > 0 ? 1.0 : 0.0;
  return out;
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
      "qat_deflate_prepare_bundle_dp",
      &qat_deflate_prepare_bundle_dp,
      py::arg("bundle"),
      py::arg("dynamic_huffman") = true,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_prepare_bundle_from_file_dp",
      &qat_deflate_prepare_bundle_from_file_dp,
      py::arg("path"),
      py::arg("dynamic_huffman") = true,
      py::arg("max_instances") = 16,
      py::arg("bandwidth_gbps") = 0.0,
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
      "qat_deflate_decompress_prepared_into_dp",
      &qat_deflate_decompress_prepared_into_dp,
      py::arg("payload"),
      py::arg("output"),
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress_prepared_window_dp",
      &qat_deflate_decompress_prepared_window_dp,
      py::arg("payload"),
      py::arg("indices"),
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress_prepared_window_into_dp",
      &qat_deflate_decompress_prepared_window_into_dp,
      py::arg("payload"),
      py::arg("indices"),
      py::arg("output"),
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress_prepared_window_with_profile_into_dp",
      &qat_deflate_decompress_prepared_window_with_profile_into_dp,
      py::arg("payload"),
      py::arg("indices"),
      py::arg("output"),
      py::arg("inflight") = 2,
      py::arg("batch") = 16,
      py::arg("max_instances") = 16);
  m.def(
      "qat_deflate_profile_prepared_window_dp",
      &qat_deflate_profile_prepared_window_dp,
      py::arg("payload"),
      py::arg("inflight") = 32,
      py::arg("batch") = 32,
      py::arg("max_instances") = 16,
      py::arg("loops") = 1);
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
  m.def(
      "qat_deflate_last_profile_dp",
      &qat_deflate_last_profile_dp);
  m.def(
      "qat_deflate_last_prepare_profile_dp",
      &qat_deflate_last_prepare_profile_dp);
}
