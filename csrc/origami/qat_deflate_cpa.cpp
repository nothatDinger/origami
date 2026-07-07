#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

extern "C" {
#include <cpa.h>
#include <cpa_dc.h>
#include <icp_sal_poll.h>
#include <icp_sal_user.h>
#include <qae_mem.h>
}

namespace {

namespace py = pybind11;

void check_cpu_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
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

struct QatCallbackTag {
  std::atomic<bool> done{false};
  CpaStatus status{CPA_STATUS_FAIL};
};

void qat_callback(void* callback_tag, CpaStatus status) {
  auto* tag = static_cast<QatCallbackTag*>(callback_tag);
  tag->status = status;
  tag->done.store(true, std::memory_order_release);
}

void wait_for_qat(CpaInstanceHandle instance, QatCallbackTag& tag, const char* what) {
  const auto start = std::chrono::steady_clock::now();
  while (!tag.done.load(std::memory_order_acquire)) {
    const CpaStatus status = icp_sal_DcPollInstance(instance, 0);
    if (status != CPA_STATUS_SUCCESS && status != CPA_STATUS_RETRY) {
      throw_qat(status, "icp_sal_DcPollInstance");
    }
    if (std::chrono::steady_clock::now() - start > std::chrono::seconds(30)) {
      throw std::runtime_error(std::string(what) + " timed out while polling QAT");
    }
    if (status == CPA_STATUS_RETRY) {
      std::this_thread::yield();
    }
  }
  require_qat(tag.status, what);
}

struct QatBuffer {
  void ensure(CpaInstanceHandle instance, int node, size_t size) {
    Cpa32U metadata_size = 0;
    require_qat(cpaDcBufferListGetMetaSize(instance, 1, &metadata_size),
                "cpaDcBufferListGetMetaSize");
    if (metadata_size > metadata_capacity) {
      if (metadata != nullptr) {
        qaeMemFreeNUMA(&metadata);
      }
      metadata = qaeMemAllocNUMA(metadata_size, node, 64);
      if (metadata == nullptr) {
        throw std::runtime_error("qaeMemAllocNUMA failed for QAT metadata");
      }
      metadata_capacity = metadata_size;
    }
    if (metadata != nullptr) {
      std::memset(metadata, 0, metadata_capacity);
    }

    if (size > data_capacity) {
      if (data != nullptr) {
        qaeMemFreeNUMA(&data);
      }
      data = qaeMemAllocNUMA(size, node, 64);
      if (data == nullptr) {
        throw std::runtime_error("qaeMemAllocNUMA failed for QAT data");
      }
      data_capacity = size;
    }

    flat[0].dataLenInBytes = static_cast<Cpa32U>(size);
    flat[0].pData = static_cast<Cpa8U*>(data);
    list.numBuffers = 1;
    list.pBuffers = flat;
    list.pUserData = nullptr;
    list.pPrivateMetaData = metadata;
  }

  QatBuffer(const QatBuffer&) = delete;
  QatBuffer& operator=(const QatBuffer&) = delete;

  QatBuffer() = default;

  ~QatBuffer() {
    if (data != nullptr) {
      qaeMemFreeNUMA(&data);
    }
    if (metadata != nullptr) {
      qaeMemFreeNUMA(&metadata);
    }
  }

  size_t metadata_capacity = 0;
  size_t data_capacity = 0;
  void* metadata = nullptr;
  void* data = nullptr;
  CpaFlatBuffer flat[1]{};
  CpaBufferList list{};
};

struct QatInstance {
  CpaInstanceHandle handle = nullptr;
  int node = 0;
  std::mutex mutex;
  QatBuffer src_buffer;
  QatBuffer dst_buffer;
  std::vector<std::unique_ptr<QatBuffer>> intermediate_buffers;
  std::vector<CpaBufferList*> intermediate_buffer_ptrs;
};

size_t dynamic_intermediate_bytes() {
  constexpr size_t kDefaultBytes = 32ULL << 20;
  constexpr size_t kMinBytes = 1ULL << 20;
  const char* value = std::getenv("CACHEGEN_QAT_DYNAMIC_INTERMEDIATE_BYTES");
  if (value == nullptr || value[0] == '\0') {
    return kDefaultBytes;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || parsed < kMinBytes) {
    return kDefaultBytes;
  }
  return static_cast<size_t>(parsed);
}

class QatState {
 public:
  QatState() {
    require_qat(icp_sal_userStart("cachegen_cpu_qat"), "icp_sal_userStart");

    Cpa16U count = 0;
    require_qat(cpaDcGetNumInstances(&count), "cpaDcGetNumInstances");
    if (count == 0) {
      throw std::runtime_error("QAT data compression has no available instances");
    }

    std::vector<CpaInstanceHandle> handles(count);
    require_qat(cpaDcGetInstances(count, handles.data()), "cpaDcGetInstances");
    instances.reserve(count);
    for (Cpa16U idx = 0; idx < count; ++idx) {
      require_qat(cpaDcSetAddressTranslation(
                      handles[idx], reinterpret_cast<CpaVirtualToPhysical>(qaeVirtToPhysNUMA)),
                  "cpaDcSetAddressTranslation");

      CpaInstanceInfo2 info{};
      require_qat(cpaDcInstanceGetInfo2(handles[idx], &info), "cpaDcInstanceGetInfo2");
      if (info.isOffloaded != CPA_TRUE) {
        continue;
      }

      auto instance = std::make_unique<QatInstance>();
      instance->handle = handles[idx];
      instance->node = static_cast<int>(info.nodeAffinity);

      Cpa16U intermediate_count = 0;
      const CpaStatus intermediate_status =
          cpaDcGetNumIntermediateBuffers(handles[idx], &intermediate_count);
      if (intermediate_status != CPA_STATUS_UNSUPPORTED) {
        require_qat(intermediate_status, "cpaDcGetNumIntermediateBuffers");
      } else {
        intermediate_count = 0;
      }
      if (intermediate_count > 0) {
        const size_t intermediate_bytes = dynamic_intermediate_bytes();
        instance->intermediate_buffers.reserve(intermediate_count);
        instance->intermediate_buffer_ptrs.reserve(intermediate_count);
        for (Cpa16U buffer_idx = 0; buffer_idx < intermediate_count; ++buffer_idx) {
          auto buffer = std::make_unique<QatBuffer>();
          buffer->ensure(handles[idx], instance->node, intermediate_bytes);
          instance->intermediate_buffer_ptrs.push_back(&buffer->list);
          instance->intermediate_buffers.push_back(std::move(buffer));
        }
      }

      require_qat(
          cpaDcStartInstance(
              handles[idx],
              intermediate_count,
              intermediate_count > 0 ? instance->intermediate_buffer_ptrs.data() : nullptr),
          "cpaDcStartInstance");
      instances.push_back(std::move(instance));
    }

    if (instances.empty()) {
      throw std::runtime_error("QAT data compression has no offloaded instances");
    }
  }

  QatState(const QatState&) = delete;
  QatState& operator=(const QatState&) = delete;

  ~QatState() {
    for (auto& instance : instances) {
      if (instance->handle != nullptr) {
        cpaDcStopInstance(instance->handle);
      }
    }
    icp_sal_userStop();
  }

  QatInstance& pick(int64_t job) {
    return *instances[static_cast<size_t>(job % static_cast<int64_t>(instances.size()))];
  }

  size_t size() const {
    return instances.size();
  }

 private:
  std::vector<std::unique_ptr<QatInstance>> instances;
};

QatState& qat_state() {
  static QatState state;
  return state;
}

enum class QatCodec {
  DeflateRaw,
  Zlib,
  Lz4Raw,
};

CpaDcCompLZ4BlockMaxSize lz4_block_size_for(int64_t input_size) {
  if (input_size <= 64 * 1024) {
    return CPA_DC_LZ4_MAX_BLOCK_SIZE_64K;
  }
  if (input_size <= 256 * 1024) {
    return CPA_DC_LZ4_MAX_BLOCK_SIZE_256K;
  }
  if (input_size <= 1024 * 1024) {
    return CPA_DC_LZ4_MAX_BLOCK_SIZE_1M;
  }
  if (input_size <= 4 * 1024 * 1024) {
    return CPA_DC_LZ4_MAX_BLOCK_SIZE_4M;
  }
  throw std::runtime_error("QAT LZ4 chunk size must be <= 4 MiB");
}

CpaDcNsSetupData make_setup(
    QatCodec codec,
    CpaDcSessionDir direction,
    int64_t input_size,
    CpaDcHuffType huff_type) {
  CpaDcNsSetupData setup{};
  setup.compLevel = CPA_DC_L1;
  setup.compType = codec == QatCodec::Lz4Raw ? CPA_DC_LZ4 : CPA_DC_DEFLATE;
  setup.huffType = codec == QatCodec::Lz4Raw ? CPA_DC_HT_STATIC : huff_type;
  setup.autoSelectBestHuffmanTree = CPA_DC_ASB_DISABLED;
  setup.sessDirection = direction;
  setup.sessState = CPA_DC_STATELESS;
  setup.windowSize = CPA_DC_WINSIZE_32K;
  setup.minMatch = CPA_DC_MIN_3_BYTE_MATCH;
  setup.lz4BlockMaxSize =
      codec == QatCodec::Lz4Raw ? lz4_block_size_for(input_size) : CPA_DC_LZ4_MAX_BLOCK_SIZE_64K;
  setup.lz4BlockChecksum = CPA_FALSE;
  setup.lz4BlockIndependence = CPA_TRUE;
  setup.checksum = CPA_DC_NONE;
  setup.accumulateXXHash = CPA_FALSE;
  return setup;
}

CpaDcOpData make_op_data(bool compress) {
  CpaDcOpData op{};
  op.flushFlag = CPA_DC_FLUSH_FINAL;
  // This platform enforces compress-and-verify for QAT compression.
  op.compressAndVerify = compress ? CPA_TRUE : CPA_FALSE;
  op.compressAndVerifyAndRecover = CPA_FALSE;
  op.integrityCrcCheck = CPA_FALSE;
  op.verifyHwIntegrityCrcs = CPA_FALSE;
  op.integrityCrcSize = CPA_DC_INTEGRITY_CRC32;
  op.pCrcData = nullptr;
  return op;
}

uint32_t adler32_bytes(const uint8_t* data, int64_t size) {
  constexpr uint32_t kMod = 65521U;
  uint32_t a = 1U;
  uint32_t b = 0U;
  for (int64_t idx = 0; idx < size; ++idx) {
    a += data[idx];
    if (a >= kMod) {
      a -= kMod;
    }
    b += a;
    if (b >= kMod) {
      b %= kMod;
    }
  }
  return (b << 16) | a;
}

int64_t compressed_bound(
    QatInstance& instance,
    QatCodec codec,
    int64_t input_size,
    CpaDcHuffType huff_type) {
  Cpa32U output_bound = 0;
  if (codec == QatCodec::Lz4Raw) {
    require_qat(
        cpaDcLZ4CompressBound(
            instance.handle, static_cast<Cpa32U>(input_size), &output_bound),
        "cpaDcLZ4CompressBound");
  } else {
    require_qat(
        cpaDcDeflateCompressBound(
            instance.handle, huff_type, static_cast<Cpa32U>(input_size), &output_bound),
        "cpaDcDeflateCompressBound");
  }
  return output_bound;
}

std::vector<uint8_t> compress_chunk(
    QatInstance& instance,
    QatCodec codec,
    const uint8_t* input,
    int64_t input_size,
    CpaDcHuffType huff_type) {
  std::lock_guard<std::mutex> guard(instance.mutex);
  const int64_t output_bound = compressed_bound(instance, codec, input_size, huff_type) + 128;
  instance.src_buffer.ensure(instance.handle, instance.node, static_cast<size_t>(input_size));
  instance.dst_buffer.ensure(instance.handle, instance.node, static_cast<size_t>(output_bound));
  std::memcpy(instance.src_buffer.data, input, static_cast<size_t>(input_size));
  instance.src_buffer.flat[0].dataLenInBytes = static_cast<Cpa32U>(input_size);
  instance.dst_buffer.flat[0].dataLenInBytes = static_cast<Cpa32U>(output_bound);

  CpaDcNsSetupData setup = make_setup(codec, CPA_DC_DIR_COMPRESS, input_size, huff_type);
  Cpa32U header_bytes = 0;
  if (codec == QatCodec::Zlib) {
    auto* out = static_cast<uint8_t*>(instance.dst_buffer.data);
    out[0] = 0x78U;
    out[1] = 0x01U;
    header_bytes = 2;
  }

  CpaBufferList dst_list = instance.dst_buffer.list;
  CpaFlatBuffer dst_flat = instance.dst_buffer.flat[0];
  dst_flat.pData = static_cast<Cpa8U*>(instance.dst_buffer.data) + header_bytes;
  dst_flat.dataLenInBytes = static_cast<Cpa32U>(output_bound) - header_bytes;
  dst_list.pBuffers = &dst_flat;

  CpaDcOpData op = make_op_data(true);
  CpaDcRqResults results{};
  QatCallbackTag tag;
  const CpaStatus status = cpaDcNsCompressData(
      instance.handle, &setup, &instance.src_buffer.list, &dst_list, &op, &results, qat_callback, &tag);
  require_qat(status, "cpaDcNsCompressData submit");
  wait_for_qat(instance.handle, tag, "cpaDcNsCompressData");
  if (results.status != CPA_DC_OK) {
    throw std::runtime_error(
        "QAT deflate compression request status " + std::to_string(results.status));
  }
  if (results.consumed != static_cast<Cpa32U>(input_size)) {
    throw std::runtime_error("QAT deflate did not consume the full input chunk");
  }

  Cpa32U footer_bytes = 0;
  if (codec == QatCodec::Zlib) {
    const uint32_t checksum = adler32_bytes(input, input_size);
    auto* footer = static_cast<uint8_t*>(instance.dst_buffer.data) + header_bytes + results.produced;
    footer[0] = static_cast<uint8_t>((checksum >> 24) & 0xFFU);
    footer[1] = static_cast<uint8_t>((checksum >> 16) & 0xFFU);
    footer[2] = static_cast<uint8_t>((checksum >> 8) & 0xFFU);
    footer[3] = static_cast<uint8_t>(checksum & 0xFFU);
    footer_bytes = 4;
  }

  std::vector<uint8_t> output(header_bytes + results.produced + footer_bytes);
  std::memcpy(output.data(), instance.dst_buffer.data, output.size());
  return output;
}

void decompress_chunk(
    QatInstance& instance,
    QatCodec codec,
    const uint8_t* input,
    int64_t input_size,
    uint8_t* output,
    int64_t output_size,
    CpaDcHuffType huff_type) {
  std::lock_guard<std::mutex> guard(instance.mutex);
  const uint8_t* compressed_input = input;
  int64_t compressed_input_size = input_size;
  if (codec == QatCodec::Zlib) {
    if (input_size < 6 || input[0] != 0x78U) {
      throw std::runtime_error("QAT zlib input does not look like a zlib stream");
    }
    compressed_input = input + 2;
    compressed_input_size = input_size - 6;
  }
  instance.src_buffer.ensure(instance.handle, instance.node, static_cast<size_t>(compressed_input_size));
  instance.dst_buffer.ensure(instance.handle, instance.node, static_cast<size_t>(output_size) + 64);
  std::memcpy(instance.src_buffer.data, compressed_input, static_cast<size_t>(compressed_input_size));
  instance.src_buffer.flat[0].dataLenInBytes = static_cast<Cpa32U>(compressed_input_size);
  instance.dst_buffer.flat[0].dataLenInBytes = static_cast<Cpa32U>(output_size + 64);

  CpaDcNsSetupData setup = make_setup(codec, CPA_DC_DIR_DECOMPRESS, output_size, huff_type);
  CpaDcOpData op = make_op_data(false);
  CpaDcRqResults results{};
  QatCallbackTag tag;
  const CpaStatus status = cpaDcNsDecompressData(
      instance.handle, &setup, &instance.src_buffer.list, &instance.dst_buffer.list, &op, &results, qat_callback, &tag);
  require_qat(status, "cpaDcNsDecompressData submit");
  wait_for_qat(instance.handle, tag, "cpaDcNsDecompressData");
  if (results.status != CPA_DC_OK) {
    throw std::runtime_error(
        "QAT deflate decompression request status " + std::to_string(results.status));
  }
  if (results.produced != static_cast<Cpa32U>(output_size)) {
    throw std::runtime_error("QAT deflate decoded size did not match the expected chunk size");
  }
  std::memcpy(output, instance.dst_buffer.data, static_cast<size_t>(output_size));
}

}  // namespace

std::vector<torch::Tensor> qat_compress(
    torch::Tensor input,
    int64_t chunk_bytes,
    QatCodec codec,
    CpaDcHuffType huff_type = CPA_DC_HT_STATIC) {
  check_cpu_contiguous(input, "input");
  TORCH_CHECK(input.element_size() == 1, "input must have a one-byte dtype");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= std::numeric_limits<Cpa32U>::max(),
              "chunk_bytes must fit in a QAT request");

  const int64_t input_bytes = input.numel();
  const int64_t chunks = (input_bytes + chunk_bytes - 1) / chunk_bytes;
  auto lengths = torch::empty({chunks}, torch::dtype(torch::kInt32));
  auto* lengths_ptr = lengths.data_ptr<int32_t>();
  const auto* input_ptr = static_cast<const uint8_t*>(input.data_ptr());
  auto& state = qat_state();

  std::vector<std::vector<uint8_t>> compressed(static_cast<size_t>(chunks));
  at::parallel_for(0, chunks, 1, [&](int64_t begin, int64_t end) {
    for (int64_t chunk = begin; chunk < end; ++chunk) {
      const int64_t offset = chunk * chunk_bytes;
      const int64_t size = std::min<int64_t>(chunk_bytes, input_bytes - offset);
      auto& instance = state.pick(chunk);
      compressed[static_cast<size_t>(chunk)] =
          compress_chunk(instance, codec, input_ptr + offset, size, huff_type);
      lengths_ptr[chunk] = static_cast<int32_t>(compressed[static_cast<size_t>(chunk)].size());
    }
  });

  int64_t total = 0;
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    total += lengths_ptr[chunk];
  }
  auto bytestream = torch::empty({total}, torch::dtype(torch::kUInt8));
  auto* bytestream_ptr = bytestream.data_ptr<uint8_t>();
  int64_t offset = 0;
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    const auto& data = compressed[static_cast<size_t>(chunk)];
    std::memcpy(bytestream_ptr + offset, data.data(), data.size());
    offset += static_cast<int64_t>(data.size());
  }
  return {bytestream, lengths};
}

torch::Tensor qat_decompress(
    torch::Tensor bytestream,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes,
    QatCodec codec,
    CpaDcHuffType huff_type = CPA_DC_HT_STATIC) {
  check_cpu_contiguous(bytestream, "bytestream");
  check_cpu_contiguous(lengths, "lengths");
  TORCH_CHECK(bytestream.scalar_type() == torch::kUInt8, "bytestream must be uint8");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt32, "lengths must be int32");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be one-dimensional");
  TORCH_CHECK(output_bytes >= 0, "output_bytes must be non-negative");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= std::numeric_limits<Cpa32U>::max(),
              "chunk_bytes must fit in a QAT request");

  const int64_t chunks = lengths.numel();
  TORCH_CHECK(chunks == (output_bytes + chunk_bytes - 1) / chunk_bytes,
              "length count must match output_bytes and chunk_bytes");
  auto output = torch::empty({output_bytes}, torch::dtype(torch::kUInt8));
  const auto* bytestream_ptr = bytestream.data_ptr<uint8_t>();
  const auto* lengths_ptr = lengths.data_ptr<int32_t>();
  auto* output_ptr = output.data_ptr<uint8_t>();

  std::vector<int64_t> offsets(static_cast<size_t>(chunks + 1), 0);
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    TORCH_CHECK(lengths_ptr[chunk] >= 0, "lengths must be non-negative");
    offsets[static_cast<size_t>(chunk + 1)] =
        offsets[static_cast<size_t>(chunk)] + lengths_ptr[chunk];
  }
  TORCH_CHECK(offsets.back() == bytestream.numel(), "lengths must sum to bytestream size");

  auto& state = qat_state();
  at::parallel_for(0, chunks, 1, [&](int64_t begin, int64_t end) {
    for (int64_t chunk = begin; chunk < end; ++chunk) {
      const int64_t output_offset = chunk * chunk_bytes;
      const int64_t output_size = std::min<int64_t>(chunk_bytes, output_bytes - output_offset);
      auto& instance = state.pick(chunk);
      decompress_chunk(
          instance,
          codec,
          bytestream_ptr + offsets[static_cast<size_t>(chunk)],
          lengths_ptr[chunk],
          output_ptr + output_offset,
          output_size,
          huff_type);
    }
  });
  return output;
}

std::vector<torch::Tensor> qat_deflate_compress(torch::Tensor input, int64_t chunk_bytes) {
  return qat_compress(input, chunk_bytes, QatCodec::DeflateRaw);
}

std::vector<torch::Tensor> qat_deflate_compress_opts(
    torch::Tensor input,
    int64_t chunk_bytes,
    bool dynamic_huffman) {
  return qat_compress(
      input,
      chunk_bytes,
      QatCodec::DeflateRaw,
      dynamic_huffman ? CPA_DC_HT_FULL_DYNAMIC : CPA_DC_HT_STATIC);
}

torch::Tensor qat_deflate_decompress(
    torch::Tensor bytestream,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes) {
  return qat_decompress(bytestream, lengths, output_bytes, chunk_bytes, QatCodec::DeflateRaw);
}

torch::Tensor qat_deflate_decompress_opts(
    torch::Tensor bytestream,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes,
    bool dynamic_huffman) {
  return qat_decompress(
      bytestream,
      lengths,
      output_bytes,
      chunk_bytes,
      QatCodec::DeflateRaw,
      dynamic_huffman ? CPA_DC_HT_FULL_DYNAMIC : CPA_DC_HT_STATIC);
}

std::vector<torch::Tensor> qat_zlib_compress(torch::Tensor input, int64_t chunk_bytes) {
  return qat_compress(input, chunk_bytes, QatCodec::Zlib);
}

torch::Tensor qat_zlib_decompress(
    torch::Tensor bytestream,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes) {
  return qat_decompress(bytestream, lengths, output_bytes, chunk_bytes, QatCodec::Zlib);
}

std::vector<torch::Tensor> qat_lz4_compress(torch::Tensor input, int64_t chunk_bytes) {
  return qat_compress(input, chunk_bytes, QatCodec::Lz4Raw);
}

torch::Tensor qat_lz4_decompress(
    torch::Tensor bytestream,
    torch::Tensor lengths,
    int64_t output_bytes,
    int64_t chunk_bytes) {
  return qat_decompress(bytestream, lengths, output_bytes, chunk_bytes, QatCodec::Lz4Raw);
}

int64_t qat_deflate_instance_count() {
  return static_cast<int64_t>(qat_state().size());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qat_deflate_compress",
      &qat_deflate_compress,
      py::arg("input"),
      py::arg("chunk_bytes") = 1 << 20,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_compress_opts",
      &qat_deflate_compress_opts,
      py::arg("input"),
      py::arg("chunk_bytes") = 1 << 20,
      py::arg("dynamic_huffman") = false,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress",
      &qat_deflate_decompress,
      py::arg("bytestream"),
      py::arg("lengths"),
      py::arg("output_bytes"),
      py::arg("chunk_bytes") = 1 << 20,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_deflate_decompress_opts",
      &qat_deflate_decompress_opts,
      py::arg("bytestream"),
      py::arg("lengths"),
      py::arg("output_bytes"),
      py::arg("chunk_bytes") = 1 << 20,
      py::arg("dynamic_huffman") = false,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_zlib_compress",
      &qat_zlib_compress,
      py::arg("input"),
      py::arg("chunk_bytes") = 1 << 20,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_zlib_decompress",
      &qat_zlib_decompress,
      py::arg("bytestream"),
      py::arg("lengths"),
      py::arg("output_bytes"),
      py::arg("chunk_bytes") = 1 << 20,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_lz4_compress",
      &qat_lz4_compress,
      py::arg("input"),
      py::arg("chunk_bytes") = 1 << 20,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "qat_lz4_decompress",
      &qat_lz4_decompress,
      py::arg("bytestream"),
      py::arg("lengths"),
      py::arg("output_bytes"),
      py::arg("chunk_bytes") = 1 << 20,
      py::call_guard<py::gil_scoped_release>());
  m.def("qat_deflate_instance_count", &qat_deflate_instance_count);
}
