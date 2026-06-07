// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace {

struct ChunkSpec {
  int64_t head_start;
  int64_t head_end;
  int64_t channel_start;
  int64_t channel_end;
  int64_t token_start;
  int64_t token_end;
};

void check_cpu_u8(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dtype() == torch::kUInt8, name, " must be uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cpu_i64(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dtype() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bits(const int64_t bits) {
  TORCH_CHECK(bits == 2 || bits == 4 || bits == 8,
              "Origami bitpack supports only 2, 4, or 8 bit symbols");
}

ChunkSpec read_spec(const int64_t* row,
                    const int64_t token_count,
                    const int64_t num_heads,
                    const int64_t head_dim) {
  ChunkSpec spec{row[0], row[1], row[2], row[3], row[4], row[5]};
  TORCH_CHECK(0 <= spec.head_start && spec.head_start < spec.head_end &&
                  spec.head_end <= num_heads,
              "invalid head range in Origami chunk spec");
  TORCH_CHECK(0 <= spec.channel_start && spec.channel_start < spec.channel_end &&
                  spec.channel_end <= head_dim,
              "invalid channel range in Origami chunk spec");
  TORCH_CHECK(0 <= spec.token_start && spec.token_start < spec.token_end &&
                  spec.token_end <= token_count,
              "invalid token range in Origami chunk spec");
  return spec;
}

int64_t spec_symbol_count(const ChunkSpec& spec) {
  return (spec.head_end - spec.head_start) *
         (spec.channel_end - spec.channel_start) *
         (spec.token_end - spec.token_start);
}

int64_t packed_bytes_for_symbols(const int64_t symbol_count, const int64_t bits) {
  return (symbol_count * bits + 7) / 8;
}

inline int64_t logical_index(const int64_t token,
                             const int64_t head,
                             const int64_t channel,
                             const int64_t num_heads,
                             const int64_t head_dim) {
  return ((token * num_heads) + head) * head_dim + channel;
}

void pack_spec_scalar(const uint8_t* src,
                      uint8_t* dst,
                      const ChunkSpec& spec,
                      const int64_t bits,
                      const int64_t num_heads,
                      const int64_t head_dim) {
  if (bits == 8) {
    int64_t out = 0;
    for (int64_t head = spec.head_start; head < spec.head_end; ++head) {
      for (int64_t channel = spec.channel_start; channel < spec.channel_end;
           ++channel) {
        for (int64_t token = spec.token_start; token < spec.token_end; ++token) {
          dst[out++] = src[logical_index(token, head, channel, num_heads, head_dim)];
        }
      }
    }
    return;
  }

  uint8_t current = 0;
  int slot = 0;
  int64_t out = 0;
  const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
  for (int64_t head = spec.head_start; head < spec.head_end; ++head) {
    for (int64_t channel = spec.channel_start; channel < spec.channel_end;
         ++channel) {
      for (int64_t token = spec.token_start; token < spec.token_end; ++token) {
        const uint8_t value =
            src[logical_index(token, head, channel, num_heads, head_dim)] & mask;
        current = static_cast<uint8_t>(current | (value << (slot * bits)));
        ++slot;
        if (slot * bits == 8) {
          dst[out++] = current;
          current = 0;
          slot = 0;
        }
      }
    }
  }
  if (slot != 0) {
    dst[out++] = current;
  }
}

void unpack_spec_scalar(const uint8_t* src,
                        const int64_t src_bytes,
                        uint8_t* dst,
                        const ChunkSpec& spec,
                        const int64_t bits,
                        const int64_t num_heads,
                        const int64_t head_dim) {
  if (bits == 8) {
    int64_t in = 0;
    for (int64_t head = spec.head_start; head < spec.head_end; ++head) {
      for (int64_t channel = spec.channel_start; channel < spec.channel_end;
           ++channel) {
        for (int64_t token = spec.token_start; token < spec.token_end; ++token) {
          TORCH_CHECK(in < src_bytes, "Origami chunk is shorter than expected");
          dst[logical_index(token, head, channel, num_heads, head_dim)] = src[in++];
        }
      }
    }
    return;
  }

  const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
  int slot = 0;
  int64_t in = 0;
  uint8_t current = src_bytes > 0 ? src[0] : 0;
  for (int64_t head = spec.head_start; head < spec.head_end; ++head) {
    for (int64_t channel = spec.channel_start; channel < spec.channel_end;
         ++channel) {
      for (int64_t token = spec.token_start; token < spec.token_end; ++token) {
        TORCH_CHECK(in < src_bytes, "Origami chunk is shorter than expected");
        const uint8_t value = static_cast<uint8_t>((current >> (slot * bits)) & mask);
        dst[logical_index(token, head, channel, num_heads, head_dim)] = value;
        ++slot;
        if (slot * bits == 8) {
          ++in;
          slot = 0;
          current = in < src_bytes ? src[in] : 0;
        }
      }
    }
  }
}

std::string runtime_isa() {
#if defined(__x86_64__) || defined(_M_X64)
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
    return "avx512";
  }
  if (__builtin_cpu_supports("avx2")) {
    return "avx2";
  }
#endif
  return "scalar";
}

}  // namespace

std::vector<torch::Tensor> pack_head_channel_chunks(torch::Tensor symbols,
                                                    int64_t bits,
                                                    int64_t token_count,
                                                    int64_t num_heads,
                                                    int64_t head_dim,
                                                    torch::Tensor chunk_specs) {
  check_bits(bits);
  check_cpu_u8(symbols, "symbols");
  check_cpu_i64(chunk_specs, "chunk_specs");
  TORCH_CHECK(symbols.dim() == 1, "symbols must be flattened uint8 data");
  TORCH_CHECK(chunk_specs.dim() == 2 && chunk_specs.size(1) == 6,
              "chunk_specs must have shape [N, 6]");
  TORCH_CHECK(token_count > 0 && num_heads > 0 && head_dim > 0,
              "symbol shape must be positive");
  TORCH_CHECK(symbols.numel() >= token_count * num_heads * head_dim,
              "symbols tensor is smaller than token_count*num_heads*head_dim");

  const auto* spec_ptr = chunk_specs.data_ptr<int64_t>();
  const auto* src = symbols.data_ptr<uint8_t>();
  std::vector<torch::Tensor> outputs;
  outputs.reserve(static_cast<size_t>(chunk_specs.size(0)));
  for (int64_t idx = 0; idx < chunk_specs.size(0); ++idx) {
    const ChunkSpec spec =
        read_spec(spec_ptr + idx * 6, token_count, num_heads, head_dim);
    const int64_t bytes = packed_bytes_for_symbols(spec_symbol_count(spec), bits);
    auto output = torch::empty({bytes}, torch::dtype(torch::kUInt8));
    pack_spec_scalar(src, output.data_ptr<uint8_t>(), spec, bits, num_heads, head_dim);
    outputs.push_back(output);
  }
  return outputs;
}

torch::Tensor unpack_head_channel_chunks(std::vector<torch::Tensor> chunks,
                                          int64_t bits,
                                          int64_t token_count,
                                          int64_t num_heads,
                                          int64_t head_dim,
                                          torch::Tensor chunk_specs) {
  check_bits(bits);
  check_cpu_i64(chunk_specs, "chunk_specs");
  TORCH_CHECK(chunk_specs.dim() == 2 && chunk_specs.size(1) == 6,
              "chunk_specs must have shape [N, 6]");
  TORCH_CHECK(static_cast<int64_t>(chunks.size()) == chunk_specs.size(0),
              "chunk count must match chunk_specs rows");
  TORCH_CHECK(token_count > 0 && num_heads > 0 && head_dim > 0,
              "symbol shape must be positive");

  auto output = torch::zeros({token_count * num_heads * head_dim},
                             torch::dtype(torch::kUInt8));
  auto* dst = output.data_ptr<uint8_t>();
  const auto* spec_ptr = chunk_specs.data_ptr<int64_t>();
  for (int64_t idx = 0; idx < chunk_specs.size(0); ++idx) {
    auto chunk = chunks[static_cast<size_t>(idx)].detach().cpu().to(torch::kUInt8).contiguous();
    check_cpu_u8(chunk, "chunk");
    const ChunkSpec spec =
        read_spec(spec_ptr + idx * 6, token_count, num_heads, head_dim);
    const int64_t expected_bytes = packed_bytes_for_symbols(spec_symbol_count(spec), bits);
    TORCH_CHECK(chunk.numel() == expected_bytes,
                "Origami packed chunk byte size mismatch: got ", chunk.numel(),
                ", expected ", expected_bytes);
    unpack_spec_scalar(chunk.data_ptr<uint8_t>(), chunk.numel(), dst, spec, bits,
                       num_heads, head_dim);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_head_channel_chunks", &pack_head_channel_chunks,
        py::arg("symbols"), py::arg("bits"), py::arg("token_count"),
        py::arg("num_heads"), py::arg("head_dim"), py::arg("chunk_specs"));
  m.def("unpack_head_channel_chunks", &unpack_head_channel_chunks,
        py::arg("chunks"), py::arg("bits"), py::arg("token_count"),
        py::arg("num_heads"), py::arg("head_dim"), py::arg("chunk_specs"));
  m.def("cpu_isa", &runtime_isa);
}
