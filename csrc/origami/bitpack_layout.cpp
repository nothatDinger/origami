// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>

#include <array>
#include <cstdint>
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

struct SourceLayout {
  int64_t rank;
  std::array<int64_t, 4> shape;
  std::array<int64_t, 4> stride;
  int64_t token_axis;
  int64_t head_axis;
  int64_t channel_axis;
  int64_t layer_axis;
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

int64_t product_shape(const SourceLayout& layout) {
  int64_t product = 1;
  for (int64_t idx = 0; idx < layout.rank; ++idx) {
    product *= layout.shape[idx];
  }
  return product;
}

bool axis_in_rank(const int64_t axis, const int64_t rank) {
  return axis >= 0 && axis < rank;
}

void check_unique_axis(const int64_t lhs,
                       const int64_t rhs,
                       const char* lhs_name,
                       const char* rhs_name) {
  TORCH_CHECK(lhs != rhs, "Origami source layout axes ", lhs_name, " and ",
              rhs_name, " must be unique");
}

SourceLayout read_source_layout(torch::Tensor source_shape,
                                torch::Tensor axis_positions,
                                const int64_t token_count,
                                const int64_t num_heads,
                                const int64_t head_dim) {
  check_cpu_i64(source_shape, "source_shape");
  check_cpu_i64(axis_positions, "axis_positions");
  TORCH_CHECK(source_shape.dim() == 1,
              "source_shape must be a one-dimensional int64 tensor");
  TORCH_CHECK(axis_positions.dim() == 1 && axis_positions.numel() == 4,
              "axis_positions must be [token_axis, head_axis, head_dim_axis, layer_axis]");
  TORCH_CHECK(source_shape.numel() == 3 || source_shape.numel() == 4,
              "Origami source layout supports rank 3 or rank 4");

  SourceLayout layout{};
  layout.rank = source_shape.numel();
  layout.shape = {1, 1, 1, 1};
  layout.stride = {1, 1, 1, 1};
  const auto* shape_ptr = source_shape.data_ptr<int64_t>();
  for (int64_t idx = 0; idx < layout.rank; ++idx) {
    TORCH_CHECK(shape_ptr[idx] > 0, "source_shape dimensions must be positive");
    layout.shape[idx] = shape_ptr[idx];
  }
  layout.stride[layout.rank - 1] = 1;
  for (int64_t idx = layout.rank - 2; idx >= 0; --idx) {
    layout.stride[idx] = layout.stride[idx + 1] * layout.shape[idx + 1];
  }

  const auto* axes = axis_positions.data_ptr<int64_t>();
  layout.token_axis = axes[0];
  layout.head_axis = axes[1];
  layout.channel_axis = axes[2];
  layout.layer_axis = axes[3];
  TORCH_CHECK(axis_in_rank(layout.token_axis, layout.rank),
              "token axis must be present in source layout");
  TORCH_CHECK(axis_in_rank(layout.head_axis, layout.rank),
              "head axis must be present in source layout");
  TORCH_CHECK(axis_in_rank(layout.channel_axis, layout.rank),
              "head_dim axis must be present in source layout");
  TORCH_CHECK(layout.layer_axis == -1 || axis_in_rank(layout.layer_axis, layout.rank),
              "layer axis must be -1 or present in source layout");
  check_unique_axis(layout.token_axis, layout.head_axis, "token", "head");
  check_unique_axis(layout.token_axis, layout.channel_axis, "token", "head_dim");
  check_unique_axis(layout.head_axis, layout.channel_axis, "head", "head_dim");
  if (layout.layer_axis != -1) {
    check_unique_axis(layout.layer_axis, layout.token_axis, "layer", "token");
    check_unique_axis(layout.layer_axis, layout.head_axis, "layer", "head");
    check_unique_axis(layout.layer_axis, layout.channel_axis, "layer", "head_dim");
    TORCH_CHECK(layout.shape[layout.layer_axis] == 1,
                "per-layer Origami native layout requires layer axis size 1");
  }

  TORCH_CHECK(layout.shape[layout.token_axis] == token_count,
              "source token axis size must match token_count");
  TORCH_CHECK(layout.shape[layout.head_axis] == num_heads,
              "source head axis size must match num_heads");
  TORCH_CHECK(layout.shape[layout.channel_axis] == head_dim,
              "source head_dim axis size must match head_dim");
  return layout;
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

inline int64_t source_index(const SourceLayout& layout,
                            const int64_t token,
                            const int64_t head,
                            const int64_t channel) {
  std::array<int64_t, 4> coords{0, 0, 0, 0};
  coords[layout.token_axis] = token;
  coords[layout.head_axis] = head;
  coords[layout.channel_axis] = channel;
  if (layout.layer_axis != -1) {
    coords[layout.layer_axis] = 0;
  }
  int64_t index = 0;
  for (int64_t axis = 0; axis < layout.rank; ++axis) {
    index += coords[axis] * layout.stride[axis];
  }
  return index;
}

void pack_spec_scalar(const uint8_t* src,
                      uint8_t* dst,
                      const ChunkSpec& spec,
                      const int64_t bits,
                      const SourceLayout& layout) {
  if (bits == 8) {
    int64_t out = 0;
    for (int64_t head = spec.head_start; head < spec.head_end; ++head) {
      for (int64_t channel = spec.channel_start; channel < spec.channel_end;
           ++channel) {
        for (int64_t token = spec.token_start; token < spec.token_end; ++token) {
          dst[out++] = src[source_index(layout, token, head, channel)];
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
        const uint8_t value = src[source_index(layout, token, head, channel)] & mask;
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
                        const SourceLayout& layout) {
  if (bits == 8) {
    int64_t in = 0;
    for (int64_t head = spec.head_start; head < spec.head_end; ++head) {
      for (int64_t channel = spec.channel_start; channel < spec.channel_end;
           ++channel) {
        for (int64_t token = spec.token_start; token < spec.token_end; ++token) {
          TORCH_CHECK(in < src_bytes, "Origami chunk is shorter than expected");
          dst[source_index(layout, token, head, channel)] = src[in++];
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
        dst[source_index(layout, token, head, channel)] = value;
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

std::vector<torch::Tensor> pack_canonical_storage_chunks_impl(
    torch::Tensor symbols,
    int64_t bits,
    torch::Tensor source_shape,
    torch::Tensor axis_positions,
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
              "canonical symbol shape must be positive");
  const SourceLayout layout = read_source_layout(
      source_shape, axis_positions, token_count, num_heads, head_dim);
  TORCH_CHECK(symbols.numel() == product_shape(layout),
              "symbols tensor size must match source_shape product");

  const auto* spec_ptr = chunk_specs.data_ptr<int64_t>();
  const auto* src = symbols.data_ptr<uint8_t>();
  std::vector<torch::Tensor> outputs;
  outputs.reserve(static_cast<size_t>(chunk_specs.size(0)));
  for (int64_t idx = 0; idx < chunk_specs.size(0); ++idx) {
    const ChunkSpec spec =
        read_spec(spec_ptr + idx * 6, token_count, num_heads, head_dim);
    const int64_t bytes = packed_bytes_for_symbols(spec_symbol_count(spec), bits);
    auto output = torch::empty({bytes}, torch::dtype(torch::kUInt8));
    pack_spec_scalar(src, output.data_ptr<uint8_t>(), spec, bits, layout);
    outputs.push_back(output);
  }
  return outputs;
}

torch::Tensor unpack_canonical_storage_chunks_impl(
    std::vector<torch::Tensor> chunks,
    int64_t bits,
    torch::Tensor source_shape,
    torch::Tensor axis_positions,
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
              "canonical symbol shape must be positive");
  const SourceLayout layout = read_source_layout(
      source_shape, axis_positions, token_count, num_heads, head_dim);

  auto output = torch::zeros({product_shape(layout)}, torch::dtype(torch::kUInt8));
  auto* dst = output.data_ptr<uint8_t>();
  const auto* spec_ptr = chunk_specs.data_ptr<int64_t>();
  for (int64_t idx = 0; idx < chunk_specs.size(0); ++idx) {
    auto chunk = chunks[static_cast<size_t>(idx)]
                     .detach()
                     .cpu()
                     .to(torch::kUInt8)
                     .contiguous();
    check_cpu_u8(chunk, "chunk");
    const ChunkSpec spec =
        read_spec(spec_ptr + idx * 6, token_count, num_heads, head_dim);
    const int64_t expected_bytes = packed_bytes_for_symbols(spec_symbol_count(spec), bits);
    TORCH_CHECK(chunk.numel() == expected_bytes,
                "Origami packed chunk byte size mismatch: got ", chunk.numel(),
                ", expected ", expected_bytes);
    unpack_spec_scalar(chunk.data_ptr<uint8_t>(), chunk.numel(), dst, spec, bits,
                       layout);
  }
  return output;
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

std::vector<torch::Tensor> pack_canonical_storage_chunks(
    torch::Tensor symbols,
    int64_t bits,
    torch::Tensor source_shape,
    torch::Tensor axis_positions,
    int64_t token_count,
    int64_t num_heads,
    int64_t head_dim,
    torch::Tensor chunk_specs) {
  return pack_canonical_storage_chunks_impl(symbols, bits, source_shape,
                                            axis_positions, token_count,
                                            num_heads, head_dim, chunk_specs);
}

torch::Tensor unpack_canonical_storage_chunks(std::vector<torch::Tensor> chunks,
                                              int64_t bits,
                                              torch::Tensor source_shape,
                                              torch::Tensor axis_positions,
                                              int64_t token_count,
                                              int64_t num_heads,
                                              int64_t head_dim,
                                              torch::Tensor chunk_specs) {
  return unpack_canonical_storage_chunks_impl(chunks, bits, source_shape,
                                              axis_positions, token_count,
                                              num_heads, head_dim, chunk_specs);
}

std::vector<torch::Tensor> pack_head_channel_chunks(torch::Tensor symbols,
                                                    int64_t bits,
                                                    int64_t token_count,
                                                    int64_t num_heads,
                                                    int64_t head_dim,
                                                    torch::Tensor chunk_specs) {
  auto source_shape = torch::tensor({token_count, num_heads, head_dim},
                                    torch::dtype(torch::kInt64));
  auto axis_positions = torch::tensor({0, 1, 2, -1}, torch::dtype(torch::kInt64));
  return pack_canonical_storage_chunks_impl(symbols, bits, source_shape,
                                            axis_positions, token_count,
                                            num_heads, head_dim, chunk_specs);
}

torch::Tensor unpack_head_channel_chunks(std::vector<torch::Tensor> chunks,
                                          int64_t bits,
                                          int64_t token_count,
                                          int64_t num_heads,
                                          int64_t head_dim,
                                          torch::Tensor chunk_specs) {
  auto source_shape = torch::tensor({token_count, num_heads, head_dim},
                                    torch::dtype(torch::kInt64));
  auto axis_positions = torch::tensor({0, 1, 2, -1}, torch::dtype(torch::kInt64));
  return unpack_canonical_storage_chunks_impl(chunks, bits, source_shape,
                                              axis_positions, token_count,
                                              num_heads, head_dim, chunk_specs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_canonical_storage_chunks", &pack_canonical_storage_chunks,
        py::arg("symbols"), py::arg("bits"), py::arg("source_shape"),
        py::arg("axis_positions"), py::arg("token_count"),
        py::arg("num_heads"), py::arg("head_dim"), py::arg("chunk_specs"));
  m.def("unpack_canonical_storage_chunks", &unpack_canonical_storage_chunks,
        py::arg("chunks"), py::arg("bits"), py::arg("source_shape"),
        py::arg("axis_positions"), py::arg("token_count"),
        py::arg("num_heads"), py::arg("head_dim"), py::arg("chunk_specs"));
  m.def("pack_head_channel_chunks", &pack_head_channel_chunks,
        py::arg("symbols"), py::arg("bits"), py::arg("token_count"),
        py::arg("num_heads"), py::arg("head_dim"), py::arg("chunk_specs"));
  m.def("unpack_head_channel_chunks", &unpack_head_channel_chunks,
        py::arg("chunks"), py::arg("bits"), py::arg("token_count"),
        py::arg("num_heads"), py::arg("head_dim"), py::arg("chunk_specs"));
  m.def("cpu_isa", &runtime_isa);
}
