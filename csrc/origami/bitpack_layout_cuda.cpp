// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

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
  int64_t shape0;
  int64_t shape1;
  int64_t shape2;
  int64_t shape3;
  int64_t stride0;
  int64_t stride1;
  int64_t stride2;
  int64_t stride3;
  int64_t token_axis;
  int64_t head_axis;
  int64_t channel_axis;
  int64_t layer_axis;
};

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_i64(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cpu_i64(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  check_i64(tensor, name);
}

void check_bits(const int64_t bits) {
  TORCH_CHECK(bits >= 1 && bits <= 8,
              "Origami CUDA bitpack supports only 1..8 bit symbols");
}

int64_t shape_at(const SourceLayout& layout, int64_t axis) {
  if (axis == 0) {
    return layout.shape0;
  }
  if (axis == 1) {
    return layout.shape1;
  }
  if (axis == 2) {
    return layout.shape2;
  }
  return layout.shape3;
}

int64_t product_shape(const SourceLayout& layout) {
  int64_t product = 1;
  for (int64_t idx = 0; idx < layout.rank; ++idx) {
    product *= shape_at(layout, idx);
  }
  return product;
}

SourceLayout read_source_layout(at::Tensor source_shape,
                                at::Tensor axis_positions,
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

  std::array<int64_t, 4> shape{1, 1, 1, 1};
  std::array<int64_t, 4> stride{1, 1, 1, 1};
  const auto* shape_ptr = source_shape.data_ptr<int64_t>();
  const int64_t rank = source_shape.numel();
  for (int64_t idx = 0; idx < rank; ++idx) {
    TORCH_CHECK(shape_ptr[idx] > 0, "source_shape dimensions must be positive");
    shape[idx] = shape_ptr[idx];
  }
  stride[rank - 1] = 1;
  for (int64_t idx = rank - 2; idx >= 0; --idx) {
    stride[idx] = stride[idx + 1] * shape[idx + 1];
  }

  const auto* axes = axis_positions.data_ptr<int64_t>();
  SourceLayout layout{
      rank,
      shape[0],
      shape[1],
      shape[2],
      shape[3],
      stride[0],
      stride[1],
      stride[2],
      stride[3],
      axes[0],
      axes[1],
      axes[2],
      axes[3],
  };
  TORCH_CHECK(layout.token_axis >= 0 && layout.token_axis < rank,
              "token axis must be present in source layout");
  TORCH_CHECK(layout.head_axis >= 0 && layout.head_axis < rank,
              "head axis must be present in source layout");
  TORCH_CHECK(layout.channel_axis >= 0 && layout.channel_axis < rank,
              "head_dim axis must be present in source layout");
  TORCH_CHECK(layout.layer_axis == -1 ||
                  (layout.layer_axis >= 0 && layout.layer_axis < rank),
              "layer axis must be -1 or present in source layout");
  TORCH_CHECK(shape_at(layout, layout.token_axis) == token_count,
              "source token axis size must match token_count");
  TORCH_CHECK(shape_at(layout, layout.head_axis) == num_heads,
              "source head axis size must match num_heads");
  TORCH_CHECK(shape_at(layout, layout.channel_axis) == head_dim,
              "source head_dim axis size must match head_dim");
  if (layout.layer_axis != -1) {
    TORCH_CHECK(shape_at(layout, layout.layer_axis) == 1,
                "per-layer Origami native layout requires layer axis size 1");
  }
  return layout;
}

std::vector<ChunkSpec> read_chunk_specs(at::Tensor chunk_specs,
                                        int64_t token_count,
                                        int64_t num_heads,
                                        int64_t head_dim) {
  check_i64(chunk_specs, "chunk_specs");
  TORCH_CHECK(chunk_specs.dim() == 2 && chunk_specs.size(1) == 6,
              "chunk_specs must have shape [N, 6]");
  auto specs_cpu = chunk_specs.device().is_cpu() ? chunk_specs : chunk_specs.cpu();
  const auto* ptr = specs_cpu.data_ptr<int64_t>();
  std::vector<ChunkSpec> specs;
  specs.reserve(static_cast<size_t>(specs_cpu.size(0)));
  for (int64_t idx = 0; idx < specs_cpu.size(0); ++idx) {
    const int64_t* row = ptr + idx * 6;
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
    specs.push_back(spec);
  }
  return specs;
}

int64_t spec_symbol_count(const ChunkSpec& spec) {
  return (spec.head_end - spec.head_start) *
         (spec.channel_end - spec.channel_start) *
         (spec.token_end - spec.token_start);
}

int64_t packed_bytes_for_symbols(const int64_t symbol_count, const int64_t bits) {
  return (symbol_count * bits + 7) / 8;
}

}  // namespace

void origami_unpack_canonical_storage_chunks_launch(
    const uint8_t* flat_chunks,
    const int64_t* chunk_offsets,
    const int64_t* bits_per_chunk,
    const int64_t* chunk_specs,
    uint8_t* output,
    int64_t rank,
    int64_t shape0,
    int64_t shape1,
    int64_t shape2,
    int64_t shape3,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t stride3,
    int64_t token_axis,
    int64_t head_axis,
    int64_t channel_axis,
    int64_t layer_axis,
    int64_t num_chunks,
    int64_t max_symbols,
    cudaStream_t stream);

void origami_cachegen_unpack_dequantize_launch(
    const uint8_t* bytestream,
    const int64_t* offsets,
    const int64_t* bits_per_layer,
    const int64_t* key_bins,
    const int64_t* value_bins,
    const void* max_key,
    const void* max_value,
    void* output,
    bool max_is_half,
    int64_t layers,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    cudaStream_t stream);

void origami_cachegen_unpack_dequantize_to_kv_cache_launch(
    const uint8_t* bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int64_t key_bits,
    int64_t value_bits,
    int64_t key_bins,
    int64_t value_bins,
    const void* max_key,
    const void* max_value,
    void* kv_cache,
    const int64_t* block_ids,
    bool max_is_half,
    bool output_is_bfloat16,
    bool cache_blocks_first,
    int64_t num_cache_blocks,
    int64_t block_count,
    int64_t block_size,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    cudaStream_t stream);

void origami_kivi_dequantize_to_kv_cache_launch(
    const uint8_t* bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int64_t bits,
    int64_t group_size,
    int64_t sink_tokens,
    const void* key_scale,
    const void* key_zero,
    const void* value_scale,
    const void* value_zero,
    const void* key_sink,
    const void* value_sink,
    void* kv_cache,
    const int64_t* block_ids,
    bool metadata_is_half,
    bool output_is_bfloat16,
    bool cache_blocks_first,
    int64_t num_cache_blocks,
    int64_t block_count,
    int64_t block_size,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    cudaStream_t stream);

void origami_fused_prefix_attention_launch(
    const uint8_t* bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int codec,
    int bits,
    int64_t group_size,
    int64_t sink_tokens,
    const void* key_scale,
    const void* key_zero,
    const void* value_scale,
    const void* value_zero,
    const void* key_sink,
    const void* value_sink,
    const void* query,
    const void* kv_cache,
    const int32_t* block_table,
    void* output,
    bool query_is_bfloat16,
    bool cache_is_bfloat16,
    bool cache_blocks_first,
    int64_t num_cache_blocks,
    int64_t block_size,
    int64_t prefix_tokens,
    int64_t query_start_position,
    int64_t sequence_length,
    int64_t query_tokens,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float softmax_scale,
    cudaStream_t stream);

at::Tensor unpack_canonical_storage_chunks_cuda(
    at::Tensor flat_chunks,
    at::Tensor chunk_offsets,
    at::Tensor bits_per_chunk,
    at::Tensor source_shape,
    at::Tensor axis_positions,
    int64_t token_count,
    int64_t num_heads,
    int64_t head_dim,
    at::Tensor chunk_specs) {
  check_cuda_contiguous(flat_chunks, "flat_chunks");
  check_cuda_contiguous(chunk_offsets, "chunk_offsets");
  check_cuda_contiguous(bits_per_chunk, "bits_per_chunk");
  check_cuda_contiguous(chunk_specs, "chunk_specs");
  TORCH_CHECK(flat_chunks.scalar_type() == at::kByte,
              "flat_chunks must be uint8");
  check_i64(chunk_offsets, "chunk_offsets");
  check_i64(bits_per_chunk, "bits_per_chunk");
  check_i64(chunk_specs, "chunk_specs");
  TORCH_CHECK(chunk_specs.dim() == 2 && chunk_specs.size(1) == 6,
              "chunk_specs must have shape [N, 6]");
  const int64_t num_chunks = chunk_specs.size(0);
  TORCH_CHECK(chunk_offsets.numel() == num_chunks + 1,
              "chunk_offsets length must be chunk count + 1");
  TORCH_CHECK(bits_per_chunk.numel() == num_chunks,
              "bits_per_chunk length must equal chunk count");
  TORCH_CHECK(token_count > 0 && num_heads > 0 && head_dim > 0,
              "canonical symbol shape must be positive");

  const SourceLayout layout = read_source_layout(
      source_shape, axis_positions, token_count, num_heads, head_dim);
  const std::vector<ChunkSpec> specs =
      read_chunk_specs(chunk_specs, token_count, num_heads, head_dim);

  auto offsets_cpu = chunk_offsets.cpu();
  auto bits_cpu = bits_per_chunk.cpu();
  const auto* offsets_ptr = offsets_cpu.data_ptr<int64_t>();
  const auto* bits_ptr = bits_cpu.data_ptr<int64_t>();
  int64_t max_symbols = 0;
  for (int64_t idx = 0; idx < num_chunks; ++idx) {
    check_bits(bits_ptr[idx]);
    const int64_t expected =
        packed_bytes_for_symbols(spec_symbol_count(specs[static_cast<size_t>(idx)]),
                                 bits_ptr[idx]);
    TORCH_CHECK(offsets_ptr[idx + 1] - offsets_ptr[idx] == expected,
                "Origami CUDA packed chunk byte size mismatch");
    max_symbols = std::max(
        max_symbols, spec_symbol_count(specs[static_cast<size_t>(idx)]));
  }

  const c10::cuda::CUDAGuard device_guard(flat_chunks.device());
  auto output = at::empty(
      {product_shape(layout)},
      at::TensorOptions().dtype(at::kByte).device(flat_chunks.device()));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      output.data_ptr<uint8_t>(),
      0,
      static_cast<size_t>(output.numel()),
      stream));
  origami_unpack_canonical_storage_chunks_launch(
      flat_chunks.data_ptr<uint8_t>(),
      chunk_offsets.data_ptr<int64_t>(),
      bits_per_chunk.data_ptr<int64_t>(),
      chunk_specs.data_ptr<int64_t>(),
      output.data_ptr<uint8_t>(),
      layout.rank,
      layout.shape0,
      layout.shape1,
      layout.shape2,
      layout.shape3,
      layout.stride0,
      layout.stride1,
      layout.stride2,
      layout.stride3,
      layout.token_axis,
      layout.head_axis,
      layout.channel_axis,
      layout.layer_axis,
      num_chunks,
      max_symbols,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor cachegen_unpack_dequantize_cuda(
    at::Tensor bytestream,
    at::Tensor offsets,
    at::Tensor bits_per_layer,
    at::Tensor key_bins,
    at::Tensor value_bins,
    at::Tensor max_key,
    at::Tensor max_value,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim) {
  check_cuda_contiguous(bytestream, "bytestream");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(bits_per_layer, "bits_per_layer");
  check_cuda_contiguous(key_bins, "key_bins");
  check_cuda_contiguous(value_bins, "value_bins");
  check_cuda_contiguous(max_key, "max_key");
  check_cuda_contiguous(max_value, "max_value");
  TORCH_CHECK(bytestream.scalar_type() == at::kByte,
              "bytestream must be uint8");
  check_i64(offsets, "offsets");
  check_i64(bits_per_layer, "bits_per_layer");
  check_i64(key_bins, "key_bins");
  check_i64(value_bins, "value_bins");
  TORCH_CHECK(max_key.scalar_type() == max_value.scalar_type(),
              "max_key and max_value must have the same dtype");
  TORCH_CHECK(max_key.scalar_type() == at::kHalf ||
                  max_key.scalar_type() == at::kFloat,
              "max tensors must be float16 or float32");
  TORCH_CHECK(bits_per_layer.numel() % 2 == 0,
              "bits_per_layer length must be even");
  const int64_t layers = bits_per_layer.numel() / 2;
  TORCH_CHECK(key_bins.numel() == layers, "key_bins length must match layers");
  TORCH_CHECK(value_bins.numel() == layers, "value_bins length must match layers");
  TORCH_CHECK(offsets.numel() == bits_per_layer.numel() + 1,
              "offsets length must be bits_per_layer length + 1");
  TORCH_CHECK(max_key.numel() == layers * tokens,
              "max_key must have layers * tokens values");
  TORCH_CHECK(max_value.numel() == layers * tokens,
              "max_value must have layers * tokens values");
  TORCH_CHECK(tokens > 0 && heads > 0 && head_dim > 0,
              "tokens, heads, and head_dim must be positive");

  auto bits_cpu = bits_per_layer.cpu();
  const auto* bits_ptr = bits_cpu.data_ptr<int64_t>();
  for (int64_t idx = 0; idx < bits_cpu.numel(); ++idx) {
    check_bits(bits_ptr[idx]);
  }

  const c10::cuda::CUDAGuard device_guard(bytestream.device());
  auto output = at::empty(
      {layers, 2, heads, tokens, head_dim},
      at::TensorOptions().dtype(at::kHalf).device(bytestream.device()));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  origami_cachegen_unpack_dequantize_launch(
      bytestream.data_ptr<uint8_t>(),
      offsets.data_ptr<int64_t>(),
      bits_per_layer.data_ptr<int64_t>(),
      key_bins.data_ptr<int64_t>(),
      value_bins.data_ptr<int64_t>(),
      max_key.data_ptr(),
      max_value.data_ptr(),
      output.data_ptr(),
      max_key.scalar_type() == at::kHalf,
      layers,
      tokens,
      heads,
      head_dim,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void cachegen_unpack_dequantize_to_kv_cache_cuda(
    at::Tensor bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int64_t key_bits,
    int64_t value_bits,
    int64_t key_bins,
    int64_t value_bins,
    at::Tensor max_key,
    at::Tensor max_value,
    at::Tensor kv_cache,
    at::Tensor block_ids,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim) {
  check_cuda_contiguous(bytestream, "bytestream");
  check_cuda_contiguous(max_key, "max_key");
  check_cuda_contiguous(max_value, "max_value");
  check_cuda_contiguous(kv_cache, "kv_cache");
  check_cuda_contiguous(block_ids, "block_ids");
  TORCH_CHECK(bytestream.scalar_type() == at::kByte,
              "bytestream must be uint8");
  TORCH_CHECK(kv_cache.scalar_type() == at::kHalf ||
                  kv_cache.scalar_type() == at::kBFloat16,
              "kv_cache must be float16 or bfloat16");
  check_i64(block_ids, "block_ids");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids must be CUDA");
  TORCH_CHECK(max_key.scalar_type() == max_value.scalar_type(),
              "max_key and max_value must have the same dtype");
  TORCH_CHECK(max_key.scalar_type() == at::kHalf ||
                  max_key.scalar_type() == at::kFloat,
              "max tensors must be float16 or float32");
  check_bits(key_bits);
  check_bits(value_bits);
  TORCH_CHECK(tokens > 0 && heads > 0 && head_dim > 0,
              "tokens, heads, and head_dim must be positive");
  TORCH_CHECK(key_bins > 1 && value_bins > 1,
              "CacheGen bins must be greater than one");
  TORCH_CHECK(key_offset >= 0 && key_bytes >= 0 &&
                  value_offset >= 0 && value_bytes >= 0,
              "invalid CacheGen stream offsets");
  TORCH_CHECK(key_offset + key_bytes <= bytestream.numel() &&
                  value_offset + value_bytes <= bytestream.numel(),
              "CacheGen stream ranges exceed bytestream length");
  TORCH_CHECK(max_key.numel() >= tokens && max_value.numel() >= tokens,
              "max tensors must contain at least tokens values");
  TORCH_CHECK(kv_cache.dim() == 5,
              "kv_cache must have rank 5");

  bool cache_blocks_first = false;
  int64_t num_cache_blocks = 0;
  int64_t block_size = 0;
  if (kv_cache.size(1) == 2) {
    // [num_blocks, 2, block_size, heads, head_dim]
    cache_blocks_first = true;
    num_cache_blocks = kv_cache.size(0);
    block_size = kv_cache.size(2);
    TORCH_CHECK(kv_cache.size(3) == heads && kv_cache.size(4) == head_dim,
                "kv_cache head shape mismatch");
  } else if (kv_cache.size(0) == 2) {
    // [2, num_blocks, block_size, heads, head_dim]
    cache_blocks_first = false;
    num_cache_blocks = kv_cache.size(1);
    block_size = kv_cache.size(2);
    TORCH_CHECK(kv_cache.size(3) == heads && kv_cache.size(4) == head_dim,
                "kv_cache head shape mismatch");
  } else {
    TORCH_CHECK(false,
                "kv_cache must be [num_blocks,2,block,heads,dim] or "
                "[2,num_blocks,block,heads,dim]");
  }
  TORCH_CHECK(block_size > 0, "kv_cache block size must be positive");
  const int64_t block_count = block_ids.numel();
  TORCH_CHECK(block_count > 0, "block_ids must be non-empty");
  TORCH_CHECK(tokens <= block_count * block_size,
              "tokens exceed destination block capacity");

  const c10::cuda::CUDAGuard device_guard(bytestream.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  origami_cachegen_unpack_dequantize_to_kv_cache_launch(
      bytestream.data_ptr<uint8_t>(),
      key_offset,
      key_bytes,
      value_offset,
      value_bytes,
      key_bits,
      value_bits,
      key_bins,
      value_bins,
      max_key.data_ptr(),
      max_value.data_ptr(),
      kv_cache.data_ptr(),
      block_ids.data_ptr<int64_t>(),
      max_key.scalar_type() == at::kHalf,
      kv_cache.scalar_type() == at::kBFloat16,
      cache_blocks_first,
      num_cache_blocks,
      block_count,
      block_size,
      tokens,
      heads,
      head_dim,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kivi_dequantize_to_kv_cache_cuda(
    at::Tensor bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int64_t bits,
    int64_t group_size,
    int64_t sink_tokens,
    at::Tensor key_scale,
    at::Tensor key_zero,
    at::Tensor value_scale,
    at::Tensor value_zero,
    at::Tensor key_sink,
    at::Tensor value_sink,
    at::Tensor kv_cache,
    at::Tensor block_ids,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim) {
  check_cuda_contiguous(bytestream, "bytestream");
  check_cuda_contiguous(key_scale, "key_scale");
  check_cuda_contiguous(key_zero, "key_zero");
  check_cuda_contiguous(value_scale, "value_scale");
  check_cuda_contiguous(value_zero, "value_zero");
  check_cuda_contiguous(key_sink, "key_sink");
  check_cuda_contiguous(value_sink, "value_sink");
  check_cuda_contiguous(kv_cache, "kv_cache");
  check_cuda_contiguous(block_ids, "block_ids");
  TORCH_CHECK(bytestream.scalar_type() == at::kByte,
              "bytestream must be uint8");
  TORCH_CHECK(kv_cache.scalar_type() == at::kHalf ||
                  kv_cache.scalar_type() == at::kBFloat16,
              "kv_cache must be float16 or bfloat16");
  check_i64(block_ids, "block_ids");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids must be CUDA");
  TORCH_CHECK(key_scale.scalar_type() == key_zero.scalar_type() &&
                  key_scale.scalar_type() == value_scale.scalar_type() &&
                  key_scale.scalar_type() == value_zero.scalar_type() &&
                  key_scale.scalar_type() == key_sink.scalar_type() &&
                  key_scale.scalar_type() == value_sink.scalar_type(),
              "KIVI metadata tensors must have the same dtype");
  TORCH_CHECK(key_scale.scalar_type() == at::kHalf ||
                  key_scale.scalar_type() == at::kFloat,
              "KIVI metadata tensors must be float16 or float32");
  check_bits(bits);
  TORCH_CHECK(group_size > 0, "KIVI group_size must be positive");
  TORCH_CHECK(sink_tokens >= 0 && sink_tokens <= tokens,
              "KIVI sink_tokens must be in [0, tokens]");
  TORCH_CHECK(tokens > 0 && heads > 0 && head_dim > 0,
              "tokens, heads, and head_dim must be positive");
  TORCH_CHECK(key_offset >= 0 && key_bytes >= 0 &&
                  value_offset >= 0 && value_bytes >= 0,
              "invalid KIVI stream offsets");
  TORCH_CHECK(key_offset + key_bytes <= bytestream.numel() &&
                  value_offset + value_bytes <= bytestream.numel(),
              "KIVI stream ranges exceed bytestream length");
  TORCH_CHECK(kv_cache.dim() == 5, "kv_cache must have rank 5");

  const int64_t body_tokens = tokens - sink_tokens;
  const int64_t channels = heads * head_dim;
  const int64_t key_groups =
      (body_tokens + group_size - 1) / group_size;
  const int64_t value_groups =
      (channels + group_size - 1) / group_size;
  TORCH_CHECK(key_scale.numel() >= key_groups * channels &&
                  key_zero.numel() >= key_groups * channels,
              "KIVI key metadata is too small");
  TORCH_CHECK(value_scale.numel() >= body_tokens * value_groups &&
                  value_zero.numel() >= body_tokens * value_groups,
              "KIVI value metadata is too small");
  TORCH_CHECK(key_sink.numel() >= sink_tokens * channels &&
                  value_sink.numel() >= sink_tokens * channels,
              "KIVI sink metadata is too small");

  bool cache_blocks_first = false;
  int64_t num_cache_blocks = 0;
  int64_t block_size = 0;
  if (kv_cache.size(1) == 2) {
    cache_blocks_first = true;
    num_cache_blocks = kv_cache.size(0);
    block_size = kv_cache.size(2);
    TORCH_CHECK(kv_cache.size(3) == heads && kv_cache.size(4) == head_dim,
                "kv_cache head shape mismatch");
  } else if (kv_cache.size(0) == 2) {
    cache_blocks_first = false;
    num_cache_blocks = kv_cache.size(1);
    block_size = kv_cache.size(2);
    TORCH_CHECK(kv_cache.size(3) == heads && kv_cache.size(4) == head_dim,
                "kv_cache head shape mismatch");
  } else {
    TORCH_CHECK(false,
                "kv_cache must be [num_blocks,2,block,heads,dim] or "
                "[2,num_blocks,block,heads,dim]");
  }
  TORCH_CHECK(block_size > 0, "kv_cache block size must be positive");
  const int64_t block_count = block_ids.numel();
  TORCH_CHECK(block_count > 0, "block_ids must be non-empty");
  TORCH_CHECK(tokens <= block_count * block_size,
              "tokens exceed destination block capacity");

  const c10::cuda::CUDAGuard device_guard(bytestream.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  origami_kivi_dequantize_to_kv_cache_launch(
      bytestream.data_ptr<uint8_t>(),
      key_offset,
      key_bytes,
      value_offset,
      value_bytes,
      bits,
      group_size,
      sink_tokens,
      key_scale.data_ptr(),
      key_zero.data_ptr(),
      value_scale.data_ptr(),
      value_zero.data_ptr(),
      key_sink.data_ptr(),
      value_sink.data_ptr(),
      kv_cache.data_ptr(),
      block_ids.data_ptr<int64_t>(),
      key_scale.scalar_type() == at::kHalf,
      kv_cache.scalar_type() == at::kBFloat16,
      cache_blocks_first,
      num_cache_blocks,
      block_count,
      block_size,
      tokens,
      heads,
      head_dim,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_prefix_attention_cuda(
    at::Tensor bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int64_t codec,
    int64_t bits,
    int64_t group_size,
    int64_t sink_tokens,
    at::Tensor key_scale,
    at::Tensor key_zero,
    at::Tensor value_scale,
    at::Tensor value_zero,
    at::Tensor key_sink,
    at::Tensor value_sink,
    at::Tensor query,
    at::Tensor kv_cache,
    at::Tensor block_table,
    int64_t prefix_tokens,
    int64_t query_start_position,
    int64_t sequence_length,
    double softmax_scale,
    at::Tensor output) {
  check_cuda_contiguous(bytestream, "bytestream");
  check_cuda_contiguous(key_scale, "key_scale");
  check_cuda_contiguous(key_zero, "key_zero");
  check_cuda_contiguous(value_scale, "value_scale");
  check_cuda_contiguous(value_zero, "value_zero");
  check_cuda_contiguous(key_sink, "key_sink");
  check_cuda_contiguous(value_sink, "value_sink");
  check_cuda_contiguous(query, "query");
  check_cuda_contiguous(kv_cache, "kv_cache");
  check_cuda_contiguous(block_table, "block_table");
  check_cuda_contiguous(output, "output");
  TORCH_CHECK(bytestream.scalar_type() == at::kByte,
              "bytestream must be uint8");
  TORCH_CHECK(codec == 0 || codec == 1,
              "codec must be 0 (KIVI) or 1 (KVQuant)");
  TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8,
              "fused attention supports 2, 3, 4, or 8 bits");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(prefix_tokens > 0, "prefix_tokens must be positive");
  TORCH_CHECK(sink_tokens >= 0 && sink_tokens <= prefix_tokens,
              "sink_tokens must be in [0, prefix_tokens]");
  TORCH_CHECK(key_offset >= 0 && key_bytes >= 0 && value_offset >= 0 &&
                  value_bytes >= 0,
              "invalid compressed stream offsets");
  TORCH_CHECK(key_offset + key_bytes <= bytestream.numel() &&
                  value_offset + value_bytes <= bytestream.numel(),
              "compressed stream ranges exceed bytestream length");
  TORCH_CHECK(query.dim() == 3,
              "query must have shape [query_tokens, query_heads, head_dim]");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "output must have the same shape as query");
  TORCH_CHECK(query.scalar_type() == at::kHalf ||
                  query.scalar_type() == at::kBFloat16,
              "query must be float16 or bfloat16");
  TORCH_CHECK(output.scalar_type() == query.scalar_type(),
              "output dtype must match query dtype");
  TORCH_CHECK(kv_cache.scalar_type() == at::kHalf ||
                  kv_cache.scalar_type() == at::kBFloat16,
              "kv_cache must be float16 or bfloat16");
  TORCH_CHECK(block_table.scalar_type() == at::kInt,
              "block_table must be int32");
  TORCH_CHECK(block_table.dim() == 1,
              "block_table must be one-dimensional");
  TORCH_CHECK(kv_cache.dim() == 5, "kv_cache must have rank 5");

  const auto device = bytestream.device();
  for (const auto& tensor : {key_scale, key_zero, value_scale, value_zero,
                             key_sink, value_sink, query, kv_cache,
                             block_table, output}) {
    TORCH_CHECK(tensor.device() == device,
                "all fused-attention tensors must be on the same CUDA device");
  }
  for (const auto& tensor : {key_scale, key_zero, value_scale, value_zero,
                             key_sink, value_sink}) {
    TORCH_CHECK(tensor.scalar_type() == at::kHalf,
                "compressed metadata tensors must be float16");
  }

  const int64_t query_tokens = query.size(0);
  const int64_t query_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  TORCH_CHECK(query_tokens > 0 && query_heads > 0,
              "query token and head counts must be positive");
  TORCH_CHECK(head_dim == 128,
              "the first Origami fused kernel supports head_dim 128 only");

  bool cache_blocks_first = false;
  int64_t num_cache_blocks = 0;
  int64_t block_size = 0;
  int64_t kv_heads = 0;
  // FlashAttention exposes its paged cache as [2, blocks, block, heads, dim].
  // Check this layout first because num_cache_blocks == 2 is otherwise
  // ambiguous with the legacy [blocks, 2, block, heads, dim] representation.
  if (kv_cache.size(0) == 2) {
    cache_blocks_first = false;
    num_cache_blocks = kv_cache.size(1);
    block_size = kv_cache.size(2);
    kv_heads = kv_cache.size(3);
    TORCH_CHECK(kv_cache.size(4) == head_dim,
                "kv_cache head_dim does not match query");
  } else if (kv_cache.size(1) == 2) {
    cache_blocks_first = true;
    num_cache_blocks = kv_cache.size(0);
    block_size = kv_cache.size(2);
    kv_heads = kv_cache.size(3);
    TORCH_CHECK(kv_cache.size(4) == head_dim,
                "kv_cache head_dim does not match query");
  } else {
    TORCH_CHECK(false,
                "kv_cache must be [num_blocks,2,block,heads,dim] or "
                "[2,num_blocks,block,heads,dim]");
  }
  TORCH_CHECK(kv_heads > 0 && query_heads % kv_heads == 0,
              "query_heads must be divisible by kv_heads");
  TORCH_CHECK(block_size > 0 && num_cache_blocks > 0,
              "kv_cache block dimensions must be positive");
  TORCH_CHECK(query_start_position >= prefix_tokens,
              "query must start at or after the compressed prefix");
  TORCH_CHECK(sequence_length > 0 &&
                  query_start_position + query_tokens <= sequence_length,
              "query range exceeds sequence length");
  const int64_t logical_blocks =
      (sequence_length + block_size - 1) / block_size;
  TORCH_CHECK(block_table.numel() >= logical_blocks,
              "block_table is too short for sequence_length");

  const int64_t channels = kv_heads * head_dim;
  const int64_t body_tokens = prefix_tokens - sink_tokens;
  const auto packed_bytes = [bits](int64_t symbols) {
    return bits == 3 ? (symbols + 1) / 2 : (symbols * bits + 7) / 8;
  };
  const int64_t stream_tokens = codec == 0 ? body_tokens : prefix_tokens;
  TORCH_CHECK(key_bytes >= packed_bytes(stream_tokens * channels) &&
                  value_bytes >= packed_bytes(stream_tokens * channels),
              "compressed K/V streams are smaller than their declared shape");
  if (codec == 0) {
    const int64_t key_groups =
        (body_tokens + group_size - 1) / group_size;
    const int64_t value_groups =
        (channels + group_size - 1) / group_size;
    TORCH_CHECK(key_scale.numel() >= key_groups * channels &&
                    key_zero.numel() >= key_groups * channels,
                "KIVI key metadata is too small");
    TORCH_CHECK(value_scale.numel() >= body_tokens * value_groups &&
                    value_zero.numel() >= body_tokens * value_groups,
                "KIVI value metadata is too small");
    TORCH_CHECK(key_sink.numel() >= sink_tokens * channels &&
                    value_sink.numel() >= sink_tokens * channels,
                "KIVI sink tensors are too small");
  } else {
    TORCH_CHECK(key_scale.numel() >= channels && key_zero.numel() >= channels,
                "KVQuant key metadata is too small");
    TORCH_CHECK(value_scale.numel() >= prefix_tokens &&
                    value_zero.numel() >= prefix_tokens,
                "KVQuant value metadata is too small");
  }

  const c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  origami_fused_prefix_attention_launch(
      bytestream.data_ptr<uint8_t>(), key_offset, key_bytes, value_offset,
      value_bytes, static_cast<int>(codec), static_cast<int>(bits), group_size,
      sink_tokens, key_scale.data_ptr(), key_zero.data_ptr(),
      value_scale.data_ptr(), value_zero.data_ptr(), key_sink.data_ptr(),
      value_sink.data_ptr(), query.data_ptr(), kv_cache.data_ptr(),
      block_table.data_ptr<int32_t>(), output.data_ptr(),
      query.scalar_type() == at::kBFloat16,
      kv_cache.scalar_type() == at::kBFloat16, cache_blocks_first,
      num_cache_blocks, block_size, prefix_tokens, query_start_position,
      sequence_length, query_tokens, query_heads, kv_heads, head_dim,
      static_cast<float>(softmax_scale), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unpack_canonical_storage_chunks_cuda",
        &unpack_canonical_storage_chunks_cuda);
  m.def("cachegen_unpack_dequantize_cuda", &cachegen_unpack_dequantize_cuda);
  m.def("cachegen_unpack_dequantize_to_kv_cache_cuda",
        &cachegen_unpack_dequantize_to_kv_cache_cuda);
  m.def("kivi_dequantize_to_kv_cache_cuda",
        &kivi_dequantize_to_kv_cache_cuda);
  m.def("fused_prefix_attention_cuda", &fused_prefix_attention_cuda);
}
