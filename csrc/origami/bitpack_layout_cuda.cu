// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

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

__device__ __forceinline__ int64_t layout_stride_at(const SourceLayout layout,
                                                    int64_t axis) {
  if (axis == 0) {
    return layout.stride0;
  }
  if (axis == 1) {
    return layout.stride1;
  }
  if (axis == 2) {
    return layout.stride2;
  }
  return layout.stride3;
}

__device__ __forceinline__ int64_t source_index(const SourceLayout layout,
                                                int64_t token,
                                                int64_t head,
                                                int64_t channel) {
  int64_t coords[4] = {0, 0, 0, 0};
  coords[layout.token_axis] = token;
  coords[layout.head_axis] = head;
  coords[layout.channel_axis] = channel;
  if (layout.layer_axis != -1) {
    coords[layout.layer_axis] = 0;
  }
  int64_t index = 0;
  for (int64_t axis = 0; axis < layout.rank; ++axis) {
    index += coords[axis] * layout_stride_at(layout, axis);
  }
  return index;
}

__device__ __forceinline__ uint8_t unpack_symbol(const uint8_t* src,
                                                 int64_t src_bytes,
                                                 int64_t symbol_index,
                                                 int bits) {
  if (bits == 8) {
    return src[symbol_index];
  }
  const int64_t bit_pos = symbol_index * static_cast<int64_t>(bits);
  const int64_t byte_idx = bit_pos >> 3;
  const int shift = static_cast<int>(bit_pos & 7);
  uint16_t packed = 0;
  if (byte_idx < src_bytes) {
    packed = src[byte_idx];
  }
  if (byte_idx + 1 < src_bytes) {
    packed = static_cast<uint16_t>(
        packed | (static_cast<uint16_t>(src[byte_idx + 1]) << 8));
  }
  return static_cast<uint8_t>((packed >> shift) & ((1u << bits) - 1u));
}

__global__ void unpack_canonical_kernel(
    const uint8_t* __restrict__ flat_chunks,
    const int64_t* __restrict__ chunk_offsets,
    const int64_t* __restrict__ bits_per_chunk,
    const int64_t* __restrict__ chunk_specs,
    uint8_t* __restrict__ output,
    SourceLayout layout,
    int64_t num_chunks) {
  const int64_t chunk_id = blockIdx.y;
  if (chunk_id >= num_chunks) {
    return;
  }
  const int64_t* row = chunk_specs + chunk_id * 6;
  const ChunkSpec spec{row[0], row[1], row[2], row[3], row[4], row[5]};
  const int bits = static_cast<int>(bits_per_chunk[chunk_id]);
  const uint8_t* src = flat_chunks + chunk_offsets[chunk_id];
  const int64_t src_bytes = chunk_offsets[chunk_id + 1] - chunk_offsets[chunk_id];
  const int64_t token_span = spec.token_end - spec.token_start;
  const int64_t channel_span = spec.channel_end - spec.channel_start;
  const int64_t per_head = channel_span * token_span;
  const int64_t symbols =
      (spec.head_end - spec.head_start) * per_head;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t local = blockIdx.x * blockDim.x + threadIdx.x;
       local < symbols;
       local += stride) {
    const int64_t head_offset = local / per_head;
    const int64_t rem = local - head_offset * per_head;
    const int64_t channel_offset = rem / token_span;
    const int64_t token_offset = rem - channel_offset * token_span;
    const int64_t head = spec.head_start + head_offset;
    const int64_t channel = spec.channel_start + channel_offset;
    const int64_t token = spec.token_start + token_offset;
    output[source_index(layout, token, head, channel)] =
        unpack_symbol(src, src_bytes, local, bits);
  }
}

template <typename T>
__device__ __forceinline__ float max_to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float max_to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename max_t>
__global__ void cachegen_unpack_dequantize_kernel(
    const uint8_t* __restrict__ bytestream,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ bits_per_layer,
    const int64_t* __restrict__ key_bins,
    const int64_t* __restrict__ value_bins,
    const max_t* __restrict__ max_key,
    const max_t* __restrict__ max_value,
    __half* __restrict__ output,
    int64_t layers,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim) {
  const int64_t channels = heads * head_dim;
  const int64_t values_per_layer = tokens * channels;
  const int64_t total = layers * 2 * values_per_layer;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t out_layer = idx / values_per_layer;
    const int64_t rem = idx - out_layer * values_per_layer;
    const int64_t token = rem / channels;
    const int64_t channel = rem - token * channels;
    const int64_t head = channel / head_dim;
    const int64_t dim = channel - head * head_dim;
    const bool is_value = out_layer >= layers;
    const int64_t layer = is_value ? out_layer - layers : out_layer;
    const int bits = static_cast<int>(bits_per_layer[out_layer]);
    const int64_t base = offsets[out_layer];
    const int64_t layer_len = offsets[out_layer + 1] - base;
    const uint8_t symbol =
        unpack_symbol(bytestream + base, layer_len, rem, bits);

    const int64_t bins = is_value ? value_bins[layer] : key_bins[layer];
    const float center = static_cast<float>(bins / 2 - 1);
    const max_t* max_ptr = is_value ? max_value : max_key;
    const float max_abs = max_to_float(max_ptr[layer * tokens + token]);
    const float value = ((static_cast<float>(symbol) - center) / center) * max_abs;

    const int64_t kv = static_cast<int64_t>(is_value);
    const int64_t out_idx =
        ((((layer * 2 + kv) * heads + head) * tokens + token) * head_dim + dim);
    output[out_idx] = __float2half(value);
  }
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
    cudaStream_t stream) {
  if (num_chunks == 0 || max_symbols == 0) {
    return;
  }
  const SourceLayout layout{
      rank,
      shape0,
      shape1,
      shape2,
      shape3,
      stride0,
      stride1,
      stride2,
      stride3,
      token_axis,
      head_axis,
      channel_axis,
      layer_axis,
  };
  const int threads = 256;
  const int blocks_x =
      static_cast<int>(std::min<int64_t>((max_symbols + threads - 1) / threads,
                                         65535));
  const dim3 grid(blocks_x, static_cast<unsigned int>(num_chunks));
  unpack_canonical_kernel<<<grid, threads, 0, stream>>>(
      flat_chunks,
      chunk_offsets,
      bits_per_chunk,
      chunk_specs,
      output,
      layout,
      num_chunks);
}

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
    cudaStream_t stream) {
  const int threads = 256;
  const int64_t total = layers * 2 * tokens * heads * head_dim;
  const int blocks = static_cast<int>(
      std::min<int64_t>((total + threads - 1) / threads, 65535));
  if (max_is_half) {
    cachegen_unpack_dequantize_kernel<__half><<<blocks, threads, 0, stream>>>(
        bytestream,
        offsets,
        bits_per_layer,
        key_bins,
        value_bins,
        static_cast<const __half*>(max_key),
        static_cast<const __half*>(max_value),
        static_cast<__half*>(output),
        layers,
        tokens,
        heads,
        head_dim);
  } else {
    cachegen_unpack_dequantize_kernel<float><<<blocks, threads, 0, stream>>>(
        bytestream,
        offsets,
        bits_per_layer,
        key_bins,
        value_bins,
        static_cast<const float*>(max_key),
        static_cast<const float*>(max_value),
        static_cast<__half*>(output),
        layers,
        tokens,
        heads,
        head_dim);
  }
}

__device__ __forceinline__ uint8_t unpack_symbol_direct(const uint8_t* src,
                                                        int64_t src_bytes,
                                                        int64_t symbol_index,
                                                        int bits) {
  if (bits == 8) {
    return src[symbol_index];
  }
  const int64_t bit_pos = symbol_index * static_cast<int64_t>(bits);
  const int64_t byte_idx = bit_pos >> 3;
  const int shift = static_cast<int>(bit_pos & 7);
  uint16_t packed = 0;
  if (byte_idx < src_bytes) {
    packed = src[byte_idx];
  }
  if (byte_idx + 1 < src_bytes) {
    packed = static_cast<uint16_t>(
        packed | (static_cast<uint16_t>(src[byte_idx + 1]) << 8));
  }
  return static_cast<uint8_t>((packed >> shift) & ((1u << bits) - 1u));
}

template <typename T>
__device__ __forceinline__ float max_to_float_direct(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float max_to_float_direct<__half>(__half value) {
  return __half2float(value);
}

template <typename out_t>
__device__ __forceinline__ out_t float_to_output(float value) {
  return static_cast<out_t>(value);
}

template <>
__device__ __forceinline__ __half float_to_output<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
float_to_output<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename max_t, typename out_t>
__global__ void cachegen_unpack_dequantize_to_kv_cache_kernel(
    const uint8_t* __restrict__ bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int key_bits,
    int value_bits,
    int64_t key_bins,
    int64_t value_bins,
    const max_t* __restrict__ max_key,
    const max_t* __restrict__ max_value,
    out_t* __restrict__ kv_cache,
    const int64_t* __restrict__ block_ids,
    bool cache_blocks_first,
    int64_t num_cache_blocks,
    int64_t block_count,
    int64_t block_size,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim) {
  const int64_t channels = heads * head_dim;
  const int64_t values_per_block = block_size * channels;
  const int64_t total = block_count * 2 * values_per_block;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t local_block = idx / (2 * values_per_block);
    const int64_t block_rem = idx - local_block * 2 * values_per_block;
    const int64_t kv = block_rem / values_per_block;
    const int64_t rem = block_rem - kv * values_per_block;
    const int64_t block_offset = rem / channels;
    const int64_t channel = rem - block_offset * channels;
    const int64_t token = local_block * block_size + block_offset;
    if (token >= tokens) {
      continue;
    }
    const int64_t head = channel / head_dim;
    const int64_t dim = channel - head * head_dim;
    const int64_t dst_block = block_ids[local_block];
    if (dst_block < 0 || dst_block >= num_cache_blocks) {
      continue;
    }
    const bool is_value = kv != 0;
    const int64_t symbol_index = token * channels + channel;
    const uint8_t* stream =
        bytestream + (is_value ? value_offset : key_offset);
    const int64_t stream_bytes = is_value ? value_bytes : key_bytes;
    const int bits = is_value ? value_bits : key_bits;
    const uint8_t symbol =
        unpack_symbol_direct(stream, stream_bytes, symbol_index, bits);
    const int64_t bins = is_value ? value_bins : key_bins;
    const float center = static_cast<float>(bins / 2 - 1);
    const max_t* max_ptr = is_value ? max_value : max_key;
    const float max_abs = max_to_float_direct(max_ptr[token]);
    const float value =
        ((static_cast<float>(symbol) - center) / center) * max_abs;
    int64_t out_idx = 0;
    if (cache_blocks_first) {
      out_idx =
          ((((dst_block * 2 + kv) * block_size + block_offset) * heads + head) *
               head_dim +
           dim);
    } else {
      out_idx =
          ((((kv * num_cache_blocks + dst_block) * block_size + block_offset) *
                heads +
            head) *
               head_dim +
           dim);
    }
    kv_cache[out_idx] = float_to_output<out_t>(value);
  }
}

template <typename out_t>
void launch_cachegen_unpack_dequantize_to_kv_cache_typed(
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
    bool cache_blocks_first,
    int64_t num_cache_blocks,
    int64_t block_count,
    int64_t block_size,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    cudaStream_t stream) {
  const int threads = 256;
  const int64_t total = block_count * 2 * block_size * heads * head_dim;
  const int blocks = static_cast<int>(
      std::min<int64_t>((total + threads - 1) / threads, 65535));
  if (max_is_half) {
    cachegen_unpack_dequantize_to_kv_cache_kernel<__half, out_t>
        <<<blocks, threads, 0, stream>>>(
            bytestream,
            key_offset,
            key_bytes,
            value_offset,
            value_bytes,
            static_cast<int>(key_bits),
            static_cast<int>(value_bits),
            key_bins,
            value_bins,
            static_cast<const __half*>(max_key),
            static_cast<const __half*>(max_value),
            static_cast<out_t*>(kv_cache),
            block_ids,
            cache_blocks_first,
            num_cache_blocks,
            block_count,
            block_size,
            tokens,
            heads,
            head_dim);
  } else {
    cachegen_unpack_dequantize_to_kv_cache_kernel<float, out_t>
        <<<blocks, threads, 0, stream>>>(
            bytestream,
            key_offset,
            key_bytes,
            value_offset,
            value_bytes,
            static_cast<int>(key_bits),
            static_cast<int>(value_bits),
            key_bins,
            value_bins,
            static_cast<const float*>(max_key),
            static_cast<const float*>(max_value),
            static_cast<out_t*>(kv_cache),
            block_ids,
            cache_blocks_first,
            num_cache_blocks,
            block_count,
            block_size,
            tokens,
            heads,
            head_dim);
  }
}

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
    cudaStream_t stream) {
  if (output_is_bfloat16) {
    launch_cachegen_unpack_dequantize_to_kv_cache_typed<__nv_bfloat16>(
        bytestream,
        key_offset,
        key_bytes,
        value_offset,
        value_bytes,
        key_bits,
        value_bits,
        key_bins,
        value_bins,
        max_key,
        max_value,
        kv_cache,
        block_ids,
        max_is_half,
        cache_blocks_first,
        num_cache_blocks,
        block_count,
        block_size,
        tokens,
        heads,
        head_dim,
        stream);
  } else {
    launch_cachegen_unpack_dequantize_to_kv_cache_typed<__half>(
        bytestream,
        key_offset,
        key_bytes,
        value_offset,
        value_bytes,
        key_bits,
        value_bits,
        key_bins,
        value_bins,
        max_key,
        max_value,
        kv_cache,
        block_ids,
        max_is_half,
        cache_blocks_first,
        num_cache_blocks,
        block_count,
        block_size,
        tokens,
        heads,
        head_dim,
        stream);
  }
}

template <typename meta_t, typename out_t>
__global__ void kivi_dequantize_to_kv_cache_kernel(
    const uint8_t* __restrict__ bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int bits,
    int64_t group_size,
    int64_t sink_tokens,
    const meta_t* __restrict__ key_scale,
    const meta_t* __restrict__ key_zero,
    const meta_t* __restrict__ value_scale,
    const meta_t* __restrict__ value_zero,
    const meta_t* __restrict__ key_sink,
    const meta_t* __restrict__ value_sink,
    out_t* __restrict__ kv_cache,
    const int64_t* __restrict__ block_ids,
    bool cache_blocks_first,
    int64_t num_cache_blocks,
    int64_t block_count,
    int64_t block_size,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim) {
  const int64_t channels = heads * head_dim;
  const int64_t values_per_block = block_size * channels;
  const int64_t total = block_count * 2 * values_per_block;
  const int64_t body_tokens = tokens - sink_tokens;
  const int64_t value_groups = (channels + group_size - 1) / group_size;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t local_block = idx / (2 * values_per_block);
    const int64_t block_rem = idx - local_block * 2 * values_per_block;
    const int64_t kv = block_rem / values_per_block;
    const int64_t rem = block_rem - kv * values_per_block;
    const int64_t block_offset = rem / channels;
    const int64_t channel = rem - block_offset * channels;
    const int64_t token = local_block * block_size + block_offset;
    if (token >= tokens) {
      continue;
    }
    const int64_t head = channel / head_dim;
    const int64_t dim = channel - head * head_dim;
    const int64_t dst_block = block_ids[local_block];
    if (dst_block < 0 || dst_block >= num_cache_blocks) {
      continue;
    }

    float value = 0.0f;
    const bool is_value = kv != 0;
    if (token < sink_tokens) {
      const int64_t sink_idx = token * channels + channel;
      const meta_t* sink = is_value ? value_sink : key_sink;
      value = max_to_float_direct(sink[sink_idx]);
    } else {
      const int64_t body_token = token - sink_tokens;
      if (body_token >= body_tokens) {
        continue;
      }
      int64_t symbol_index = 0;
      if (is_value) {
        // Value stream order: head > layer > head_dim > token.
        symbol_index = ((head * head_dim + dim) * body_tokens + body_token);
      } else {
        // Key stream order: head_dim > layer > head > token.
        symbol_index = ((dim * heads + head) * body_tokens + body_token);
      }
      const uint8_t* stream =
          bytestream + (is_value ? value_offset : key_offset);
      const int64_t stream_bytes = is_value ? value_bytes : key_bytes;
      const uint8_t symbol =
          unpack_symbol_direct(stream, stream_bytes, symbol_index, bits);
      if (is_value) {
        const int64_t group = channel / group_size;
        const int64_t meta_idx = body_token * value_groups + group;
        value =
            static_cast<float>(symbol) * max_to_float_direct(value_scale[meta_idx]) +
            max_to_float_direct(value_zero[meta_idx]);
      } else {
        const int64_t group = body_token / group_size;
        const int64_t meta_idx = group * channels + channel;
        value =
            static_cast<float>(symbol) * max_to_float_direct(key_scale[meta_idx]) +
            max_to_float_direct(key_zero[meta_idx]);
      }
    }

    int64_t out_idx = 0;
    if (cache_blocks_first) {
      out_idx =
          ((((dst_block * 2 + kv) * block_size + block_offset) * heads + head) *
               head_dim +
           dim);
    } else {
      out_idx =
          ((((kv * num_cache_blocks + dst_block) * block_size + block_offset) *
                heads +
            head) *
               head_dim +
           dim);
    }
    kv_cache[out_idx] = float_to_output<out_t>(value);
  }
}

template <typename out_t>
void launch_kivi_dequantize_to_kv_cache_typed(
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
    bool cache_blocks_first,
    int64_t num_cache_blocks,
    int64_t block_count,
    int64_t block_size,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    cudaStream_t stream) {
  const int threads = 256;
  const int64_t total = block_count * 2 * block_size * heads * head_dim;
  const int blocks = static_cast<int>(
      std::min<int64_t>((total + threads - 1) / threads, 65535));
  if (metadata_is_half) {
    kivi_dequantize_to_kv_cache_kernel<__half, out_t>
        <<<blocks, threads, 0, stream>>>(
            bytestream,
            key_offset,
            key_bytes,
            value_offset,
            value_bytes,
            static_cast<int>(bits),
            group_size,
            sink_tokens,
            static_cast<const __half*>(key_scale),
            static_cast<const __half*>(key_zero),
            static_cast<const __half*>(value_scale),
            static_cast<const __half*>(value_zero),
            static_cast<const __half*>(key_sink),
            static_cast<const __half*>(value_sink),
            static_cast<out_t*>(kv_cache),
            block_ids,
            cache_blocks_first,
            num_cache_blocks,
            block_count,
            block_size,
            tokens,
            heads,
            head_dim);
  } else {
    kivi_dequantize_to_kv_cache_kernel<float, out_t>
        <<<blocks, threads, 0, stream>>>(
            bytestream,
            key_offset,
            key_bytes,
            value_offset,
            value_bytes,
            static_cast<int>(bits),
            group_size,
            sink_tokens,
            static_cast<const float*>(key_scale),
            static_cast<const float*>(key_zero),
            static_cast<const float*>(value_scale),
            static_cast<const float*>(value_zero),
            static_cast<const float*>(key_sink),
            static_cast<const float*>(value_sink),
            static_cast<out_t*>(kv_cache),
            block_ids,
            cache_blocks_first,
            num_cache_blocks,
            block_count,
            block_size,
            tokens,
            heads,
            head_dim);
  }
}

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
    cudaStream_t stream) {
  if (output_is_bfloat16) {
    launch_kivi_dequantize_to_kv_cache_typed<__nv_bfloat16>(
        bytestream,
        key_offset,
        key_bytes,
        value_offset,
        value_bytes,
        bits,
        group_size,
        sink_tokens,
        key_scale,
        key_zero,
        value_scale,
        value_zero,
        key_sink,
        value_sink,
        kv_cache,
        block_ids,
        metadata_is_half,
        cache_blocks_first,
        num_cache_blocks,
        block_count,
        block_size,
        tokens,
        heads,
        head_dim,
        stream);
  } else {
    launch_kivi_dequantize_to_kv_cache_typed<__half>(
        bytestream,
        key_offset,
        key_bytes,
        value_offset,
        value_bytes,
        bits,
        group_size,
        sink_tokens,
        key_scale,
        key_zero,
        value_scale,
        value_zero,
        key_sink,
        value_sink,
        kv_cache,
        block_ids,
        metadata_is_half,
        cache_blocks_first,
        num_cache_blocks,
        block_count,
        block_size,
        tokens,
        heads,
        head_dim,
        stream);
  }
}
