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

template <>
__device__ __forceinline__ float max_to_float_direct<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
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

__device__ __forceinline__ uint8_t origami_fused_unpack_symbol(
    const uint8_t* src,
    int64_t src_bytes,
    int64_t symbol_index,
    int bits) {
  // The KIVI/KVQuant adapters store 3-bit values in two nibbles per byte
  // to match the upstream projects' layout. Other widths are densely packed.
  if (bits == 3) {
    const int64_t byte_idx = symbol_index >> 1;
    if (byte_idx >= src_bytes) {
      return 0;
    }
    const int shift = static_cast<int>((symbol_index & 1) * 4);
    return static_cast<uint8_t>((src[byte_idx] >> shift) & 0x07);
  }
  return unpack_symbol_direct(src, src_bytes, symbol_index, bits);
}

template <typename scalar_t, typename cache_t, int codec>
__global__ void origami_fused_prefix_attention_kernel(
    const uint8_t* __restrict__ bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int bits,
    int64_t group_size,
    int64_t sink_tokens,
    const __half* __restrict__ key_scale,
    const __half* __restrict__ key_zero,
    const __half* __restrict__ value_scale,
    const __half* __restrict__ value_zero,
    const __half* __restrict__ key_sink,
    const __half* __restrict__ value_sink,
    const scalar_t* __restrict__ query,
    const cache_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_table,
    scalar_t* __restrict__ output,
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
    float softmax_scale) {
  const int64_t query_token = blockIdx.x;
  const int64_t query_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (query_token >= query_tokens || query_head >= query_heads || tid >= head_dim) {
    return;
  }

  // Four warps process a four-token tile. K is read once per warp while all
  // 128 threads accumulate one output channel each using online softmax.
  __shared__ float tile_scores[4];
  __shared__ float tile_weights[4];
  __shared__ float shared_old_weight;
  __shared__ float shared_running_max;
  __shared__ float shared_running_sum;
  const int64_t kv_head = query_head / (query_heads / kv_heads);
  const int64_t channels = kv_heads * head_dim;
  const int64_t body_tokens = prefix_tokens - sink_tokens;
  const int64_t value_groups =
      codec == 0 ? (channels + group_size - 1) / group_size : 0;
  float accumulator = 0.0f;
  const int64_t causal_end = min(
      sequence_length - 1, query_start_position + query_token);
  if (tid == 0) {
    shared_running_max = -__int_as_float(0x7f800000);
    shared_running_sum = 0.0f;
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
  for (int64_t token_base = 0; token_base <= causal_end; token_base += 4) {
    const int64_t token = token_base + warp;
    float dot = 0.0f;
    if (token <= causal_end) {
      for (int dim = lane; dim < head_dim; dim += 32) {
        const int64_t channel = kv_head * head_dim + dim;
        float key_value = 0.0f;
        if (token < prefix_tokens) {
          if (codec == 0) {
            if (token < sink_tokens) {
              key_value = __half2float(key_sink[token * channels + channel]);
            } else {
              const int64_t body_token = token - sink_tokens;
              const int64_t symbol_index =
                  (dim * kv_heads + kv_head) * body_tokens + body_token;
              const uint8_t symbol = origami_fused_unpack_symbol(
                  bytestream + key_offset, key_bytes, symbol_index, bits);
              const int64_t meta_index =
                  (body_token / group_size) * channels + channel;
              key_value = static_cast<float>(symbol) *
                              __half2float(key_scale[meta_index]) +
                          __half2float(key_zero[meta_index]);
            }
          } else {
            const int64_t symbol_index =
                (dim * kv_heads + kv_head) * prefix_tokens + token;
            const uint8_t symbol = origami_fused_unpack_symbol(
                bytestream + key_offset, key_bytes, symbol_index, bits);
            key_value = static_cast<float>(symbol) *
                            __half2float(key_scale[channel]) +
                        __half2float(key_zero[channel]);
          }
        } else {
          const int64_t logical_block = token / block_size;
          const int64_t block_offset = token % block_size;
          const int64_t physical_block = block_table[logical_block];
          if (physical_block >= 0 && physical_block < num_cache_blocks) {
            int64_t key_index = 0;
            if (cache_blocks_first) {
              key_index = ((((physical_block * 2) * block_size + block_offset) *
                                kv_heads +
                            kv_head) *
                               head_dim +
                           dim);
            } else {
              key_index = (((physical_block * block_size + block_offset) *
                                kv_heads +
                            kv_head) *
                               head_dim +
                           dim);
            }
            key_value = max_to_float_direct(kv_cache[key_index]);
          }
        }
        const float query_value = max_to_float_direct(
            query[(query_token * query_heads + query_head) * head_dim + dim]);
        dot += query_value * key_value;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
    if (lane == 0) {
      tile_scores[warp] = token <= causal_end
                              ? dot * softmax_scale
                              : -__int_as_float(0x7f800000);
    }
    __syncthreads();

    if (tid == 0) {
      float tile_max = tile_scores[0];
      for (int index = 1; index < 4; ++index) {
        tile_max = fmaxf(tile_max, tile_scores[index]);
      }
      const float next_max = fmaxf(shared_running_max, tile_max);
      shared_old_weight = expf(shared_running_max - next_max);
      float tile_sum = 0.0f;
      for (int index = 0; index < 4; ++index) {
        const int64_t tile_token = token_base + index;
        const float weight = tile_token <= causal_end
                                 ? expf(tile_scores[index] - next_max)
                                 : 0.0f;
        tile_weights[index] = weight;
        tile_sum += weight;
      }
      shared_running_sum =
          shared_running_sum * shared_old_weight + tile_sum;
      shared_running_max = next_max;
    }
    __syncthreads();

    float tile_accumulator = 0.0f;
    const int64_t channel = kv_head * head_dim + tid;
    for (int index = 0; index < 4; ++index) {
      const int64_t value_token = token_base + index;
      if (value_token > causal_end) {
        continue;
      }
      float value_value = 0.0f;
      if (value_token < prefix_tokens) {
        if (codec == 0) {
          if (value_token < sink_tokens) {
            value_value =
                __half2float(value_sink[value_token * channels + channel]);
          } else {
            const int64_t body_token = value_token - sink_tokens;
            const int64_t symbol_index =
                (kv_head * head_dim + tid) * body_tokens + body_token;
            const uint8_t symbol = origami_fused_unpack_symbol(
                bytestream + value_offset, value_bytes, symbol_index, bits);
            const int64_t meta_index =
                body_token * value_groups + channel / group_size;
            value_value = static_cast<float>(symbol) *
                              __half2float(value_scale[meta_index]) +
                          __half2float(value_zero[meta_index]);
          }
        } else {
          const int64_t symbol_index =
              (kv_head * prefix_tokens + value_token) * head_dim + tid;
          const uint8_t symbol = origami_fused_unpack_symbol(
              bytestream + value_offset, value_bytes, symbol_index, bits);
          value_value = static_cast<float>(symbol) *
                            __half2float(value_scale[value_token]) +
                        __half2float(value_zero[value_token]);
        }
      } else {
        const int64_t logical_block = value_token / block_size;
        const int64_t block_offset = value_token % block_size;
        const int64_t physical_block = block_table[logical_block];
        if (physical_block >= 0 && physical_block < num_cache_blocks) {
          int64_t value_index = 0;
          if (cache_blocks_first) {
            value_index = (((((physical_block * 2) + 1) * block_size +
                              block_offset) *
                                 kv_heads +
                             kv_head) *
                                head_dim +
                            tid);
          } else {
            value_index = ((((num_cache_blocks + physical_block) * block_size +
                              block_offset) *
                                 kv_heads +
                             kv_head) *
                                head_dim +
                            tid);
          }
          value_value = max_to_float_direct(kv_cache[value_index]);
        }
      }
      tile_accumulator += tile_weights[index] * value_value;
    }
    accumulator = accumulator * shared_old_weight + tile_accumulator;
    __syncthreads();
  }

  output[(query_token * query_heads + query_head) * head_dim + tid] =
      float_to_output<scalar_t>(accumulator / shared_running_sum);
}

template <typename scalar_t, typename cache_t>
void launch_origami_fused_prefix_attention_typed(
    const uint8_t* bytestream,
    int64_t key_offset,
    int64_t key_bytes,
    int64_t value_offset,
    int64_t value_bytes,
    int codec,
    int bits,
    int64_t group_size,
    int64_t sink_tokens,
    const __half* key_scale,
    const __half* key_zero,
    const __half* value_scale,
    const __half* value_zero,
    const __half* key_sink,
    const __half* value_sink,
    const scalar_t* query,
    const cache_t* kv_cache,
    const int32_t* block_table,
    scalar_t* output,
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
    cudaStream_t stream) {
  const dim3 grid(query_tokens, query_heads);
  if (codec == 0) {
    origami_fused_prefix_attention_kernel<scalar_t, cache_t, 0>
        <<<grid, 128, 0, stream>>>(
            bytestream, key_offset, key_bytes, value_offset, value_bytes, bits,
            group_size, sink_tokens, key_scale, key_zero, value_scale,
            value_zero, key_sink, value_sink, query, kv_cache, block_table,
            output, cache_blocks_first, num_cache_blocks, block_size,
            prefix_tokens, query_start_position, sequence_length, query_tokens,
            query_heads, kv_heads, head_dim, softmax_scale);
  } else {
    origami_fused_prefix_attention_kernel<scalar_t, cache_t, 1>
        <<<grid, 128, 0, stream>>>(
            bytestream, key_offset, key_bytes, value_offset, value_bytes, bits,
            group_size, sink_tokens, key_scale, key_zero, value_scale,
            value_zero, key_sink, value_sink, query, kv_cache, block_table,
            output, cache_blocks_first, num_cache_blocks, block_size,
            prefix_tokens, query_start_position, sequence_length, query_tokens,
            query_heads, kv_heads, head_dim, softmax_scale);
  }
}

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
    cudaStream_t stream) {
#define ORIGAMI_LAUNCH_FUSED(QTYPE, CTYPE)                                  \
  launch_origami_fused_prefix_attention_typed<QTYPE, CTYPE>(                \
      bytestream, key_offset, key_bytes, value_offset, value_bytes, codec,  \
      bits, group_size, sink_tokens, static_cast<const __half*>(key_scale), \
      static_cast<const __half*>(key_zero),                                 \
      static_cast<const __half*>(value_scale),                              \
      static_cast<const __half*>(value_zero),                               \
      static_cast<const __half*>(key_sink),                                 \
      static_cast<const __half*>(value_sink), static_cast<const QTYPE*>(query), \
      static_cast<const CTYPE*>(kv_cache), block_table,                     \
      static_cast<QTYPE*>(output), cache_blocks_first, num_cache_blocks,    \
      block_size, prefix_tokens, query_start_position, sequence_length,     \
      query_tokens, query_heads, kv_heads, head_dim, softmax_scale, stream)
  if (query_is_bfloat16 && cache_is_bfloat16) {
    ORIGAMI_LAUNCH_FUSED(__nv_bfloat16, __nv_bfloat16);
  } else if (query_is_bfloat16) {
    ORIGAMI_LAUNCH_FUSED(__nv_bfloat16, __half);
  } else if (cache_is_bfloat16) {
    ORIGAMI_LAUNCH_FUSED(__half, __nv_bfloat16);
  } else {
    ORIGAMI_LAUNCH_FUSED(__half, __half);
  }
#undef ORIGAMI_LAUNCH_FUSED
}
