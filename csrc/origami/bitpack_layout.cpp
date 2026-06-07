// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Origami CPU bitpack/layout native integration point.
//
// The compression hot path is expected to fuse:
//   quantized symbols -> bitpack -> head-first chunk layout
// and the CPU restore hot path fuses:
//   head-first chunk layout -> unpack -> quantized symbols.
//
// The production implementation should dispatch AVX-512 first and fall back to
// AVX2. It is intentionally isolated from the Python connector so ablation
// studies can switch origami_bitpack/origami_layout_policy without touching the
// vLLM KVConnector lifecycle.

#include <cstddef>
#include <cstdint>

namespace vllm::origami {

struct OrigamiChunkLayout {
  int64_t layer_index;
  int64_t head_start;
  int64_t head_end;
  int64_t channel_start;
  int64_t channel_end;
  int64_t token_start;
  int64_t token_end;
  int64_t unpacked_bytes;
};

// TODO(origami-native): replace the Python fallback with AVX-512/AVX2 kernels.
// The ABI is kept simple: callers pass already-quantized uint8 symbols and a
// planned head/channel chunk. The kernel writes a byte-contiguous payload ready
// for QAT raw DEFLATE.
size_t origami_pack_head_first_cpu(const uint8_t* src, size_t src_bytes,
                                   const OrigamiChunkLayout&, uint8_t* dst,
                                   size_t dst_capacity) {
  if (dst_capacity < src_bytes) {
    return 0;
  }
  for (size_t i = 0; i < src_bytes; ++i) {
    dst[i] = src[i];
  }
  return src_bytes;
}

size_t origami_unpack_head_first_cpu(const uint8_t* src, size_t src_bytes,
                                     const OrigamiChunkLayout&, uint8_t* dst,
                                     size_t dst_capacity) {
  if (dst_capacity < src_bytes) {
    return 0;
  }
  for (size_t i = 0; i < src_bytes; ++i) {
    dst[i] = src[i];
  }
  return src_bytes;
}

}  // namespace vllm::origami
