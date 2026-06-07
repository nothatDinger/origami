// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Origami QAT raw-DEFLATE native integration point.
//
// Implementation notes from /home/td/dpucomp that the production backend must
// preserve:
//   * QAT Data Plane API, stateless raw DEFLATE.
//   * Dynamic Huffman mode for ratio.
//   * Pre-allocated DMA-backed buffers and prepared payload descriptors.
//   * Pre-created workers, sessions, and inflight slots; no hot-path session
//     creation and no repeated torch uint8 -> QAE staging.
//
// Python currently imports this as optional module vllm._origami_qat. If the
// module is unavailable, Origami fails explicitly unless the test/development
// config selects the zlib raw-DEFLATE backend.

#include <cstddef>
#include <cstdint>

namespace vllm::origami {

struct QatPreparedPayload {
  const uint8_t* src;
  size_t src_bytes;
  uint8_t* dst;
  size_t dst_capacity;
};

// TODO(origami-native): wire this shim to QATzip/QATlib DP sessions. Returning
// zero signals that the compiled native backend is only a scaffold.
size_t origami_qat_raw_deflate_compress(const QatPreparedPayload&) { return 0; }

size_t origami_qat_raw_deflate_decompress(const QatPreparedPayload&) { return 0; }

}  // namespace vllm::origami
