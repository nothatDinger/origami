# Origami native backends

This directory holds the native integration points for Origami KV cache transport compression. The Python connector calls optional extension module vllm._origami_qat; when that module is absent, tests must choose the explicit zlib raw-DEFLATE fallback.

The intended production path is GPU quantization, CPU fused bitpack/layout, QAT raw DEFLATE, and the reverse path with CPU or nvCOMP lossless restore depending on request-level offload policy.
