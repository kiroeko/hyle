#pragma once

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "../compute_models/d5_gguf_test_support.h"

// Original native Metal integration oracle. Primitive equation/codec tests
// remain independent; the model oracle must use Metal's activation precision
// and encoding, rather than the CPU provider's different quantized vec-dot.
namespace apple_sidecar_quant_reference {
class Backend final {
public:
  Backend() : value_(ggml_backend_metal_init()) {
    if (!value_ || !ggml_backend_is_metal(value_))
      throw std::runtime_error("sidecar reference requires original Metal");
  }
  ~Backend() { ggml_backend_free(value_); }
  ggml_backend_t get() const noexcept { return value_; }
private:
  ggml_backend_t value_;
};
struct Storage final {
  ggml_context *context = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ~Storage() { ggml_backend_buffer_free(buffer); ggml_free(context); }
};

inline std::vector<float> projection(hyle::compute::GgufQuantizedType type,
    std::span<const std::byte> packed, std::int64_t n, std::int64_t k,
    std::span<const float> input) {
  if (k <= 0 || n <= 0 || input.size() % std::size_t(k) != 0)
    throw std::runtime_error("sidecar reference geometry differs");
  Backend backend;
  Storage storage;
  storage.context = ggml_init({ggml_tensor_overhead() * 4 +
      ggml_graph_overhead_custom(8, false), nullptr, true});
  if (!storage.context)
    throw std::runtime_error("sidecar reference metadata allocation failed");
  auto *weight = ggml_new_tensor_2d(storage.context,
      d5_gguf_test::native_type(type), k, n);
  auto *activation = ggml_new_tensor_2d(storage.context, GGML_TYPE_F32, k,
      std::int64_t(input.size() / k));
  auto *output = ggml_mul_mat(storage.context, weight, activation);
  if (!ggml_prec_set_acc(output, GGML_PREC_F32))
    throw std::runtime_error("sidecar reference F32 accumulation unavailable");
  if (ggml_nbytes(weight) != packed.size())
    throw std::runtime_error("sidecar reference packed byte size differs");
  auto *graph = ggml_new_graph_custom(storage.context, 8, false);
  ggml_build_forward_expand(graph, output);
  storage.buffer = ggml_backend_alloc_ctx_tensors(storage.context, backend.get());
  if (!storage.buffer)
    throw std::runtime_error("sidecar reference device allocation failed");
  ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
  ggml_backend_tensor_set(activation, input.data(), 0, input.size_bytes());
  if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("sidecar reference original Metal program failed");
  std::vector<float> result(input.size() / k * n);
  ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
  return result;
}

inline std::vector<float> grouped(const hyle::compute::TensorDesc &description,
    std::span<const std::byte> packed, std::int64_t n, std::int64_t k,
    std::span<const float> input, std::span<const std::int32_t> ids,
    std::int64_t tokens) {
  if (tokens <= 0 || k <= 0 || n <= 0 ||
      ids.size() % std::size_t(tokens) != 0 || input.size() != ids.size() * k)
    throw std::runtime_error("sidecar grouped reference geometry differs");
  const auto slots = std::int64_t(ids.size() / tokens);
  const auto experts = description.dimensions()[0];
  for (const auto id : ids)
    if (id < 0 || id >= experts)
      throw std::runtime_error("sidecar grouped reference id out of range");
  const auto type = description.is_plain() ? GGML_TYPE_BF16 :
      d5_gguf_test::native_type(hyle::compute::gguf_quantized_type(
          description.storage_format()).value());
  Backend backend;
  Storage storage;
  storage.context = ggml_init({ggml_tensor_overhead() * 6 +
      ggml_graph_overhead_custom(8, false), nullptr, true});
  if (!storage.context)
    throw std::runtime_error("sidecar grouped metadata allocation failed");
  auto *weight = ggml_new_tensor_3d(storage.context, type, k, n, experts);
  auto *activation = ggml_new_tensor_3d(storage.context, GGML_TYPE_F32, k, slots, tokens);
  auto *indices = ggml_new_tensor_2d(storage.context, GGML_TYPE_I32, slots, tokens);
  // Plain BF16 weights multiply the declared F32 input, without silently
  // casting that activation to BF16 in an original backend's MMF shortcut.
  auto *weights = description.is_plain() ?
      ggml_cast(storage.context, weight, GGML_TYPE_F32) : weight;
  auto *output = ggml_mul_mat_id(storage.context, weights, activation, indices);
  if (!ggml_prec_set_acc(output, GGML_PREC_F32))
    throw std::runtime_error("sidecar grouped F32 accumulation unavailable");
  if (ggml_nbytes(weight) != packed.size())
    throw std::runtime_error("sidecar grouped packed byte size differs");
  auto *graph = ggml_new_graph_custom(storage.context, 8, false);
  ggml_build_forward_expand(graph, output);
  storage.buffer = ggml_backend_alloc_ctx_tensors(storage.context, backend.get());
  if (!storage.buffer)
    throw std::runtime_error("sidecar grouped device allocation failed");
  ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
  ggml_backend_tensor_set(activation, input.data(), 0, input.size_bytes());
  ggml_backend_tensor_set(indices, ids.data(), 0, ids.size_bytes());
  if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("sidecar grouped original Metal program failed");
  std::vector<float> result(ids.size() * n);
  ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
  return result;
}
} // namespace apple_sidecar_quant_reference

#define HYLE_C4_QUANTIZED_REFERENCE apple_sidecar_quant_reference::projection
#define HYLE_C4_GROUPED_REFERENCE apple_sidecar_quant_reference::grouped
