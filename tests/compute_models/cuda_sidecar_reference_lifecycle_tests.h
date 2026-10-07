#pragma once
#include <atomic>
#include <string_view>

class ReferenceCaptureObserver final {
public:
  ReferenceCaptureObserver() {
    ggml_log_get(&previous_, &previous_data_);
    ggml_log_set([](ggml_log_level level, const char *text, void *data) {
      auto &self = *static_cast<ReferenceCaptureObserver *>(data);
      if (std::string_view(text).find("CUDA graph warmup complete") !=
          std::string_view::npos)
        self.cross_program_captures.fetch_add(1, std::memory_order_relaxed);
      if (self.previous_)
        self.previous_(level, text, self.previous_data_);
    }, this);
  }
  ~ReferenceCaptureObserver() { ggml_log_set(previous_, previous_data_); }
  std::atomic<unsigned> cross_program_captures{0};
private:
  ggml_log_callback previous_ = nullptr;
  void *previous_data_ = nullptr;
};

// A fresh original program supplies the exact CUDA activation-codec result.
// This test exercises the shared integration reference's ownership, not a new
// implementation or a relaxed mathematical comparison.
inline std::vector<float> fresh_q3_reference(
    std::span<const std::byte> packed, std::span<const float> input,
    std::int64_t k) {
  cuda_sidecar_quant_reference::Backend backend;
  cuda_sidecar_quant_reference::Storage storage;
  storage.context = ggml_init({ggml_tensor_overhead() * 4 +
                                  ggml_graph_overhead_custom(8, false),
                              nullptr, true});
  auto *weight = ggml_new_tensor_2d(storage.context, GGML_TYPE_Q3_K, k, 8);
  auto *activation = ggml_new_tensor_2d(storage.context, GGML_TYPE_F32, k, 1);
  auto *output = ggml_mul_mat(storage.context, weight, activation);
  (void)ggml_prec_set_acc(output, GGML_PREC_F32);
  auto *graph = ggml_new_graph_custom(storage.context, 8, false);
  ggml_build_forward_expand(graph, output);
  storage.buffer = ggml_backend_alloc_ctx_tensors(storage.context, backend.get());
  if (!storage.buffer)
    throw std::runtime_error("fresh Q3 reference allocation failed");
  ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
  ggml_backend_tensor_set(activation, input.data(), 0, input.size_bytes());
  if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("fresh Q3 reference program failed");
  std::vector<float> result(8);
  ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
  return result;
}

TEST(CudaSidecarReference, EqualStrideCodecsCannotReuseRetiredCapture) {
  namespace fixture = cpu_c5_gguf_fixture;
  using Q = hyle::compute::GgufQuantizedType;
  const auto item = std::find_if(
      std::begin(cpu_c5_inputs::cases), std::end(cpu_c5_inputs::cases),
      [](const auto &value) {
        return std::string_view(value.id) == "g3560f36d25c826bd";
      });
  ASSERT_NE(item, std::end(cpu_c5_inputs::cases));
  ASSERT_EQ(ggml_row_size(GGML_TYPE_Q3_K, item->k),
            ggml_row_size(GGML_TYPE_IQ3_S, item->k));
  const fixture::Bank bank(*item);
  const auto activation = fixture::activation(3, item->k, 19);
  const auto input = std::span(activation).subspan(item->k, item->k);
  const auto expected = fresh_q3_reference(bank.packed, input, item->k);
  ReferenceCaptureObserver observer;
  for (int transition = 0; transition != 8; ++transition) {
    SCOPED_TRACE(transition);
    // These equal-stride source encodings share geometry but not mathematics.
    // Two calls would warm a capture in the retired thread-local backend.
    for (int warmup = 0; warmup != 2; ++warmup)
      (void)cuda_sidecar_quant_reference::projection(
          Q::kIq3S, bank.packed, 8, item->k, input);
    const auto actual = cuda_sidecar_quant_reference::projection(
        Q::kQ3K, bank.packed, 8, item->k, input);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t row = 0; row != expected.size(); ++row)
      EXPECT_FLOAT_EQ(actual[row], expected[row]) << row;
  }
  // Each helper call owns a new logical program. A warm capture would prove
  // that a retired program's backend survived, even without an address collision.
  EXPECT_EQ(observer.cross_program_captures.load(std::memory_order_relaxed), 0U);
}
