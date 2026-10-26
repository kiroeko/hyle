#include <gtest/gtest.h>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <vector>

#include "../../src/compute_ggml_runtime/gated_delta_scale.h"

extern "C" std::int32_t ggml_hyle_cuda_gdn_indexed_state_contract();

namespace {
// Direct private-native regression, not evidence that the public provider or
// Graph slot-write assembly has been wired. These fixed tables are proved here
// before upload, then stay immutable until actual native retirement.
constexpr std::int32_t indexed_contract = 0x47444931;

struct NativeGraph final {
  ggml_backend_t backend = nullptr;
  ggml_context *context = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ~NativeGraph() {
    if (backend) ggml_backend_free(backend);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (context) ggml_free(context);
  }
};

std::uint16_t bfloat_bits(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>(
      (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

std::vector<std::uint32_t> read_bits(ggml_tensor *tensor) {
  std::vector<std::uint32_t> result(ggml_nbytes(tensor) / sizeof(std::uint32_t));
  ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(result[0]));
  return result;
}

void compare_case(std::int64_t width, std::int64_t heads,
                  std::int64_t tokens, int snapshots, bool bfloat,
                  std::optional<float> scale) {
  ASSERT_EQ(ggml_hyle_cuda_gdn_indexed_state_contract(), indexed_contract);
  constexpr std::int64_t batch = 2;
  constexpr std::int64_t capacity = 17;
  const auto item = static_cast<std::size_t>(width * width * heads);
  const auto stride = item + 16; // Original slot padding is never an output.
  const auto scores = static_cast<std::size_t>(width * heads * tokens * batch);
  const auto gates = static_cast<std::size_t>(heads * tokens * batch);
  const std::array<std::int32_t, 2> sources{13, 2};
  const std::vector<std::int32_t> destinations = snapshots == 1
      ? std::vector<std::int32_t>{7, 16}
      : std::vector<std::int32_t>{7, 16, 4, 11};
  ASSERT_EQ(destinations.size(), static_cast<std::size_t>(batch * snapshots));
  auto unique = destinations;
  std::sort(unique.begin(), unique.end());
  ASSERT_EQ(std::adjacent_find(unique.begin(), unique.end()), unique.end());
  for (const auto id : destinations) {
    ASSERT_GE(id, 0); ASSERT_LT(id, capacity);
    ASSERT_EQ(std::find(sources.begin(), sources.end(), id), sources.end());
  }

  NativeGraph run;
  run.backend = ggml_backend_cuda_init(0);
  ASSERT_NE(run.backend, nullptr);
  run.context = ggml_init({.mem_size = 2 * 1024 * 1024, .mem_buffer = nullptr,
                          .no_alloc = true});
  ASSERT_NE(run.context, nullptr);
  auto *q = ggml_new_tensor_4d(run.context, GGML_TYPE_F32, width, heads, tokens, batch);
  auto *k = ggml_new_tensor_4d(run.context, GGML_TYPE_F32, width, heads, tokens, batch);
  auto *v_wide = ggml_new_tensor_4d(run.context, GGML_TYPE_F32, width, heads, tokens, batch);
  auto *g = ggml_new_tensor_4d(run.context, GGML_TYPE_F32, 1, heads, tokens, batch);
  auto *beta_wide = ggml_new_tensor_4d(run.context, GGML_TYPE_F32, 1, heads, tokens, batch);
  auto *v = bfloat ? ggml_new_tensor_4d(run.context, GGML_TYPE_BF16, width, heads, tokens, batch) : v_wide;
  auto *beta = bfloat ? ggml_new_tensor_4d(run.context, GGML_TYPE_BF16, 1, heads, tokens, batch) : beta_wide;
  auto *dense_state = ggml_new_tensor_4d(run.context, GGML_TYPE_F32, width, width, heads, batch);
  auto *backing = ggml_new_tensor_1d(run.context, GGML_TYPE_F32,
      capacity * static_cast<std::int64_t>(stride));
  auto *pool = ggml_view_4d(run.context, backing, width, width, heads, capacity,
      static_cast<std::size_t>(width) * sizeof(float),
      static_cast<std::size_t>(width * width) * sizeof(float), stride * sizeof(float), 0);
  auto *source_ids = ggml_new_tensor_1d(run.context, GGML_TYPE_I32, batch);
  auto *destination_ids = ggml_new_tensor_2d(run.context, GGML_TYPE_I32, snapshots, batch);
  auto *legacy = ggml_gated_delta_net(run.context, q, k, v_wide, g, beta_wide,
                                     dense_state, snapshots);
  auto *indexed = ggml_new_tensor_4d(run.context, GGML_TYPE_F32, width, heads, tokens, batch);
  indexed->op = GGML_OP_GATED_DELTA_NET;
  const std::array inputs{q, k, v, g, beta, pool, source_ids, destination_ids};
  for (std::size_t i = 0; i < inputs.size(); ++i) indexed->src[i] = inputs[i];
  indexed->op_params[0] = snapshots;
  indexed->op_params[4] = indexed_contract;
  hyle::compute::ggml_runtime::set_gdn_output_scale(legacy, scale);
  hyle::compute::ggml_runtime::set_gdn_output_scale(indexed, scale);
  ggml_set_output(legacy);
  ggml_set_output(indexed);
  auto *graph = ggml_new_graph_custom(run.context, 64, false);
  ggml_build_forward_expand(graph, legacy);
  ggml_build_forward_expand(graph, indexed);
  run.buffer = ggml_backend_alloc_ctx_tensors(run.context, run.backend);
  ASSERT_NE(run.buffer, nullptr);
  ASSERT_TRUE(ggml_backend_dev_supports_op(ggml_backend_get_device(run.backend), indexed));

  std::vector<float> q_values(scores), k_values(scores), v_values(scores);
  std::vector<float> decay(gates), beta_values(gates);
  for (std::size_t i = 0; i < scores; ++i) {
    q_values[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.007F;
    k_values[i] = static_cast<float>(static_cast<int>((i * 3) % 31) - 15) * 0.004F;
    v_values[i] = static_cast<float>(static_cast<int>((i * 5) % 37) - 18) * 0.025F;
  }
  for (std::size_t i = 0; i < gates; ++i) {
    decay[i] = -0.04F - static_cast<float>(i % 5) * 0.013F;
    beta_values[i] = 0.2F + static_cast<float>(i % 7) * 0.017F;
  }
  std::vector<std::uint16_t> v_bits(scores), beta_bits(gates);
  if (bfloat) {
    for (std::size_t i = 0; i < scores; ++i) {
      v_bits[i] = bfloat_bits(v_values[i]);
      v_values[i] = std::bit_cast<float>(std::uint32_t(v_bits[i]) << 16U);
    }
    for (std::size_t i = 0; i < gates; ++i) {
      beta_bits[i] = bfloat_bits(beta_values[i]);
      beta_values[i] = std::bit_cast<float>(std::uint32_t(beta_bits[i]) << 16U);
    }
  }
  std::vector<std::uint32_t> initial(static_cast<std::size_t>(capacity) * stride);
  for (std::size_t slot = 0; slot < static_cast<std::size_t>(capacity); ++slot) {
    for (std::size_t i = 0; i < stride; ++i) {
      initial[slot * stride + i] = i < item
          ? std::bit_cast<std::uint32_t>(static_cast<float>(static_cast<int>((i + slot * 7) % 41) - 20) * 0.003F)
          : 0x41f00000U + static_cast<std::uint32_t>(i - item);
    }
  }
  std::vector<std::uint32_t> dense(static_cast<std::size_t>(batch) * item);
  for (std::size_t b = 0; b < sources.size(); ++b)
    std::copy_n(initial.begin() + static_cast<std::size_t>(sources[b]) * stride,
                item, dense.begin() + b * item);
  ggml_backend_tensor_set(q, q_values.data(), 0, q_values.size() * sizeof(float));
  ggml_backend_tensor_set(k, k_values.data(), 0, k_values.size() * sizeof(float));
  ggml_backend_tensor_set(v_wide, v_values.data(), 0, v_values.size() * sizeof(float));
  ggml_backend_tensor_set(g, decay.data(), 0, decay.size() * sizeof(float));
  ggml_backend_tensor_set(beta_wide, beta_values.data(), 0, beta_values.size() * sizeof(float));
  if (bfloat) {
    ggml_backend_tensor_set(v, v_bits.data(), 0, v_bits.size() * sizeof(v_bits[0]));
    ggml_backend_tensor_set(beta, beta_bits.data(), 0, beta_bits.size() * sizeof(beta_bits[0]));
  }
  ggml_backend_tensor_set(dense_state, dense.data(), 0, dense.size() * sizeof(dense[0]));
  ggml_backend_tensor_set(backing, initial.data(), 0, initial.size() * sizeof(initial[0]));
  ggml_backend_tensor_set(source_ids, sources.data(), 0, sizeof(sources));
  ggml_backend_tensor_set(destination_ids, destinations.data(), 0, destinations.size() * sizeof(destinations[0]));
  std::vector<std::vector<std::uint32_t>> input_before;
  for (auto *tensor : {q, k, v, g, beta, dense_state, source_ids, destination_ids})
    input_before.push_back(read_bits(tensor));
  ASSERT_EQ(ggml_backend_graph_compute(run.backend, graph), GGML_STATUS_SUCCESS);
  ggml_backend_synchronize(run.backend);
  const auto legacy_bits = read_bits(legacy);
  const auto actual_y = read_bits(indexed);
  ASSERT_GE(legacy_bits.size(), scores + static_cast<std::size_t>(snapshots * batch) * item);
  ASSERT_EQ(actual_y.size(), scores);
  ASSERT_TRUE(std::equal(actual_y.begin(), actual_y.end(), legacy_bits.begin()));
  EXPECT_TRUE(std::any_of(actual_y.begin(), actual_y.end(), [](auto bits) { return (bits & 0x7fffffffU) != 0; }));
  auto expected = initial;
  for (std::size_t b = 0; b < sources.size(); ++b) {
    for (int c = 0; c < snapshots; ++c) {
      // Existing packed native tail is newest-first; the new slot table is
      // chronological over the last K boundaries, including the T>K case.
      const auto reverse = static_cast<std::size_t>(snapshots - 1 - c);
      const auto offset = scores + (reverse * sources.size() + b) * item;
      const auto destination = static_cast<std::size_t>(destinations[b * static_cast<std::size_t>(snapshots) + static_cast<std::size_t>(c)]);
      std::copy_n(legacy_bits.begin() + offset, item, expected.begin() + destination * stride);
    }
  }
  EXPECT_EQ(read_bits(backing), expected); // Includes every old/source/free slot and all padding.
  std::size_t port = 0;
  for (auto *tensor : {q, k, v, g, beta, dense_state, source_ids, destination_ids})
    EXPECT_EQ(read_bits(tensor), input_before[port++]);
}

TEST(GgmlCudaIndexedGdn, FinalStatePreservesNoncontiguousSlotsAndPadding) {
  compare_case(16, 2, 3, 1, false, std::nullopt);
}
TEST(GgmlCudaIndexedGdn, LastKIsChronologicalWhenTokensExceedK) {
  compare_case(16, 2, 4, 2, false, 0.5F);
}
TEST(GgmlCudaIndexedGdn, BfloatValuesAndBetaMatchRoundedLegacyMath) {
  compare_case(128, 1, 4, 2, true, -0.75F);
}
} // namespace
