#include "gguf_test_support.h"
#include "vulkan_program_internal.h"

#include <algorithm>
#include <utility>

namespace {
using namespace vulkan_gguf_test;

TEST(VulkanGguf, EveryCatalogFormatUsesNativeDirectRegionAndReplays) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  constexpr int k = 256;
  constexpr int n = 8;
  constexpr int m = 2;
  const auto input_desc =
      vulkan_tensor_test::dense(c::DataType::kFloat32, {m, k});
  const auto physical = weights(n, k);
  for (const auto type : selected_matrix_types()) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto weight_desc =
        c::gguf_weight_description(type, c::DataType::kFloat32, k, n).value();
    auto packed = pack(type, physical, n, k, *weight_desc.required_bytes());
    auto weight = context.upload(weight_desc, packed.bytes).value();
    auto program = v::prepare(context, c::MatMul{}, input_desc, weight_desc);
    ASSERT_TRUE(program) << program.error().message();
    ASSERT_TRUE(v::detail::Access::program(*program)->dispatches.back().region);
    auto invocation = program->create_invocation().value();
    for (int replay = 1; replay <= 2; ++replay) {
      const auto values = activations(m, k, replay);
      auto input = vulkan_tensor_test::make(context, input_desc, values);
      ASSERT_TRUE(invocation.bind(input, weight).ok());
      auto work = invocation.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      expect_native_projection(context, result->get()->tensors()[0], type,
                               values, packed, m, n, k);
    }
  }
}

TEST(VulkanGguf, Iq1SOriginalQ8ActivationCauseCapture) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  constexpr int k = 256, n = 8, m = 2;
  const auto input_desc = vulkan_tensor_test::dense(c::DataType::kFloat32, {m, k});
  const auto weight_desc = c::gguf_weight_description(c::GgufQuantizedType::kIq1S,
      c::DataType::kFloat32, k, n).value();
  auto packed = pack(c::GgufQuantizedType::kIq1S, weights(n, k), n, k, *weight_desc.required_bytes());
  auto weight = context.upload(weight_desc, packed.bytes).value();
  auto program = v::prepare(context, c::MatMul{}, input_desc, weight_desc).value();
  const auto *traits = ggml_get_type_traits(GGML_TYPE_Q8_1);
  ASSERT_NE(traits->from_float_ref, nullptr);
  ASSERT_EQ(ggml_type_size(GGML_TYPE_Q8_1), 36u);
  ASSERT_EQ(ggml_type_size(GGML_TYPE_IQ1_S), 50u);
  auto half = [](const std::byte *p) {
    ggml_fp16_t bits; std::memcpy(&bits, p, sizeof(bits));
    return ggml_fp16_to_fp32(bits);
  };
  for (int replay : {1, 2}) {
    auto x = activations(m, k, replay);
    std::vector<float> f32(x.begin(), x.end());
    std::vector<std::byte> q8(m * (k / 32) * 36);
    traits->from_float_ref(f32.data(), q8.data(), m * k);
    auto input = vulkan_tensor_test::make(context, input_desc, x);
    auto slot = program.create_invocation().value();
    ASSERT_TRUE(slot.bind(input, weight).ok());
    auto work = slot.submit().value(); auto result = work.wait().value();
    auto actual = vulkan_tensor_test::values(context, result.get()->tensors()[0]);
    const auto nominal = matmul(x, packed.decoded, m, n, k);
    double native_error = 0, nominal_error = 0;
    for (int token = 0; token < m; ++token) for (int row = 0; row < n; ++row) {
      const auto *a = packed.bytes.data() + row * 50;
      const float scale = half(a);
      double total = 0;
      for (int group = 0; group < 8; ++group) {
        std::uint16_t high; std::memcpy(&high, a + 34 + group * 2, sizeof(high));
        const float dl = scale * (2 * ((high >> 12) & 7) + 1);
        const float delta = (high & 0x8000) ? -.125F : .125F;
        const auto *b = q8.data() + (token * 8 + group) * 36;
        int dot = 0;
        for (int i = 0; i < 32; ++i) {
          const int grid = int(std::round(packed.decoded[row * k + group * 32 + i] / dl - delta));
          std::int8_t code; std::memcpy(&code, b + 4 + i, sizeof(code));
          dot += (grid + 1) * code;
        }
        total += dl * (half(b) * dot + (delta - 1) * half(b + 2));
      }
      const auto pos = token * n + row;
      native_error = std::max(native_error, std::abs(actual[pos] - total));
      nominal_error = std::max(nominal_error, std::abs(actual[pos] - nominal[pos]));
    }
    std::printf("IQ1_S_CAUSE replay=%d q8_original_max_error=%.9g nominal_f32_max_error=%.9g\n",
        replay, native_error, nominal_error);
  }
}

TEST(VulkanGguf, TargetEmbeddingFormatsUseNativeRowsAndCheckedPublication) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  constexpr int rows = 3;
  constexpr int width = 256;
  const auto ids_desc = vulkan_tensor_test::dense(c::DataType::kInt32, {2});
  for (const auto type : kEmbeddingTypes) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto stored =
        c::gguf_weight_description(type, c::DataType::kFloat32, width, rows)
            .value();
    const auto table_desc = stored.transposed().value();
    const auto physical = weights(rows, width, static_cast<int>(type));
    const auto packed =
        pack(type, physical, rows, width, *table_desc.required_bytes());
    auto table_result = context.upload(table_desc, packed.bytes);
    ASSERT_TRUE(table_result) << table_result.error().message();
    const auto table = std::move(*table_result);
    auto program = v::prepare(context, c::Gather{}, table, ids_desc);
    ASSERT_TRUE(program) << program.error().message();
    const auto &dispatches = v::detail::Access::program(*program)->dispatches;
    EXPECT_EQ(std::ranges::count_if(dispatches,
                                    [](const auto &dispatch) {
                                      return dispatch.region != nullptr;
                                    }),
              1);
    auto invocation_result = program->create_invocation();
    ASSERT_TRUE(invocation_result) << invocation_result.error().message();
    auto invocation = std::move(*invocation_result);
    constexpr std::array<std::array<std::int32_t, 2>, 2> selections{
        {{{2, 0}}, {{1, 2}}}};
    v::Tensor retained;
    for (std::size_t replay = 0; replay < selections.size(); ++replay) {
      const auto &selection = selections[replay];
      auto ids_result =
          context.upload(ids_desc, std::as_bytes(std::span(selection)));
      ASSERT_TRUE(ids_result) << ids_result.error().message();
      const auto ids = std::move(*ids_result);
      ASSERT_TRUE(invocation.bind(ids).ok());
      auto work = invocation.submit();
      ASSERT_TRUE(work) << work.error().message();
      auto result = work->wait();
      ASSERT_TRUE(result) << result.error().message();
      const auto output = result->get()->tensors()[0];
      std::vector<double> expected;
      expected.reserve(2 * width);
      for (const auto row : selection)
        for (int column = 0; column < width; ++column)
          expected.push_back(packed.decoded[row * width + column]);
      expect_close(context, output, expected);
      if (replay + 1 == selections.size())
        retained = output;
    }
    const auto before = vulkan_tensor_test::bytes(context, retained);
    constexpr std::array<std::int32_t, 2> invalid{0, rows};
    auto bad_result =
        context.upload(ids_desc, std::as_bytes(std::span(invalid)));
    ASSERT_TRUE(bad_result) << bad_result.error().message();
    const auto bad = std::move(*bad_result);
    auto failed_invocation_result = program->create_invocation();
    ASSERT_TRUE(failed_invocation_result)
        << failed_invocation_result.error().message();
    auto failed_invocation = std::move(*failed_invocation_result);
    ASSERT_TRUE(failed_invocation.bind(bad).ok());
    auto failed_work = failed_invocation.submit();
    ASSERT_TRUE(failed_work) << failed_work.error().message();
    auto failure = failed_work->wait();
    ASSERT_FALSE(failure);
    EXPECT_TRUE(failure.error().IsOutOfRange());
    EXPECT_EQ(vulkan_tensor_test::bytes(context, retained), before);

    auto good_result =
        context.upload(ids_desc, std::as_bytes(std::span(selections.front())));
    ASSERT_TRUE(good_result) << good_result.error().message();
    const auto good = std::move(*good_result);
    auto recovered_result = program->create_invocation();
    ASSERT_TRUE(recovered_result) << recovered_result.error().message();
    auto recovered = std::move(*recovered_result);
    ASSERT_TRUE(recovered.bind(good).ok());
    auto recovered_work = recovered.submit();
    ASSERT_TRUE(recovered_work) << recovered_work.error().message();
    EXPECT_TRUE(recovered_work->wait().has_value());
  }
}

TEST(VulkanGguf, PackedFormatsGroupedExpertsUseNativeMulMatId) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  for (const auto type : selected_matrix_types(true)) {
    SCOPED_TRACE(static_cast<int>(type));
    constexpr int experts = 4;
    constexpr int n = 8;
    constexpr int k = 256;
    constexpr int tokens = 2;
    constexpr int slots = 2;
    const auto weight_desc =
        c::gguf_grouped_weight_description(type,
                                           c::DataType::kFloat32, experts, n, k)
            .value();
    const auto activation_desc =
        vulkan_tensor_test::dense(c::DataType::kFloat32, {tokens, slots, k});
    const auto ids_desc =
        vulkan_tensor_test::dense(c::DataType::kInt32, {tokens, slots});
    const auto physical = weights(experts * n, k, 3);
    auto packed = pack(type, physical, experts * n, k,
                       *weight_desc.required_bytes());
    auto weight = context.upload(weight_desc, packed.bytes).value();
    const std::array<std::int32_t, tokens * slots> ids{0, 3, 1, 2};
    auto selected =
        context.upload(ids_desc, std::as_bytes(std::span(ids))).value();
    auto program = v::prepare(context, c::model_ops::grouped_matmul(),
                              weight_desc, activation_desc, ids_desc);
    ASSERT_TRUE(program) << program.error().message();
    const auto &prepared = *v::detail::Access::program(*program);
    ASSERT_TRUE(prepared.status_value.has_value());
    ASSERT_TRUE(std::ranges::any_of(prepared.dispatches, [](const auto &dispatch) {
      return static_cast<bool>(dispatch.region);
    }));
    auto invocation = program->create_invocation().value();
    for (int replay = 1; replay <= 2; ++replay) {
      const auto values = activations(tokens * slots, k, replay);
      auto activation =
          vulkan_tensor_test::make(context, activation_desc, values);
      ASSERT_TRUE(invocation.bind(weight, activation, selected).ok());
      auto work = invocation.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      expect_native_projection(context, result->get()->tensors()[0], type,
                               values, packed, tokens * slots, n, k, ids);
    }
  }
}

TEST(VulkanGguf, DenseBfloatGroupedExpertsUseNativeMulMatIdAndReplay) {
  auto context = vulkan_test::context().value();
  constexpr int experts = 3;
  constexpr int n = 2;
  constexpr int k = 32;
  constexpr int tokens = 1;
  constexpr int slots = 2;
  const auto weight_desc = vulkan_tensor_test::dense(
      c::DataType::kBFloat16, {experts, n, k});
  const auto activation_desc = vulkan_tensor_test::dense(
      c::DataType::kFloat32, {tokens, slots, k});
  const auto ids_desc =
      vulkan_tensor_test::dense(c::DataType::kInt32, {tokens, slots});
  constexpr std::array<double, experts * n> row_values{1.0, 2.0, 3.0,
                                                        4.0, -1.0, 0.5};
  std::vector<double> weight_values(experts * n * k);
  for (std::size_t row = 0; row < row_values.size(); ++row)
    std::fill_n(weight_values.begin() + row * k, k, row_values[row]);
  auto weight =
      vulkan_tensor_test::make(context, weight_desc, weight_values);
  constexpr std::array<std::int32_t, tokens * slots> ids{0, 2};
  auto selected =
      context.upload(ids_desc, std::as_bytes(std::span(ids))).value();
  auto program = v::prepare(context, c::model_ops::grouped_matmul(),
                            weight_desc, activation_desc, ids_desc);
  ASSERT_TRUE(program) << program.error().message();
  const auto &prepared = *v::detail::Access::program(*program);
  ASSERT_TRUE(prepared.status_value.has_value());
  ASSERT_TRUE(std::ranges::any_of(prepared.dispatches, [](const auto &dispatch) {
    return static_cast<bool>(dispatch.region);
  }));
  auto invocation = program->create_invocation().value();
  for (int replay = 1; replay <= 2; ++replay) {
    const std::vector<double> activation_values(tokens * slots * k,
                                                0.25 * replay);
    auto activation =
        vulkan_tensor_test::make(context, activation_desc, activation_values);
    ASSERT_TRUE(invocation.bind(weight, activation, selected).ok());
    auto work = invocation.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const std::array expected{8.0 * replay, 16.0 * replay,
                              -8.0 * replay, 4.0 * replay};
    vulkan_tensor_test::expect(context, result->get()->tensors()[0], expected);
  }
}

TEST(VulkanGguf, PlainF32GroupedExpertsPreserveOperandsOnMatrixPath) {
  auto context = vulkan_test::context().value();
  constexpr int experts = 3, columns = 2, depth = 32, tokens = 16, slots = 2;
  const auto weight_desc = vulkan_tensor_test::dense(
      c::DataType::kFloat32, {experts, columns, depth});
  const auto input_desc = vulkan_tensor_test::dense(
      c::DataType::kFloat32, {tokens, slots, depth});
  const auto ids_desc = vulkan_tensor_test::dense(
      c::DataType::kInt32, {tokens, slots});
  constexpr std::array<double, experts * columns> coefficients{1, 2, 3, 4, -1, 0.5};
  std::vector<double> weights(experts * columns * depth);
  for (std::size_t row = 0; row < coefficients.size(); ++row)
    std::fill_n(weights.begin() + row * depth, depth, coefficients[row]);
  auto weight = vulkan_tensor_test::make(context, weight_desc, weights);
  std::array<std::int32_t, tokens * slots> ids{};
  for (int token = 0; token < tokens; ++token) {
    ids[token * slots] = 0;
    ids[token * slots + 1] = 2;
  }
  auto selected = context.upload(ids_desc, std::as_bytes(std::span(ids))).value();
  auto program = v::prepare(context, c::model_ops::grouped_matmul(),
                            weight_desc, input_desc, ids_desc);
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation().value();
  for (float input : {1.0004f, 65536.f}) {
    const std::vector<double> values(tokens * slots * depth, input);
    auto tensor = vulkan_tensor_test::make(context, input_desc, values);
    ASSERT_TRUE(invocation.bind(weight, tensor, selected).ok());
    auto result = invocation.submit().value().wait();
    ASSERT_TRUE(result) << result.error().message();
    std::vector<double> expected(tokens * slots * columns);
    for (int token = 0; token < tokens; ++token)
      for (int slot = 0; slot < slots; ++slot)
        for (int column = 0; column < columns; ++column)
          expected[(token * slots + slot) * columns + column] =
              depth * double(input) * coefficients[ids[token * slots + slot] * columns + column];
    vulkan_tensor_test::expect(context, result->get()->tensors()[0], expected);
  }
}

TEST(VulkanGguf, RejectsMalformedFormatAndPackedStorageOnWrongPorts) {
  auto context = vulkan_test::context().value();
  const auto input = vulkan_tensor_test::dense(c::DataType::kFloat32, {2, 256});
  const auto malformed_format =
      c::StorageFormat::blocked(c::kGgufQ4K, 256, 143).value();
  const auto malformed =
      c::TensorDesc::blocked_strided(
          c::DataType::kFloat32, std::array<std::int64_t, 2>{256, 8},
          malformed_format, 0, std::array<std::int64_t, 2>{143, 143})
          .value();
  EXPECT_FALSE(v::prepare(context, c::MatMul{}, input, malformed));
  const auto packed = c::gguf_weight_description(c::GgufQuantizedType::kQ4K,
                                                 c::DataType::kFloat32, 256, 8)
                          .value();
  EXPECT_FALSE(v::prepare(context, c::Copy{}, packed));
  EXPECT_FALSE(v::prepare(context, c::MatMul{}, packed, input));
  EXPECT_FALSE(v::prepare(context,
                          c::MatMul{.accumulation_type = c::DataType::kFloat16},
                          input, packed));
}
} // namespace
