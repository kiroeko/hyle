#include "hyle/compute/vulkan.h"

#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "fixture.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace t = vulkan_test;

void equal_values(std::span<const float> actual,
                  std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    ASSERT_NEAR(actual[index], expected[index],
                3e-5f + 2e-5f * std::abs(expected[index]))
        << "element " << index;
}

TEST(VulkanNative, NativeOwnersTransfersAndQuota) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1024 * 1024);
  auto context = t::context(&quota);
  ASSERT_TRUE(context) << context.error().message();
  EXPECT_FALSE(context->environment().device_name.empty());
  EXPECT_FALSE(context->environment().shader_compiler.empty());
  EXPECT_EQ(context->environment().shader_revision,
            "456172ec733a135778adcd32d00e576a58232e45");
  EXPECT_TRUE(context->environment().storage_update_after_bind);
  const auto host = t::data(257, 1);
  auto tensor = t::upload(*context, t::dense({257}), host);
  ASSERT_TRUE(tensor) << tensor.error().message();
  EXPECT_GE(quota.used(), 257u * sizeof(float));
  EXPECT_GT(quota.peak(), quota.used()); // actual staging lived during upload
  auto borrowed = context->borrow_ready(
      tensor->description(), tensor->native_buffer(), tensor->native_offset(),
      host.size() * sizeof(float), tensor->native_owner());
  ASSERT_TRUE(borrowed);
  std::weak_ptr<const void> lifetime = tensor->native_owner();
  tensor = v::Tensor{};
  EXPECT_FALSE(lifetime.expired());
  auto values = t::read(*context, *borrowed);
  ASSERT_TRUE(values);
  equal_values(*values, host);
  EXPECT_FALSE(context->borrow_ready(t::dense({257}), borrowed->native_buffer(),
                                     borrowed->native_offset(), 4,
                                     borrowed->native_owner()));
  EXPECT_FALSE(context->borrow_ready(t::dense({257}), borrowed->native_buffer(),
                                     borrowed->native_offset(),
                                     host.size() * sizeof(float), {}));
  EXPECT_TRUE(context->close().ok());
  EXPECT_FALSE(context->allocate(t::dense({1})));
  EXPECT_FALSE(context->upload(*borrowed, std::as_bytes(std::span(host))).ok());
  ASSERT_TRUE(t::read(*context, *borrowed));
  borrowed = v::Tensor{};
  EXPECT_TRUE(lifetime.expired());
  EXPECT_EQ(quota.used(), 0u);
  hyle::mem::QuotaAllocator empty(hyle::mem::system_allocator(), 0);
  auto limited = t::context(&empty);
  ASSERT_TRUE(limited);
  auto refused = limited->allocate(t::dense({1}));
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().IsResourceExhausted());
  EXPECT_EQ(empty.used(), 0u);
}

TEST(VulkanTyped, MatMulDenseConstantChangingAndTransposedRhs) {
  auto context = t::context();
  ASSERT_TRUE(context) << context.error().message();
  for (const auto shape :
       {std::array{1, 67, 37}, std::array{7, 96, 35}, std::array{16, 67, 37},
        std::array{65, 65, 71}, std::array{129, 65, 129}}) {
    const auto [m, k, n] = shape;
    SCOPED_TRACE(::testing::Message() << "M=" << m << " K=" << k << " N=" << n);
    const auto a_desc = t::dense({m, k});
    const auto b_desc = t::dense({k, n});
    const auto a_host = t::data(static_cast<std::size_t>(m) * k, 3);
    auto a = t::upload(*context, a_desc, a_host);
    ASSERT_TRUE(a);
    for (int layout = 0; layout < 3; ++layout) {
      SCOPED_TRACE(::testing::Message() << "RHS layout=" << layout);
      const auto b_host = t::data(static_cast<std::size_t>(k) * n, layout + 2);
      auto physical = b_host;
      auto description = b_desc;
      if (layout == 2) {
        description = *t::dense({n, k}).transposed();
        for (int i = 0; i < k; ++i)
          for (int j = 0; j < n; ++j)
            physical[j * k + i] = b_host[i * n + j];
      }
      auto b = t::upload(*context, description, physical);
      ASSERT_TRUE(b);
      auto program =
          layout == 0 ? v::prepare(*context, c::MatMul{}, a_desc, *b)
                      : v::prepare(*context, c::MatMul{}, a_desc, description);
      ASSERT_TRUE(program) << program.error().message();
      auto invocation = program->create_invocation();
      ASSERT_TRUE(invocation) << invocation.error().message();
      ASSERT_TRUE(
          (layout == 0 ? invocation->bind(*a) : invocation->bind(*a, *b)).ok());
      auto submitted = invocation->submit();
      ASSERT_TRUE(submitted) << submitted.error().message();
      auto result = submitted->wait();
      ASSERT_TRUE(result) << result.error().message();
      auto actual = t::read(*context, result->get()->tensors()[0]);
      ASSERT_TRUE(actual);
      equal_values(*actual, t::matmul(a_host, b_host, m, k, n));
    }
  }
}

TEST(VulkanTyped, SiluSeparateAndPackedHalvesWithOddRows) {
  auto context = t::context();
  ASSERT_TRUE(context) << context.error().message();
  constexpr int rows = 3, width = 521;
  const auto x = t::data(rows * width, 4);
  const auto y = t::data(rows * width, 7);
  std::vector<float> packed(rows * width * 2), expected(rows * width);
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < width; ++column) {
      const auto index = row * width + column;
      packed[row * width * 2 + column] = x[index];
      packed[row * width * 2 + width + column] = y[index];
      expected[index] = x[index] / (1.f + std::exp(-x[index])) * y[index];
    }
  auto a = t::upload(*context, t::dense({rows, width}), x);
  auto b = t::upload(*context, t::dense({rows, width}), y);
  auto ab = t::upload(*context, t::dense({rows, width * 2}), packed);
  ASSERT_TRUE(a && b && ab);
  for (const auto mode :
       {c::GatedInputKind::kSeparate, c::GatedInputKind::kPackedHalves}) {
    auto program =
        mode == c::GatedInputKind::kSeparate
            ? v::prepare(*context, c::GatedActivation{.inputs = mode},
                         a->description(), b->description())
            : v::prepare(*context, c::GatedActivation{.inputs = mode},
                         ab->description());
    ASSERT_TRUE(program) << program.error().message();
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation) << invocation.error().message();
    ASSERT_TRUE((mode == c::GatedInputKind::kSeparate ? invocation->bind(*a, *b)
                                                      : invocation->bind(*ab))
                    .ok());
    auto run = invocation->submit();
    ASSERT_TRUE(run);
    auto result = run->wait();
    ASSERT_TRUE(result) << result.error().message();
    auto actual = t::read(*context, result->get()->tensors()[0]);
    ASSERT_TRUE(actual);
    equal_values(*actual, expected);
  }
}

TEST(VulkanTyped, RebindingChangesBothActivationAndDenseRhsOnRecordedCommands) {
  auto context = t::context();
  ASSERT_TRUE(context);
  constexpr int m = 17, k = 67, n = 37;
  const auto a_desc = t::dense({m, k});
  const auto b_desc = t::dense({k, n});
  auto program = v::prepare(*context, c::MatMul{}, a_desc, b_desc);
  ASSERT_TRUE(program) << program.error().message();
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot) << slot.error().message();
  for (int iteration = 1; iteration < 4; ++iteration) {
    const auto a_host = t::data(m * k, iteration);
    const auto b_host = t::data(k * n, iteration + 4);
    auto a = t::upload(*context, a_desc, a_host);
    auto b = t::upload(*context, b_desc, b_host);
    ASSERT_TRUE(a && b);
    ASSERT_TRUE(slot->bind(*a, *b).ok());
    auto run = slot->submit();
    ASSERT_TRUE(run);
    auto result = run->wait();
    ASSERT_TRUE(result) << result.error().message();
    auto actual = t::read(*context, result->get()->tensors()[0]);
    ASSERT_TRUE(actual);
    equal_values(*actual, t::matmul(a_host, b_host, m, k, n));
  }
}

TEST(VulkanNative, RecorderCombinesMatureFfnAndCustomSpirvWithReadyHandoff) {
  t::Ffn fixture;
  auto context = t::context();
  ASSERT_TRUE(context) << context.error().message();
  auto constants = fixture.constants(*context);
  ASSERT_TRUE(constants);
  constexpr int rows = 3;
  const auto input_desc = t::dense({rows, t::Ffn::k});
  const auto hidden_desc = t::dense({rows, t::Ffn::hidden});
  const auto output_desc = t::dense({rows, t::Ffn::output});
  auto input = t::upload(*context, input_desc, t::data(rows * t::Ffn::k, 1));
  auto gate = context->allocate(hidden_desc);
  auto up = context->allocate(hidden_desc);
  auto product = context->allocate(hidden_desc);
  auto down = context->allocate(output_desc);
  auto destination = context->allocate(output_desc);
  ASSERT_TRUE(input && gate && up && product && down && destination);
  auto borrowed = context->borrow_ready(
      input_desc, input->native_buffer(), input->native_offset(),
      *input_desc.required_bytes(), input->native_owner());
  ASSERT_TRUE(borrowed);
  auto gate_program =
      v::prepare(*context, c::MatMul{}, input_desc, (*constants)[0]);
  auto up_program =
      v::prepare(*context, c::MatMul{}, input_desc, (*constants)[1]);
  auto silu_program = v::prepare(
      *context, c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
      hidden_desc, hidden_desc);
  auto down_program =
      v::prepare(*context, c::MatMul{}, hidden_desc, (*constants)[2]);
  ASSERT_TRUE(gate_program && up_program && silu_program && down_program);
  auto kernel = t::affine(*context);
  ASSERT_TRUE(kernel) << kernel.error().message();
  auto recorder = v::Recorder::create(*context);
  ASSERT_TRUE(recorder);
  ASSERT_TRUE(
      recorder->append(*gate_program, std::array{*borrowed}, std::array{*gate})
          .ok());
  ASSERT_TRUE(
      recorder->append(*up_program, std::array{*borrowed}, std::array{*up})
          .ok());
  ASSERT_TRUE(
      recorder
          ->append(*silu_program, std::array{*gate, *up}, std::array{*product})
          .ok());
  ASSERT_TRUE(
      recorder->append(*down_program, std::array{*product}, std::array{*down})
          .ok());
  ASSERT_TRUE(recorder
                  ->dispatch(*kernel, std::array{*down, *destination},
                             {1, 1, 1}, t::count_push(rows * t::Ffn::output))
                  .ok());
  auto program = recorder->finish();
  ASSERT_TRUE(program) << program.error().message();
  for (int repetition = 0; repetition < 3; ++repetition) {
    const auto x = t::data(rows * t::Ffn::k, repetition + 1);
    ASSERT_TRUE(context->upload(*input, std::as_bytes(std::span(x))).ok());
    auto expected = fixture.oracle(x, rows);
    for (auto &value : expected)
      value = 2.f * value + 1.f;
    auto run = program->submit();
    ASSERT_TRUE(run) << run.error().message();
    ASSERT_TRUE(run->wait().ok());
    auto actual = t::read(*context, *destination);
    ASSERT_TRUE(actual);
    equal_values(*actual, expected);
  }
}

TEST(VulkanTyped, InvalidDomainsAndForeignDeviceAreRejected) {
  auto context = t::context();
  auto other = t::context();
  ASSERT_TRUE(context && other);
  const auto description = t::dense({2, 2});
  auto foreign = t::upload(*other, description, t::data(4, 1));
  ASSERT_TRUE(foreign);
  EXPECT_FALSE(v::prepare(*context, c::MatMul{}, description, *foreign));
  auto program = v::prepare(*context, c::MatMul{}, description, description);
  ASSERT_TRUE(program);
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot);
  EXPECT_FALSE(slot->submit());
  EXPECT_FALSE(slot->bind(*foreign, *foreign).ok());
  EXPECT_FALSE(v::prepare(*context, c::Unary{c::UnaryKind::kNeg}, description));
  EXPECT_FALSE(
      v::prepare(*context, c::MatMul{}, description, t::dense({3, 2})));
  EXPECT_FALSE(
      v::prepare(*context, c::MatMul{}, t::dense({0, 2}), description));
  const std::array<std::int64_t, 2> dims{2, 2};
  auto half = c::TensorDesc::dense(c::DataType::kFloat16, dims);
  ASSERT_TRUE(half);
  EXPECT_FALSE(v::prepare(*context, c::MatMul{}, *half, *half));
}

} // namespace
