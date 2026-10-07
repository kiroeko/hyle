#include <array>
#include <span>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/mem/buffer.h"

namespace {

TEST(OneDnnComposition, UnsupportedShapeAdvancesToQualifiedMatureProvider) {
  namespace c = hyle::compute;
  const auto mature = c::aten_cpu::provider();
  const std::array providers{
      c::ProviderBinding{c::onednn::provider(), 100},
      c::ProviderBinding{mature.valid() ? mature : c::native::provider(), 200}};
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                {.providers = providers});
  ASSERT_TRUE(context.has_value()) << context.error().message();

  const std::array<std::int64_t, 2> left_shape{1, 17};
  // Transposed storage is a genuine BRGEMM adapter limit, unlike M1.
  const std::array<std::int64_t, 2> right_shape{65, 17};
  const std::array<std::int64_t, 2> output_shape{1, 65};
  const auto left = hyle::compute::TensorDesc::dense(
      hyle::compute::DataType::kFloat32, left_shape);
  const auto right = hyle::compute::TensorDesc::dense(
      hyle::compute::DataType::kFloat32, right_shape);
  const auto output = hyle::compute::TensorDesc::dense(
      hyle::compute::DataType::kFloat32, output_shape);
  ASSERT_TRUE(left.has_value());
  ASSERT_TRUE(right.has_value());
  ASSERT_TRUE(output.has_value());

  auto prepared =
      hyle::compute::prepare(*context, hyle::compute::MatMul{}, *left,
                             right->transposed().value(), *output);
  if (!mature.valid()) {
    // Explicitly trimmed SDKs must not resurrect the retired Native MatMul.
    ASSERT_FALSE(prepared.has_value());
    EXPECT_TRUE(prepared.error().IsUnimplemented());
    return;
  }
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::aten_cpu::kProviderId);

  std::array<float, 17> a{};
  std::array<float, 65 * 17> b{};
  std::array<float, 65> result{};
  auto a_buffer = hyle::mem::Buffer::create(hyle::mem::system_resource(), sizeof(a), 64).value();
  auto b_buffer = hyle::mem::Buffer::create(hyle::mem::system_resource(), sizeof(b), 64).value();
  auto output_buffer = hyle::mem::Buffer::create(hyle::mem::system_resource(), sizeof(result), 64).value();
  auto invocation = prepared->create_invocation().value();
  auto queue = context->create_queue().value();
  const auto av = c::ConstTensorView::create(*left, a_buffer.view()).value();
  const auto bv = c::ConstTensorView::create(right->transposed().value(), b_buffer.view()).value();
  const auto cv = c::TensorView::create(*output, output_buffer.view()).value();
  for (int replay = 0; replay < 2; ++replay) {
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = float(int(i % 7) - 3) * (replay ? .125F : .25F);
    for (std::size_t i = 0; i < b.size(); ++i)
      b[i] = float(int((i * 7 + replay * 3) % 31) - 15) / 16.F;
    ASSERT_TRUE(a_buffer.view().write(std::as_bytes(std::span(a))).ok());
    ASSERT_TRUE(b_buffer.view().write(std::as_bytes(std::span(b))).ok());
    auto completed = invocation.submit(queue, av, bv, cv);
    ASSERT_TRUE(completed.has_value()) << completed.error().message();
    ASSERT_TRUE(completed->wait().ok());
    ASSERT_TRUE(output_buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
    for (std::size_t col = 0; col < result.size(); ++col) {
      double expected = 0;
      for (std::size_t k = 0; k < a.size(); ++k)
        expected += double(a[k]) * double(b[col * a.size() + k]);
      EXPECT_FLOAT_EQ(result[col], float(expected));
    }
  }
}

} // namespace
