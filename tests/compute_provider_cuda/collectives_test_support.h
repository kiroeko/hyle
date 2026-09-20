#pragma once

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <numeric>

namespace cccl_test {
using namespace hyle::compute;
namespace mem = hyle::mem;
template <class T> constexpr DataType type() {
  if constexpr (std::is_same_v<T, std::int32_t>)
    return DataType::kInt32;
  else
    return half_test::data_type<T>();
}
template <class T> double number(T value) {
  if constexpr (std::is_same_v<T, std::int32_t>)
    return value;
  else
    return half_test::decode(value);
}
template <class T> T encode(double value) {
  if constexpr (std::is_same_v<T, std::int32_t>)
    return static_cast<T>(value);
  else
    return half_test::encode<T>(value);
}
inline TensorDesc description(DataType dtype,
                              std::span<const std::int64_t> shape,
                              std::size_t axis, int mode) {
  if (mode == 0)
    return TensorDesc::dense(dtype, shape).value();
  std::vector<std::int64_t> strides(shape.size());
  std::int64_t step = static_cast<std::int64_t>(data_type_size(dtype)) + 1;
  for (std::size_t d = shape.size(); d-- > 0;) {
    strides[d] = mode == 2 && d == axis ? 0 : step;
    step *= std::max<std::int64_t>(shape[d], 1);
  }
  return TensorDesc::strided_bytes(dtype, shape, strides).value();
}
inline std::size_t offset(const TensorDesc &desc, std::size_t logical) {
  std::size_t result = 0;
  for (std::size_t d = desc.rank(); d-- > 0;) {
    result += logical % desc.dimensions()[d] * desc.byte_strides()[d];
    logical /= desc.dimensions()[d];
  }
  return result;
}
struct Tensor final {
  mem::Buffer buffer;
  TensorView view;
  Tensor(const TensorDesc &desc, mem::MemoryResourceRef resource,
         bool element_offset = false)
      : buffer(mem::Buffer::create(resource, desc.required_bytes().value() + 32,
                                   256)
                   .value()),
        view(TensorView::create(
                 desc,
                 buffer.view()
                     .subview(element_offset ? data_type_size(desc.data_type())
                                             : 0,
                              desc.required_bytes().value())
                     .value())
                 .value()) {
    EXPECT_TRUE(buffer.view().fill(std::byte{0xa5}).ok());
  }
  template <class T> void write(std::span<const T> data) {
    std::vector<std::byte> bytes(view.storage().size(), std::byte{0x5a});
    for (std::size_t i = 0; i < data.size(); ++i)
      std::memcpy(bytes.data() + offset(view.description(), i), &data[i],
                  sizeof(T));
    ASSERT_TRUE(view.storage().write(bytes).ok());
  }
  template <class T> std::vector<T> read() const {
    std::vector<T> result(view.description().element_count().value());
    std::vector<std::byte> bytes(view.storage().size());
    EXPECT_TRUE(view.storage().read(bytes).ok());
    for (std::size_t i = 0; i < result.size(); ++i)
      std::memcpy(&result[i], bytes.data() + offset(view.description(), i),
                  sizeof(T));
    return result;
  }
  void guard() const {
    std::array<std::byte, 16> bytes{};
    ASSERT_TRUE(buffer.read(buffer.size() - bytes.size(), bytes).ok());
    for (auto value : bytes)
      EXPECT_EQ(value, std::byte{0xa5});
  }
};
inline mem::Buffer workspace(const PreparedOperation &program,
                             mem::MemoryResourceRef memory) {
  const auto requirement = program.workspace_requirement();
  return mem::Buffer::create(memory, requirement.size, requirement.alignment)
      .value();
}
template <class T>
void scan(Context &context, std::span<const std::int64_t> shape,
          std::int32_t requested_axis, int mode) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const auto axis = static_cast<std::size_t>(
      requested_axis < 0 ? requested_axis + static_cast<int>(shape.size())
                         : requested_axis);
  SCOPED_TRACE(::testing::Message()
               << static_cast<int>(type<T>()) << '/' << shape[axis] << '/'
               << requested_axis << '/' << mode);
  const auto desc = description(type<T>(), shape, axis, mode);
  const auto dense = TensorDesc::dense(type<T>(), shape).value();
  Tensor input(desc, memory.ref(), mode == 1), output(dense, memory.ref());
  std::vector<T> data(desc.element_count().value());
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<T>(static_cast<int>(i % 29) - 14);
  input.write<T>(data);
  data = input.read<T>(); // Includes repeated storage for broadcast strides.
  std::size_t inner = 1, outer = 1;
  for (std::size_t d = 0; d < axis; ++d)
    outer *= shape[d];
  for (std::size_t d = axis + 1; d < shape.size(); ++d)
    inner *= shape[d];
  const auto width = static_cast<std::size_t>(shape[axis]);
  for (bool exclusive : {false, true}) {
    auto prepared =
        prepare(context, Scan{.axis = requested_axis, .exclusive = exclusive},
                desc, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, cuda_native::kCcclProviderId);
    EXPECT_EQ(prepared->runtime_properties().replay_class,
              ProviderReplayClass::kManaged);
    EXPECT_EQ(prepared->runtime_properties().blocks_host,
              (std::is_same_v<T, std::int32_t>));
    auto scratch = workspace(*prepared, memory.ref());
    auto slot = prepared->create_invocation(scratch.view()).value();
    auto done = slot.submit(queue, input.view.as_const(), output.view);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = output.read<T>();
    for (std::size_t group = 0; group < outer; ++group)
      for (std::size_t column = 0; column < inner; ++column) {
        double sum = 0;
        for (std::size_t pos = 0; pos < width; ++pos) {
          const auto index = (group * width + pos) * inner + column;
          const auto expected = exclusive ? sum : sum + data[index];
          sum += data[index];
          ASSERT_EQ(actual[index], expected) << index;
        }
      }
    output.guard();
  }
}
template <class T>
void topk(Context &context, std::span<const std::int64_t> shape,
          std::int32_t requested_axis, int mode, std::size_t k) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const auto axis = static_cast<std::size_t>(
      requested_axis < 0 ? requested_axis + static_cast<int>(shape.size())
                         : requested_axis);
  SCOPED_TRACE(::testing::Message()
               << static_cast<int>(type<T>()) << '/' << shape[axis] << '/'
               << requested_axis << '/' << mode << '/' << k);
  const auto desc = description(type<T>(), shape, axis, mode);
  Tensor input(desc, memory.ref(), mode == 1);
  std::vector<T> data(desc.element_count().value());
  for (std::size_t i = 0; i < data.size(); ++i) {
    if constexpr (std::is_same_v<T, std::int32_t>) {
      constexpr std::array<std::int32_t, 8> pattern{
          INT32_MAX, INT32_MIN, 16777217, 16777216, 0, -1, 0, INT32_MAX};
      data[i] = pattern[i % pattern.size()];
    } else {
      constexpr auto inf = std::numeric_limits<double>::infinity();
      constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
      constexpr std::array pattern{3.,   -0., 0.,   nan, inf,
                                   -inf, 3.,  -nan, -7., 1.};
      data[i] = encode<T>(pattern[i % pattern.size()]);
      if (i % 10 == 7) {
        if constexpr (std::is_same_v<T, float>)
          data[i] = std::bit_cast<float>(0xffc001abU);
        else if constexpr (std::is_same_v<T, hyle::simd::float16_t>)
          data[i] = T::from_bits(0xfeab);
        else
          data[i] = T::from_bits(0xffeb);
      }
    }
  }
  input.write<T>(data);
  data = input.read<T>();
  std::size_t inner = 1, outer = 1;
  for (std::size_t d = 0; d < axis; ++d)
    outer *= shape[d];
  for (std::size_t d = axis + 1; d < shape.size(); ++d)
    inner *= shape[d];
  const auto width = static_cast<std::size_t>(shape[axis]);
  for (bool largest : {false, true}) {
    TopK operation{.count = k,
                   .axis = requested_axis,
                   .largest = largest,
                   .sorted = largest};
    std::array<TensorDesc, 2> outputs;
    ASSERT_TRUE(infer_output_descriptions(Operation(operation),
                                          std::span(&desc, 1), outputs)
                    .ok());
    Tensor values(outputs[0], memory.ref()), indices(outputs[1], memory.ref());
    auto prepared = prepare(context, operation, desc, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, cuda_native::kCcclProviderId);
    auto scratch = workspace(*prepared, memory.ref());
    auto slot = prepared->create_invocation(scratch.view()).value();
    const auto in = input.view.as_const();
    const std::array out{values.view, indices.view};
    auto done = slot.submit(queue, std::span(&in, 1), out);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = values.read<T>();
    const auto selected = indices.read<std::int32_t>();
    for (std::size_t group = 0; group < outer; ++group)
      for (std::size_t column = 0; column < inner; ++column) {
        std::vector<std::size_t> order(width);
        std::iota(order.begin(), order.end(), 0);
        const auto source_index = [&](std::size_t pos) {
          return (group * width + pos) * inner + column;
        };
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
          const auto x = number(data[source_index(a)]),
                     y = number(data[source_index(b)]);
          if (std::isnan(x))
            return false;
          if (std::isnan(y))
            return true;
          return largest ? x > y : x < y;
        });
        for (std::size_t pos = 0; pos < k; ++pos) {
          const auto output_index = (group * k + pos) * inner + column;
          ASSERT_EQ(selected[output_index], order[pos]);
          ASSERT_EQ(std::memcmp(&actual[output_index],
                                &data[source_index(order[pos])], sizeof(T)),
                    0);
        }
      }
    values.guard();
    indices.guard();
  }
}
template <class Bits, class Index>
void gather(Context &context, DataType dtype,
            std::span<const std::int64_t> shape, std::int32_t requested_axis,
            std::span<const std::int64_t> index_shape, int source_mode,
            int index_mode) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const auto axis = static_cast<std::size_t>(
      requested_axis < 0 ? requested_axis + static_cast<int>(shape.size())
                         : requested_axis);
  SCOPED_TRACE(::testing::Message()
               << static_cast<int>(dtype) << '/' << sizeof(Index) << '/'
               << requested_axis << '/' << source_mode << '/' << index_mode);
  const auto desc = description(dtype, shape, axis, source_mode);
  const auto idesc = description(
      sizeof(Index) == 8 ? DataType::kInt64 : DataType::kInt32, index_shape,
      index_shape.empty() ? 0 : index_shape.size() - 1, index_mode);
  Tensor source(desc, memory.ref(), true), indices(idesc, memory.ref(), true);
  std::vector<Bits> data(desc.element_count().value());
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = dtype == DataType::kBool
                  ? static_cast<Bits>(i & 1u)
                  : static_cast<Bits>(0x7fc000018123ULL + i * 0x9e3779b1ULL);
  std::vector<Index> ids(idesc.element_count().value());
  for (std::size_t i = 0; i < ids.size(); ++i)
    ids[i] = static_cast<Index>((i * 3 + 1) % shape[axis]);
  source.write<Bits>(data);
  indices.write<Index>(ids);
  data = source.read<Bits>();
  ids = indices.read<Index>();
  auto program = prepare(context, Gather{.axis = requested_axis}, desc, idesc,
                         memory.ref());
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->provider().id, cuda_native::kCcclProviderId);
  EXPECT_TRUE(program->runtime_properties().blocks_host);
  auto scratch = workspace(*program, memory.ref());
  auto slot = program->create_invocation(scratch.view()).value();
  Tensor output(program->output_description(0).value(), memory.ref());
  auto done = slot.submit(queue, source.view.as_const(),
                          indices.view.as_const(), output.view);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto actual = output.read<Bits>();
  std::size_t outer = 1, inner = 1;
  for (std::size_t d = 0; d < axis; ++d)
    outer *= shape[d];
  for (std::size_t d = axis + 1; d < shape.size(); ++d)
    inner *= shape[d];
  for (std::size_t group = 0; group < outer; ++group)
    for (std::size_t selected = 0; selected < ids.size(); ++selected)
      for (std::size_t col = 0; col < inner; ++col)
        ASSERT_EQ(actual[(group * ids.size() + selected) * inner + col],
                  data[(group * shape[axis] + ids[selected]) * inner + col]);
  EXPECT_EQ(source.read<Bits>(), data);
  EXPECT_EQ(indices.read<Index>(), ids);
  output.guard();
}
} // namespace cccl_test
