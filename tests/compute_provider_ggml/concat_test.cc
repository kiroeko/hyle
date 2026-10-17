#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"

namespace {
using namespace hyle::compute;

TEST(GgmlCpuConcat, BalancedPortsPreserveAllStorageBitsAxesAndEmptyInputs) {
  auto context =
      Context::create(cpu::device(), ggml_cpu::provider({.native_threads = 1}))
          .value();
  auto queue = context.create_queue().value();
  for (int raw = int(DataType::kBool); raw <= int(DataType::kUInt64); ++raw) {
    const auto type = static_cast<DataType>(raw);
    const auto scalar_bytes = data_type_size(type);
    for (int axis : {0, 1, 2}) {
      for (int count : {1, 3, 4, 7}) {
        SCOPED_TRACE(testing::Message() << raw << '/' << axis << '/' << count);
        std::vector<TensorDesc> descriptions;
        std::vector<hyle::mem::Buffer> buffers;
        std::vector<ConstTensorView> sources;
        std::vector<std::size_t> widths;
        std::vector<bool> padded;
        std::array<std::int64_t, 3> output_shape{2, 3, 5};
        output_shape[axis] = 0;
        std::size_t inner = 1, outer = 1;
        for (int i = 0; i < axis; ++i)
          outer *= output_shape[i];
        for (int i = axis + 1; i < 3; ++i)
          inner *= output_shape[i];
        for (int port = 0; port < count + (count > 1); ++port) {
          auto shape = output_shape;
          const bool empty = count > 1 && port == 1;
          shape[axis] = empty ? 0 : 1 + port % 3;
          const bool strided = !empty && port % 2 == 0;
          const std::array<std::int64_t, 3> strides{
              shape[1] * shape[2] * std::int64_t(scalar_bytes) * 2,
              shape[2] * std::int64_t(scalar_bytes) * 2,
              std::int64_t(scalar_bytes) * 2};
          descriptions.push_back(
              strided ? TensorDesc::strided_bytes(type, shape, strides).value()
                      : TensorDesc::dense(type, shape).value());
          buffers.push_back(hyle::mem::Buffer::create(
                                hyle::mem::system_resource(),
                                descriptions.back().required_bytes().value(),
                                64)
                                .value());
          sources.push_back(ConstTensorView::create(descriptions.back(),
                                                    buffers.back().view())
                                .value());
          widths.push_back(std::size_t(shape[axis]) * inner);
          padded.push_back(strided);
          output_shape[axis] += shape[axis];
        }
        const auto output_desc = TensorDesc::dense(type, output_shape).value();
        auto output =
            hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                      output_desc.required_bytes().value(), 64)
                .value();
        const auto destination =
            TensorView::create(output_desc, output.view()).value();
        auto prepared =
            prepare(context, Operation(Concat{.axis = axis - 3}), descriptions,
                    std::span(&output_desc, 1), hyle::mem::system_resource());
        ASSERT_TRUE(prepared) << prepared.error().message();
        auto invocation = prepared->create_invocation();
        ASSERT_TRUE(invocation) << invocation.error().message();
        for (int replay = 0; replay < 2; ++replay) {
          std::vector<std::vector<std::byte>> logical;
          for (std::size_t port = 0; port < sources.size(); ++port) {
            logical.emplace_back(outer * widths[port] * scalar_bytes);
            auto &bytes = logical.back();
            for (std::size_t i = 0; i < bytes.size(); ++i)
              bytes[i] = std::byte((i * 37 + port * 61 + replay * 113) & 255);
            std::vector<std::byte> physical(buffers[port].size(),
                                            std::byte{0xA5});
            for (std::size_t i = 0; i < bytes.size() / scalar_bytes; ++i)
              std::memcpy(physical.data() +
                              i * scalar_bytes * (padded[port] ? 2 : 1),
                          bytes.data() + i * scalar_bytes, scalar_bytes);
            ASSERT_TRUE(buffers[port].view().write(physical).ok());
          }
          std::vector<std::byte> expected;
          for (std::size_t row = 0; row < outer; ++row)
            for (std::size_t port = 0; port < sources.size(); ++port) {
              const auto size = widths[port] * scalar_bytes;
              expected.insert(expected.end(),
                              logical[port].begin() + row * size,
                              logical[port].begin() + (row + 1) * size);
            }
          auto done =
              invocation->submit(queue, sources, std::span(&destination, 1));
          ASSERT_TRUE(done) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
          std::vector<std::byte> actual(output.size());
          ASSERT_TRUE(output.view().read(actual).ok());
          EXPECT_EQ(actual, expected);
        }
      }
    }
  }
  ASSERT_TRUE(context.close().ok());
}
} // namespace
