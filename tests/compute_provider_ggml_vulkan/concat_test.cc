#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "ggml-backend.h"
#include "metadata_arena.h"
#include "native_device.h"
#include "region.h"

namespace {
namespace c = hyle::compute;
namespace d = c::ggml_vulkan::detail;

struct Ports final {
  d::NativeDevice device;
  d::MetadataArena arena;
  std::vector<std::shared_ptr<ggml_backend_buffer>> buffers;
  std::vector<ggml_tensor *> tensors;
  std::vector<d::RegionBinding> bindings;

  Ports(d::NativeDevice owner, std::size_t count)
      : device(std::move(owner)),
        arena(d::MetadataArena::create(count * ggml_tensor_overhead(),
                                       hyle::mem::system_allocator())
                  .value()) {}

  void add(const c::TensorDesc &description) {
    auto active = device.activate();
    const auto bytes = description.required_bytes().value();
    auto *allocation = device.allocate_buffer(bytes).value();
    buffers.emplace_back(allocation, [owner = device](auto *buffer) {
      auto release_scope = owner.activate();
      ggml_backend_buffer_free(buffer);
    });
    const auto type =
        description.data_type() == c::DataType::kFloat32    ? GGML_TYPE_F32
        : description.data_type() == c::DataType::kFloat16  ? GGML_TYPE_F16
        : description.data_type() == c::DataType::kBFloat16 ? GGML_TYPE_BF16
                                                            : GGML_TYPE_I32;
    auto *tensor = ggml_new_tensor_1d(
        arena.get(), type, std::int64_t(description.element_count().value()));
    ASSERT_EQ(ggml_backend_tensor_alloc(
                  allocation, tensor, ggml_backend_buffer_get_base(allocation)),
              GGML_STATUS_SUCCESS);
    tensors.push_back(tensor);
    bindings.push_back(
        {device.buffer_handle(allocation), 0, bytes, buffers.back()});
  }

  void write(std::size_t port, std::span<const std::byte> bytes) {
    auto active = device.activate();
    ggml_backend_tensor_set(tensors[port], bytes.data(), 0, bytes.size());
  }

  std::vector<std::byte> read(std::size_t port) {
    auto active = device.activate();
    std::vector<std::byte> bytes(bindings[port].capacity);
    ggml_backend_tensor_get(tensors[port], bytes.data(), 0, bytes.size());
    return bytes;
  }
};

TEST(GgmlVulkanConcat, BalancedPortsPreserveNativeStorageBitsAndAxes) {
  auto device = d::NativeDevice::create().value();
  for (auto type : {c::DataType::kFloat32, c::DataType::kFloat16,
                    c::DataType::kBFloat16, c::DataType::kInt32}) {
    const auto scalar_bytes = c::data_type_size(type);
    for (int axis : {0, 1, 2}) {
      for (int count : {1, 3, 4, 7}) {
        SCOPED_TRACE(testing::Message()
                     << int(type) << '/' << axis << '/' << count);
        d::RegionRecipe recipe;
        std::vector<std::uint32_t> inputs;
        std::vector<std::size_t> widths;
        std::array<std::int64_t, 3> output_shape{2, 3, 5};
        output_shape[axis] = 0;
        std::size_t inner = 1, outer = 1;
        for (int i = 0; i < axis; ++i)
          outer *= output_shape[i];
        for (int i = axis + 1; i < 3; ++i)
          inner *= output_shape[i];
        Ports ports(device, count + 1);
        for (int port = 0; port < count; ++port) {
          auto shape = output_shape;
          shape[axis] = 1 + port % 3;
          recipe.values.push_back(c::TensorDesc::dense(type, shape).value());
          recipe.inputs.push_back(port);
          inputs.push_back(port);
          ports.add(recipe.values.back());
          widths.push_back(std::size_t(shape[axis]) * inner);
          output_shape[axis] += shape[axis];
        }
        recipe.values.push_back(
            c::TensorDesc::dense(type, output_shape).value());
        ports.add(recipe.values.back());
        recipe.nodes.push_back({c::Operation(c::Concat{.axis = axis - 3}),
                                inputs,
                                {std::uint32_t(count)}});
        recipe.outputs = {std::uint32_t(count)};
        auto region = d::Region::create(device, recipe);
        ASSERT_TRUE(region) << region.error().message();
        ASSERT_TRUE(region->bind(ports.bindings).ok());
        for (int replay = 0; replay < 2; ++replay) {
          std::vector<std::vector<std::byte>> logical;
          for (int port = 0; port < count; ++port) {
            logical.emplace_back(outer * widths[port] * scalar_bytes);
            for (std::size_t i = 0; i < logical.back().size(); ++i)
              logical.back()[i] =
                  std::byte((i * 37 + port * 61 + replay * 113) & 255);
            ports.write(port, logical.back());
          }
          std::vector<std::byte> expected;
          for (std::size_t row = 0; row < outer; ++row)
            for (int port = 0; port < count; ++port) {
              const auto size = widths[port] * scalar_bytes;
              expected.insert(expected.end(),
                              logical[port].begin() + row * size,
                              logical[port].begin() + (row + 1) * size);
            }
          ASSERT_TRUE(region->submit().status.ok());
          ASSERT_TRUE(region->wait().status.ok());
          EXPECT_EQ(ports.read(count), expected);
        }
      }
    }
  }
}

TEST(GgmlVulkanConcat, RepeatedPortsKeepPublishedIntermediateAndDownstream) {
  auto device = d::NativeDevice::create().value();
  const auto input = c::TensorDesc::dense(c::DataType::kFloat32,
                                          std::array<std::int64_t, 2>{2, 7})
                         .value();
  const auto output = c::TensorDesc::dense(c::DataType::kFloat32,
                                           std::array<std::int64_t, 2>{2, 490})
                          .value();
  d::RegionRecipe recipe{
      .values = {input, output, output},
      .nodes = {{c::Operation(c::Concat{.axis = 1}),
                 std::vector<std::uint32_t>(70, 0),
                 {1}},
                {c::Operation(c::Unary{c::UnaryKind::kNeg}), {1}, {2}}},
      .inputs = {0},
      .outputs = {1, 2}};
  auto region = d::Region::create(device, recipe);
  ASSERT_TRUE(region) << region.error().message();
  Ports ports(device, 3);
  for (const auto &description : recipe.values)
    ports.add(description);
  ASSERT_TRUE(region->bind(ports.bindings).ok());
  for (int replay = 0; replay < 2; ++replay) {
    std::array<float, 14> source;
    for (std::size_t i = 0; i < source.size(); ++i)
      source[i] = float(int(i) - 5 + replay * 17);
    ports.write(0, std::as_bytes(std::span(source)));
    ASSERT_TRUE(region->submit().status.ok());
    ASSERT_TRUE(region->wait().status.ok());
    const auto positive = ports.read(1), negative = ports.read(2);
    for (std::size_t i = 0; i < 980; ++i) {
      const auto reference = source[(i / 490) * 7 + i % 7];
      float actual, negated;
      std::memcpy(&actual, positive.data() + i * sizeof(float), sizeof(float));
      std::memcpy(&negated, negative.data() + i * sizeof(float), sizeof(float));
      EXPECT_EQ(actual, reference);
      EXPECT_EQ(negated, -reference);
    }
  }
}

TEST(GgmlVulkanConcat, RetainsDenseNonemptyAdmissionBoundary) {
  const std::array<std::int64_t, 2> shape{2, 3}, strides{24, 8};
  const auto dense = c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
  const auto strided =
      c::TensorDesc::strided_bytes(c::DataType::kFloat32, shape, strides)
          .value();
  const auto empty = c::TensorDesc::dense(c::DataType::kFloat32,
                                          std::array<std::int64_t, 2>{2, 0})
                         .value();
  const c::Operation operation(c::Concat{.axis = 1});
  EXPECT_FALSE(
      d::region_supports(operation, std::array{strided}, std::array{dense}));
  EXPECT_FALSE(d::region_supports(operation, std::array{dense, empty},
                                  std::array{dense}));
}
} // namespace
