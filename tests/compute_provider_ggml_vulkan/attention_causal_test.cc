#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "ggml-backend.h"
#include "metadata_arena.h"
#include "native_device.h"
#include "region.h"

namespace {
namespace c = hyle::compute;
namespace d = c::ggml_vulkan::detail;
using Half = hyle::simd::float16_t;

struct AttentionPorts final {
  d::NativeDevice device;
  d::MetadataArena arena;
  std::vector<std::shared_ptr<ggml_backend_buffer>> owners;
  std::vector<ggml_tensor *> tensors;
  std::vector<d::RegionBinding> bindings;

  AttentionPorts(d::NativeDevice owner,
                 std::span<const c::TensorDesc> descriptions)
      : device(std::move(owner)),
        arena(d::MetadataArena::create(descriptions.size() *
                                           ggml_tensor_overhead(),
                                       hyle::mem::system_allocator())
                  .value()) {
    auto active = device.activate();
    for (const auto &description : descriptions) {
      const auto bytes = description.required_bytes().value();
      auto *allocation = device.allocate_buffer(bytes).value();
      owners.emplace_back(allocation, [owner = device](auto *buffer) {
        auto scope = owner.activate();
        ggml_backend_buffer_free(buffer);
      });
      auto *tensor = ggml_new_tensor_1d(
          arena.get(),
          description.data_type() == c::DataType::kFloat32 ? GGML_TYPE_F32
                                                           : GGML_TYPE_F16,
          std::int64_t(bytes / c::data_type_size(description.data_type())));
      if (ggml_backend_tensor_alloc(allocation, tensor,
                                    ggml_backend_buffer_get_base(allocation)) !=
          GGML_STATUS_SUCCESS)
        throw std::bad_alloc();
      tensors.push_back(tensor);
      bindings.push_back(
          {device.buffer_handle(allocation), 0, bytes, owners.back()});
    }
  }

  void write(std::size_t port, const c::TensorDesc &description,
             std::span<const double> logical) {
    std::vector<std::byte> bytes(description.required_bytes().value(),
                                 std::byte{0x7f});
    for (std::size_t i = 0; i < logical.size(); ++i) {
      auto index = i;
      std::size_t offset = 0;
      for (std::size_t axis = description.rank(); axis-- > 0;) {
        const auto extent = std::size_t(description.dimensions()[axis]);
        offset +=
            index % extent * std::size_t(description.byte_strides()[axis]);
        index /= extent;
      }
      if (description.data_type() == c::DataType::kFloat32) {
        const auto value = static_cast<float>(logical[i]);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
      } else {
        const auto value = half_test::encode<Half>(logical[i]);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
      }
    }
    auto active = device.activate();
    ggml_backend_tensor_set(tensors[port], bytes.data(), 0, bytes.size());
  }

  std::vector<double> read(const c::TensorDesc &description) {
    auto active = device.activate();
    std::vector<std::byte> bytes(bindings.back().capacity);
    ggml_backend_tensor_get(tensors.back(), bytes.data(), 0, bytes.size());
    std::vector<double> result(description.element_count().value());
    for (std::size_t i = 0; i < result.size(); ++i) {
      if (description.data_type() == c::DataType::kFloat32) {
        float value;
        std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
        result[i] = value;
      } else {
        Half value;
        std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
        result[i] = half_test::decode(value);
      }
    }
    return result;
  }
};

std::vector<double> pattern(const c::TensorDesc &description, int seed) {
  std::vector<double> values(description.element_count().value());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = double(int((i * 7 + seed * 13) % 23) - 11) / 32.;
  return values;
}

TEST(GgmlVulkanCausalAttention,
     OrdinaryGqaBatchAndScalesMatchIndependentOracle) {
  auto created = d::NativeDevice::create();
  ASSERT_TRUE(created) << created.error().message();
  auto device = *created;
  struct Case {
    c::DataType type;
    std::int64_t batch, queries, keys, heads, kv_heads, depth;
    float scale;
  };
  for (const auto test :
       {Case{c::DataType::kFloat32, 2, 17, 31, 8, 2, 64, .125F},
        Case{c::DataType::kFloat16, 2, 5, 11, 4, 1, 128, -.125F},
        Case{c::DataType::kFloat32, 1, 3, 7, 2, 2, 256, 0.F},
        Case{c::DataType::kFloat16, 1, 1, 31, 8, 2, 64, .125F},
        Case{c::DataType::kFloat32, 2, 5, 11, 4, 1, 128, -.125F},
        Case{c::DataType::kFloat16, 1, 3, 7, 2, 2, 256, 0.F}}) {
    SCOPED_TRACE(testing::Message()
                 << int(test.type) << '/' << test.batch << '/' << test.queries
                 << '/' << test.keys << '/' << test.depth << '/' << test.scale);
    const auto query =
        c::TensorDesc::dense(test.type, std::array{test.batch, test.queries,
                                                   test.heads, test.depth})
            .value();
    const auto scalar = std::int64_t(c::data_type_size(test.type));
    // Padded batch pitch makes a mistaken dense-cache or batch0 address
    // visible.
    const auto cache =
        c::TensorDesc::strided_bytes(
            test.type,
            std::array{test.batch, test.keys, test.kv_heads, test.depth},
            std::array{(test.keys + 5) * test.kv_heads * test.depth * scalar,
                       test.kv_heads * test.depth * scalar, test.depth * scalar,
                       scalar})
            .value();
    const c::Operation operation(c::ScaledDotProductAttention{
        .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
        .scale = test.scale,
        .causal = true});
    d::RegionRecipe recipe{.values = {query, cache, cache, query},
                           .nodes = {{operation, {0, 1, 2}, {3}}},
                           .inputs = {0, 1, 2},
                           .outputs = {3}};
    auto region = d::Region::create(device, recipe);
    ASSERT_TRUE(region) << region.error().message();
    // Ordinary fixed-shape attention must not be mistaken for batch request0.
    ASSERT_TRUE(region->validate_batch_layout({}, {}).ok());
    ASSERT_TRUE(region->update_batch_layout({}, {}).ok());
    const std::array descriptions{query, cache, cache, query};
    AttentionPorts first(device, descriptions), second(device, descriptions);
    std::vector<double> retained;
    for (int replay = 0; replay < 2; ++replay) {
      auto q = pattern(query, 1 + replay * 3),
           k = pattern(cache, 2 + replay * 3),
           v = pattern(cache, 3 + replay * 3);
      auto &ports = replay ? second : first;
      ports.write(0, query, q);
      ports.write(1, cache, k);
      ports.write(2, cache, v);
      ASSERT_TRUE(region->bind(ports.bindings).ok());
      ASSERT_TRUE(region->submit().status.ok());
      ASSERT_TRUE(region->wait().status.ok());
      const auto actual = ports.read(query);
      for (std::int64_t b = 0; b < test.batch; ++b)
        for (std::int64_t row = 0; row < test.queries; ++row)
          for (std::int64_t head = 0; head < test.heads; ++head) {
            const auto kv_head = head / (test.heads / test.kv_heads);
            const auto end = test.keys - test.queries + row + 1;
            std::vector<double> logits(end);
            for (std::int64_t key = 0; key < end; ++key) {
              double dot = 0;
              for (std::int64_t depth = 0; depth < test.depth; ++depth)
                dot += q[((b * test.queries + row) * test.heads + head) *
                             test.depth +
                         depth] *
                       k[((b * test.keys + key) * test.kv_heads + kv_head) *
                             test.depth +
                         depth];
              logits[key] = dot * double(test.scale);
            }
            const auto maximum =
                *std::max_element(logits.begin(), logits.end());
            double sum = 0;
            for (auto &value : logits) {
              value = std::exp(value - maximum);
              sum += value;
            }
            for (std::int64_t depth = 0; depth < test.depth; ++depth) {
              double expected = 0;
              for (std::int64_t key = 0; key < end; ++key)
                expected +=
                    logits[key] / sum *
                    v[((b * test.keys + key) * test.kv_heads + kv_head) *
                          test.depth +
                      depth];
              expected =
                  test.type == c::DataType::kFloat32
                      ? double(float(expected))
                      : half_test::decode(half_test::encode<Half>(expected));
              const auto at =
                  ((b * test.queries + row) * test.heads + head) * test.depth +
                  depth;
              const auto tolerance = test.type == c::DataType::kFloat32
                                         ? 6e-6 * std::abs(expected) + 2e-6
                                         : .001 * std::abs(expected) + 2e-7;
              EXPECT_NEAR(actual[at], expected, tolerance);
            }
          }
      if (replay == 0)
        retained = actual;
      else
        EXPECT_EQ(first.read(query), retained);
    }
  }
}

TEST(GgmlVulkanCausalAttention, ColdGallocrStorageHasNoQuadraticMask) {
  auto device = d::NativeDevice::create().value();
  const auto query =
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array<std::int64_t, 4>{1, 1025, 8, 64})
          .value();
  const auto cache =
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array<std::int64_t, 4>{1, 1041, 2, 64})
          .value();
  d::RegionRecipe recipe{
      .values = {query, cache, cache, query},
      .nodes = {{c::Operation(c::ScaledDotProductAttention{
                     .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                     .scale = .125F,
                     .causal = true}),
                 {0, 1, 2},
                 {3}}},
      .inputs = {0, 1, 2},
      .outputs = {3}};
  auto region = d::Region::create(device, recipe);
  ASSERT_TRUE(region) << region.error().message();
  // This is native gallocr backing capacity, not total/peak VRAM. NativeWork's
  // internal flash split-K buffers and caller ports are separate owners.
  std::cout << "causal_attention_gallocr_bytes=" << region->storage_bytes()
            << " caller_output_bytes=" << query.required_bytes().value()
            << '\n';
  EXPECT_LE(region->storage_bytes(), query.required_bytes().value());
  const std::array descriptions{query, cache, cache, query};
  AttentionPorts ports(device, descriptions);
  std::size_t caller_requested = 0, caller_native_capacity = 0;
  for (std::size_t i = 0; i < descriptions.size(); ++i) {
    caller_requested += descriptions[i].required_bytes().value();
    caller_native_capacity +=
        ggml_backend_buffer_get_size(ports.owners[i].get());
  }
  std::cout << "causal_attention_caller_requested_bytes=" << caller_requested
            << " caller_native_buffer_capacity_bytes=" << caller_native_capacity
            << '\n';
  ports.write(0, query, std::vector<double>(query.element_count().value(), 0.));
  ports.write(1, cache, std::vector<double>(cache.element_count().value(), 0.));
  ports.write(2, cache,
              std::vector<double>(cache.element_count().value(), .25));
  ASSERT_TRUE(region->bind(ports.bindings).ok());
  ASSERT_TRUE(region->submit().status.ok());
  ASSERT_TRUE(region->wait().status.ok());
  for (const auto value : ports.read(query))
    EXPECT_NEAR(value, .25, 4e-6);
}

TEST(GgmlVulkanCausalAttention, ExistingLayoutAndDtypeAdmissionRemainExplicit) {
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto query =
        c::TensorDesc::dense(type, std::array<std::int64_t, 4>{1, 3, 4, 64})
            .value();
    const auto cache =
        c::TensorDesc::dense(type, std::array<std::int64_t, 4>{1, 7, 2, 64})
            .value();
    for (auto layout : {c::AttentionLayout::kBatchSequenceHeadDepth,
                        c::AttentionLayout::kBatchHeadSequenceDepth}) {
      const c::Operation operation(
          c::ScaledDotProductAttention{.layout = layout, .causal = true});
      EXPECT_EQ(d::region_supports(operation, std::array{query, cache, cache},
                                   std::array{query}),
                type != c::DataType::kBFloat16 &&
                    layout == c::AttentionLayout::kBatchSequenceHeadDepth);
    }
  }
}
} // namespace
