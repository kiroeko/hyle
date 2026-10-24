#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"
#include "../compute_models/row_cache_fixture.h"
#if defined(HYLE_ROW_CACHE_DEFAULT_ROUTE)
#include "hyle/compute/cuda.h"
#endif

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using Encoding = c::RowCacheEncoding;

using row_cache_test::input_values;
using row_cache_test::reference;
std::vector<std::uint16_t> selected_values(Encoding encoding,
    const row_cache_test::Reference &reference, std::span<const std::int32_t> ids) {
  std::vector<std::uint16_t> output;
  for (auto id : ids) {
    const auto first = reference.decoded.begin() + std::size_t(id) * encoding.depth();
    output.insert(output.end(), first, first + encoding.depth());
  }
  return output;
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> dims) {
  return c::TensorDesc::dense(type, {dims.begin(), dims.size()}).value();
}
struct Tensor {
  m::Buffer storage;
  c::TensorView view;
  Tensor(c::TensorDesc desc, m::MemoryResourceRef resource, std::size_t offset = 0)
      : storage(m::Buffer::create(resource, (std::max)(std::size_t{1}, *desc.required_bytes() + offset), 256).value()),
        view(c::TensorView::create(desc, storage.view().subview(offset, *desc.required_bytes()).value()).value()) {}
  template <class T> void write(const std::vector<T> &v) {
    ASSERT_TRUE(view.storage().write(std::as_bytes(std::span(v))).ok());
  }
  template <class T> std::vector<T> read() {
    std::vector<T> v(view.storage().size() / sizeof(T));
    EXPECT_TRUE(view.storage().read(std::as_writable_bytes(std::span(v))).ok());
    return v;
  }
};
c::Context context() {
  return c::Context::create(c::cuda::device(0).value(), c::cuda_native::provider()).value();
}

TEST(CudaRowCache, EncodesAndSelectsCanonicalRowsAcrossSlots) {
  auto owner = context();
  auto resource = c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kDevice).value();
  for (auto encoding : {Encoding::float8_block64_tail64(), Encoding::float4_block32()}) {
    SCOPED_TRACE(encoding.depth());
    const auto source = dense(c::DataType::kBFloat16, {5, encoding.depth()});
    const auto bank = c::row_cache_carrier_description(encoding, 5).value();
    const auto idesc = dense(c::DataType::kInt32, {2, 3});
    const auto out = dense(c::DataType::kBFloat16, {2, 3, encoding.depth()});
    Tensor input(source, resource.ref()), ids(idesc, resource.ref());
    std::array encoded{Tensor(bank, resource.ref(), 1), Tensor(bank, resource.ref(), 1)};
    std::array output{Tensor(out, resource.ref()), Tensor(out, resource.ref())};
    const auto encoder = c::prepare(owner, c::model_ops::row_cache_encode(encoding),
        std::array{source}, std::array{bank}, resource.ref()).value();
    const auto decoder = c::prepare(owner, c::model_ops::row_cache_gather_decode(encoding),
        std::array{bank, idesc}, std::array{out}, resource.ref()).value();
    EXPECT_EQ(encoder.provider().id, c::cuda_native::kProviderId);
    EXPECT_EQ(encoder.workspace_requirement().size, 0U);
    std::array encode{encoder.create_invocation().value(), encoder.create_invocation().value()};
    std::array decode{decoder.create_invocation().value(), decoder.create_invocation().value()};
    std::array queue{owner.create_queue().value(), owner.create_queue().value()};
    std::vector<std::uint16_t> old;
    for (int phase = 0; phase < 2; ++phase) {
      input.write(input_values(encoding, phase));
      std::vector<std::int32_t> selected{4, 1, 3, 0, 3, 2};
      if (phase) std::reverse(selected.begin(), selected.end());
      ids.write(selected);
      const auto expected = reference(encoding, input_values(encoding, phase));
      ASSERT_TRUE(encode[phase].submit(queue[phase], std::array{input.view.as_const()},
          std::array{encoded[phase].view})->wait().ok());
      EXPECT_EQ(encoded[phase].read<std::uint8_t>(), expected.bytes);
      ASSERT_TRUE(decode[phase].submit(queue[phase],
          std::array{encoded[phase].view.as_const(), ids.view.as_const()},
          std::array{output[phase].view})->wait().ok());
      EXPECT_EQ(output[phase].read<std::uint16_t>(), selected_values(encoding, expected, selected));
      if (phase) EXPECT_EQ(output[0].read<std::uint16_t>(), old);
      else old = output[0].read<std::uint16_t>();
    }
    // Last ID invalid: no earlier, valid output row may be published either.
    ids.write(std::vector<std::int32_t>{4, 1, 3, 0, 3, 5});
    const auto sentinel = output[1].read<std::uint16_t>();
    auto failed = decoder.create_invocation().value();
    auto queue_failed = owner.create_queue().value();
    auto done = failed.submit(queue_failed,
        std::array{encoded[1].view.as_const(), ids.view.as_const()}, std::array{output[1].view});
    ASSERT_TRUE(done); EXPECT_TRUE(done->wait().IsOutOfRange());
    EXPECT_EQ(output[1].read<std::uint16_t>(), sentinel);
    EXPECT_EQ(output[0].read<std::uint16_t>(), old);
    // Retry on a fresh completion sequence, without mutating the old bank.
    ids.write(std::vector<std::int32_t>{0, 0, 1, 2, 3, 4});
    auto retry = decoder.create_invocation().value(); auto queue_retry = owner.create_queue().value();
    ASSERT_TRUE(retry.submit(queue_retry, std::array{encoded[1].view.as_const(), ids.view.as_const()},
        std::array{output[1].view})->wait().ok());
    auto nonfinite = input_values(encoding, 0); nonfinite.back() = 0x7f80;
    input.write(nonfinite);
    const auto old_bank = encoded[0].read<std::uint8_t>();
    auto invalid = encoder.create_invocation().value(); auto invalid_queue = owner.create_queue().value();
    auto rejected = invalid.submit(invalid_queue, std::array{input.view.as_const()}, std::array{encoded[0].view});
    ASSERT_TRUE(rejected); EXPECT_TRUE(rejected->wait().IsInvalidArgument());
    EXPECT_EQ(encoded[0].read<std::uint8_t>(), old_bank);
  }
}

TEST(CudaRowCache, GraphPublishesEncodedAndSelectedRowsWithoutDecodingHistory) {
  auto owner = context();
  auto resource = c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kDevice).value();
  for (auto encoding : {Encoding::float8_block64_tail64(), Encoding::float4_block32()}) {
    auto builder = g::GraphBuilder::create().value();
    const auto input_desc = dense(c::DataType::kBFloat16, {5, encoding.depth()});
    const auto ids_desc = dense(c::DataType::kInt32, {3});
    const auto output_desc = dense(c::DataType::kBFloat16, {3, encoding.depth()});
    const auto x = builder.input(input_desc).value(), ids = builder.input(ids_desc).value();
    const auto bank = builder.add(c::model_ops::row_cache_encode(encoding), x).value();
    const auto y = builder.add(c::model_ops::row_cache_gather_decode(encoding), bank, ids).value();
    ASSERT_TRUE(builder.publish(bank).ok()); ASSERT_TRUE(builder.publish(y).ok());
    const auto logical = std::move(builder).finish().value();
    const std::array domains{g::StorageDomain{.resource = resource.ref()}};
    const std::array targets{g::ExecutionTarget{.source = hyle::graph::ExecutionSource::acquire(owner).value(), .default_storage = g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
    auto compiled = g::compile(logical, {.storage_domains = domains, .targets = targets, .lanes = lanes});
    ASSERT_TRUE(compiled) << compiled.error().message();
    for (std::size_t i = 0; i < compiled->report.step_count(); ++i)
      EXPECT_EQ(compiled->report.step(i)->provider.id, c::cuda_native::kProviderId);
    auto invocation = compiled->executable.create_invocation().value();
    auto queue = owner.create_queue().value();
    Tensor input(input_desc, resource.ref()), indices(ids_desc, resource.ref());
    Tensor bytes(c::row_cache_carrier_description(encoding, 5).value(), resource.ref());
    Tensor output(output_desc, resource.ref());
    ASSERT_TRUE(invocation.bind(x, input.view.as_const()).ok());
    ASSERT_TRUE(invocation.bind(ids, indices.view.as_const()).ok());
    ASSERT_TRUE(invocation.bind(bank, bytes.view).ok()); ASSERT_TRUE(invocation.bind(y, output.view).ok());
    for (int phase = 0; phase < 2; ++phase) {
      input.write(input_values(encoding, phase));
      const std::vector<std::int32_t> selected{4, phase, 4}; indices.write(selected);
      ASSERT_TRUE(invocation.submit(queue)->wait().ok());
      const auto expected = reference(encoding, input_values(encoding, phase));
      EXPECT_EQ(bytes.read<std::uint8_t>(), expected.bytes);
      EXPECT_EQ(output.read<std::uint16_t>(), selected_values(encoding, expected, selected));
    }
  }
}

TEST(CudaRowCache, ScalarEmptyUncheckedAndIeeePayloadsKeepTheirContract) {
  auto owner = context();
  auto resource = c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kDevice).value();
  for (auto encoding : {Encoding::float8_block64_tail64(), Encoding::float4_block32()}) {
    const bool four = encoding.depth() == 128;
    const auto bank_desc = c::row_cache_carrier_description(encoding, 1).value();
    const auto id_desc = dense(c::DataType::kInt32, {});
    const auto out_desc = dense(c::DataType::kBFloat16, {encoding.depth()});
    Tensor bank(bank_desc, resource.ref(), 1), ids(id_desc, resource.ref()), output(out_desc, resource.ref());
    auto reference = row_cache_test::reference(encoding, input_values(encoding, 0));
    reference.bytes.resize(encoding.row_bytes());
    if (four) {
      reference.bytes[0] = 0x91; // +/-0.5 times UE8M0=0 (2^-127).
      reference.bytes[1] = 0x80; // +/-0.
      reference.bytes[64] = 0;
      reference.bytes[65] = 255; // IEEE UE8M0 NaN, not an index error.
    } else {
      reference.bytes[0] = 0x7f; reference.bytes[1] = 0xff;
      reference.bytes[576] = 127;
      reference.bytes[577] = 255;
    }
    bank.write(reference.bytes); ids.write(std::vector<std::int32_t>{0});
    auto prepared = c::prepare(owner, c::model_ops::row_cache_gather_decode(
        encoding, c::IndexBoundsPolicy::kUnchecked), std::array{bank_desc, id_desc},
        std::array{out_desc}, resource.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    auto unchecked = std::move(*prepared);
    auto invocation = unchecked.create_invocation().value(); auto queue = owner.create_queue().value();
    ASSERT_TRUE(invocation.submit(queue, std::array{bank.view.as_const(), ids.view.as_const()},
        std::array{output.view})->wait().ok());
    const auto actual = output.read<std::uint16_t>();
    if (four) {
      EXPECT_EQ(actual[0], row_cache_test::bfloat(std::ldexp(.5, -127)));
      EXPECT_EQ(actual[1], row_cache_test::bfloat(std::ldexp(-.5, -127)));
      EXPECT_EQ(actual[2], 0U); EXPECT_EQ(actual[3], 0x8000U);
    } else {
      EXPECT_TRUE(std::isnan(row_cache_test::decode(actual[0])));
      EXPECT_TRUE(std::isnan(row_cache_test::decode(actual[1])));
      for (unsigned d = 448; d < 512; ++d)
        EXPECT_EQ(actual[d], reference.decoded[d]);
    }
    EXPECT_TRUE(std::isnan(row_cache_test::decode(actual[four ? 32 : 64])));
    const auto checked = c::prepare(owner, c::model_ops::row_cache_gather_decode(encoding),
        std::array{bank_desc, id_desc}, std::array{out_desc}, resource.ref()).value();
    ids.write(std::vector<std::int32_t>{-1});
    auto invalid = checked.create_invocation().value(); auto invalid_queue = owner.create_queue().value();
    auto failed = invalid.submit(invalid_queue, std::array{bank.view.as_const(), ids.view.as_const()}, std::array{output.view});
    ASSERT_TRUE(failed); EXPECT_TRUE(failed->wait().IsOutOfRange());
    EXPECT_EQ(output.read<std::uint16_t>(), actual);
    const auto empty_bank = c::row_cache_carrier_description(encoding, 0).value();
    const auto empty_ids = dense(c::DataType::kInt32, {0});
    const auto empty_output = dense(c::DataType::kBFloat16, {0, encoding.depth()});
    Tensor none(empty_bank, resource.ref()), no_ids(empty_ids, resource.ref()), no_output(empty_output, resource.ref());
    auto empty_prepared = c::prepare(owner, c::model_ops::row_cache_gather_decode(encoding),
        std::array{empty_bank, empty_ids}, std::array{empty_output}, resource.ref());
    ASSERT_TRUE(empty_prepared) << empty_prepared.error().message();
    auto empty = std::move(*empty_prepared);
    auto empty_invocation = empty.create_invocation().value(); auto empty_queue = owner.create_queue().value();
    ASSERT_TRUE(empty_invocation.submit(empty_queue, std::array{none.view.as_const(), no_ids.view.as_const()},
        std::array{no_output.view})->wait().ok());
  }
}

#if defined(HYLE_ROW_CACHE_DEFAULT_ROUTE)
TEST(CudaRowCache, DefaultRosterAdmitsBothEncodings) {
  auto owner = c::cuda::create_context().value();
  auto resource = c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kDevice).value();
  for (auto encoding : {Encoding::float8_block64_tail64(), Encoding::float4_block32()}) {
    const auto source = dense(c::DataType::kBFloat16, {1, encoding.depth()});
    const auto bank = c::row_cache_carrier_description(encoding, 1).value();
    const auto ids = dense(c::DataType::kInt32, {});
    const auto out = dense(c::DataType::kBFloat16, {encoding.depth()});
    auto encode = c::prepare(owner, c::model_ops::row_cache_encode(encoding),
        std::array{source}, std::array{bank}, resource.ref());
    auto decode = c::prepare(owner, c::model_ops::row_cache_gather_decode(encoding),
        std::array{bank, ids}, std::array{out}, resource.ref());
    ASSERT_TRUE(encode) << encode.error().message(); ASSERT_TRUE(decode) << decode.error().message();
    EXPECT_EQ(encode->provider().id, c::cuda_native::kProviderId);
    EXPECT_EQ(decode->provider().id, c::cuda_native::kProviderId);
  }
}
#endif
} // namespace
