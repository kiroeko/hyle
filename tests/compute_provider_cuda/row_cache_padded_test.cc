#include <gtest/gtest.h>
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/executable.h"
#include "../compute_models/row_cache_fixture.h"
#include "hyle/compute/batch_layout.h"
#include "../../src/compute_batch/row_cache_map.h"
#include "../compute_models/row_cache_paged_indexer_fixture.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using Encoding = c::RowCacheEncoding;
struct Tensor final {
  m::Buffer owner;
  c::TensorView view;
  Tensor(c::TensorDesc desc, m::MemoryResourceRef resource,
         std::size_t offset = 0)
      : owner(m::Buffer::create(
                  resource,
                  (std::max)(std::size_t{1}, *desc.required_bytes() + offset),
                  256)
                  .value()),
        view(c::TensorView::create(
                 desc,
                 owner.view().subview(offset, *desc.required_bytes()).value())
                 .value()) {}
  template <class T> void write(const std::vector<T> &v) {
    ASSERT_TRUE(view.storage().write(std::as_bytes(std::span(v))).ok());
  }
  template <class T> std::vector<T> read() {
    std::vector<T> result(view.storage().size() / sizeof(T));
    EXPECT_TRUE(
        view.storage().read(std::as_writable_bytes(std::span(result))).ok());
    return result;
  }
};
c::TensorDesc indices_description(std::int64_t selected) {
  return c::TensorDesc::dense(c::DataType::kInt32,
                              std::array<std::int64_t, 2>{2, selected})
      .value();
}
std::vector<std::uint16_t> expected(const row_cache_test::Reference &ref,
                                    Encoding encoding,
                                    const std::vector<std::int32_t> &ids) {
  std::vector<std::uint16_t> result;
  for (auto id : ids) {
    if (id == -1)
      result.insert(result.end(), encoding.depth(), std::uint16_t{0});
    else
      result.insert(result.end(),
                    ref.decoded.begin() + std::size_t(id) * encoding.depth(),
                    ref.decoded.begin() +
                        std::size_t(id + 1) * encoding.depth());
  }
  return result;
}

TEST(CudaPaddedRowCache, DefaultDirectPadsOnlyMinusOneAndProtectsAllOutput) {
  auto context = c::cuda::create_context().value();
  auto memory =
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
          .value();
  for (auto encoding :
       {Encoding::float8_block64_tail64(), Encoding::float4_block32()}) {
    SCOPED_TRACE(encoding.depth());
    const auto bank_desc =
        c::row_cache_carrier_description(encoding, 5).value();
    const auto id_desc = indices_description(4);
    const auto op = c::model_ops::row_cache_gather_decode_padded(encoding);
    const auto out_desc =
        c::infer_single_output(op, std::array{bank_desc, id_desc}).value();
    Tensor bank(bank_desc, memory.ref(), 1), ids(id_desc, memory.ref());
    std::array outputs{Tensor(out_desc, memory.ref()),
                       Tensor(out_desc, memory.ref())};
    auto prepared = c::prepare(context, op, std::array{bank_desc, id_desc},
                               std::array{out_desc}, memory.ref())
                        .value();
    EXPECT_EQ(prepared.provider().id, c::cuda_native::kProviderId);
    EXPECT_EQ(prepared.workspace_requirement().size, 0U);
    std::array slots{prepared.create_invocation().value(),
                     prepared.create_invocation().value()};
    std::array queues{context.create_queue().value(),
                      context.create_queue().value()};
    std::vector<std::uint16_t> old;
    for (int phase = 0; phase < 2; ++phase) {
      const auto ref = row_cache_test::reference(
          encoding, row_cache_test::input_values(encoding, phase));
      bank.write(ref.bytes);
      std::vector<std::int32_t> selected{4, -1, 1, -1, 0, 3, -1, 4};
      if (phase)
        std::reverse(selected.begin(), selected.end());
      ids.write(selected);
      auto done = slots[phase].submit(
          queues[phase], std::array{bank.view.as_const(), ids.view.as_const()},
          std::array{outputs[phase].view});
      ASSERT_TRUE(done);
      ASSERT_TRUE(done->wait().ok());
      EXPECT_EQ(outputs[phase].read<std::uint16_t>(),
                expected(ref, encoding, selected));
      if (!phase)
        old = outputs[0].read<std::uint16_t>();
      else
        EXPECT_EQ(outputs[0].read<std::uint16_t>(), old);
    }
    const auto sentinel = outputs[1].read<std::uint16_t>();
    for (auto bad : {std::int32_t{-2}, std::int32_t{5}}) {
      ids.write(std::vector<std::int32_t>{0, -1, 1, 2, 3, 4, -1, bad});
      auto failed = prepared.create_invocation().value();
      auto queue = context.create_queue().value();
      auto done = failed.submit(
          queue, std::array{bank.view.as_const(), ids.view.as_const()},
          std::array{outputs[1].view});
      ASSERT_TRUE(done);
      EXPECT_TRUE(done->wait().IsOutOfRange());
      EXPECT_EQ(outputs[1].read<std::uint16_t>(), sentinel);
      EXPECT_EQ(outputs[0].read<std::uint16_t>(), old);
    }
    // The original checked operation still rejects -1 rather than padding it.
    ids.write(std::vector<std::int32_t>(8, -1));
    auto original =
        c::prepare(context, c::model_ops::row_cache_gather_decode(encoding),
                   std::array{bank_desc, id_desc}, std::array{out_desc},
                   memory.ref())
            .value();
    auto failed = original.create_invocation().value();
    auto queue = context.create_queue().value();
    auto done = failed.submit(
        queue, std::array{bank.view.as_const(), ids.view.as_const()},
        std::array{outputs[1].view});
    ASSERT_TRUE(done);
    EXPECT_TRUE(done->wait().IsOutOfRange());
    EXPECT_EQ(outputs[1].read<std::uint16_t>(), sentinel);
  }
}

TEST(CudaPaddedRowCache,
     DefaultGraphPublishesPaddedRowsAcrossSlotsAndRebinding) {
  auto context = c::cuda::create_context().value();
  auto memory =
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
          .value();
  for (auto encoding :
       {Encoding::float8_block64_tail64(), Encoding::float4_block32()}) {
    SCOPED_TRACE(encoding.depth());
    const auto bank_desc =
        c::row_cache_carrier_description(encoding, 5).value();
    const auto id_desc = indices_description(4);
    auto builder = g::GraphBuilder::create().value();
    auto b = builder.input(bank_desc).value(),
         i = builder.input(id_desc).value();
    auto decoded =
        builder
            .add(c::model_ops::row_cache_gather_decode_padded(encoding), b, i)
            .value();
    auto widened = builder.add(c::Cast{c::DataType::kFloat32}, decoded).value();
    ASSERT_TRUE(builder.publish(decoded).ok());
    ASSERT_TRUE(builder.publish(widened).ok());
    const auto graph = std::move(builder).finish().value();
    const std::array storage{g::StorageDomain{memory.ref()}};
    const std::array targets{
        g::ExecutionTarget{hyle::graph::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
    auto compiled = g::compile(graph, {storage, targets, {}}).value();
    const auto out_desc = graph.value_description(decoded).value();
    const auto wide_desc = graph.value_description(widened).value();
    Tensor bank(bank_desc, memory.ref(), 1), ids(id_desc, memory.ref());
    std::array output{Tensor(out_desc, memory.ref()),
                      Tensor(out_desc, memory.ref())};
    std::array wide{Tensor(wide_desc, memory.ref()),
                    Tensor(wide_desc, memory.ref())};
    std::array slots{compiled.executable.create_invocation().value(),
                     compiled.executable.create_invocation().value()};
    auto queue = context.create_queue().value();
    std::vector<std::uint16_t> old;
    for (int phase = 0; phase < 2; ++phase) {
      auto ref = row_cache_test::reference(
          encoding, row_cache_test::input_values(encoding, phase));
      bank.write(ref.bytes);
      std::vector<std::int32_t> selected{-1, 4, 0, -1, 2, 2, -1, 1};
      if (phase)
        std::reverse(selected.begin(), selected.end());
      ids.write(selected);
      auto &slot = slots[phase];
      ASSERT_TRUE(gb::bind(slot, b, bank.view.as_const()).ok());
      ASSERT_TRUE(gb::bind(slot, i, ids.view.as_const()).ok());
      ASSERT_TRUE(gb::bind(slot, decoded, output[phase].view).ok());
      ASSERT_TRUE(gb::bind(slot, widened, wide[phase].view).ok());
      auto done = slot.submit(queue);
      ASSERT_TRUE(done);
      ASSERT_TRUE(done->wait().ok());
      const auto values = expected(ref, encoding, selected);
      EXPECT_EQ(output[phase].read<std::uint16_t>(), values);
      const auto actual = wide[phase].read<float>();
      ASSERT_EQ(actual.size(), values.size());
      for (std::size_t j = 0; j < values.size(); ++j) {
        if (std::isnan(row_cache_test::decode(values[j])))
          EXPECT_TRUE(std::isnan(actual[j]));
        else
          EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[j]),
                    std::uint32_t(values[j]) << 16);
      }
      if (!phase)
        old = output[0].read<std::uint16_t>();
      else
        EXPECT_EQ(output[0].read<std::uint16_t>(), old);
    }
  }
}

TEST(CudaPaddedRowCache, EmptyHistoryAllPaddingAndEmptyIdsNeedNoBankRead) {
  auto context = c::cuda::create_context().value();
  auto memory =
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
          .value();
  for (auto encoding :
       {Encoding::float8_block64_tail64(), Encoding::float4_block32()})
    for (const auto geometry :
         {std::array<std::int64_t, 2>{0, 4}, std::array<std::int64_t, 2>{1, 4},
          std::array<std::int64_t, 2>{0, 0}}) {
      SCOPED_TRACE(::testing::Message()
                   << encoding.depth() << " rows=" << geometry[0]
                   << " selected=" << geometry[1]);
      const auto bank_desc =
          c::row_cache_carrier_description(encoding, geometry[0]).value();
      const auto id_desc = indices_description(geometry[1]);
      const auto op = c::model_ops::row_cache_gather_decode_padded(encoding);
      const auto out_desc =
          c::infer_single_output(op, std::array{bank_desc, id_desc}).value();
      Tensor bank(bank_desc, memory.ref()), ids(id_desc, memory.ref()),
          output(out_desc, memory.ref());
      bank.write(std::vector<std::uint8_t>(
          std::size_t(geometry[0]) * encoding.row_bytes(), 255));
      ids.write(std::vector<std::int32_t>(std::size_t(2 * geometry[1]), -1));
      auto prepared = c::prepare(context, op, std::array{bank_desc, id_desc},
                                 std::array{out_desc}, memory.ref())
                          .value();
      auto slot = prepared.create_invocation().value();
      auto queue = context.create_queue().value();
      auto done = slot.submit(
          queue, std::array{bank.view.as_const(), ids.view.as_const()},
          std::array{output.view});
      ASSERT_TRUE(done);
      ASSERT_TRUE(done->wait().ok());
      EXPECT_EQ(output.read<std::uint16_t>(),
                std::vector<std::uint16_t>(
                    std::size_t(2 * geometry[1]) * encoding.depth(), 0));
    }
}

TEST(CudaPaddedRowCache, OwnedMetadataMapsVisibleRangesAndRejectsOtherErrors) {
  auto context = c::cuda::create_context().value();
  auto memory =
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
          .value();
  const auto encoding = Encoding::float4_block32();
  auto schema = c::RowCacheBatchSchema::create({.encoding = encoding,
                                                .max_requests = 3,
                                                .max_query_rows = 4,
                                                .max_kv_rows = 8,
                                                .max_pages = 5,
                                                .max_page_refs = 4,
                                                .page_size = 2})
                    .value();
  const auto offsets = schema.metadata_offsets();
  const c::detail::RowCachePaddedMap operation{
      {offsets.query_indptr, offsets.kv_indptr, offsets.page_indices,
       offsets.last_page_lengths, offsets.bytes, 2, 5},
      4};
  const auto id_desc = c::TensorDesc::dense(c::DataType::kInt32,
                                            std::array<std::int64_t, 2>{4, 4})
                           .value();
  const auto range_desc =
      c::TensorDesc::dense(c::DataType::kInt32,
                           std::array<std::int64_t, 2>{4, 2})
          .value();
  const auto metadata_desc =
      c::TensorDesc::dense(
          c::DataType::kUInt8,
          std::array<std::int64_t, 1>{std::int64_t(offsets.bytes)})
          .value();
  const auto mask_desc =
      c::TensorDesc::dense(c::DataType::kFloat32, id_desc.dimensions()).value();
  const auto bank_desc = c::row_cache_carrier_description(encoding, 10).value();
  const auto decode = c::model_ops::row_cache_gather_decode_padded(encoding);
  const auto out_desc =
      c::infer_single_output(decode, std::array{bank_desc, id_desc}).value();
  Tensor ids(id_desc, memory.ref()), ranges(range_desc, memory.ref()),
      metadata(metadata_desc, memory.ref());
  Tensor mapped(id_desc, memory.ref()), mask(mask_desc, memory.ref()),
      bank(bank_desc, memory.ref(), 1), output(out_desc, memory.ref());
  auto mapper = c::prepare(context, c::Operation::custom(operation),
                           std::array{id_desc, range_desc, metadata_desc},
                           std::array{id_desc, mask_desc}, memory.ref())
                    .value();
  auto decoder = c::prepare(context, decode, std::array{bank_desc, id_desc},
                            std::array{out_desc}, memory.ref())
                     .value();
  auto mapping = mapper.create_invocation().value();
  auto decoding = decoder.create_invocation().value();
  auto ref = row_cache_test::reference(
      encoding, row_cache_test::input_values(encoding, 0));
  const auto next = row_cache_test::reference(
      encoding, row_cache_test::input_values(encoding, 1));
  ref.bytes.insert(ref.bytes.end(), next.bytes.begin(), next.bytes.end());
  ref.decoded.insert(ref.decoded.end(), next.decoded.begin(),
                     next.decoded.end());
  bank.write(ref.bytes);
  const std::array<std::int32_t, 4> q{0, 2, 2, 4}, kv{0, 2, 3, 4},
      pages{4, 1, 3, 0};
  const std::array<std::int32_t, 3> tails{1, 2, 2};
  auto layout = schema.version({q, kv, pages, tails}).value();
  auto owned = layout.metadata().value();
  std::vector<std::uint8_t> bytes(std::size_t(offsets.bytes));
  ASSERT_TRUE(
      owned.view()->read(std::as_writable_bytes(std::span(bytes))).ok());
  metadata.write(bytes);
  const std::vector<std::int32_t> selected{0, 1, 2,  -1, 2,  2, 0, 3,
                                           1, 0, -1, 3,  -1, 0, 1, 3};
  const std::vector<std::int32_t> visible{0, 1, 1, 3, 0, 2, 0, 0};
  ids.write(selected);
  ranges.write(visible);
  const std::vector<std::int32_t> physical{8, -1, -1, -1, 2,  2,  -1, -1,
                                           1, 0,  -1, -1, -1, -1, -1, -1};
  auto queue = context.create_queue().value();
  auto first =
      mapping.submit(queue,
                     std::array{ids.view.as_const(), ranges.view.as_const(),
                                metadata.view.as_const()},
                     std::array{mapped.view, mask.view});
  ASSERT_TRUE(first);
  auto last = decoding.submit(
      queue, std::array{bank.view.as_const(), mapped.view.as_const()},
      std::array{output.view});
  ASSERT_TRUE(last);
  ASSERT_TRUE(last->wait().ok());
  ASSERT_TRUE(first->wait().ok());
  EXPECT_EQ(mapped.read<std::int32_t>(), physical);
  const auto actual_mask = mask.read<float>();
  for (std::size_t i = 0; i < physical.size(); ++i)
    EXPECT_EQ(actual_mask[i],
              physical[i] < 0 ? -std::numeric_limits<float>::infinity() : 0.F);
  const auto sentinel = expected(ref, encoding, physical);
  EXPECT_EQ(output.read<std::uint16_t>(), sentinel);
  for (bool invalid_range : {false, true}) {
    auto bad_ids = selected;
    auto bad_ranges = visible;
    if (invalid_range)
      bad_ranges.back() = 3;
    else
      bad_ids.back() = 4;
    ids.write(bad_ids);
    ranges.write(bad_ranges);
    auto map_slot = mapper.create_invocation().value();
    auto decode_slot = decoder.create_invocation().value();
    auto failed_queue = context.create_queue().value();
    auto mapped_done =
        map_slot.submit(failed_queue,
                        std::array{ids.view.as_const(), ranges.view.as_const(),
                                   metadata.view.as_const()},
                        std::array{mapped.view, mask.view});
    ASSERT_TRUE(mapped_done);
    auto done = decode_slot.submit(
        failed_queue, std::array{bank.view.as_const(), mapped.view.as_const()},
        std::array{output.view});
    ASSERT_TRUE(done);
    EXPECT_TRUE(done->wait().IsOutOfRange());
    mapped_done->wait().IgnoreError();
    EXPECT_EQ(mapped.read<std::int32_t>().back(), -2);
    EXPECT_EQ(output.read<std::uint16_t>(), sentinel);
  }
}

TEST(CudaPaddedRowCache,
     TypedGraphFirstPrefillAndEmptyHistoryKeepExactFootprint) {
  namespace f = row_cache_paged_indexer_test;
  namespace gb = g::batch;
  auto context = c::cuda::create_context().value();
  auto memory =
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
          .value();
  for (auto encoding :
       {Encoding::float8_block64_tail64(), Encoding::float4_block32()}) {
    auto schema = c::RowCacheBatchSchema::create(f::capacity(encoding)).value();
    const auto id_desc = c::TensorDesc::dense(c::DataType::kInt32,
                                              std::array<std::int64_t, 2>{3, 3})
                             .value();
    const auto range_desc =
        c::TensorDesc::dense(c::DataType::kInt32,
                             std::array<std::int64_t, 2>{3, 2})
            .value();
    auto builder = g::GraphBuilder::create().value();
    auto ids = builder.input(id_desc).value(),
         ranges = builder.input(range_desc).value();
    const auto macro =
        gb::add_row_cache_gather_padded(builder, schema, ids, ranges, 4)
            .value();
    ASSERT_TRUE(builder.publish(macro.output()).ok());
    ASSERT_TRUE(builder.publish(macro.mask()).ok());
    auto graph = std::move(builder).finish().value();
    const std::array storage{g::StorageDomain{memory.ref()}};
    const std::array targets{
        g::ExecutionTarget{hyle::graph::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
    auto compiled = g::compile(graph, {storage, targets, {}}).value();
    auto queue = context.create_queue().value();
    const std::array lanes{g::QueueBinding{
        .lane = g::ExecutionLaneId{0}, .queue = &queue}};
    auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}})
                    .value();
    std::vector<std::uint16_t> source(std::size_t(10) * encoding.depth());
    for (std::size_t i = 0; i < source.size(); ++i)
      source[i] = row_cache_test::bfloat(double(int((i * 7) % 31) - 15) / 8);
    auto reference = row_cache_test::reference(
        encoding, std::span(source).first(5 * encoding.depth()));
    const auto tail = row_cache_test::reference(
        encoding, std::span(source).subspan(5 * encoding.depth()));
    reference.bytes.insert(reference.bytes.end(), tail.bytes.begin(),
                           tail.bytes.end());
    reference.decoded.insert(reference.decoded.end(), tail.decoded.begin(),
                             tail.decoded.end());
    reference.bytes.resize(8 * encoding.row_bytes());
    reference.decoded.resize(8 * encoding.depth());
    auto pool = c::LeasePool::create(memory.ref(), 1u << 20).value();
    for (bool empty : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << encoding.depth() << " empty=" << empty);
      auto arena = c::PagedArena::create(
                       schema, memory.ref(),
                       {.max_backing_bytes = 8 * encoding.row_bytes() + 1})
                       .value();
      std::vector<c::ReadLease> ready;
      if (!empty)
        ready = f::upload(arena, reference.bytes, encoding.row_bytes());
      const auto layout = f::layout(schema, empty);
      ASSERT_TRUE(gb::prepare(slot, graph, lanes, {}, std::array{gb::RowCacheBinding{macro.reader(), layout}})
                      .ok());
      ASSERT_TRUE(
          gb::bind(slot, macro.bank_input(), f::frame(arena, layout, ready, empty))
              .ok());
      const std::vector<std::int32_t> selected{-1, 0, 1, 0, 1, 2, -1, 0, 1};
      const std::vector<std::int32_t> visible =
          empty ? std::vector<std::int32_t>(6)
                : std::vector<std::int32_t>{0, 0, 0, 1, 0, 1};
      ASSERT_TRUE(slot.bind(ids, id_desc, row_cache_paged_test::load(
                                pool, std::as_bytes(std::span(selected))))
                      .ok());
      ASSERT_TRUE(slot.bind(ranges, range_desc, row_cache_paged_test::load(
                                pool, std::as_bytes(std::span(visible))))
                      .ok());
      const auto out_desc = graph.value_description(macro.output()).value(),
                 mask_desc = graph.value_description(macro.mask()).value();
      ASSERT_TRUE(slot.bind(macro.output(), out_desc, pool.allocate(*out_desc.required_bytes()).value())
                      .ok());
      ASSERT_TRUE(slot.bind(macro.mask(), mask_desc, pool.allocate(*mask_desc.required_bytes()).value())
                      .ok());
      // The empty layout admits no page ReadLease at all. Hold every actual
      // page writer while the operation runs to verify that boundary.
      std::vector<c::PagedPageWriter> writers;
      if (empty)
        for (unsigned page = 0; page < 4; ++page)
          writers.push_back(arena.acquire(page).value());
      ASSERT_TRUE(slot.prepare_replay(lanes).ok());auto done = slot.submit(lanes);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto decoded =
          f::read<std::uint16_t>(slot.result(macro.output()).value());
      const auto mask = f::read<float>(slot.result(macro.mask()).value());
      for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t k = 0; k < 3; ++k) {
          const auto id = selected[row * 3 + k];
          const bool valid =
              id >= visible[row * 2] && id < visible[row * 2 + 1];
          EXPECT_EQ(mask[row * 3 + k],
                    valid ? 0.F : -std::numeric_limits<float>::infinity());
          for (std::size_t d = 0; d < encoding.depth(); ++d) {
            const auto wanted =
                valid ? reference.decoded[std::size_t(f::physical(row, id)) *
                                              encoding.depth() +
                                          d]
                      : 0;
            EXPECT_EQ(decoded[(row * 3 + k) * encoding.depth() + d], wanted);
          }
        }
    }
    ASSERT_TRUE(slot.drain().ok());
  }
}
} // namespace
