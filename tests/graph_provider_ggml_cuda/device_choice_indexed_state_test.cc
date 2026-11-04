#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include "hyle/compute/model_ops.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/slots.h"
#include "../../src/graph/slots_region.h"
#include "../compute_models/gdn_snapshot_fixture.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;
namespace m = hyle::mem;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::move(*value);
}
void check(hyle::Status status) { if (!status.ok()) throw std::runtime_error(std::string(status.message())); }
constexpr std::size_t kSlots = 12, kStride = 4096, kSnapshots = 2;
constexpr std::array<std::uint32_t, 8> kSources{1, 3, 0, 2, 5, 7, 8, 9};
constexpr std::array<std::uint32_t, 2> kDestinations{4, 6};
struct Math final {
  bool convolution;
  c::TensorDesc item, output;
  std::vector<c::TensorDesc> inputs;
  std::vector<std::vector<float>> values;
  std::vector<std::vector<float>> initial, expected_output, expected_state;
};
Math math(c::Context &context, m::MemoryResourceRef resource, bool convolution, std::size_t choices) {
  Math result{convolution};
  result.initial.resize(choices); result.expected_output.resize(choices); result.expected_state.resize(choices);
  if (!convolution) {
    gdn_snapshot_fixture::Fixture fixture(9, 1, 4, 2, 2, 16);
    result.item = dense(c::DataType::kFloat32, {2, 16, 16});
    result.output = fixture.inputs[2];
    result.inputs.assign(fixture.inputs.begin(), fixture.inputs.begin() + 5);
    result.values.assign(fixture.data.begin(), fixture.data.begin() + 5);
    for (std::size_t choice = 0; choice < choices; ++choice) {
      result.initial[choice] = fixture.data[5];
      for (auto &x : result.initial[choice]) x = x / (choice + 1.F) + choice / 16.F;
    }
    for (std::size_t choice = 0; choice < choices; ++choice) {
      fixture.data[5] = result.initial[choice];
      std::array<m::Buffer, 6> storage;
      std::array<c::ConstTensorView, 6> inputs;
      for (std::size_t p = 0; p < 6; ++p) {
        storage[p] = upload<float>(resource, fixture.data[p]);
        inputs[p] = take(c::ConstTensorView::create(fixture.inputs[p], storage[p].view()));
      }
      const auto descriptions = fixture.outputs(kSnapshots);
      std::array out{output_buffer(resource, descriptions[0]), output_buffer(resource, descriptions[1])};
      const std::array outputs{take(c::TensorView::create(descriptions[0], out[0].view())),
          take(c::TensorView::create(descriptions[1], out[1].view()))};
      auto prepared = take(c::prepare(context, c::Operation(c::GatedDeltaNet{.snapshot_count = kSnapshots}), inputs, outputs));
      auto workspace = ggml_cuda_test::workspace_buffer(prepared, resource);
      auto invocation = take(prepared.create_invocation(workspace.view()));
      auto queue = take(context.create_queue());
      auto done = take(invocation.submit(queue, inputs, outputs)); check(done.wait());
      result.expected_output[choice] = half_test::read<float>(out[0]);
      result.expected_state[choice] = half_test::read<float>(out[1]);
    }
  } else {
    constexpr std::size_t channels = 128, history = 3, tokens = 4;
    result.item = dense(c::DataType::kFloat32, {history, channels});
    result.output = dense(c::DataType::kFloat32, {1, tokens, channels});
    result.inputs = {result.output, dense(c::DataType::kFloat32, {channels, history + 1})};
    result.values.resize(2); result.values[0].resize(tokens * channels); result.values[1].resize(channels * (history + 1));
    for (std::size_t p = 0; p < 2; ++p) for (std::size_t j = 0; j < result.values[p].size(); ++j)
      result.values[p][j] = static_cast<float>(static_cast<int>((j + p * 7) % 31) - 15) / 32.F;
    for (std::size_t choice = 0; choice < choices; ++choice) {
      auto &initial = result.initial[choice]; initial.resize(history * channels);
      for (std::size_t j = 0; j < initial.size(); ++j)
        initial[j] = static_cast<float>(static_cast<int>((j + choice * 17) % 41) - 20) / 64.F;
      auto logical = initial; logical.insert(logical.end(), result.values[0].begin(), result.values[0].end());
      auto samples = upload<float>(resource, logical), weights = upload<float>(resource, result.values[1]);
      auto output = output_buffer(resource, result.output);
      const std::array inputs{take(c::ConstTensorView::create(dense(c::DataType::kFloat32, {1, history + tokens, channels}), samples.view())),
          take(c::ConstTensorView::create(result.inputs[1], weights.view()))};
      check(ggml_cuda_tensor_test::execute(context, c::model_ops::causal_conv1d(), inputs,
          take(c::TensorView::create(result.output, output.view()))));
      result.expected_output[choice] = half_test::read<float>(output);
      for (std::size_t prefix = 0; prefix < kSnapshots; ++prefix) {
        const auto start = (tokens - kSnapshots + 1 + prefix) * channels;
        result.expected_state[choice].insert(result.expected_state[choice].end(),
            logical.begin() + start, logical.begin() + start + history * channels);
      }
    }
  }
  return result;
}
enum class Fault { kNone, kLeftId, kRightId, kPriorError, kWithdraw };

void exercise(bool convolution, std::int32_t selector, Fault fault = Fault::kNone, bool nested = true,
    std::size_t choices = 2, std::size_t corrupt_index = 0) {
  SCOPED_TRACE(convolution ? "conv" : "GDN");
  SCOPED_TRACE(selector);
  auto context = take(g::cuda::create_context());
  auto device = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  auto staging = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
  SCOPED_TRACE(choices);
  if (!nested) choices = 1;
  const auto reference = math(context, device.ref(), convolution, choices);
  const auto elements = take(reference.item.element_count());
  const auto bytes = take(reference.item.required_bytes());
  const auto schema = take(s::Schema::create(reference.item, kSlots, kStride));
  std::vector<std::byte> before(kSlots * kStride, std::byte{0xa5});
  for (std::size_t p = 0; p < choices; ++p) {
    const auto raw = std::as_bytes(std::span(reference.initial[p]));
    std::copy(raw.begin(), raw.end(), before.begin() + kSources[p] * kStride);
  }
  auto backing = take(m::Buffer::create(device.ref(), before.size(), 256)); check(backing.view().write(before));
  // Diagnostic aliases only, never bound as pool authority. Mutation below
  // occurs after prepare, before submit, while the exact writers are exclusive.
  const auto diagnostic_write = backing.view(); const m::ConstBufferView diagnostic = backing.view();
  auto pool = take(c::LeasePool::adopt(std::move(backing), {.max_blocks = 2 * kSlots}));
  const auto layout = take(c::LeaseSlotLayout::create(pool, reference.item, kSlots, kStride));
  std::array<c::WriteLease, kSlots> blocks;
  std::array<c::ReadLease, kSlots> padding, unused;
  for (std::size_t slot = 0; slot < kSlots; ++slot) {
    blocks[slot] = take(pool.allocate(bytes, 256)); EXPECT_EQ(take(layout.locate(blocks[slot])).index, slot);
    auto gap = take(pool.allocate(kStride - bytes, 256)); padding[slot] = take(gap.publish());
  }
  std::vector<c::ReadLease> sources(choices);
  std::array<c::WriteLease, 2> destinations;
  for (std::size_t p = 0; p < choices; ++p) sources[p] = take(blocks[kSources[p]].publish());
  for (std::size_t p = 0; p < 2; ++p) destinations[p] = std::move(blocks[kDestinations[p]]);
  for (std::size_t slot = 0; slot < kSlots; ++slot) if (blocks[slot].valid()) unused[slot] = take(blocks[slot].publish());

  auto builder = take(g::GraphBuilder::create());
  const auto capacity = take(builder.input(take(schema.capacity_description())));
  const auto ids = take(builder.input(dense(c::DataType::kInt32, {static_cast<std::int64_t>(choices)})));
  const auto dst = take(builder.input(dense(c::DataType::kInt32, {2})));
  const auto control = nested ? take(builder.input(dense(c::DataType::kInt32, {1}))) : g::ValueId{};
  g::ValueId error_table, error_ids, error_output;
  g::NodeId error_node;
  if (fault == Fault::kPriorError) {
    error_table = take(builder.input(dense(c::DataType::kFloat32, {1, 1})));
    error_ids = take(builder.input(dense(c::DataType::kInt32, {1})));
    error_node = take(builder.add_node(c::Operation(c::Gather{.axis = 0,
        .bounds = c::IndexBoundsPolicy::kChecked}), std::array{error_table, error_ids}));
    error_output = take(builder.result(error_node));
    check(builder.publish(error_output));
  }
  const auto first_node = take(builder.add_node(c::Operation(c::Gather{.axis = 0,
      .bounds = c::IndexBoundsPolicy::kChecked}), std::array{capacity, ids}));
  const auto first = take(builder.result(first_node));
  if (fault == Fault::kPriorError) check(builder.depends_on(first_node, error_node));
  auto gathered = nested ? take(builder.add(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}, first, control)) : first;
  const auto reshape = [&](g::ValueId value, std::initializer_list<std::int64_t> dimensions) {
    return take(builder.add(take(c::View::reshape(take(c::TensorShape::create(dimensions)))), value));
  };
  std::vector<g::ValueId> arguments;
  for (const auto &description : reference.inputs) arguments.push_back(take(builder.input(description)));
  g::ValueId output, states;
  if (!convolution) {
    gathered = reshape(gathered, {1, 2, 16, 16});
    arguments.push_back(gathered);
    const auto node = take(builder.add_node(c::Operation(c::GatedDeltaNet{.snapshot_count = kSnapshots}), arguments));
    output = take(builder.result(node, 0)); states = reshape(take(builder.result(node, 1)), {2, static_cast<std::int64_t>(elements)});
  } else {
    gathered = reshape(gathered, {1, 3, 128});
    const auto samples = take(builder.add(c::Concat{.axis = 1}, gathered, arguments[0]));
    output = take(builder.add(c::model_ops::causal_conv1d(), samples, arguments[1]));
    std::vector<g::ValueId> windows;
    for (std::int64_t row = 3; row <= 4; ++row)
      windows.push_back(reshape(take(builder.add(take(c::View::slice(1, row, 3)), samples)), {1, 1, 3, 128}));
    states = reshape(take(builder.result(take(builder.add_node(c::Operation(c::Concat{.axis = 1}), windows)))), {2, 384});
  }
  const auto store = take(builder.add_node(c::Operation::custom(s::Store{schema}), std::array{states, dst}));
  check(builder.publish(output));
  const auto graph = take(std::move(builder).finish());
  auto compiled = take(g::cuda::compile(graph, context, device.ref()));
  const c::RoutePolicyId policy{0x48594c4547474355ULL, convolution ? 7u : 6u, 1};
  std::size_t indexed = 0;
  for (std::size_t step = 0; step < compiled.report.step_count(); ++step)
    if (take(compiled.report.step(step)).policy == policy) ++indexed;
  ASSERT_EQ(indexed, 1u) << "a payload Gather/Store fallback is not this test's indexed implementation";
  auto invocation = take(compiled.executable.create_invocation()); auto queue = take(context.create_queue());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  std::vector<m::Buffer> values(reference.values.size());
  for (std::size_t p = 0; p < values.size(); ++p) {
    values[p] = upload<float>(device.ref(), reference.values[p]);
    check(invocation.bind(arguments[p], take(c::ConstTensorView::create(reference.inputs[p], values[p].view()))));
  }
  auto selector_storage = upload<std::int32_t>(device.ref(), std::array{selector});
  if (nested) check(invocation.bind(control, take(c::ConstTensorView::create(dense(c::DataType::kInt32, {1}), selector_storage.view()))));
  auto output_storage = output_buffer(device.ref(), reference.output);
  check(invocation.bind(output, take(c::TensorView::create(reference.output, output_storage.view()))));
  m::Buffer bad_table, bad_ids, bad_output;
  if (fault == Fault::kPriorError) {
    bad_table = upload<float>(device.ref(), std::array{1.F}); bad_ids = upload<std::int32_t>(device.ref(), std::array<std::int32_t, 1>{1});
    bad_output = output_buffer(device.ref(), dense(c::DataType::kFloat32, {1, 1}));
    check(invocation.bind(error_table, take(c::ConstTensorView::create(dense(c::DataType::kFloat32, {1, 1}), bad_table.view()))));
    check(invocation.bind(error_ids, take(c::ConstTensorView::create(dense(c::DataType::kInt32, {1}), bad_ids.view()))));
    check(invocation.bind(error_output, take(c::TensorView::create(dense(c::DataType::kFloat32, {1, 1}), bad_output.view()))));
  }
  const std::array associations{s::SourceBinding{store, take(graph.producer(first))}};
  check(s::prepare(invocation, graph, queues, associations, {.host_workspace = staging.ref()}));
  const auto source_span = nested ? std::span<const c::ReadLease>(sources) : std::span<const c::ReadLease>(&sources[0], 1);
  auto transaction = take(s::SlotBindings::create(layout, source_span, destinations));
  check(s::bind(invocation, graph, store, std::move(transaction)));
  check(invocation.prepare_replay(queues, {.mode = g::ReplayMode::kRequireCapture}));
  for (const auto slot : kDestinations) {
    check(take(diagnostic_write.subview(slot * kStride, bytes)).fill(std::byte{0x3d}));
    std::fill_n(before.begin() + slot * kStride, bytes, std::byte{0x3d});
  }
  if (fault == Fault::kLeftId || fault == Fault::kRightId) {
    c::ConstTensorView metadata;
    check(g::detail::RegionAccess::visit(invocation, store,
        +[](void *opaque, const g::detail::RegionAccessView &view) noexcept -> hyle::Status {
          const auto *extension = view.prepared_vtable && view.prepared_vtable->query_extension
              ? static_cast<const s::detail::StoreRegionVTable *>(view.prepared_vtable->query_extension(view.prepared, s::detail::kStoreBindingExtensionId)) : nullptr;
          if (!extension || !extension->binding) return hyle::Status::Unimplemented("no original Store binding");
          auto *binding = extension->binding(view.invocation);
          if (!binding || !binding->valid()) return hyle::Status::FailedPrecondition("no original bound transaction");
          *static_cast<c::ConstTensorView *>(opaque) = binding->configuration().source_indices;
          return hyle::Status::Ok();
        }, &metadata));
    ASSERT_EQ(metadata.description(), dense(c::DataType::kInt32, {static_cast<std::int64_t>(choices)}));
    std::vector<std::int32_t> corrupted(kSources.begin(), kSources.begin() + choices);
    corrupted[fault == Fault::kLeftId ? corrupt_index : choices - 1] = static_cast<std::int32_t>(kSlots);
    const auto *pointer = take(c::cuda::pointer<std::int32_t>(metadata.storage()));
    // Deliberate post-prepare corruption of this exact private metadata only;
    // it is never a Source or a legitimate ordinary binding operation.
    struct Copy final { std::int32_t *destination; const std::int32_t *source; std::size_t count; } copy_args{
        const_cast<std::int32_t *>(pointer), corrupted.data(), choices};
    auto copy = take(c::cuda::submit(queue,
        +[](cudaStream_t stream, void *opaque) noexcept -> hyle::Status {
          const auto &args = *static_cast<const Copy *>(opaque);
          return c::cuda::status(cudaMemcpyAsync(args.destination, args.source,
              args.count * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
        }, &copy_args));
    check(copy.wait()); copy = {};
  }
  const bool invalid = selector < 0 || static_cast<std::size_t>(selector) >= choices ||
      fault == Fault::kLeftId || fault == Fault::kRightId || fault == Fault::kPriorError;
  if (fault == Fault::kWithdraw) {
    check(invocation.reset()); auto returned = take(s::take(invocation, store));
    EXPECT_EQ(returned.destination_count(), 2u);
    EXPECT_FALSE(invocation.submit(queues));
  } else {
    auto done = take(invocation.submit(queues)); const auto status = done.wait(); done = {};
    if (invalid) {
      EXPECT_TRUE(status.IsOutOfRange()) << status.message();
      EXPECT_FALSE(invocation.output(output));
      (void)take(s::take(invocation, store)); // Retired recovery, never publish.
    } else {
      check(status); const auto choice = nested ? static_cast<std::size_t>(selector) : 0u;
      const auto actual_output = half_test::read<float>(output_storage);
      ASSERT_EQ(actual_output.size(), reference.expected_output[choice].size());
      for (std::size_t element = 0; element < actual_output.size(); ++element)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_output[element]),
            std::bit_cast<std::uint32_t>(reference.expected_output[choice][element])) << element;
      auto returned = take(s::take(invocation, store));
      EXPECT_EQ(returned.source_count(), choices);
      for (std::size_t row = 0; row < 2; ++row) {
        auto ready = take(returned.destinations()[row].publish());
        const auto expected = std::as_bytes(std::span(reference.expected_state[choice]).subspan(row * elements, elements));
        std::vector<std::byte> actual(bytes); check(take(ready.view()).read(actual)); EXPECT_TRUE(std::ranges::equal(actual, expected));
        std::copy(expected.begin(), expected.end(), before.begin() + kDestinations[row] * kStride);
      }
    }
  }
  std::vector<std::byte> after(before.size()); check(diagnostic.read(after)); EXPECT_EQ(after, before);
}

TEST(DeviceChoiceIndexedState, BothCheckedSelectorsUseOriginalIndexedMathAndStore) {
  for (const bool convolution : {false, true}) for (const auto selector : {0, 1}) exercise(convolution, selector);
}
TEST(DeviceChoiceIndexedState, InvalidSelectorLeavesPreparedPoisonedStoreBytesUnchanged) {
  for (const bool convolution : {false, true}) for (const auto selector : {-1, 2}) exercise(convolution, selector);
}
TEST(DeviceChoiceIndexedState, BothSelectedAndUnselectedMetadataIdsAreChecked) {
  for (const bool convolution : {false, true}) for (const auto selector : {0, 1})
    for (const auto fault : {Fault::kLeftId, Fault::kRightId}) exercise(convolution, selector, fault);
}
TEST(DeviceChoiceIndexedState, EarlierGraphDeviceErrorSuppressesIndexedStore) {
  for (const bool convolution : {false, true}) exercise(convolution, 0, Fault::kPriorError);
}
TEST(DeviceChoiceIndexedState, UnsubmittedWithdrawalRetiresWithoutWritingOrPublishing) {
  for (const bool convolution : {false, true}) exercise(convolution, 0, Fault::kWithdraw);
}
TEST(DeviceChoiceIndexedState, OriginalOneSourceIndexedPathStillQualifies) {
  for (const bool convolution : {false, true}) exercise(convolution, 0, Fault::kNone, false);
}
TEST(DeviceChoiceIndexedState, MultipleCapacitiesUseAllCheckedSourcesAndOriginalCapturedMath) {
  for (const bool convolution : {false, true}) for (const auto choices : {1u, 4u, 8u})
    for (std::uint32_t selector = 0; selector < choices; ++selector)
      exercise(convolution, static_cast<std::int32_t>(selector), Fault::kNone, true, choices);
}
TEST(DeviceChoiceIndexedState, MultipleCapacityBoundsAndEveryUnselectedSourceAreChecked) {
  for (const bool convolution : {false, true}) for (const auto choices : {1u, 4u, 8u}) {
    for (const auto selector : {-1, static_cast<std::int32_t>(choices)})
      exercise(convolution, selector, Fault::kNone, true, choices);
    for (std::size_t bad = 0; bad < choices; ++bad)
      exercise(convolution, static_cast<std::int32_t>((bad + 1) % choices), Fault::kLeftId, true, choices, bad);
    exercise(convolution, 0, Fault::kPriorError, true, choices);
    exercise(convolution, 0, Fault::kWithdraw, true, choices);
  }
}
} // namespace
