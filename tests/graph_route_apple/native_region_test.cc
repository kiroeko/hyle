#include <gtest/gtest.h>

#include "hyle/graph/apple.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/detail/region_adapter.h"
#include "../../src/graph/execution_source_internal.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/mem/stats.h"

#include <array>
#include <bit>
#include <cstring>
#include <iostream>

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace m = hyle::mem;

const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
    std::array<std::int64_t, 1>{4}).value();

template<class T, std::size_t N>
c::ReadLease input(const a::MemoryResource &memory, const c::TensorDesc &desc,
                   const std::array<T, N> &values) {
  auto array = memory.allocate(desc).value();
  auto host = m::Buffer::create(m::system_resource(), sizeof(values), alignof(T)).value();
  EXPECT_TRUE(host.view().write(std::as_bytes(std::span(values))).ok());
  const auto view = c::ConstTensorView::create(desc, host.view()).value();
  EXPECT_TRUE(memory.copy_from_host(array, view).ok());
  return memory.import_readonly(array, desc).value();
}
template<std::size_t N>
void expect(c::ConstTensorView tensor, const std::array<float, N> &wanted) {
  std::array<float, N> actual{};
  ASSERT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(actual))).ok());
  for (std::size_t i = 0; i < N; ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(wanted[i])) << i;
}
hyle::Status submit(g::Invocation &invocation) {
  auto status = invocation.prepare_replay({});
  EXPECT_TRUE(status.ok()) << status.message();
  if (!status.ok()) return status;
  auto accepted = invocation.submit();
  if (!accepted) { ADD_FAILURE() << accepted.error().message(); return std::move(accepted).error(); }
  return accepted->wait();
}

struct ConstantHost final : m::MemoryResource {
  unsigned reads = 0;
  bool fail_read = false;
  std::string_view name() const noexcept override { return "Apple cold constant host"; }
  m::MemoryProperties properties() const noexcept override { return m::system_resource().properties(); }
  hyle::Result<m::ResourceAllocation> allocate(std::size_t bytes, std::size_t alignment) noexcept override {
    return m::system_resource().resource().allocate(bytes, alignment);
  }
  void deallocate(m::ResourceAllocation allocation) noexcept override {
    m::system_resource().resource().deallocate(allocation);
  }
  hyle::Status read(void *handle, std::size_t offset, std::span<std::byte> bytes) noexcept override {
    ++reads;
    if (fail_read) return hyle::Status::Unimplemented("injected Apple constant read failure");
    return m::system_resource().resource().read(handle, offset, bytes);
  }
  hyle::Status write(void *handle, std::size_t offset, std::span<const std::byte> bytes) noexcept override {
    return m::system_resource().resource().write(handle, offset, bytes);
  }
};

TEST(AppleNativeRegion, NativeOnlyHostConstantsShareAcrossExactRoutesAndOutliveSource) {
  auto context = a::Context::create().value();
  RecordProperty("device", context.environment()->device_name);
  auto memory = a::MemoryResource::create(context).value();
  const auto source = g::apple::execution_source(context).value();
  auto builder = g::GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 2}).value();
  const auto spec = g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(rows), g::Dimension::fixed(4)}).value();
  const auto x = builder.input(spec).value();
  const auto weight = builder.constant(description).value();
  const auto add = builder.add_node(c::Operation(c::Binary{c::BinaryKind::kAdd}),
      std::array{x, weight}, std::array{spec}).value();
  const auto sum = builder.result(add).value();
  const auto neg = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{sum}, std::array{spec}).value();
  const auto output = builder.result(neg).value();
  ASSERT_TRUE(builder.publish(output).ok());
  ASSERT_TRUE(builder.publish(weight).ok());
  auto graph = std::move(builder).finish().value();
  ConstantHost host;
  auto weights = m::Buffer::create(m::MemoryResourceRef(host), 16, 4).value();
  const std::array<float, 4> values{0.5f, -0.25f, 1.25f, 2.0f};
  ASSERT_TRUE(weights.view().write(std::as_bytes(std::span(values))).ok());
  const std::array constants{g::ConstantBinding{weight,
      c::ConstTensorView::create(description, weights.view()).value(), g::ConstantPolicy::kMaterialize}};
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}}; // No CPU Context.
  const std::array one{g::ShapeBinding{rows, 1}}, two{g::ShapeBinding{rows, 2}};
  const std::array variants{g::ExactShapeVariant{one}, g::ExactShapeVariant{two}};
  auto compiled = g::compile(graph, {domains, targets},
      {.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants, .constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(host.reads, 1U);
  EXPECT_EQ(compiled->report.total_compiled_storage().materialized_constant_bytes, 16U);
  ASSERT_TRUE(weights.reset().ok());
  auto first = compiled->executable.create_invocation({.shape = one, .resources = g::ResourceOptions{}}).value();
  auto second = compiled->executable.create_invocation({.shape = two, .resources = g::ResourceOptions{}}).value();
  auto desc_one = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 4}).value();
  auto desc_two = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{2, 4}).value();
  auto a_input = input(memory, desc_one, std::array<float, 4>{0.25f, 1.5f, -2.75f, 0.0f});
  auto b_input = input(memory, desc_two, std::array<float, 8>{-2.0f, 3.25f, 1.5f, -0.5f,
                                                           1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_TRUE(first.bind(x, desc_one, a_input).ok());
  ASSERT_TRUE(second.bind(x, desc_two, b_input).ok());
  for (int replay = 0; replay < 2; ++replay) {
    ASSERT_TRUE(submit(first).ok());
    ASSERT_TRUE(submit(second).ok());
    expect(first.output(output).value(), std::array<float, 4>{-0.75f, -1.25f, 1.5f, -2.0f});
    expect(second.output(output).value(), std::array<float, 8>{1.5f, -3.0f, -2.75f, -1.5f,
                                                              -1.5f, -1.75f, -4.25f, -6.0f});
    const auto left = first.output(weight).value(), right = second.output(weight).value();
    EXPECT_EQ(left.storage().resource(), right.storage().resource());
    EXPECT_EQ(left.storage().native_handle(), right.storage().native_handle());
    EXPECT_EQ(left.storage().offset(), right.storage().offset());
    expect(left, values);
  }
  EXPECT_EQ(host.reads, 1U);
}

TEST(AppleNativeRegion, ColdConstantBatchChecksReadonlyTargetsAndKeepsReadErrorsFatal) {
  auto context = a::Context::create().value();
  auto memory = a::MemoryResource::create(context).value();
  const auto source = g::apple::execution_source(context).value();
  ConstantHost host;
  const auto raw = c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 1>{3}).value();
  auto bytes = m::Buffer::create(m::MemoryResourceRef(host), 3, 1).value();
  const std::array payload{std::byte{1}, std::byte{127}, std::byte{255}};
  ASSERT_TRUE(bytes.view().write(payload).ok());
  auto destination = m::Buffer::create(memory.ref(), 3, 1).value();
  const auto raw_source = c::ConstTensorView::create(raw, bytes.view()).value();
  const auto raw_target = c::TensorView::create(raw, destination.view()).value();
  const std::array odd{g::detail::ConstantCopy{raw_target, raw_source}};
  EXPECT_TRUE(g::detail::ExecutionSourceAccess::can_copy_constant(source, odd[0]).value());
  EXPECT_EQ(g::detail::ExecutionSourceAccess::constant_copy_scratch_bytes(source, odd).value(), 0U);
  ASSERT_TRUE(g::detail::ExecutionSourceAccess::copy_constants(source, odd, m::system_allocator()).ok());
  std::array<std::byte, 3> observed{};
  ASSERT_TRUE(destination.view().read(observed).ok());
  EXPECT_EQ(observed, payload);
  EXPECT_EQ(host.reads, 1U);

  auto owner = input(memory, description, std::array<float, 4>{7, 8, 9, 10});
  const auto readonly = owner.view().value();
  // Imported storage cannot allocate or grant write authority, even though
  // its actual shared Metal backing is CPU-accessible and still alive.
  EXPECT_FALSE(m::Buffer::create(readonly.resource(), 16, 4));
  auto host_values = m::Buffer::create(m::MemoryResourceRef(host), 16, 4).value();
  const std::array<float, 4> sentinel{11, 12, 13, 14};
  ASSERT_TRUE(host_values.view().write(std::as_bytes(std::span(sentinel))).ok());
  const auto typed_host = c::ConstTensorView::create(description, host_values.view()).value();
  // Exercise the provider-author adoption seam adversarially: this second
  // wrapper retains no new native allocation/free claim (the import's
  // deallocator is readonly). Its real resource must still deny copy writes.
  auto attempted_writer = m::Buffer::adopt(readonly.resource(),
      {readonly.native_handle(), readonly.size(), 1}).value();
  const auto forbidden_target = c::TensorView::create(description, attempted_writer.view()).value();
  const g::detail::ConstantCopy forbidden{forbidden_target, typed_host};
  EXPECT_FALSE(g::detail::ExecutionSourceAccess::can_copy_constant(source, forbidden).value());
  ASSERT_TRUE(destination.view().fill(std::byte{45}).ok());
  const auto before_reject = host.reads;
  const std::array invalid_batch{odd[0], forbidden};
  EXPECT_TRUE(g::detail::ExecutionSourceAccess::copy_constants(
      source, invalid_batch, m::system_allocator()).IsUnimplemented());
  EXPECT_EQ(host.reads, before_reject);
  ASSERT_TRUE(destination.view().read(observed).ok());
  EXPECT_EQ(observed, (std::array{std::byte{45}, std::byte{45}, std::byte{45}}));

  // A readonly source is legal. A second actual writable MemoryResource on
  // this same Metal Device can receive its bytes without a CPU Context.
  auto other_memory = a::MemoryResource::create(context).value();
  auto replica = m::Buffer::create(other_memory.ref(), 16, 4).value();
  const std::array native_copy{g::detail::ConstantCopy{
      c::TensorView::create(description, replica.view()).value(), owner.tensor(description).value()}};
  ASSERT_TRUE(g::detail::ExecutionSourceAccess::can_copy_constant(source, native_copy[0]).value());
  ASSERT_TRUE(g::detail::ExecutionSourceAccess::copy_constants(source, native_copy, m::system_allocator()).ok());
  expect(native_copy[0].destination.as_const(), std::array<float, 4>{7, 8, 9, 10});
  auto builder = g::GraphBuilder::create().value();
  const auto weight = builder.constant(description).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, weight).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  const std::array constants{g::ConstantBinding{weight, typed_host, g::ConstantPolicy::kMaterialize}};
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  const auto reads = host.reads;
  host.fail_read = true;
  auto failed = g::compile(graph, {domains, targets}, {.constants = constants});
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsUnimplemented()) << failed.error().message();
  EXPECT_EQ(host.reads, reads + 1);
  expect(owner.tensor(description).value(), std::array<float, 4>{7, 8, 9, 10});
}

TEST(AppleNativeRegion, ReadonlyNativeOutputsRejectMutableMemAndBlitAccess) {
  auto context = a::Context::create().value();
  auto memory = a::MemoryResource::create(context).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, x).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{g::apple::execution_source(context).value(), g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {domains, targets}).value();
  auto invocation = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  auto owned_input = input(memory, description, std::array<float, 4>{1, -2, 3, -4});
  ASSERT_TRUE(invocation.bind(x, description, owned_input).ok());
  ASSERT_TRUE(submit(invocation).ok());
  auto result = invocation.result(output).value();
  auto retained = result.storage_lease().value();
  const auto readonly = retained.view().value();
  const std::array<float, 4> wanted{-1, 2, -3, 4};
  EXPECT_NE(readonly.resource(), memory.ref());
  expect(retained.tensor(description).value(), wanted);
  auto read_map = readonly.map_host().value();
  std::array<float, 4> mapped{};
  ASSERT_EQ(read_map.bytes().size(), sizeof(mapped));
  std::memcpy(mapped.data(), read_map.bytes().data(), sizeof(mapped));
  EXPECT_EQ(mapped, wanted);
  auto writable_read_map = read_map.writable_bytes();
  ASSERT_FALSE(writable_read_map);
  EXPECT_TRUE(writable_read_map.error().IsPermissionDenied());
  ASSERT_TRUE(read_map.close().ok());

  // Adversarially use the provider-author seam without replacing the actual
  // resource, native handle or origin. The retained output owns every byte.
  auto attempted_writer = m::Buffer::adopt(readonly.resource(),
      {readonly.native_handle(), readonly.offset() + readonly.size(), 1}).value();
  const auto forbidden = attempted_writer.view().subview(readonly.offset(), readonly.size()).value();
  EXPECT_EQ(forbidden.resource(), readonly.resource());
  EXPECT_EQ(forbidden.native_handle(), readonly.native_handle());
  EXPECT_EQ(forbidden.offset(), readonly.offset());
  for (const auto access : {m::HostAccess::kWrite, m::HostAccess::kReadWrite}) {
    auto mapping = forbidden.map_host(access);
    ASSERT_FALSE(mapping);
    EXPECT_TRUE(mapping.error().IsPermissionDenied());
  }
  const std::array<float, 4> replacement{11, 12, 13, 14};
  auto host = m::Buffer::create(m::system_resource(), sizeof(replacement), alignof(float)).value();
  ASSERT_TRUE(host.view().write(std::as_bytes(std::span(replacement))).ok());
  EXPECT_TRUE(forbidden.write(std::as_bytes(std::span(replacement))).IsPermissionDenied());
  EXPECT_TRUE(forbidden.fill(std::byte{53}).IsPermissionDenied());
  EXPECT_TRUE(forbidden.copy_from(host.view()).IsPermissionDenied());
  auto native = memory.binding(retained, description).value();
  EXPECT_TRUE(memory.copy_to_host(native,
      c::TensorView::create(description, forbidden).value()).IsPermissionDenied());

  // Ordinary native allocations keep writable host mappings and Mem copies.
  auto writable = m::Buffer::create(memory.ref(), sizeof(replacement), alignof(float)).value();
  ASSERT_TRUE(writable.view().fill(std::byte{0}).ok());
  auto write_map = writable.view().map_host(m::HostAccess::kWrite).value();
  auto write_bytes = write_map.writable_bytes().value();
  std::memcpy(write_bytes.data(), replacement.data(), sizeof(replacement));
  ASSERT_TRUE(write_map.flush().ok());
  ASSERT_TRUE(write_map.close().ok());
  ASSERT_TRUE(writable.view().write(std::as_bytes(std::span(replacement))).ok());
  ASSERT_TRUE(writable.view().copy_from(host.view()).ok());
  expect(c::ConstTensorView::create(description, writable.view()).value(), replacement);

  // Native blits bypass map_host, so both fixed and deferred destination
  // proofs must independently reject the same actual readonly resource.
  auto copy = a::detail::prepare_copy_recording(context, 1).value();
  const std::array<c::detail::CopyBackendRange, 1> denied{{{forbidden, writable.view()}}};
  EXPECT_TRUE(a::detail::bind_copy_recording(*copy, denied).IsPermissionDenied());
  EXPECT_FALSE(a::detail::submit_copy_recording(*copy));
  EXPECT_TRUE(a::detail::bind_result_copy_recording(*copy, denied).IsPermissionDenied());
  EXPECT_FALSE(a::detail::submit_copy_recording(*copy));
  EXPECT_EQ(a::detail::copy_recording_report(*copy).native_calls, 0U);
  expect(retained.tensor(description).value(), wanted);

  // Rejection did not poison the original copy carrier. A true retained
  // readonly output remains a legal source on that original Metal stream.
  const std::array<c::detail::CopyBackendRange, 1> allowed{{{writable.view(), readonly}}};
  ASSERT_TRUE(a::detail::bind_copy_recording(*copy, allowed).ok());
  auto completed = a::detail::submit_copy_recording(*copy).value();
  ASSERT_TRUE(completed.wait().ok());
  expect(c::ConstTensorView::create(description, writable.view()).value(), wanted);
  expect(retained.tensor(description).value(), wanted);
}

TEST(AppleNativeRegion, UnifiedInvocationKeepsNativeFusionOwnersAndBorrowedLimits) {
  auto context = a::Context::create().value();
  RecordProperty("device", context.environment()->device_name);
  RecordProperty("os_runtime", context.environment()->os_runtime);
  auto memory = a::MemoryResource::create(context).value();
  const auto source = g::apple::execution_source(context).value();
  auto duplicate = g::apple::execution_source(context).value();
  EXPECT_TRUE(source.same_owner(duplicate));
  EXPECT_FALSE(source.validate_resource(m::system_resource()).ok());
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto constant = builder.constant(description).value();
  const auto sum = builder.add(c::Binary{c::BinaryKind::kAdd}, x, constant).value();
  const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, sum).value();
  const auto shape = c::TensorShape::create(std::array<std::int64_t, 2>{2, 2}).value();
  const auto alias = builder.add(c::View::reshape(shape).value(), x).value();
  for (const auto value : {sum, negative, alias}) ASSERT_TRUE(builder.publish(value).ok());
  auto graph = std::move(builder).finish().value();
  auto weights = input(memory, description, std::array<float, 4>{0.5f, -0.25f, 1.25f, 2.0f});
  const std::array constants{g::ConstantBinding{constant, weights.tensor(description).value(), g::ConstantPolicy::kBorrow}};
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {domains, targets}, {.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.region_count(), 1U);
  EXPECT_EQ(compiled->executable.required_queue_count(), 0U);
  EXPECT_EQ(compiled->executable.workspace_bytes(), 0U);
  auto first = compiled->executable.create_invocation().value();
  auto second = compiled->executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  EXPECT_FALSE(first.result(negative)); // A cold deferred port has no result/array yet.
  auto raw = input(memory, description, std::array<float, 4>{0.25f, 1.5f, -2.75f, 0.0f});
  ASSERT_TRUE(first.bind(x, raw.tensor(description).value()).ok());
  ASSERT_TRUE(submit(first).ok());
  auto old = first.result(negative).value();
  expect(old.tensor().value(), std::array<float, 4>{-0.75f, -1.25f, 1.5f, -2.0f});
  auto raw_alias = first.result(alias).value();
  expect(raw_alias.tensor().value(), std::array<float, 4>{0.25f,1.5f,-2.75f,0.0f});
  auto denied = raw_alias.storage_lease();
  ASSERT_FALSE(denied); // A borrowed alias cannot create independent payload ownership.
  EXPECT_FALSE(first.bind(x, raw.tensor(description).value()).ok());
  c::ReadLease escaped;
  for (const auto values : {std::array<float, 4>{-2.0f, 3.25f, 1.5f, -0.5f},
                            std::array<float, 4>{1.25f, -4.0f, 0.25f, 8.0f}}) {
    auto owned = input(memory, description, values);
    ASSERT_TRUE(second.bind(x, description, owned).ok());
    owned.reset(); // Only the actual binding/MLX array owns this input now.
    ASSERT_TRUE(submit(second).ok());
    const std::array<float, 4> wanted{values[0]+0.5f, values[1]-0.25f, values[2]+1.25f, values[3]+2.0f};
    auto result = second.result(negative).value();
    expect(result.tensor().value(), std::array<float, 4>{-wanted[0], -wanted[1], -wanted[2], -wanted[3]});
    auto input_alias = second.result(alias).value();
    escaped = input_alias.storage_lease().value();
    expect(escaped.tensor(description).value(), values);
  }
  expect(old.tensor().value(), std::array<float, 4>{-0.75f, -1.25f, 1.5f, -2.0f});
  first = {}; second = {}; compiled->executable = {}; graph = {};
  ASSERT_TRUE(context.close().ok());
  expect(old.tensor().value(), std::array<float, 4>{-0.75f, -1.25f, 1.5f, -2.0f});
  expect(escaped.tensor(description).value(), std::array<float, 4>{1.25f, -4.0f, 0.25f, 8.0f});
  EXPECT_FALSE(g::apple::execution_source(context));
}

TEST(AppleNativeRegion, DeferredNativePortsResolveOnOriginalOwnerAndKeepFailure) {
  auto context = a::Context::create().value();
  auto memory = a::MemoryResource::create(context).value();
  const auto ids_description = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto ids = builder.input(ids_description).value();
  const auto selected_node = builder.add_node(c::Operation(c::Gather{.axis=0}), std::array{x, ids}).value();
  const auto selected = builder.result(selected_node).value();
  const auto negative_node = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{selected}).value();
  const auto negative = builder.result(negative_node).value();
  ASSERT_TRUE(builder.depends_on(negative_node, selected_node).ok());
  ASSERT_TRUE(builder.publish(selected).ok()); ASSERT_TRUE(builder.publish(negative).ok());
  auto graph = std::move(builder).finish().value();
  const auto source = g::apple::execution_source(context).value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  {
    // This dependency is legal inside one native region. Exercise deferred
    // Graph ports separately with cold fragment selection, not a product cut.
    auto fused = g::compile(graph, {domains, targets});
    ASSERT_TRUE(fused) << fused.error().message();
    EXPECT_EQ(fused->executable.region_count(), 1U);
  }
  auto capability = g::detail::ExecutionSourceAccess::region(source);
  ASSERT_TRUE(capability) << capability.error().message();
  auto native_context = g::detail::ExecutionSourceAccess::native_context(source);
  ASSERT_TRUE(native_context) << native_context.error().message();
  struct FragmentSelection final {
    const g::RegionProviderVTable *native = nullptr;
    void *context = nullptr;
    std::array<g::NodeId, 2> nodes{};
  } selection{static_cast<const g::RegionProviderVTable *>(capability->native_interface()),
              *native_context, {selected_node, negative_node}};
  ASSERT_NE(selection.native, nullptr);
  const g::RegionProviderVTable fragments{
      +[](void *opaque, const g::RegionDiscoveryRequest &request,
          m::AllocatorRef, g::RegionProposalWriter &writer) noexcept -> hyle::Status {
        const auto &chosen = *static_cast<const FragmentSelection *>(opaque);
        for (const auto node : chosen.nodes) {
          if (!request.allowed_nodes.empty() && !request.allowed_nodes[node.ordinal()])
            continue;
          HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({.nodes = std::span(&node, 1),
              .route = 1, .replay_class = c::ProviderReplayClass::kManaged,
              .implementation = c::ProviderImplementationId{1}, .preference = 0}));
          if (!more) break;
        }
        return hyle::Status::Ok();
      },
      +[](void *opaque, const g::RegionCompileRequest &request, m::AllocatorRef allocator,
          std::span<g::RegionInputRequirement> inputs,
          std::span<g::RegionOutputRequirement> outputs,
          g::RegionResourceProposal *proposal) noexcept -> hyle::Status {
        const auto &chosen = *static_cast<const FragmentSelection *>(opaque);
        return chosen.native->propose_resources(chosen.context, request, allocator,
                                                inputs, outputs, proposal);
      },
      +[](void *opaque, const g::RegionCompileRequest &request, m::AllocatorRef allocator,
          g::ProviderRegionPrepared *prepared) noexcept -> c::ProviderPrepareOutcome {
        const auto &chosen = *static_cast<const FragmentSelection *>(opaque);
        return chosen.native->compile(chosen.context, request, allocator, prepared);
      }};
  std::array adapters{g::detail::RegionAdapter{.source = std::move(*capability),
      .target = g::ExecutionTargetId{0}, .interface = &fragments, .context = &selection}};
  {
    auto invalid = adapters;
    invalid[0].provider_index = 0; // Native sources have no Compute provider slot.
    auto rejected = g::detail::compile_with_adapters(graph, {domains, targets}, {}, invalid);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument()) << rejected.error().message();
  }
  {
    auto invalid = adapters;
    invalid[0].target = g::ExecutionTargetId{1};
    auto rejected = g::detail::compile_with_adapters(graph, {domains, targets}, {}, invalid);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument()) << rejected.error().message();
  }
  {
    // A second real owner on the same device cannot borrow this target's
    // admission simply because its interface has the same implementation.
    auto foreign_context = a::Context::create().value();
    auto foreign_source = g::apple::execution_source(foreign_context).value();
    auto foreign = g::detail::ExecutionSourceAccess::region(foreign_source);
    ASSERT_TRUE(foreign) << foreign.error().message();
    auto invalid = adapters;
    invalid[0].source = std::move(*foreign);
    auto rejected = g::detail::compile_with_adapters(graph, {domains, targets}, {}, invalid);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument()) << rejected.error().message();
  }
  {
    // The cold override table is configuration, never the source capability.
    auto forged = g::detail::GraphRegionSource::native(source, &fragments);
    ASSERT_TRUE(forged) << forged.error().message();
    auto invalid = adapters;
    invalid[0].source = std::move(*forged);
    auto rejected = g::detail::compile_with_adapters(graph, {domains, targets}, {}, invalid);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument()) << rejected.error().message();
  }
  auto compiled = g::detail::compile_with_adapters(graph, {domains, targets}, {}, adapters);
  ASSERT_TRUE(compiled) << compiled.error().message();
  // These compile-local controls are not an executor or a replay owner.
  adapters = {}; selection = {};
  EXPECT_EQ(compiled->executable.region_count(), 2U);
  EXPECT_EQ(compiled->executable.workspace_bytes(), 0U);
  auto slot = compiled->executable.create_invocation({.resources=g::ResourceOptions{}}).value();
  c::ReadLease retained;
  for (int phase = 0; phase < 2; ++phase) {
    auto values = input(memory, description, phase == 0 ? std::array<float, 4>{0.25f,-1.5f,3.75f,2.0f}
                                                       : std::array<float, 4>{4.0f,0.5f,-2.25f,-0.0f});
    auto indices = input(memory, ids_description, std::array<std::int32_t, 2>{3, 1});
    ASSERT_TRUE(slot.bind(x, description, values).ok());
    ASSERT_TRUE(slot.bind(ids, ids_description, indices).ok());
    ASSERT_TRUE(submit(slot).ok());
    const auto wanted = phase == 0 ? std::array<float, 2>{-2.0f,1.5f} : std::array<float, 2>{0.0f,-0.5f};
    auto result = slot.result(negative).value(); expect(result.tensor().value(), wanted);
    if (phase == 0) retained = result.storage_lease().value();
  }
  expect(retained.tensor(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{2}).value()).value(), std::array<float, 2>{-2.0f,1.5f});
  auto failed = compiled->executable.create_invocation({.resources=g::ResourceOptions{}}).value();
  auto values = input(memory, description, std::array<float, 4>{1,2,3,4});
  auto bad = input(memory, ids_description, std::array<std::int32_t, 2>{0,4});
  ASSERT_TRUE(failed.bind(x, description, values).ok());
  ASSERT_TRUE(failed.bind(ids, ids_description, bad).ok());
  ASSERT_TRUE(failed.prepare_replay({}).ok());
  auto accepted = failed.submit();
  ASSERT_TRUE(accepted) << accepted.error().message();
  const auto terminal = accepted->wait();
  EXPECT_TRUE(terminal.IsOutOfRange()) << terminal.message();
  EXPECT_FALSE(failed.result(selected)); EXPECT_FALSE(failed.result(negative));
}

TEST(AppleNativeRegion, OwnedViewFeedbackKeepsOneExactLeaseOwner) {
  auto counted = m::StatsAllocator::create(m::system_allocator()).value();
  {
    auto context = a::Context::create().value();
    auto memory = a::MemoryResource::create(context).value();
    const auto source = g::apple::execution_source(context, {.allocator = counted.get()}).value();
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(description).value();
    const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, x).value();
    const auto shape = c::TensorShape::create(std::array<std::int64_t, 2>{2, 2}).value();
    const auto alias = builder.add(c::View::reshape(shape).value(), x).value();
    ASSERT_TRUE(builder.publish(negative).ok());
    ASSERT_TRUE(builder.publish(alias).ok());
    auto graph = std::move(builder).finish().value();
    const std::array domains{g::StorageDomain{memory.ref()}};
    const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
    auto compiled = g::compile(graph, {domains, targets});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->executable.region_count(), 1U);
    auto slot = compiled->executable.create_invocation({.resources = g::ResourceOptions{}}).value();

    // The readable input is a strict subrange of a larger, independently owned
    // allocation. A view result must preserve this exact permission, not export
    // the whole Metal allocation through another owner wrapper.
    const auto backing_description = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 1>{8}).value();
    auto backing = input(memory, backing_description,
        std::array<float, 8>{99.0f, 98.0f, 0.25f, -1.5f, 3.75f, -0.0f, 97.0f, 96.0f});
    auto current = backing.subview(2 * sizeof(float), 4 * sizeof(float)).value();
    backing.reset();
    const auto original = current.view().value();
    const auto generation = current.generation();
    const std::array<float, 4> wanted{0.25f, -1.5f, 3.75f, -0.0f};
    const auto feedback = [&]() -> hyle::Status {
      HYLE_RETURN_IF_ERROR(slot.bind(x, description, current));
      HYLE_RETURN_IF_ERROR(submit(slot));
      auto result = slot.result(alias);
      if (!result) return std::move(result).error();
      auto next = result->storage_lease();
      if (!next) return std::move(next).error();
      expect(next->tensor(description).value(), wanted);
      const auto physical = next->view().value();
      EXPECT_EQ(physical.native_handle(), original.native_handle());
      EXPECT_EQ(physical.offset(), original.offset());
      EXPECT_EQ(physical.size(), original.size());
      current = std::move(*next);
      return hyle::Status::Ok();
    };
    // Warm the normal slot/report capacities before comparing live metadata.
    ASSERT_TRUE(feedback().ok());
    ASSERT_TRUE(feedback().ok());
    const auto baseline = counted.snapshot();
    std::array<m::AllocStatsSnapshot, 4> samples{};
    for (std::size_t round = 0; round < 64; ++round) {
      const auto status = feedback();
      ASSERT_TRUE(status.ok()) << status.message();
      if (round == 7 || round == 15 || round == 31 || round == 63) {
        const auto index = round == 7 ? 0U : round == 15 ? 1U : round == 31 ? 2U : 3U;
        samples[index] = counted.snapshot();
      }
    }
    std::cout << "owned_alias_feedback baseline=" << baseline.live_count << '/' << baseline.live_bytes;
    for (const auto &sample : samples) std::cout << " live=" << sample.live_count << '/' << sample.live_bytes;
    std::cout << '\n';
    EXPECT_EQ(current.view()->resource(), original.resource());
    EXPECT_EQ(current.generation(), generation);
    EXPECT_LE(samples.back().live_count, baseline.live_count);
    EXPECT_LE(samples.back().live_bytes, baseline.live_bytes);

    // A genuine computed result still owns its new Array, independently of the
    // input/view lease. This must not be flattened merely by graph provenance.
    auto result = slot.result(negative).value();
    auto computed = result.storage_lease().value();
    expect(computed.tensor(description).value(), std::array<float, 4>{-0.25f, 1.5f, -3.75f, 0.0f});
    EXPECT_NE(computed.view()->native_handle(), original.native_handle());
    result = {};
    slot = {};
    compiled->executable = {};
    graph = {};
    ASSERT_TRUE(context.close().ok());
    expect(current.tensor(description).value(), wanted);
    expect(computed.tensor(description).value(), std::array<float, 4>{-0.25f, 1.5f, -3.75f, 0.0f});
  }
  EXPECT_EQ(counted.snapshot().live_count, 0U);
}
} // namespace
