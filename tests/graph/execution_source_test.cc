#include <gtest/gtest.h>

#include <array>
#include <utility>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/operation_access.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/detail/execution_source.h"
#include "hyle/graph/detail/region_access.h"
#include "../../src/graph/execution_source_internal.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;

hyle::Result<c::Context> source_context() {
  return c::Context::create(c::cpu::device(), c::native::provider());
}
hyle::Result<g::Graph> source_graph() {
  HYLE_ASSIGN_OR_RETURN(auto description, c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 1>{4}));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(description));
  // This fixture installs only the native provider; its F32 Exp carrier is
  // retained, while ordinary F32 Neg is supplied by other CPU providers.
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(
      c::Operation(c::Unary{c::UnaryKind::kExp}), std::array{input}));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  return std::move(builder).finish();
}

TEST(GraphExecutionSource, ExactOwnerSurvivesContextMove) {
  auto made = source_context(); ASSERT_TRUE(made);
  auto first = g::ExecutionSource::acquire(*made); ASSERT_TRUE(first);
  auto independent = g::ExecutionSource::acquire(*made); ASSERT_TRUE(independent);
  EXPECT_TRUE(first->same_owner(*independent));
  c::Context moved = std::move(*made);
  EXPECT_FALSE(made->valid());
  EXPECT_TRUE(first->valid()); EXPECT_TRUE(first->is_open());
  auto after_move = g::ExecutionSource::acquire(moved); ASSERT_TRUE(after_move);
  EXPECT_TRUE(first->same_owner(*after_move));
  auto other = source_context(); ASSERT_TRUE(other);
  auto foreign = g::ExecutionSource::acquire(*other); ASSERT_TRUE(foreign);
  EXPECT_FALSE(first->same_owner(*foreign));
  EXPECT_TRUE(first->same_device(*foreign));
  ASSERT_TRUE(moved.close().ok());
  EXPECT_TRUE(first->valid()); EXPECT_FALSE(first->is_open());
  EXPECT_FALSE(independent->is_open()); EXPECT_FALSE(after_move->is_open());
  EXPECT_TRUE(foreign->is_open());
}

TEST(GraphExecutionSource, RejectsForeignQueueAndProviderCapability) {
  auto first = source_context(); ASSERT_TRUE(first);
  auto second = source_context(); ASSERT_TRUE(second);
  auto source = g::ExecutionSource::acquire(*first); ASSERT_TRUE(source);
  auto queue = first->create_queue(); ASSERT_TRUE(queue);
  auto foreign_queue = second->create_queue(); ASSERT_TRUE(foreign_queue);
  EXPECT_TRUE(g::detail::ExecutionSourceAccess::owns(*source, *queue));
  EXPECT_FALSE(g::detail::ExecutionSourceAccess::owns(*source, *foreign_queue));
  const auto id = c::detail::kProjectedInputReadExtensionId;
  auto capability = c::detail::ProviderExtensionAccess::query(*first, 0, id);
  auto foreign = c::detail::ProviderExtensionAccess::query(*second, 0, id);
  ASSERT_TRUE(capability); ASSERT_TRUE(foreign);
  EXPECT_TRUE(g::detail::ExecutionSourceAccess::owns(*source, *capability));
  EXPECT_FALSE(g::detail::ExecutionSourceAccess::owns(*source, *foreign));
  auto rejected = g::detail::GraphRegionSource::shared(*source, *foreign);
  ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsInvalidArgument());
  auto retained = g::detail::GraphRegionSource::shared(*source, *capability);
  ASSERT_TRUE(retained); EXPECT_FALSE(retained->is_native());
  ASSERT_NE(retained->shared_extension(), nullptr);
  EXPECT_EQ(retained->provider_instance_id(), capability->provider_instance_id());

  auto graph = source_graph(); ASSERT_TRUE(graph);
  const std::array domains{g::StorageDomain{hyle::mem::system_resource()}};
  const std::array targets{g::ExecutionTarget{*source, g::StorageDomainId{0}}};
  auto compiled = g::compile(*graph, {domains, targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_TRUE(g::detail::RegionAccess::validate_queue(
      compiled->executable, g::ExecutionLaneId{0}, *queue).ok());
  EXPECT_TRUE(g::detail::RegionAccess::validate_queue(
      compiled->executable, g::ExecutionLaneId{0}, *foreign_queue).IsInvalidArgument());
}

TEST(GraphExecutionSource, CompileRejectsSeparatelyAcquiredDuplicateOwner) {
  auto context = source_context(); ASSERT_TRUE(context);
  auto first = g::ExecutionSource::acquire(*context); ASSERT_TRUE(first);
  auto second = g::ExecutionSource::acquire(*context); ASSERT_TRUE(second);
  auto graph = source_graph(); ASSERT_TRUE(graph);
  const std::array domains{g::StorageDomain{hyle::mem::system_resource()}};
  const std::array targets{
      g::ExecutionTarget{*first, g::StorageDomainId{0}},
      g::ExecutionTarget{*second, g::StorageDomainId{0}}};
  auto compiled = g::compile(*graph, {domains, targets});
  ASSERT_FALSE(compiled); EXPECT_TRUE(compiled.error().IsInvalidArgument());
}

TEST(GraphExecutionSource, ClosingEndpointDoesNotReopenRetainedSource) {
  auto context = source_context(); ASSERT_TRUE(context);
  auto source = g::ExecutionSource::acquire(*context); ASSERT_TRUE(source);
  auto graph = source_graph(); ASSERT_TRUE(graph);
  auto capability = c::detail::ProviderExtensionAccess::query(
      *context, 0, c::detail::kProjectedInputReadExtensionId);
  ASSERT_TRUE(capability);
  ASSERT_TRUE(context->close().ok());
  EXPECT_TRUE(source->valid()); EXPECT_FALSE(source->is_open());
  const std::array domains{g::StorageDomain{hyle::mem::system_resource()}};
  const std::array targets{g::ExecutionTarget{*source, g::StorageDomainId{0}}};
  auto compiled = g::compile(*graph, {domains, targets});
  ASSERT_FALSE(compiled); EXPECT_TRUE(compiled.error().IsFailedPrecondition());
  bool invoked = false;
  const auto status = g::detail::ExecutionSourceAccess::call(*source,
      [](void *value) noexcept { *static_cast<bool *>(value) = true; return hyle::Status::Ok(); },
      &invoked);
  EXPECT_TRUE(status.IsFailedPrecondition()); EXPECT_FALSE(invoked);
  auto sibling = c::detail::ProviderExtensionAccess::query(
      *capability, c::detail::kProjectedInputReadExtensionId);
  ASSERT_FALSE(sibling); EXPECT_TRUE(sibling.error().IsFailedPrecondition());
}
} // namespace
