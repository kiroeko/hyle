#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>

#include <mlx/array.h>

#include "hyle/compute/apple_memory.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace cd = c::detail;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace mx = mlx::core;
using hyle::Status;

Status expect(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

hyle::Result<c::ResourceBudget> budget(m::MemoryResourceRef resource,
                                      std::size_t backing_bytes) {
  const std::array limits{
      c::ResourceBudgetLimit{resource, backing_bytes},
      c::ResourceBudgetLimit{m::system_resource(), 1u << 20}};
  return c::ResourceBudget::create({.limits = limits});
}

Status expect_charge(const c::ResourceBudget &scope, m::MemoryResourceRef resource,
                     std::size_t bytes) {
  HYLE_ASSIGN_OR_RETURN(const auto usage, scope.usage(resource));
  return expect(usage.live_bytes == bytes, "retained backing charge differs");
}

template<std::size_t N>
Status expect_values(const c::ReadLease &lease, const std::array<float, N> &wanted) {
  HYLE_ASSIGN_OR_RETURN(auto view, lease.view());
  HYLE_RETURN_IF_ERROR(expect(view.size() == sizeof(wanted), "readonly byte extent differs"));
  std::array<float, N> actual{};
  HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(actual))));
  return expect(actual == wanted, "retained native payload differs");
}

template<std::size_t N>
Status initialize(const a::MemoryResource &memory, const mx::array &destination,
                  const c::TensorDesc &description, const std::array<float, N> &values) {
  HYLE_ASSIGN_OR_RETURN(auto host,
      m::Buffer::create(m::system_resource(), sizeof(values), alignof(float)));
  HYLE_RETURN_IF_ERROR(host.view().write(std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto source, c::ConstTensorView::create(description, host.view()));
  return memory.copy_from_host(destination, source);
}

Status shared_backing(const a::Context &context, const a::MemoryResource &memory) {
  HYLE_ASSIGN_OR_RETURN(auto whole_description, c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto half_description, c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 1>{32}));
  std::array<float, 64> values{};
  std::array<float, 32> left_values{}, right_values{}, negative{};
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i + 1);
  for (std::size_t i = 0; i < left_values.size(); ++i) {
    left_values[i] = values[i]; right_values[i] = values[i + 32]; negative[i] = -values[i];
  }
  HYLE_ASSIGN_OR_RETURN(auto whole, memory.allocate(whole_description));
  HYLE_RETURN_IF_ERROR(initialize(memory, whole, whole_description, values));
  mx::array left(mx::Shape{32}, mx::float32, nullptr, {});
  mx::array right(mx::Shape{32}, mx::float32, nullptr, {});
  left.copy_shared_buffer(whole, left.strides(), left.flags(), left.size(), 0);
  right.copy_shared_buffer(whole, right.strides(), right.flags(), right.size(), 32);
  left.set_status(mx::array::available); right.set_status(mx::array::available);
  HYLE_ASSIGN_OR_RETURN(const auto whole_storage, memory.storage_info(whole));
  HYLE_ASSIGN_OR_RETURN(const auto left_storage, memory.storage_info(left));
  HYLE_ASSIGN_OR_RETURN(const auto right_storage, memory.storage_info(right));
  HYLE_RETURN_IF_ERROR(expect(whole_storage.backing_bytes >= sizeof(values) &&
      left_storage.buffer == whole_storage.buffer && right_storage.buffer == whole_storage.buffer &&
      left_storage.offset_bytes == 0 && right_storage.offset_bytes == sizeof(left_values),
      "shared Data aliases do not describe the expected original backing"));

  HYLE_ASSIGN_OR_RETURN(auto first, memory.import_readonly(left, half_description));
  HYLE_ASSIGN_OR_RETURN(auto second, memory.import_readonly(right, half_description));
  HYLE_ASSIGN_OR_RETURN(const auto first_backing, cd::LeaseAccess::backing(first));
  HYLE_ASSIGN_OR_RETURN(const auto second_backing, cd::LeaseAccess::backing(second));
  HYLE_RETURN_IF_ERROR(expect(first_backing.identity != second_backing.identity &&
      first_backing.bytes == whole_storage.backing_bytes &&
      second_backing.bytes == whole_storage.backing_bytes,
      "independent imports must retain the complete original backing"));
  HYLE_ASSIGN_OR_RETURN(auto tiny, budget(memory.ref(), whole_storage.backing_bytes - 1));
  const auto denied = cd::LeaseAccess::retain_budget(first, memory.ref(), tiny);
  HYLE_RETURN_IF_ERROR(expect(denied.IsResourceExhausted(), "undersized alias budget was accepted"));
  HYLE_RETURN_IF_ERROR(expect_charge(tiny, memory.ref(), 0));
  HYLE_RETURN_IF_ERROR(expect_values(first, left_values));
  HYLE_RETURN_IF_ERROR(expect_values(second, right_values));
  HYLE_RETURN_IF_ERROR(expect(!first.try_write() && !second.try_write(),
      "failed retention admission changed readonly permissions"));

  HYLE_ASSIGN_OR_RETURN(auto cache, budget(memory.ref(), whole_storage.backing_bytes));
  HYLE_RETURN_IF_ERROR(cd::LeaseAccess::retain_budget(first, memory.ref(), cache));
  HYLE_RETURN_IF_ERROR(cd::LeaseAccess::retain_budget(second, memory.ref(), cache));
  HYLE_ASSIGN_OR_RETURN(auto slice, first.subview(0, sizeof(float)));
  HYLE_RETURN_IF_ERROR(cd::LeaseAccess::retain_budget(slice, memory.ref(), cache));
  HYLE_RETURN_IF_ERROR(expect_charge(cache, memory.ref(), whole_storage.backing_bytes));

  // Real native use of one imported domain. Bound sources and the original
  // Completion retain it; no synthetic completion or delayed-retirement hook.
  {
    const std::array native_inputs{left};
    HYLE_ASSIGN_OR_RETURN(auto program,
        context.prepare(c::Unary{c::UnaryKind::kNeg}, std::span<const mx::array>(native_inputs)));
    HYLE_ASSIGN_OR_RETURN(auto invocation, a::LeasedInvocation::create(program, memory));
    std::array inputs{a::LeasedInput{half_description, first}};
    HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
    HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit());
    inputs[0].source.reset(); first.reset(); second.reset(); slice.reset();
    HYLE_RETURN_IF_ERROR(expect_charge(cache, memory.ref(), whole_storage.backing_bytes));
    HYLE_ASSIGN_OR_RETURN(auto result, completion.wait());
    HYLE_ASSIGN_OR_RETURN(auto output, memory.import_readonly(result->arrays()[0], half_description));
    HYLE_RETURN_IF_ERROR(expect_values(output, negative));
  }
  HYLE_RETURN_IF_ERROR(expect_charge(cache, memory.ref(), 0));
  HYLE_ASSIGN_OR_RETURN(auto original, memory.import_readonly(whole, whole_description));
  HYLE_RETURN_IF_ERROR(expect_values(original, values));
  std::printf("PASS Apple shared native backing: two independent imports/one %zu-byte charge/real native use/readonly quota failure\n",
              whole_storage.backing_bytes);
  return Status::Ok();
}

Status graph_output(const a::Context &context, const a::MemoryResource &memory) {
  HYLE_ASSIGN_OR_RETURN(auto description, c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 1>{32}));
  std::array<float, 32> values{}, wanted{};
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i + 1); wanted[i] = -values[i];
  }
  c::ReadLease escaped;
  c::ResourceBudget cache;
  std::size_t full_backing_bytes = 0;
  {
    HYLE_ASSIGN_OR_RETURN(auto array, memory.allocate(description));
    HYLE_RETURN_IF_ERROR(initialize(memory, array, description, values));
    HYLE_ASSIGN_OR_RETURN(auto input_owner, memory.import_readonly(array, description));
    HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
    HYLE_ASSIGN_OR_RETURN(const auto input, builder.input(description));
    HYLE_ASSIGN_OR_RETURN(const auto output, builder.add(c::Unary{c::UnaryKind::kNeg}, input));
    HYLE_RETURN_IF_ERROR(builder.publish(output));
    HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(auto source, g::apple::execution_source(context));
    const std::array domains{g::StorageDomain{memory.ref()}};
    const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
    HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(graph, {domains, targets}));
    HYLE_ASSIGN_OR_RETURN(auto invocation,
        compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(invocation.bind(input, description, input_owner));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay({}));
    HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit());
    HYLE_RETURN_IF_ERROR(completion.wait());
    HYLE_ASSIGN_OR_RETURN(auto result, invocation.result(output));
    HYLE_ASSIGN_OR_RETURN(auto ordinary, result.storage_lease());
    HYLE_RETURN_IF_ERROR(expect_values(ordinary, wanted));
    HYLE_ASSIGN_OR_RETURN(const auto backing, cd::LeaseAccess::backing(ordinary));
    full_backing_bytes = backing.bytes;
    HYLE_RETURN_IF_ERROR(expect(full_backing_bytes >= sizeof(wanted), "native result backing is too small"));
    HYLE_ASSIGN_OR_RETURN(auto tiny, budget(memory.ref(), full_backing_bytes - 1));
    const auto denied = result.storage_lease(tiny);
    HYLE_RETURN_IF_ERROR(expect(!denied && denied.error().IsResourceExhausted(),
        "undersized Graph output budget was accepted"));
    HYLE_RETURN_IF_ERROR(expect_charge(tiny, memory.ref(), 0));
    HYLE_RETURN_IF_ERROR(expect_values(ordinary, wanted));
    HYLE_ASSIGN_OR_RETURN(auto still_readable, result.tensor());
    HYLE_RETURN_IF_ERROR(expect(still_readable.description() == description,
        "failed Graph retention changed the original output description"));
    HYLE_ASSIGN_OR_RETURN(cache, budget(memory.ref(), full_backing_bytes));
    HYLE_ASSIGN_OR_RETURN(escaped, result.storage_lease(cache));
    HYLE_RETURN_IF_ERROR(expect_values(escaped, wanted));
    HYLE_RETURN_IF_ERROR(expect_charge(cache, memory.ref(), full_backing_bytes));
  }
  // Only the independent readonly lease remains: every Graph, Invocation,
  // GraphOutput and Completion above has been destroyed.
  HYLE_RETURN_IF_ERROR(expect_values(escaped, wanted));
  HYLE_RETURN_IF_ERROR(expect_charge(cache, memory.ref(), full_backing_bytes));
  escaped.reset();
  HYLE_RETURN_IF_ERROR(expect_charge(cache, memory.ref(), 0));
  std::printf("PASS Apple Graph retained output: quota rejection preserves result/independent %zu-byte backing owner survives all Graph handles\n",
              full_backing_bytes);
  return Status::Ok();
}

Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(context));
  HYLE_RETURN_IF_ERROR(shared_backing(context, memory));
  HYLE_RETURN_IF_ERROR(graph_output(context, memory));
  return context.close();
}
} // namespace

int main() {
  try {
    const auto status = run();
    if (status.ok()) return 0;
    std::fprintf(stderr, "FAIL %.*s\n", static_cast<int>(status.message().size()), status.message().data());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
  }
  return 1;
}
