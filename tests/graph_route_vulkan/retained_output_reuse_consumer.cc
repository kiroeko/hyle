// Actual Vulkan Graph Invocation reuse while independent output leases live.
// Downloads are oracle observations only; no shadow output preserves a version.
#include <hyle/compute/vulkan_memory.h>
#include <hyle/compute/detail/lease_access.h>
#include <hyle/graph/executable.h>
#include <hyle/graph/result.h>
#include <hyle/graph/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
using hyle::Result;
using hyle::Status;
using Bytes = std::vector<std::byte>;
constexpr std::size_t elements = 32, slice_first = 8, slice_elements = 16;

Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
struct NativeRange final {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
};
struct Generation final {
  c::ReadLease whole, slice;
  NativeRange range;
  std::uint64_t whole_generation = 0, slice_generation = 0;
};
struct Ports final { g::ValueId input, whole, slice; };
struct Admitted final {
  // Three independent exports, not copies/subviews of one lease domain.
  std::array<c::ReadLease, 3> leases;
  std::array<const void *, 3> domains{}, blocks{};
  std::array<std::uint64_t, 3> generations{};
};

Result<c::ResourceBudget> retention_budget(const v::MemoryResource &device, std::size_t bytes) {
  const std::array limits{c::ResourceBudgetLimit{device.ref(), bytes},
      c::ResourceBudgetLimit{hyle::mem::system_resource(), 1u << 20}};
  return c::ResourceBudget::create({.limits = limits});
}
Status charged(const c::ResourceBudget &budget, const v::MemoryResource &device, std::size_t bytes) {
  HYLE_ASSIGN_OR_RETURN(auto usage, budget.usage(device.ref()));
  return require(usage.live_bytes == bytes, "canonical retained root requested-byte charge differs");
}
Result<Admitted> admit_three(const g::GraphOutput &whole, const g::GraphOutput &slice,
                            const c::ResourceBudget &budget, const v::MemoryResource &device,
                            std::size_t expected_charge) {
  Admitted result;
  // Slice first must still charge the complete original Vk storage block.
  HYLE_ASSIGN_OR_RETURN(result.leases[0], slice.storage_lease(budget));
  HYLE_RETURN_IF_ERROR(charged(budget, device, expected_charge));
  HYLE_ASSIGN_OR_RETURN(result.leases[1], whole.storage_lease(budget));
  HYLE_RETURN_IF_ERROR(charged(budget, device, expected_charge));
  HYLE_ASSIGN_OR_RETURN(result.leases[2], whole.storage_lease(budget));
  HYLE_RETURN_IF_ERROR(charged(budget, device, expected_charge));
  for (std::size_t i = 0; i < result.leases.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto backing, c::detail::LeaseAccess::backing(result.leases[i]));
    result.domains[i] = backing.identity;
    result.blocks[i] = c::detail::LeaseAccess::identity(result.leases[i]);
    result.generations[i] = result.leases[i].generation();
    for (std::size_t previous = 0; previous < i; ++previous)
      HYLE_RETURN_IF_ERROR(require(result.domains[i] != result.domains[previous] &&
          result.blocks[i] != result.blocks[previous],
          "fresh exports were merged into one F1 block/domain instead of budget-only canonical accounting"));
  }
  return result;
}

Result<NativeRange> native_range(const v::MemoryResource &device,
                               const c::ReadLease &lease,
                               const c::TensorDesc &description) {
  HYLE_ASSIGN_OR_RETURN(auto tensor, device.binding(lease, description));
  HYLE_ASSIGN_OR_RETURN(auto storage, device.storage_info(tensor));
  return NativeRange{storage.buffer, storage.offset_bytes};
}
Result<Bytes> read(const v::Context &context, const v::MemoryResource &device,
                   const c::ReadLease &lease, const c::TensorDesc &description) {
  HYLE_ASSIGN_OR_RETURN(auto size, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto storage, lease.view());
  HYLE_RETURN_IF_ERROR(require(lease.size() == size && storage.size() == size &&
      storage.resource() == device.ref(), "output lease lost its exact native resource or byte extent"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, device.binding(lease, description));
  Bytes bytes(size);
  HYLE_RETURN_IF_ERROR(context.download(tensor, bytes));
  return bytes;
}
Status readonly(c::ReadLease &lease, std::uint64_t generation) {
  auto writer = lease.try_write();
  HYLE_RETURN_IF_ERROR(require(!writer && lease.valid() && lease.generation() == generation,
      "retained native output gained writer permission or changed generation"));
  return Status::Ok();
}
Status observe(const v::Context &context, const v::MemoryResource &device,
               Generation &generation, const c::TensorDesc &whole,
               const c::TensorDesc &slice, const Bytes &expected,
               std::size_t scalar_bytes) {
  HYLE_RETURN_IF_ERROR(readonly(generation.whole, generation.whole_generation));
  HYLE_RETURN_IF_ERROR(readonly(generation.slice, generation.slice_generation));
  HYLE_ASSIGN_OR_RETURN(auto actual, read(context, device, generation.whole, whole));
  HYLE_ASSIGN_OR_RETURN(auto actual_slice, read(context, device, generation.slice, slice));
  const auto subspan = std::span(expected).subspan(slice_first * scalar_bytes, slice_elements * scalar_bytes);
  HYLE_RETURN_IF_ERROR(require(actual == expected &&
      actual_slice.size() == subspan.size() &&
      std::equal(actual_slice.begin(), actual_slice.end(), subspan.begin()),
      "native retained whole output or its published offset alias changed bytes"));
  HYLE_ASSIGN_OR_RETURN(auto whole_range, native_range(device, generation.whole, whole));
  HYLE_ASSIGN_OR_RETURN(auto slice_range, native_range(device, generation.slice, slice));
  HYLE_RETURN_IF_ERROR(require(whole_range.buffer == generation.range.buffer &&
      whole_range.offset == generation.range.offset &&
      slice_range.buffer == whole_range.buffer &&
      slice_range.offset == whole_range.offset + slice_first * scalar_bytes,
      "retained output/offset alias lost its original native backing relationship"));
  return Status::Ok();
}
Status observe_admitted(const v::Context &context, const v::MemoryResource &device,
                        Admitted &admitted, const c::TensorDesc &whole,
                        const c::TensorDesc &slice, const Bytes &expected,
                        std::size_t scalar_bytes) {
  for (std::size_t i = 0; i < admitted.leases.size(); ++i) {
    auto &lease = admitted.leases[i];
    HYLE_ASSIGN_OR_RETURN(auto backing, c::detail::LeaseAccess::backing(lease));
    HYLE_RETURN_IF_ERROR(require(backing.identity == admitted.domains[i] &&
        c::detail::LeaseAccess::identity(lease) == admitted.blocks[i],
        "budget admission/replay changed an existing F1 block or domain identity"));
    HYLE_RETURN_IF_ERROR(readonly(lease, admitted.generations[i]));
    HYLE_ASSIGN_OR_RETURN(auto bytes, read(context, device, lease, i == 0 ? slice : whole));
    const auto wanted = i == 0 ? std::span(expected).subspan(slice_first * scalar_bytes, slice_elements * scalar_bytes)
                               : std::span(expected);
    HYLE_RETURN_IF_ERROR(require(bytes.size() == wanted.size() &&
        std::equal(bytes.begin(), bytes.end(), wanted.begin()),
        "budgeted native independent export changed its complete readonly bytes"));
  }
  return Status::Ok();
}

Status stale_submit_rejected(g::Invocation &invocation) {
  // The no-argument convenience overload prepares commands itself. Exercise
  // the strict prepared submission contract, which must not rearm implicitly.
  auto stale = invocation.submit(std::span<const g::QueueBinding>{});
  if (stale) {
    stale->wait().IgnoreError();
    return Status::Internal("unbound or rebound-but-unprepared Invocation admitted stale submission");
  }
  return require(stale.error().IsFailedPrecondition(), "stale submission returned an unexpected error category");
}
template<class Inspect>
Result<Generation> execute(g::Invocation &invocation, const Ports &ports,
                           const c::TensorDesc &input_description,
                           const c::TensorDesc &whole_description,
                           const c::TensorDesc &slice_description,
                           const c::ReadLease &input, const v::MemoryResource &device, Inspect inspect) {
  HYLE_RETURN_IF_ERROR(invocation.bind(ports.input, input_description, input));
  HYLE_RETURN_IF_ERROR(stale_submit_rejected(invocation));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay({}));
  Generation result;
  {
    HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(std::span<const g::QueueBinding>{}));
    HYLE_RETURN_IF_ERROR(completion.wait());
    {
      HYLE_ASSIGN_OR_RETURN(auto whole, invocation.result(ports.whole));
      HYLE_ASSIGN_OR_RETURN(auto slice, invocation.result(ports.slice));
      HYLE_ASSIGN_OR_RETURN(auto whole_desc, whole.description());
      HYLE_ASSIGN_OR_RETURN(auto slice_desc, slice.description());
      HYLE_RETURN_IF_ERROR(require(whole_desc == whole_description && slice_desc == slice_description,
          "generated output descriptions changed across native replay"));
      HYLE_ASSIGN_OR_RETURN(result.whole, whole.storage_lease());
      HYLE_ASSIGN_OR_RETURN(result.slice, slice.storage_lease());
      HYLE_ASSIGN_OR_RETURN(result.range, native_range(device, result.whole, whole_description));
      result.whole_generation = result.whole.generation(); result.slice_generation = result.slice.generation();
      HYLE_RETURN_IF_ERROR(inspect(whole, slice, result));
      // Public result handles still pin their original Invocation. Independent
      // leases become the only escaped observations when this scope ends.
      HYLE_RETURN_IF_ERROR(require(!invocation.unbind(ports.input).ok(),
          "live GraphOutput observation allowed its original input to be withdrawn"));
    }
  }
  HYLE_RETURN_IF_ERROR(require(!invocation.in_flight(), "successful original Completion did not retire native work"));
  return result;
}

Bytes expected(const std::array<float, elements> &input, bool bfloat) {
  Bytes bytes(input.size() * (bfloat ? sizeof(std::uint16_t) : sizeof(float)));
  for (std::size_t i = 0; i < input.size(); ++i) {
    if (bfloat) {
      // Every fixture value is exactly representable as BF16. This oracle
      // therefore does not introduce a conversion-tolerance policy.
      const auto bits = std::bit_cast<std::uint32_t>(input[i]);
      const auto word = static_cast<std::uint16_t>(bits >> 16);
      const auto source = std::as_bytes(std::span(&word, 1));
      std::copy(source.begin(), source.end(), bytes.begin() + static_cast<std::ptrdiff_t>(i * sizeof(word)));
    } else {
      const float value = -input[i];
      const auto source = std::as_bytes(std::span(&value, 1));
      std::copy(source.begin(), source.end(), bytes.begin() + static_cast<std::ptrdiff_t>(i * sizeof(value)));
    }
  }
  return bytes;
}

Status exercise(const v::Context &context, const v::MemoryResource &device,
                const g::ExecutionSource &source, bool bfloat) {
  const auto dtype = bfloat ? c::DataType::kBFloat16 : c::DataType::kFloat32;
  const auto scalar_bytes = bfloat ? sizeof(std::uint16_t) : sizeof(float);
  HYLE_ASSIGN_OR_RETURN(auto input_description, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{elements}));
  HYLE_ASSIGN_OR_RETURN(auto whole_description, c::TensorDesc::dense(dtype,
      std::array<std::int64_t, 1>{elements}));
  HYLE_ASSIGN_OR_RETURN(auto slice_description, c::TensorDesc::dense(dtype,
      std::array<std::int64_t, 1>{slice_elements}));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  Ports ports;
  HYLE_ASSIGN_OR_RETURN(ports.input, builder.input(input_description));
  if (bfloat) {
    // The original native route selects its actual BF16 conversion/managed
    // implementation. The test does not replace it with a host conversion.
    HYLE_ASSIGN_OR_RETURN(ports.whole, builder.add(c::Cast{dtype}, ports.input));
  } else {
    HYLE_ASSIGN_OR_RETURN(ports.whole, builder.add(c::Unary{c::UnaryKind::kNeg}, ports.input));
  }
  HYLE_ASSIGN_OR_RETURN(auto slice_op, c::View::slice(0, slice_first, slice_elements));
  HYLE_ASSIGN_OR_RETURN(ports.slice, builder.add(slice_op, ports.whole));
  HYLE_RETURN_IF_ERROR(builder.publish(ports.whole));
  HYLE_RETURN_IF_ERROR(builder.publish(ports.slice));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto calculation, graph.producer(ports.whole));
  const std::array domains{g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(graph, {domains, targets}));
  bool native_calculation = false;
  for (std::size_t index = 0; index < compiled.report.step_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto step, compiled.report.step(index));
    for (std::size_t node = 0; node < step.node_count; ++node) {
      HYLE_ASSIGN_OR_RETURN(auto logical, compiled.report.step_node(index, node));
      if (logical != calculation) continue;
      HYLE_RETURN_IF_ERROR(require(step.target == g::ExecutionTargetId{0} &&
          step.kind == g::CompileStepKind::kProviderRegion &&
          step.runtime.mode == c::ProviderRuntimeMode::kDelegated,
          "calculation escaped the original delegated Vulkan source"));
      native_calculation = true;
    }
  }
  HYLE_RETURN_IF_ERROR(require(native_calculation && compiled.executable.required_queue_count() == 0,
      "actual Vulkan calculation or its original no-caller-Queue route is missing"));
  HYLE_ASSIGN_OR_RETURN(auto invocation,
      compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  std::array<c::ReadLease, 3> inputs;
  std::array<Bytes, 3> wanted;
  for (std::size_t generation = 0; generation < inputs.size(); ++generation) {
    std::array<float, elements> values;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const float magnitude = float(i % 16) * .5F + .25F + float(generation) * 2.F;
      values[i] = i % 2 ? -magnitude : magnitude;
    }
    HYLE_ASSIGN_OR_RETURN(auto native, context.upload(input_description, std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(inputs[generation], device.import_readonly(native));
    wanted[generation] = expected(values, bfloat);
  }
  // One Invocation, one compiled Graph/source and the original input resource
  // serve every generation. No second execution slot or payload copy is used.
  c::ResourceBudget tiny, one_root, two_roots;
  std::size_t root_bytes = 0;
  std::array<Admitted, 3> one_admitted, two_admitted;
  HYLE_ASSIGN_OR_RETURN(auto first, execute(invocation, ports, input_description,
      whole_description, slice_description, inputs[0], device,
      [&](const g::GraphOutput &whole, const g::GraphOutput &slice, Generation &ordinary) -> Status {
        HYLE_ASSIGN_OR_RETURN(auto tensor, device.binding(ordinary.whole, whole_description));
        HYLE_ASSIGN_OR_RETURN(auto storage, device.storage_info(tensor));
        HYLE_ASSIGN_OR_RETURN(auto domain, c::detail::LeaseAccess::backing(ordinary.whole));
        HYLE_ASSIGN_OR_RETURN(auto logical, whole_description.required_bytes());
        // These whole32 fixtures are exact, zero-origin owned roots. Public
        // capacity is requested bytes; allocation_bytes is a separate physical
        // VkMemory requirement and is deliberately not the retained quota.
        HYLE_RETURN_IF_ERROR(require(storage.offset_bytes == 0 && storage.capacity_bytes == logical &&
            domain.bytes == storage.capacity_bytes && domain.resource == device.ref(),
            "whole-output fixture does not expose its exact requested root capacity"));
        root_bytes = static_cast<std::size_t>(storage.capacity_bytes);
        HYLE_ASSIGN_OR_RETURN(tiny, retention_budget(device, root_bytes - 1));
        HYLE_ASSIGN_OR_RETURN(one_root, retention_budget(device, root_bytes));
        HYLE_ASSIGN_OR_RETURN(two_roots, retention_budget(device, root_bytes * 2));
        const auto whole_domain = domain.identity;
        const auto whole_block = c::detail::LeaseAccess::identity(ordinary.whole);
        auto denied = slice.storage_lease(tiny);
        HYLE_RETURN_IF_ERROR(require(!denied && denied.error().IsResourceExhausted(),
            "slice retention accepted less than its complete owned root requested bytes"));
        HYLE_RETURN_IF_ERROR(charged(tiny, device, 0));
        HYLE_ASSIGN_OR_RETURN(auto still_whole, whole.tensor());
        HYLE_ASSIGN_OR_RETURN(auto still_slice, slice.tensor());
        HYLE_ASSIGN_OR_RETURN(auto after, c::detail::LeaseAccess::backing(ordinary.whole));
        HYLE_RETURN_IF_ERROR(require(still_whole.description() == whole_description &&
            still_slice.description() == slice_description && after.identity == whole_domain &&
            c::detail::LeaseAccess::identity(ordinary.whole) == whole_block,
            "quota rejection changed original output observations or F1/domain identity"));
        HYLE_RETURN_IF_ERROR(observe(context, device, ordinary, whole_description, slice_description, wanted[0], scalar_bytes));
        HYLE_ASSIGN_OR_RETURN(one_admitted[0], admit_three(whole, slice, one_root, device, root_bytes));
        HYLE_ASSIGN_OR_RETURN(two_admitted[0], admit_three(whole, slice, two_roots, device, root_bytes));
        HYLE_RETURN_IF_ERROR(observe_admitted(context, device, one_admitted[0], whole_description, slice_description, wanted[0], scalar_bytes));
        return observe_admitted(context, device, two_admitted[0], whole_description, slice_description, wanted[0], scalar_bytes);
      }));
  HYLE_RETURN_IF_ERROR(observe(context, device, first, whole_description, slice_description, wanted[0], scalar_bytes));
  HYLE_RETURN_IF_ERROR(invocation.unbind(ports.input));
  HYLE_RETURN_IF_ERROR(stale_submit_rejected(invocation));
  HYLE_ASSIGN_OR_RETURN(auto second, execute(invocation, ports, input_description,
      whole_description, slice_description, inputs[1], device,
      [&](const g::GraphOutput &whole, const g::GraphOutput &slice, Generation &ordinary) -> Status {
        HYLE_ASSIGN_OR_RETURN(two_admitted[1], admit_three(whole, slice, two_roots, device, root_bytes * 2));
        HYLE_RETURN_IF_ERROR(charged(one_root, device, root_bytes));
        auto denied = whole.storage_lease(one_root);
        HYLE_RETURN_IF_ERROR(require(!denied && denied.error().IsResourceExhausted(),
            "second live generation incorrectly shared the first generation's canonical budget key"));
        HYLE_RETURN_IF_ERROR(charged(one_root, device, root_bytes));
        HYLE_RETURN_IF_ERROR(observe(context, device, first, whole_description, slice_description, wanted[0], scalar_bytes));
        HYLE_RETURN_IF_ERROR(observe_admitted(context, device, one_admitted[0], whole_description, slice_description, wanted[0], scalar_bytes));
        HYLE_RETURN_IF_ERROR(observe_admitted(context, device, two_admitted[0], whole_description, slice_description, wanted[0], scalar_bytes));
        HYLE_RETURN_IF_ERROR(observe(context, device, ordinary, whole_description, slice_description, wanted[1], scalar_bytes));
        // Drop two fresh domains, keeping the independent budgeted slice: its
        // token must still retain the original complete root charge.
        one_admitted[0].leases[1].reset(); one_admitted[0].leases[2].reset();
        HYLE_RETURN_IF_ERROR(charged(one_root, device, root_bytes));
        one_admitted[0].leases[0].reset();
        HYLE_RETURN_IF_ERROR(charged(one_root, device, 0));
        // Retry these original successful GraphOutput handles, without a new
        // submission, executor, output copy or caller-supplied annotation.
        HYLE_ASSIGN_OR_RETURN(one_admitted[1], admit_three(whole, slice, one_root, device, root_bytes));
        HYLE_RETURN_IF_ERROR(observe_admitted(context, device, one_admitted[1], whole_description, slice_description, wanted[1], scalar_bytes));
        return observe_admitted(context, device, two_admitted[1], whole_description, slice_description, wanted[1], scalar_bytes);
      }));
  HYLE_RETURN_IF_ERROR(require(first.range.buffer != second.range.buffer || first.range.offset != second.range.offset,
      "two simultaneously retained distinct generations use the same output range"));
  HYLE_RETURN_IF_ERROR(observe(context, device, first, whole_description, slice_description, wanted[0], scalar_bytes));
  HYLE_RETURN_IF_ERROR(observe(context, device, second, whole_description, slice_description, wanted[1], scalar_bytes));
  first.whole.reset(); first.slice.reset();
  two_admitted[0] = {};
  HYLE_RETURN_IF_ERROR(charged(two_roots, device, root_bytes));
  HYLE_RETURN_IF_ERROR(invocation.unbind(ports.input));
  HYLE_RETURN_IF_ERROR(stale_submit_rejected(invocation));
  HYLE_ASSIGN_OR_RETURN(auto third, execute(invocation, ports, input_description,
      whole_description, slice_description, inputs[2], device,
      [&](const g::GraphOutput &whole, const g::GraphOutput &slice, Generation &ordinary) -> Status {
        HYLE_RETURN_IF_ERROR(charged(two_roots, device, root_bytes));
        HYLE_ASSIGN_OR_RETURN(two_admitted[2], admit_three(whole, slice, two_roots, device, root_bytes * 2));
        one_admitted[1] = {};
        HYLE_RETURN_IF_ERROR(charged(one_root, device, 0));
        HYLE_ASSIGN_OR_RETURN(one_admitted[2], admit_three(whole, slice, one_root, device, root_bytes));
        HYLE_RETURN_IF_ERROR(observe(context, device, ordinary, whole_description, slice_description, wanted[2], scalar_bytes));
        return observe_admitted(context, device, two_admitted[1], whole_description, slice_description, wanted[1], scalar_bytes);
      }));
  HYLE_RETURN_IF_ERROR(require(second.range.buffer != third.range.buffer || second.range.offset != third.range.offset,
      "third generation overwrote the still-retained second generation"));
  HYLE_RETURN_IF_ERROR(require(third.range.buffer == first.range.buffer && third.range.offset == first.range.offset,
      "released first generation was not reused as the original block's bounded spare"));
  HYLE_RETURN_IF_ERROR(observe(context, device, second, whole_description, slice_description, wanted[1], scalar_bytes));
  HYLE_RETURN_IF_ERROR(observe(context, device, third, whole_description, slice_description, wanted[2], scalar_bytes));
  HYLE_RETURN_IF_ERROR(observe_admitted(context, device, two_admitted[1], whole_description, slice_description, wanted[1], scalar_bytes));
  HYLE_RETURN_IF_ERROR(observe_admitted(context, device, two_admitted[2], whole_description, slice_description, wanted[2], scalar_bytes));
  HYLE_RETURN_IF_ERROR(observe_admitted(context, device, one_admitted[2], whole_description, slice_description, wanted[2], scalar_bytes));
  invocation = {}; compiled.executable = {}; graph = {};
  HYLE_RETURN_IF_ERROR(observe(context, device, second, whole_description, slice_description, wanted[1], scalar_bytes));
  HYLE_RETURN_IF_ERROR(observe(context, device, third, whole_description, slice_description, wanted[2], scalar_bytes));
  HYLE_RETURN_IF_ERROR(charged(two_roots, device, root_bytes * 2));
  HYLE_RETURN_IF_ERROR(charged(one_root, device, root_bytes));
  two_admitted[1] = {};
  HYLE_RETURN_IF_ERROR(charged(two_roots, device, root_bytes));
  two_admitted[2].leases[1].reset(); two_admitted[2].leases[2].reset();
  HYLE_RETURN_IF_ERROR(charged(two_roots, device, root_bytes));
  two_admitted[2].leases[0].reset();
  HYLE_RETURN_IF_ERROR(charged(two_roots, device, 0));
  one_admitted[2] = {};
  HYLE_RETURN_IF_ERROR(charged(one_root, device, 0));
  HYLE_RETURN_IF_ERROR(charged(tiny, device, 0));
  return Status::Ok();
}

Status device_buffer_io(const v::MemoryResource &device) {
  HYLE_RETURN_IF_ERROR(require(device.ref().properties().has(hyle::mem::MemoryProperty::kDeviceAccessible) &&
      !device.ref().properties().has(hyle::mem::MemoryProperty::kHostAccessible),
      "Vulkan device fixture must retain its actual device-only memory properties"));
  constexpr std::size_t bytes = 2u * 1024u * 1024u + 2u;
  HYLE_ASSIGN_OR_RETURN(auto buffer, hyle::mem::Buffer::create(device.ref(), bytes, 4));
  HYLE_RETURN_IF_ERROR(require(buffer.size() == bytes && buffer.resource() == device.ref(),
      "device Buffer exposes native padding as logical capacity"));
  Bytes expected(bytes, std::byte{0x31}), actual(bytes);
  const auto observe_buffer = [&]() -> Status {
    HYLE_RETURN_IF_ERROR(buffer.view().read(actual));
    return require(actual == expected, "public device Buffer IO changed bytes outside its exact logical request");
  };
  // This whole owning allocation crosses two full staging chunks and has a
  // two-byte logical tail. Only the provider can prove its private padding.
  HYLE_RETURN_IF_ERROR(buffer.view().fill(std::byte{0x31}));
  HYLE_RETURN_IF_ERROR(observe_buffer());
  for (std::size_t i = 0; i < expected.size(); ++i)
    expected[i] = static_cast<std::byte>((i * 37u + i / 1024u + 11u) & 255u);
  HYLE_RETURN_IF_ERROR(buffer.view().write(expected));
  HYLE_RETURN_IF_ERROR(observe_buffer());

  HYLE_ASSIGN_OR_RETURN(auto aligned, buffer.view().subview(4, 12));
  const std::array<std::byte, 12> pattern{std::byte{0xE1}, std::byte{0x12}, std::byte{0xD3}, std::byte{0x24},
      std::byte{0xC5}, std::byte{0x36}, std::byte{0xB7}, std::byte{0x48},
      std::byte{0xA9}, std::byte{0x5A}, std::byte{0x9B}, std::byte{0x6C}};
  HYLE_RETURN_IF_ERROR(aligned.write(pattern));
  std::copy(pattern.begin(), pattern.end(), expected.begin() + 4);
  HYLE_RETURN_IF_ERROR(observe_buffer());
  HYLE_RETURN_IF_ERROR(aligned.fill(std::byte{0x5D}));
  std::fill_n(expected.begin() + 4, 12, std::byte{0x5D});
  HYLE_RETURN_IF_ERROR(observe_buffer());

  HYLE_ASSIGN_OR_RETURN(auto partial, buffer.view().subview(1, 3));
  const std::array<std::byte, 3> forbidden{std::byte{0xF1}, std::byte{0xF2}, std::byte{0xF3}};
  std::array<std::byte, 3> denied_read{std::byte{0xA7}, std::byte{0xA7}, std::byte{0xA7}};
  const auto untouched_read = denied_read;
  HYLE_RETURN_IF_ERROR(require(partial.write(forbidden).IsUnimplemented(),
      "unaligned partial device write acquired an unproved read/modify/write permission"));
  HYLE_RETURN_IF_ERROR(require(partial.fill(std::byte{0xF4}).IsUnimplemented(),
      "unaligned partial device fill acquired an unproved read/modify/write permission"));
  HYLE_RETURN_IF_ERROR(require(partial.read(denied_read).IsUnimplemented() && denied_read == untouched_read,
      "unaligned partial device read changed its caller destination or was unexpectedly admitted"));
  HYLE_RETURN_IF_ERROR(observe_buffer());
  // A rejected range is a local capability failure, not a failed native
  // controller. Subsequent ordinary aligned operations must still work.
  HYLE_RETURN_IF_ERROR(aligned.fill(std::byte{0x44}));
  std::fill_n(expected.begin() + 4, 12, std::byte{0x44});
  HYLE_RETURN_IF_ERROR(observe_buffer());
  HYLE_RETURN_IF_ERROR(aligned.write(pattern));
  std::copy(pattern.begin(), pattern.end(), expected.begin() + 4);
  std::array<std::byte, 12> aligned_read{};
  HYLE_RETURN_IF_ERROR(aligned.read(aligned_read));
  HYLE_RETURN_IF_ERROR(require(aligned_read == pattern, "aligned device IO stayed poisoned after a rejected partial range"));
  HYLE_RETURN_IF_ERROR(observe_buffer());
  HYLE_ASSIGN_OR_RETURN(auto empty, buffer.view().subview(1, 0));
  HYLE_RETURN_IF_ERROR(empty.write(std::span<const std::byte>{}));
  HYLE_RETURN_IF_ERROR(empty.read(std::span<std::byte>{}));
  HYLE_RETURN_IF_ERROR(empty.fill(std::byte{0xFF}));
  HYLE_RETURN_IF_ERROR(observe_buffer());

  // A standalone BF16-like scalar owns its two logical bytes. Its private
  // four-byte Vulkan transfer tail never becomes an observable Mem subview.
  HYLE_ASSIGN_OR_RETURN(auto scalar, hyle::mem::Buffer::create(device.ref(), 2, 4));
  HYLE_RETURN_IF_ERROR(require(scalar.size() == 2 && !scalar.view().subview(0, 4),
      "two-byte scalar exposed its provider-private padded transfer capacity"));
  HYLE_RETURN_IF_ERROR(scalar.view().fill(std::byte{0x62}));
  std::array<std::byte, 2> scalar_read{};
  HYLE_RETURN_IF_ERROR(scalar.view().read(scalar_read));
  HYLE_RETURN_IF_ERROR(require(scalar_read == std::array<std::byte, 2>{std::byte{0x62}, std::byte{0x62}},
      "whole two-byte owning scalar fill/read differs"));
  const std::array<std::uint16_t, 1> bf16_word{0x3F80};
  HYLE_RETURN_IF_ERROR(scalar.view().write(std::as_bytes(std::span(bf16_word))));
  std::array<std::uint16_t, 1> actual_word{};
  HYLE_RETURN_IF_ERROR(scalar.view().read(std::as_writable_bytes(std::span(actual_word))));
  HYLE_RETURN_IF_ERROR(require(actual_word == bf16_word, "whole two-byte owning scalar write/read differs"));
  HYLE_ASSIGN_OR_RETURN(auto scalar_empty, scalar.view().subview(1, 0));
  HYLE_RETURN_IF_ERROR(scalar_empty.write(std::span<const std::byte>{}));
  HYLE_RETURN_IF_ERROR(scalar_empty.read(std::span<std::byte>{}));
  return scalar_empty.fill(std::byte{0xFF});
}

Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_RETURN_IF_ERROR(require(!context.environment().software_device, "retained-output consumer requires an actual Vulkan device"));
  std::printf("Vulkan device: %s; driver: %s\n", context.environment().device_name.c_str(), context.environment().driver_info.c_str());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto source, g::vulkan::execution_source(context));
  HYLE_RETURN_IF_ERROR(exercise(context, device, source, false));
  HYLE_RETURN_IF_ERROR(exercise(context, device, source, true));
  HYLE_RETURN_IF_ERROR(device_buffer_io(device));
  std::printf("PASS Vulkan 3 cases: F32 Neg/BF16 Cast retained readonly generations, canonical whole/slice/repeat root budgets and bounded spare reuse; device Buffer IO 2MiB+2/2B whole tails, aligned subranges, unaligned rejection/recovery and empty ranges\n");
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (status.ok()) return 0;
  std::fprintf(stderr, "Vulkan retained-output reuse: %.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
