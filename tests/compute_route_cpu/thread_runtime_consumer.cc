#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <latch>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/quota_allocator.h"

#if defined(HYLE_G10_TEST_VENDOR)
#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include "../../src/compute_provider_native/default_rules.h"
#endif

namespace {
namespace c = hyle::compute;
namespace cpu = c::cpu;
namespace mem = hyle::mem;

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

void check(hyle::Status status, std::string_view operation) {
  if (!status.ok())
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(status.message()));
}

template <class T> T take(hyle::Result<T> result) {
  if (!result)
    check(result.error(), "result");
  return std::move(result).value();
}

cpu::RuntimeThreadInfo info() { return take(cpu::runtime_thread_info()); }

void require_open() {
  const auto value = info();
  require(value.backend != cpu::RuntimeThreadBackend::kUnavailable,
          "this scenario requires an actual ATen thread runtime");
  require(value.configuration_open && !value.usage_started,
          "fresh process must have an unused startup window");
  require(!value.configured_intra_op_threads && !value.intra_op_limit &&
              !value.inter_op_limit,
          "open observation must leave limits unknown");
}

void require_configured(std::uint32_t threads, bool used) {
  const auto value = info();
  require(!value.configuration_open && value.usage_started == used,
          "configuration/first-use state is incorrect");
  require(value.configured_intra_op_threads == threads &&
              value.intra_op_limit == threads && value.inter_op_limit &&
              *value.inter_op_limit > 0,
          "configuration must match the actual caller limit");
}

template <class T> mem::Buffer buffer(std::span<const T> values) {
  auto result = take(mem::Buffer::create(mem::system_resource(),
                                        values.size_bytes(), 64));
  check(result.view().write(std::as_bytes(values)), "initialize input");
  return result;
}

// Real ordinary programs, independent output/workspace and checked completion.
// Shared runtime limits do not assert a total OS-thread cap across these callers.
void run_phase(c::Context &context, c::ConstTensorView input,
               std::span<const float> values, bool expect_aten) {
  auto queue = take(context.create_queue());
  for (const auto kind : {c::UnaryKind::kSilu, c::UnaryKind::kAbs}) {
    auto prepared = take(c::prepare(context, c::Unary{kind}, input.description()));
    if (kind == c::UnaryKind::kSilu && expect_aten) {
      require(prepared.provider().id == c::aten_cpu::kProviderId &&
                  prepared.runtime_properties().uses_private_threads,
              "ordinary SiLU must exercise the actual shared ATen runtime");
    }
    if (kind == c::UnaryKind::kAbs)
      require(prepared.provider().id == c::ggml_cpu::kProviderId,
              "ordinary Abs must exercise the independent ggml native budget");
    const auto desc = take(prepared.output_description());
    auto output = take(mem::Buffer::create(mem::system_resource(),
                                           take(desc.required_bytes()), 64));
    auto view = take(c::TensorView::create(desc, output.view()));
    const auto need = prepared.workspace_requirement();
    mem::Buffer workspace;
    if (need.size)
      workspace = take(mem::Buffer::create(mem::system_resource(), need.size,
                                           (std::max)(need.alignment, std::size_t{64})));
    auto invocation = take(prepared.create_invocation(workspace.view()));
    for (int replay = 0; replay < 3; ++replay) {
      {
        auto done = take(invocation.submit(queue, input, view));
        check(done.wait(), "phase completion");
      }
      std::vector<float> actual(values.size());
      check(output.view().read(std::as_writable_bytes(std::span(actual))),
            "read phase output");
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = values[i];
        const double expected = kind == c::UnaryKind::kAbs
                                    ? std::abs(x)
                                    : x / (1.0 + std::exp(-x));
        require(std::abs(actual[i] - expected) <= 2e-6 * (1.0 + std::abs(expected)),
                "phase numerical oracle failed");
      }
    }
  }
  check(queue.close(), "phase queue close");
  check(queue.drain(), "phase queue drain");
}

void startup() {
  require_open();
  require(cpu::configure_runtime_threads(0).IsInvalidArgument(), "zero is invalid");
  require(cpu::configure_runtime_threads((std::numeric_limits<std::uint32_t>::max)())
              .IsInvalidArgument(),
          "out-of-int-range count is invalid");
  require_open();
  check(cpu::configure_runtime_threads(2), "startup configure");
  require_configured(2, false);
  check(cpu::configure_runtime_threads(2), "same-value startup request");
  require(cpu::configure_runtime_threads(1).IsFailedPrecondition(),
          "conflicting startup request must not change the runtime");

  auto first = take(cpu::create_context({.native_threads = 1}));
  auto second = take(cpu::create_context({.native_threads = 2}));
  const auto desc = take(c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 1024}));
  std::vector<float> values(64 * 1024);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 32.0F;
  auto input_buffer = buffer<float>(values);
  const auto input = take(c::ConstTensorView::create(desc, input_buffer.view()));
  std::latch start(1);
  std::array<std::exception_ptr, 2> failures{};
  const auto run = [&](std::size_t index, c::Context &context) {
    start.wait();
    try {
      require_configured(2, true);
      run_phase(context, input, values, true);
    } catch (...) {
      failures[index] = std::current_exception();
    }
  };
  std::jthread a([&] { run(0, first); });
  std::jthread b;
  try {
    b = std::jthread([&] { run(1, second); });
  } catch (...) {
    start.count_down();
    throw;
  }
  start.count_down();
  a.join();
  b.join();
  for (const auto &failure : failures)
    if (failure)
      std::rethrow_exception(failure);
  check(first.close(), "first context close");
  check(second.close(), "second context close");
  check(first.drain(), "first context drain");
  check(second.drain(), "second context drain");
  first = {};
  second = {};
  require_configured(2, true);
  check(cpu::configure_runtime_threads(2), "same-value after use");
  require(cpu::configure_runtime_threads(1).IsFailedPrecondition(),
          "context destruction must not reopen configuration");
}

void query_first() {
  require_open();
  require_open();
  hyle::Status configured;
  std::jthread other([&] { configured = cpu::configure_runtime_threads(2); });
  other.join();
  check(configured, "configuration on a different startup caller");
  // If the open query initialized this caller's old OpenMP TLS, this reports
  // the stale limit instead of the configuration published by the other caller.
  require_configured(2, false);
}

void late_context(bool aten) {
  require_open();
  {
    auto context = take(c::Context::create(
        cpu::device(), aten ? c::aten_cpu::provider() : c::native::provider()));
    require(cpu::configure_runtime_threads(2).IsFailedPrecondition(),
            "expert provider first use must close configuration");
    if (!aten) {
      const auto desc = take(c::TensorDesc::dense(
          c::DataType::kBFloat16, std::array<std::int64_t, 2>{1, 128}));
      const std::array descriptions{desc};
      auto prepared = take(c::prepare(
          context, c::model_ops::block_quantization_roundtrip(
                       c::model_ops::BlockQuantizationFormat::kFloat8E4M3Block128),
          descriptions, descriptions, mem::system_resource()));
      require(prepared.provider().id == c::native::kProviderId &&
                  prepared.runtime_properties().uses_private_threads,
              "Native low-precision bypass must exercise its original runtime");
      const std::vector<std::uint16_t> values(128, 0x3f80);
      auto input = buffer<std::uint16_t>(values);
      auto output = buffer<std::uint16_t>(std::vector<std::uint16_t>(128));
      const auto x = take(c::ConstTensorView::create(desc, input.view()));
      const auto y = take(c::TensorView::create(desc, output.view()));
      auto queue = take(context.create_queue());
      {
        auto invocation = take(prepared.create_invocation());
        for (int replay = 0; replay < 3; ++replay) {
          auto done = take(invocation.submit(queue, x, y));
          check(done.wait(), "Native low-precision completion");
          std::array<std::uint16_t, 128> actual{};
          check(output.view().read(std::as_writable_bytes(std::span(actual))),
                "Native low-precision output");
          require(std::ranges::all_of(actual, [](auto value) { return value == 0x3f80; }),
                  "Native low-precision numerical oracle failed");
        }
      }
      check(queue.close(), "Native queue close");
      check(queue.drain(), "Native queue drain");
    }
    check(context.close(), "expert context close");
    check(context.drain(), "expert context drain");
  }
  const auto value = info();
  require(!value.configuration_open && value.usage_started &&
              !value.configured_intra_op_threads && value.intra_op_limit,
          "unconfigured first use must remain distinguishable after destruction");
  require(cpu::configure_runtime_threads(1).IsFailedPrecondition(),
          "late request must remain refused after all contexts are gone");
}

void race(bool first_use) {
  require_open();
  std::latch start(1);
  std::array<hyle::Status, 2> statuses;
  c::Context context;
  std::jthread a([&] {
    start.wait();
    statuses[0] = cpu::configure_runtime_threads(2);
  });
  std::jthread b;
  try {
    b = std::jthread([&] {
      start.wait();
      if (first_use) {
        auto result = c::Context::create(cpu::device(), c::native::provider());
        if (result)
          context = std::move(*result);
        else
          statuses[1] = result.error();
      } else {
        statuses[1] = cpu::configure_runtime_threads(1);
      }
    });
  } catch (...) {
    start.count_down();
    throw;
  }
  start.count_down();
  a.join();
  b.join();
  const auto value = info();
  require(!value.configuration_open, "racing startup must finish closed");
  if (first_use) {
    check(statuses[1], "racing first use");
    require(context.valid() && value.usage_started, "racing context was not created");
    require(statuses[0].ok() || statuses[0].IsFailedPrecondition(),
            "configure/first-use race returned an unrelated error");
    require(value.configured_intra_op_threads.has_value() == statuses[0].ok(),
            "only a successful configuration may publish a requested value");
    if (statuses[0].ok())
      require(value.configured_intra_op_threads == 2 && value.intra_op_limit == 2,
              "winning configuration was not applied");
    check(context.close(), "racing context close");
    check(context.drain(), "racing context drain");
  } else {
    require(statuses[0].ok() != statuses[1].ok(),
            "exactly one conflicting request must succeed");
    const auto loser = statuses[0].ok() ? statuses[1] : statuses[0];
    require(loser.IsFailedPrecondition(), "losing request must be refused");
    const std::uint32_t winner = statuses[0].ok() ? 2u : 1u;
    require_configured(winner, false);
  }
}

void metadata_failure() {
  require_open();
  mem::QuotaAllocator quota(mem::system_allocator(), 0);
  auto failed = cpu::create_context({.metadata_allocator = quota.get()});
  require(!failed && failed.error().IsResourceExhausted(),
          "the ordinary factory must exercise its outer metadata OOM");
  const auto value = info();
  require(!value.configuration_open && value.usage_started &&
              !value.configured_intra_op_threads,
          "ordinary Context allocation failure must preserve the closed window");
  require(cpu::configure_runtime_threads(1).IsFailedPrecondition(),
          "failed Context attempt must not reopen configuration");
}

void disabled() {
  const auto value = info();
  require(value.backend == cpu::RuntimeThreadBackend::kUnavailable &&
              !value.configuration_open && !value.usage_started &&
              !value.configured_intra_op_threads && !value.intra_op_limit &&
              !value.inter_op_limit,
          "disabled runtime must report absence rather than zero limits");
  require(cpu::configure_runtime_threads(2).IsUnimplemented(),
          "disabled runtime must refuse valid configuration honestly");
  require(cpu::configure_runtime_threads(0).IsInvalidArgument(),
          "disabled runtime must still reject an invalid count");
  auto context = take(cpu::create_context({.native_threads = 2}));
  const auto desc = take(c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  const std::vector<float> values(64, -1.0F);
  auto input = buffer<float>(values);
  const auto x = take(c::ConstTensorView::create(desc, input.view()));
  // SiLU is not part of the feature-OFF claim; the original ggml budget still works.
  auto prepared = take(c::prepare(context, c::Unary{c::UnaryKind::kAbs}, desc));
  auto output = buffer<float>(std::vector<float>(64));
  const auto y = take(c::TensorView::create(desc, output.view()));
  auto queue = take(context.create_queue());
  const auto need = prepared.workspace_requirement();
  mem::Buffer workspace;
  if (need.size)
    workspace = take(mem::Buffer::create(mem::system_resource(), need.size,
                                         (std::max)(need.alignment, std::size_t{64})));
  {
    auto invocation = take(prepared.create_invocation(workspace.view()));
    auto done = take(invocation.submit(queue, x, y));
    check(done.wait(), "feature-OFF ordinary completion");
    std::array<float, 64> actual{};
    check(output.view().read(std::as_writable_bytes(std::span(actual))),
          "feature-OFF output");
    require(std::ranges::all_of(actual, [](float v) { return v == 1.0F; }),
            "feature-OFF numerical oracle failed");
  }
  check(queue.close(), "feature-OFF queue close");
  check(queue.drain(), "feature-OFF queue drain");
  check(context.close(), "feature-OFF context close");
  check(context.drain(), "feature-OFF context drain");
  require(info().backend == cpu::RuntimeThreadBackend::kUnavailable,
          "ordinary use must not invent an unavailable runtime");
}

#if defined(HYLE_G10_TEST_VENDOR)
void foreign_runtime() {
  require_open();
  if (info().backend == cpu::RuntimeThreadBackend::kOpenMP) {
    check(cpu::configure_runtime_threads(2), "OpenMP startup configuration");
    require(!c::native::internal::qualified_default_environment(),
            "explicit non-one runtime must not inherit the old default recipe");
    at::set_num_threads(1);
    const auto changed = info();
    require(changed.configured_intra_op_threads == 2 && changed.intra_op_limit == 1,
            "foreign drift must remain observable without repairing it");
    require(cpu::configure_runtime_threads(2).IsFailedPrecondition(),
            "same-value request must detect drift instead of invoking a setter");
    require(at::get_num_threads() == 1,
            "a refused request must preserve the actual foreign setting");
  } else {
    // Native's original singleton pool cannot resize after actual ATen math.
    // Its void setter can only warn; HYLE must not publish a false success.
    at::set_num_threads(2);
    const auto x = at::ones({512, 512});
    const auto y = at::matmul(x, x);
    require(y.sum().item<float>() == 512.0F * 512.0F * 512.0F,
            "foreign Native math oracle failed");
    require_open();
    require(cpu::configure_runtime_threads(1).IsFailedPrecondition(),
            "a consumed Native pool must not claim a successful resize");
    const auto value = info();
    require(!value.configuration_open && !value.usage_started &&
                !value.configured_intra_op_threads && value.intra_op_limit == 2,
            "failed foreign setter must close the window without publishing requested");
    require(cpu::configure_runtime_threads(2).IsFailedPrecondition(),
            "partial setter failure must not reopen the configuration window");
  }
  require(!c::native::internal::qualified_default_environment(),
          "failed/drifted runtime must not qualify the old default recipe");
}

void qualified_drift(bool require_eligible) {
  require_open();
  require(info().backend == cpu::RuntimeThreadBackend::kOpenMP,
          "qualified drift requires the original OpenMP setting mechanism");
  check(cpu::configure_runtime_threads(1), "one-thread configuration");
  const bool eligible = c::native::internal::qualified_default_environment();
  std::printf("initial_default_recipe_eligible=%u\n", static_cast<unsigned>(eligible));
  require(!require_eligible || eligible,
          "this host/placement must actually exercise the original qualified recipe");
  auto context = take(cpu::create_context());
  const auto desc = take(c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 2>{32, 2048}));
  constexpr std::uint64_t recipe_namespace = 0x48594c4543505552ULL;
  auto before = take(c::prepare(context, c::Binary{c::BinaryKind::kAdd}, desc, desc));
  require(!eligible || (before.provider().id == c::native::kProviderId &&
                           before.route_policy().namespace_id == recipe_namespace),
          "qualified Context must first exercise a cached Native recipe");
  at::set_num_threads(2);
  const auto value = info();
  require(value.configured_intra_op_threads == 1 && value.intra_op_limit == 2,
          "foreign drift must be detected before new preparation");
  auto after = take(c::prepare(context, c::Binary{c::BinaryKind::kAdd}, desc, desc));
  require(after.route_policy().namespace_id != recipe_namespace,
          "an existing Context must not publish a stale qualified recipe after drift");
  check(context.close(), "qualified Context close");
  check(context.drain(), "qualified Context drain");
}
#endif
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view mode = argc == 2 ? argv[1] : "--startup";
    if (mode == "--startup") startup();
    else if (mode == "--query-first") query_first();
    else if (mode == "--late-native") late_context(false);
    else if (mode == "--late-aten") late_context(true);
    else if (mode == "--race-use") race(true);
    else if (mode == "--race-config") race(false);
    else if (mode == "--metadata-oom") metadata_failure();
    else if (mode == "--disabled") disabled();
#if defined(HYLE_G10_TEST_VENDOR)
    else if (mode == "--foreign") foreign_runtime();
    else if (mode == "--qualified-drift") {
      if (info().backend != cpu::RuntimeThreadBackend::kOpenMP) {
        std::puts("NOT_APPLICABLE: foreign reconfiguration drift requires OpenMP; Native late-setter refusal has its own scenario");
        return 77;
      }
      qualified_drift(false);
    }
    else if (mode == "--qualified-drift-required") qualified_drift(true);
#endif
    else throw std::runtime_error("unknown runtime consumer scenario");
    const auto value = info();
    std::printf("PASS CPU runtime %.*s backend=%u configured=%u observed=%u used=%u\n",
                static_cast<int>(mode.size()), mode.data(),
                static_cast<unsigned>(value.backend),
                value.configured_intra_op_threads.value_or(0),
                value.intra_op_limit.value_or(0), static_cast<unsigned>(value.usage_started));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL CPU runtime: %s\n", error.what());
    return 1;
  }
}
