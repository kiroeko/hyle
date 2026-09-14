#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/custom_kernel.h"

namespace custom_test {

using namespace hyle::compute;
using hyle::Result;
using hyle::Status;

inline constexpr ProviderIdentity identity{
    {0x435553544f4dULL, 1}, "owning-affine", "1"};
inline constexpr ProviderRuntimeProperties runtime{
    .replay_class = ProviderReplayClass::kPrepared, .blocks_host = true};

struct Affine final {
  static constexpr OperationId operation_id{0x435553544f4dULL, 2};
  static constexpr std::string_view operation_name = "affine-test";
  float bias = 0;
  OperationTraits operation_traits() const noexcept {
    return {
        .inputs = {1, 1}, .outputs = 1, .alias = AliasPolicy::kMayAliasInput0};
  }
  Status
  infer_output_descriptions(std::span<const TensorDesc> inputs,
                            std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1)
      return Status::InvalidArgument("affine arity");
    outputs[0] = inputs[0];
    return Status::Ok();
  }
};

enum class Fault {
  kNone,
  kUnsupported,
  kResourceDecline,
  kFatalUnsupported,
  kInvalidOutcome,
  kMissingKernel,
  kRuntimeMismatch,
  kInvocation,
  kRejected,
  kAcceptedError,
  kDelegatedProposal,
};

struct Probe final {
  std::atomic<int> configurations{0}, kernels{0}, invocations{0};
  std::atomic<int> proposals{0}, preparations{0}, executions{0};
  Fault fault = Fault::kNone;
  bool concurrent = true;
};

struct Config final {
  Probe *probe = nullptr;
  float scale = 2;
  explicit Config(Probe &p) noexcept : probe(&p) { ++probe->configurations; }
  ~Config() { --probe->configurations; }
};

struct alignas(64) Kernel final {
  const Config *config = nullptr;
  float bias = 0;
  Kernel(const Config &c, float b) noexcept : config(&c), bias(b) {
    ++c.probe->kernels;
  }
  Kernel(Kernel &&other) noexcept
      : config(std::exchange(other.config, nullptr)), bias(other.bias) {}
  ~Kernel() {
    if (config != nullptr)
      --config->probe->kernels;
  }

  struct Invocation final {
    Probe *probe;
    int runs = 0;
    explicit Invocation(Probe &p) noexcept : probe(&p) { ++p.invocations; }
    Invocation(Invocation &&other) noexcept
        : probe(std::exchange(other.probe, nullptr)), runs(other.runs) {}
    ~Invocation() {
      if (probe != nullptr)
        --probe->invocations;
    }
  };
  Result<Invocation> create_invocation(hyle::mem::BufferView,
                                       hyle::mem::AllocatorRef) const noexcept {
    if (config->probe->fault == Fault::kInvocation)
      return std::unexpected(
          Status::ResourceExhausted("custom invocation failure"));
    return Invocation(*config->probe);
  }
  AliasSupport alias_support(std::size_t output,
                             std::size_t input) const noexcept {
    return {output == 0 && input == 0, 0};
  }
  KernelCommandResult
  enqueue(Invocation &invocation, Queue &,
          std::span<const ConstTensorView> inputs,
          std::span<const TensorView> outputs) const noexcept {
    if (config->probe->fault == Fault::kRejected)
      return KernelCommandResult::rejected(
          Status::DataLoss("before acceptance"));
    auto source = inputs[0].storage().map_host();
    if (!source)
      return KernelCommandResult::rejected(std::move(source).error());
    auto destination =
        outputs[0].storage().map_host(hyle::mem::HostAccess::kWrite);
    if (!destination)
      return KernelCommandResult::rejected(std::move(destination).error());
    auto bytes = destination->writable_bytes();
    if (!bytes)
      return KernelCommandResult::rejected(std::move(bytes).error());
    const auto *in = reinterpret_cast<const float *>(source->bytes().data());
    auto *out = reinterpret_cast<float *>(bytes->data());
    const auto count = inputs[0].description().element_count().value();
    for (std::size_t index = 0; index < count; ++index)
      out[index] = in[index] * config->scale + bias;
    ++invocation.runs;
    ++config->probe->executions;
    auto done = destination->close();
    if (done.ok())
      done = source->close();
    if (config->probe->fault == Fault::kAcceptedError)
      done = Status::DataLoss("after acceptance");
    return KernelCommandResult::accepted_command(std::move(done));
  }
};

struct alignas(128) Factory final {
  std::unique_ptr<Config> config;
  Result<KernelProposal>
  propose(const KernelRequest &request, std::span<ResourceRequirement> inputs,
          std::span<ResourceRequirement> outputs) const noexcept {
    ++config->probe->proposals;
    if (request.device.backend_id() != cpu::kBackendId ||
        !request.operation.custom_as<Affine>())
      return std::unexpected(Status::Unimplemented("not an affine kernel"));
    if (request.inputs.size() != 1 || request.outputs.size() != 1 ||
        !request.inputs[0].is_dense() ||
        request.inputs[0].data_type() != DataType::kFloat32 ||
        request.inputs[0] != request.outputs[0])
      return std::unexpected(
          Status::Unimplemented("requires dense f32 tensors"));
    for (auto port : {inputs, outputs})
      port[0] = {.required_properties =
                     hyle::mem::MemoryProperty::kHostAccessible,
                 .minimum_alignment = alignof(float),
                 .exact_resource = hyle::mem::system_resource()};
    auto facts = runtime;
    facts.supports_concurrent_invocations = config->probe->concurrent;
    if (config->probe->fault == Fault::kDelegatedProposal)
      facts.mode = ProviderRuntimeMode::kDelegated;
    return KernelProposal{.route = 7,
                          .runtime = facts,
                          .cost = {.workspace = WorkspaceRequirement{},
                                   .persistent_bytes = 0,
                                   .copy_bytes = 0,
                                   .repack_bytes = 0,
                                   .synchronization_count = 0,
                                   .dispatch_count = 1}};
  }
  KernelPreparation prepare(const KernelPrepareRequest &request,
                            hyle::mem::AllocatorRef allocator) const noexcept {
    ++config->probe->preparations;
    const auto operation = request.operation.custom_as<Affine>();
    if (request.route != 7 || !operation ||
        request.resources.inputs[0] != hyle::mem::system_resource())
      return KernelPreparation::error(
          Status::Internal("route/resource changed"));
    auto properties =
        KernelProperties{.runtime = runtime, .persistent_bytes = 0};
    properties.runtime.supports_concurrent_invocations =
        config->probe->concurrent;
    if (config->probe->fault == Fault::kRuntimeMismatch)
      properties.runtime.replay_class = ProviderReplayClass::kManaged;
    auto kernel = make_custom_kernel(Kernel(*config, operation->bias),
                                     properties, allocator);
    if (!kernel)
      return KernelPreparation::error(std::move(kernel).error());
    auto result = KernelPreparation::prepared(std::move(*kernel));
    switch (config->probe->fault) {
    case Fault::kUnsupported:
      result.outcome = ProviderPrepareOutcome::unsupported(
          Status::Unimplemented("declined"));
      break;
    case Fault::kResourceDecline:
      result.outcome = ProviderPrepareOutcome::resource_declined(
          Status::ResourceExhausted("declined"));
      break;
    case Fault::kFatalUnsupported:
      result.outcome =
          ProviderPrepareOutcome::error(Status::Unimplemented("fatal"));
      break;
    case Fault::kInvalidOutcome:
      result.outcome.status = Status::DataLoss("invalid prepared status");
      break;
    case Fault::kMissingKernel:
      result.kernel.reset();
      break;
    default:
      break;
    }
    return result;
  }
};

inline Result<Provider> provider(Probe &probe,
                                 hyle::mem::AllocatorRef allocator = {}) {
  return make_custom_provider(identity, cpu::kBackendId,
                              Factory{std::make_unique<Config>(probe)},
                              allocator);
}

inline TensorDesc description() {
  return TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{4})
      .value();
}

struct Tensors final {
  hyle::mem::Buffer input =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64).value();
  hyle::mem::Buffer output =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64).value();
  ConstTensorView in =
      ConstTensorView::create(description(), input.view()).value();
  TensorView out = TensorView::create(description(), output.view()).value();
  Status fill() {
    const std::array<float, 4> values{1, 2, 3, 4};
    return input.view().write(std::as_bytes(std::span(values)));
  }
  std::array<float, 4> read() {
    std::array<float, 4> values{};
    output.view().read(std::as_writable_bytes(std::span(values))).IgnoreError();
    return values;
  }
};

} // namespace custom_test
