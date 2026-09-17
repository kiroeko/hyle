#pragma once

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_ROUTE_APPLE_API
#elif defined(HYLE_COMPUTE_ROUTE_APPLE_BUILD)
#define HYLE_COMPUTE_ROUTE_APPLE_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_ROUTE_APPLE_API HYLE_IMPORT
#endif

#if defined(HYLE_COMPUTE_APPLE)
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <mlx/array.h>
#include <mlx/stream.h>

#include "hyle/compute/operation.h"
#include "hyle/compute/result.h"
#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::apple {
namespace detail {
struct Runtime;
struct ProgramState;
struct InvocationState;
struct ResultValues;
} // namespace detail

// MLX types are intentional in this native route header, never in the common
// Compute/Graph contract. A region is pure, uses the supplied GPU stream, and
// captures only immutable, successfully evaluated constants. Do not change
// MLX process-wide compiler settings or invoke this Context from its callable.
using RegionFunction = std::function<Result<std::vector<mlx::core::array>>(
    const std::vector<mlx::core::array> &, mlx::core::Stream)>;

// Shape/dtype only; Graph and other metadata-driven consumers do not need a
// dummy host payload. Preparation creates its own isolated native inputs.
struct InputSpec final {
  mlx::core::Shape shape;
  mlx::core::Dtype dtype = mlx::core::float32;
};

// Captured once when creating the native Context. Apple driver/runtime is
// shipped with macOS; its OS build is distinct from the compiled SDK/Metal
// compiler and deployment floor. Fixed implementation rules may match these
// facts; unknown combinations retain the legal MLX baseline, not a SOTA claim.
struct Environment final {
  std::string device_name;
  std::string architecture;
  std::string os_runtime;
  std::string mlx_version;
  std::string mlx_revision;
  std::string metal_compiler;
  std::uint32_t build_sdk = 0;
  std::uint32_t deployment_target = 0;
};

// MLX affine Q4: eight unsigned nibbles, lowest logical K in the low nibble
// of each uint32 word, physical [N,K/8]. Scales/biases are separate F32 arrays
// [N,K/64], with decoded weight q*scale+bias. This is not Marlin/GGUF packing.
inline constexpr StorageFormatId kAffineQ4Format{0x48594c454150504cULL,
                                                 0x4d4c585134000001ULL};
HYLE_COMPUTE_ROUTE_APPLE_API Result<TensorDesc>
affine_q4_description(std::int64_t output_channels,
                      std::int64_t input_channels) noexcept;

// Cold typed lowering, useful inside a native region and shared by Graph.
// Fixed MLX rules cover the documented typed tensor, floating arithmetic,
// gate/norm and F32 MatMul/affine Q4 domains. Native arrays own their layouts;
// this is not a Mem buffer adapter. Unsupported domains return an error.
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::vector<mlx::core::array>>
lower(const Operation &operation, std::span<const mlx::core::array> inputs,
      mlx::core::Stream stream) noexcept;

class Program;
class Invocation;

// Original MLX owners: no shadow output allocation or payload copy. Available
// only through a successful OwnedResult lease. Do not mutate aliased native
// arrays while a Program, invocation or another result still uses them.
class Results final {
public:
  Results() noexcept = default;
  Results(const Results &) = delete;
  Results &operator=(const Results &) = delete;
  Results(Results &&) noexcept = default;
  Results &operator=(Results &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_APPLE_API
  std::span<const mlx::core::array> arrays() const noexcept;

private:
  friend class Invocation;
  explicit Results(std::shared_ptr<detail::ResultValues> values) noexcept
      : values_(std::move(values)) {}
  std::shared_ptr<detail::ResultValues> values_;
};

// One owned submission thread and one native GPU stream. All compile, eval
// and checked synchronization occur there. Calls on separate handles may run
// concurrently; one Context executes admitted regions in FIFO order.
class Context final {
public:
  Context() noexcept = default;
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_APPLE_API Context &operator=(Context &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~Context();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Context> create() noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  // Borrowed immutable facts, valid while this Context retains its state.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<const Environment &>
  environment() const noexcept;

  // Cold compile and isolated zero-input warmup for these exact shapes/dtypes.
  // The callable runs only while preparing a MLX compiled region; this is a
  // managed replay contract, not allocation-free recorded commands.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<Program>
  compile(RegionFunction region,
          std::span<const mlx::core::array> input_examples) const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<Program>
  compile(RegionFunction region,
          std::span<const InputSpec> inputs) const noexcept;

  // Natural single-operation preparation; returns the same native Program
  // as a whole-region compile. Examples must already be successfully ready.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<Program>
  prepare(const Operation &operation,
          std::span<const mlx::core::array> input_examples) const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<Program>
  prepare(const Operation &operation,
          std::span<const InputSpec> inputs) const noexcept;
  template <class Descriptor>
    requires(!std::same_as<Descriptor, Operation>) &&
            std::constructible_from<Operation, Descriptor>
  Result<Program>
  prepare(Descriptor operation,
          std::span<const mlx::core::array> inputs) const noexcept;
  template <class Descriptor>
    requires(!std::same_as<Descriptor, Operation>) &&
            std::constructible_from<Operation, Descriptor>
  Result<Program> prepare(Descriptor operation,
                          std::span<const InputSpec> inputs) const noexcept;

  // Stops admission, drains every accepted prefix, joins the native owner.
  // Idempotent. Destruction performs this same explicit lifetime duty. Result
  // leases survive close; programs/slots reject new submissions afterwards.
  HYLE_COMPUTE_ROUTE_APPLE_API Status close() noexcept;

private:
  explicit Context(std::shared_ptr<detail::Runtime> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::Runtime> state_;
};

class Program final {
public:
  Program() noexcept = default;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_ROUTE_APPLE_API Result<Invocation>
  create_invocation(mem::AllocatorRef result_allocator =
                        mem::system_allocator()) const noexcept;

private:
  friend class Context;
  explicit Program(std::shared_ptr<detail::ProgramState> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::ProgramState> state_;
};

class Invocation final {
public:
  Invocation() noexcept = default;
  Invocation(const Invocation &) = delete;
  Invocation &operator=(const Invocation &) = delete;
  HYLE_COMPUTE_ROUTE_APPLE_API Invocation(Invocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Invocation &operator=(Invocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~Invocation();
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  // Ready, successfully evaluated arrays only. Pending inputs are rejected
  // without an implicit wait/copy. Retains native owners, not a borrowed span.
  // bind/submit are checked against in-flight use; moving a handle needs
  // external synchronization. No writer may mutate retained native payloads.
  HYLE_COMPUTE_ROUTE_APPLE_API Status
  bind(std::span<const mlx::core::array> inputs) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<OwnedResult<Results>> submit() noexcept;

private:
  friend class Program;
  explicit Invocation(detail::InvocationState *state) noexcept
      : state_(state) {}
  detail::InvocationState *state_ = nullptr;
};

template <class Descriptor>
  requires(!std::same_as<Descriptor, Operation>) &&
          std::constructible_from<Operation, Descriptor>
Result<Program>
Context::prepare(Descriptor operation,
                 std::span<const mlx::core::array> inputs) const noexcept {
  return prepare(Operation(std::move(operation)), inputs);
}
template <class Descriptor>
  requires(!std::same_as<Descriptor, Operation>) &&
          std::constructible_from<Operation, Descriptor>
Result<Program>
Context::prepare(Descriptor operation,
                 std::span<const InputSpec> inputs) const noexcept {
  return prepare(Operation(std::move(operation)), inputs);
}
} // namespace hyle::compute::apple
#endif
