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
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include <mlx/array.h>
#include <mlx/stream.h>

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
using RegionFunction = std::function<std::vector<mlx::core::array>(
    const std::vector<mlx::core::array> &, mlx::core::Stream)>;

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

  // Cold compile and isolated zero-input warmup for these exact shapes/dtypes.
  // The callable runs only while preparing a MLX compiled region; this is a
  // managed replay contract, not allocation-free recorded commands.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<Program>
  compile(RegionFunction region,
          std::span<const mlx::core::array> input_examples) const noexcept;

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
  HYLE_COMPUTE_ROUTE_APPLE_API Result<Invocation> create_invocation(
      mem::AllocatorRef result_allocator = mem::system_allocator()) const noexcept;

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
  explicit Invocation(detail::InvocationState *state) noexcept : state_(state) {}
  detail::InvocationState *state_ = nullptr;
};
} // namespace hyle::compute::apple
#endif
