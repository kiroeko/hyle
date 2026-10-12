#pragma once

#include "hyle/graph/executable.h"
#include "hyle/mem/std_allocator.h"

#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace hyle::graph {

// Cold concrete snapshot for native lowering. Tensor/node/resource identities
// remain those of the logical Graph. Shape expressions and symbolic Views use
// the common compiler's exact binding, guards and operation inference.
struct ExactGraph final {
  Graph graph;
  SpecializationSignature signature;
};

HYLE_GRAPH_API Result<ExactGraph>
specialize_exact(const Graph &graph, std::span<const ShapeBinding> shape = {},
                 mem::AllocatorRef allocator = mem::system_allocator()) noexcept;

namespace detail {
// Cold selection only: validates identity, independent bindings and guards.
HYLE_GRAPH_API Result<SpecializationSignature>
native_shape_signature(const Graph &, std::span<const ShapeBinding>,
                       mem::AllocatorRef) noexcept;
} // namespace detail

// A finite cold artifact table, not a runtime executor. A selected shared
// artifact creates its own ordinary native Invocation and independent slot.
// Missing points never trigger JIT, padding or implicit cache growth.
template <class Artifact> class ExactNativeVariants final {
  static_assert(std::is_nothrow_copy_constructible_v<Artifact>,
                "a native artifact must be a nonthrowing shared handle");
public:
  ExactNativeVariants() noexcept = default;

  std::size_t variant_count() const noexcept { return artifacts_.size(); }

  Result<SpecializationSignature> signature(std::size_t index) const noexcept {
    if (index >= signatures_.size())
      return std::unexpected(Status::OutOfRange("native variant index is out of range"));
    return signatures_[index];
  }

  // Cold, before create_invocation. Reset/submit retain the selected shape.
  Result<Artifact> select(std::span<const ShapeBinding> shape = {}) const noexcept {
    HYLE_ASSIGN_OR_RETURN(auto requested,
                          detail::native_shape_signature(graph_, shape, allocator_));
    for (std::size_t i = 0; i < signatures_.size(); ++i)
      if (requested == signatures_[i]) return artifacts_[i];
    return std::unexpected(Status::Unimplemented("shape is outside the prepared exact native variants"));
  }

  // Cold typed extension: prepare all concrete points before constructing any
  // native artifact. Allocator backing outlives the family, selected artifacts
  // and retained signatures. The factory receives an immutable concrete Graph with
  // original value identities; no callback is retained by this family.
  template <class Factory>
  static Result<ExactNativeVariants> prepare(
      const Graph &graph, std::span<const ExactShapeVariant> variants,
      Factory &&factory, mem::AllocatorRef allocator = mem::system_allocator()) noexcept {
    if (!graph || !allocator.valid() || variants.empty())
      return std::unexpected(Status::InvalidArgument("native variants require a graph, allocator and finite nonempty table"));
    try {
      ExactNativeVariants family(allocator);
      family.graph_ = graph;
      family.allocator_ = allocator;
      std::vector<Graph, mem::StdAllocator<Graph>> concrete{mem::StdAllocator<Graph>(allocator)};
      concrete.reserve(variants.size());
      family.signatures_.reserve(variants.size());
      family.artifacts_.reserve(variants.size());
      for (const auto &variant : variants) {
        HYLE_ASSIGN_OR_RETURN(auto point, specialize_exact(graph, variant.shape, allocator));
        for (const auto &signature : family.signatures_)
          if (signature == point.signature)
            return std::unexpected(Status::InvalidArgument("duplicate exact native variant"));
        family.signatures_.push_back(std::move(point.signature));
        concrete.push_back(std::move(point.graph));
      }
      for (const auto &point : concrete) {
        HYLE_ASSIGN_OR_RETURN(auto artifact, factory(point));
        family.artifacts_.push_back(std::move(artifact));
      }
      return family;
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("native variant table allocation failed"));
    } catch (...) {
      return std::unexpected(Status::Internal("native variant preparation failed"));
    }
  }

private:
  explicit ExactNativeVariants(mem::AllocatorRef allocator) noexcept
      : allocator_(allocator), signatures_(mem::StdAllocator<SpecializationSignature>(allocator)),
        artifacts_(mem::StdAllocator<Artifact>(allocator)) {}
  Graph graph_;
  mem::AllocatorRef allocator_ = mem::system_allocator();
  std::vector<SpecializationSignature, mem::StdAllocator<SpecializationSignature>> signatures_{
      mem::StdAllocator<SpecializationSignature>(mem::system_allocator())};
  std::vector<Artifact, mem::StdAllocator<Artifact>> artifacts_{
      mem::StdAllocator<Artifact>(mem::system_allocator())};
};

} // namespace hyle::graph
