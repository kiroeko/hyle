#pragma once

// Qwen-private cold graph construction. This borrows the original geometry,
// precision, weight bindings and Backend. It owns no Model, Session, request,
// Invocation or execution policy; literal ownership moves to the prepared graph.
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "weights.h"
#include "precision_profile.h"
#include "../../inference/graph_bindings.h"
#include "hyle/compute/operation.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/graph/graph.h"

namespace qwen35 {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
using V = g::ValueId;
using D = g::Dimension;
using Type = c::DataType;
constexpr D f(std::int64_t value) noexcept { return D::fixed(value); }

struct PackedProjectionNodes final {
  std::array<g::NodeId, 4> chain;
  std::size_t count = 4;
};

struct GraphAssembly final {
  using Location = hyle::inference::GraphBindings::Location;
  struct Scope final {
    GraphAssembly &assembly;
    Location previous;
    Scope(GraphAssembly &selected, Location location);
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope();
  };

  std::vector<Tensor> literals;

  explicit GraphAssembly(const hyle::inference::Backend &environment, PrecisionProfile precision = {},
                         const c::ResourceBudget &budget = {});
  GraphAssembly(const GraphAssembly &) = delete;
  GraphAssembly &operator=(const GraphAssembly &) = delete;
  static Location primary();
  Location cpu() const;
  Tensor &literal_at(Location location, c::TensorDesc description);
  Tensor &literal(c::TensorDesc description);
  Type predicate_type() const;
  Type activation_type() const noexcept;
  void node(g::NodeId node);
  void external(V value, g::StorageDomainId storage = g::StorageDomainId{0});
  std::span<const g::ConstantBinding> constants() const noexcept;
  g::CompileOptions options(g::CompileOptions options) const;
  void bind_constant(V value, Tensor &tensor);

private:
  hyle::inference::GraphBindings bindings_;
  const hyle::inference::Backend &backend_;
  const c::ResourceBudget budget_;
  const PrecisionProfile precision_;
  const bool mixed_;
  const Type predicate_type_;
  Location location_ = primary();
};

struct Recipe final {
  g::GraphBuilder builder;
  std::vector<g::TensorSpec> specs;
  std::map<Tensor *, V> operands;
  std::vector<PackedProjectionNodes> packed_projection_nodes;
  std::map<std::string, LoadedWeight> &bank;
  const ProjectionBindings &projections;
  const WeightMeanings *meanings = nullptr;
  GraphAssembly &assembly;
  const ModelGeometry &geometry;
  const std::int64_t hidden;
  g::DimensionId last_row;
  D token_rows, packed_rows, sample_rows;
  Recipe(std::map<std::string, LoadedWeight> &selected, const ProjectionBindings &selected_projections,
         GraphAssembly &graph_assembly, const ModelGeometry &model_geometry,
         std::size_t max_query_rows, std::size_t requests = 1,
         const WeightMeanings *parameter_meanings = nullptr);
  static g::TensorSpec dense_spec(Type type, std::span<const D> dimensions);
  static g::TensorSpec dense_spec(Type type, std::initializer_list<D> dimensions);
  Type activation_type() const noexcept;
  const g::TensorSpec &spec(V value) const;
  V remember(V value, g::TensorSpec description);
  V input(g::TensorSpec description);
  V input(c::TensorDesc description);
  V constant(Tensor &tensor);
  V operand(Tensor &tensor);
  V weight_absolute(const std::string &name);
  V weight(const std::string &name);
  prepared::ParameterMeaning meaning(const std::string &name) const;
  V embedding(V tokens, const std::string &name);
  V scalar(float value, Type type = Type::kFloat32);
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments,
                    std::initializer_list<g::TensorSpec> outputs,
                    g::NodeId *node_id = nullptr);
  // Only concrete constant transforms use the Compute-only inference shortcut.
  // Symbolic operations declare their real logical outputs; each exact child
  // still runs the authoritative Compute inference during Graph compilation.
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments,
                    g::NodeId *node_id = nullptr);
  template<class O> V one(O operation, std::initializer_list<V> arguments,
                          g::NodeId *node_id = nullptr) {
    return op(c::Operation(operation), arguments, node_id).at(0);
  }
  template<class O> V one(O operation, std::initializer_list<V> arguments,
                          g::TensorSpec output, g::NodeId *node_id = nullptr) {
    return op(c::Operation(operation), arguments, {output}, node_id).at(0);
  }
  template<class O> V same(O operation, std::initializer_list<V> arguments, V source) {
    return one(operation, arguments, dense_spec(spec(source).data_type(), spec(source).dimensions()));
  }
  V cast(V x, Type type);
  V host_predicate(V value);
  V unary(c::UnaryKind kind, V x);
  V binary(c::BinaryKind kind, V x, V y);
  V reshape(V x, std::initializer_list<D> dimensions);
  V copy(V x);
  V concat(std::span<const V> inputs, int axis, g::TensorSpec output);
  V slice(V x, int axis, D first, D count);
  V linear(V x, const std::string &name);
  const ProjectionBinding &projection(const std::string &name) const;
  V emit_projection(V x, const ProjectionBinding &binding, bool scores);
  V linear_absolute(V x, const std::string &name);
  V scores(V x, const std::string &name);
  V norm_absolute(V x, const std::string &name);
  V norm(V x, const std::string &name);
  V decay(const std::string &name);
  V l2(V x);
  V gdn_key_rows(V rows);
  void publish(V value);
};

} // namespace qwen35
