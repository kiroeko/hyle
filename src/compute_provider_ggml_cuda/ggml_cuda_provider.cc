#include "program.h"
#include "hyle/core/environment.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <ggml-backend.h>
#include <ggml-backend-impl.h>
#include <ggml-alloc.h>
#include <ggml-cuda.h>
#include <ggml.h>
#include <ggml-impl.h>

#include "hyle/compute/detail/capture_bindings.h"
#include "hyle/compute/route_policy.h"
#include "hyle/container/small_vector.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/compute/model_ops.h"
#include "metadata_arena.h"
#include "operation_cold.h"
#include "index_guard.h"
#include "native_bridge.h"
#include "native_context.h"
#include "tensor_program.h"
#include "indexed_state.h"
#include "indexed_conv.h"
#include "sink_mask.h"

namespace hyle::compute::ggml_cuda {
namespace {

inline constexpr std::string_view kProviderVersion =
    HYLE_GGML_CUDA_PROVIDER_VERSION;

bool default_cublas_precision() noexcept {
#if HYLE_OS_WINDOWS
  std::size_t bytes = 0;
  return ::getenv_s(&bytes, nullptr, 0, "GGML_CUDA_CUBLAS_COMPUTE_TYPE") == 0 &&
         bytes == 0;
#else
  return std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") == nullptr;
#endif
}

struct Extension final {
  ProviderExtensionId id{};
  const void *table = nullptr;
};
struct Context final {
  CUcontext primary = nullptr;
  int device = -1;
  bool bfloat16_attention = false;
  Extension extension;
};
struct Program final {
  struct WorkspaceChunk final {
    std::size_t offset = 0;
    std::size_t bytes = 0;
  };
  struct IndexCheck final {
    std::uint32_t input = 0;
    std::size_t node = 0;
    std::size_t count = 0;
    std::int64_t extent = 0;
  };

  CUcontext primary = nullptr;
  int device = -1;
  internal::Recipe recipe;
  std::vector<std::size_t> alignments;
  std::vector<std::vector<float>> hadamards;
  std::vector<std::vector<std::int32_t>> snapshot_indices;
  std::vector<IndexCheck> index_checks;
  bool transpose_constant = false; // Only the cold constant factory sets it.
  std::vector<WorkspaceChunk> workspace_chunks{};
  WorkspaceRequirement workspace{};
  struct PackedNode final {
    std::size_t ordinal = 0;
    internal::PackedScratchPlan plan;
    std::uint64_t streams = 0;
  };
  struct PackedArena final { int stream = -1; std::size_t bytes = 0; };
  std::vector<PackedNode> packed_nodes{};
  std::vector<PackedArena> packed_arenas{};
};

// The allocator type is private planning state. Tensor buffers returned from
// it retain the canonical CUDA type/interface and never own caller payload.
// Keep the native context and sizing callbacks: changing the maximum size
// between measurement and creation would change the native chunk partition.
struct NativeWorkspace final : ggml_backend_buffer_type {
  int ordinal = -1;
  std::vector<Program::WorkspaceChunk> *measurement = nullptr;
  std::span<const Program::WorkspaceChunk> expected;
  std::byte *base = nullptr;
  std::size_t capacity = 0;
  std::size_t span = 0;
  std::size_t next = 0;
  bool layout_mismatch = false;

  void initialize(int device_ordinal) noexcept {
    ordinal = device_ordinal;
    static_cast<ggml_backend_buffer_type &>(*this) =
        *ggml_backend_cuda_buffer_type(device_ordinal);
    iface.alloc_buffer = &allocate;
  }

  static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type,
                                        std::size_t bytes) noexcept {
    auto &owner = *static_cast<NativeWorkspace *>(type);
    try {
      const auto alignment = ggml_backend_buft_get_alignment(type);
      if (alignment == 0 || !std::has_single_bit(alignment) ||
          owner.span > SIZE_MAX - (alignment - 1))
        return nullptr;
      const auto offset = (owner.span + alignment - 1) & ~(alignment - 1);
      if (bytes > SIZE_MAX - offset)
        return nullptr;
      if (!owner.measurement &&
          (owner.next >= owner.expected.size() ||
           owner.expected[owner.next].offset != offset ||
           owner.expected[owner.next].bytes != bytes ||
           offset > owner.capacity || bytes > owner.capacity - offset)) {
        owner.layout_mismatch = true;
        return nullptr;
      }
      std::unique_ptr<ggml_backend_buffer,
                      decltype(&ggml_backend_buffer_free)> buffer(
          bridge::binding_buffer(owner.ordinal, bytes),
          &ggml_backend_buffer_free);
      if (!buffer)
        return nullptr;
      if (owner.measurement)
        owner.measurement->push_back({offset, bytes});
      else
        bridge::bind_buffer(buffer.get(),
                            owner.base ? owner.base + offset : nullptr);
      owner.span = offset + bytes;
      ++owner.next;
      return buffer.release();
    } catch (...) {
      // Native reserve owns partial chunk cleanup only after this callback
      // returns. Never unwind through its C allocation loop.
      return nullptr;
    }
  }
};

// Device adapters resolve BufferView offsets before this comparison. Inspect
// only the actual used workspace prefix, not unused backing capacity. Equal
// allocation handles alone neither prove overlap nor cover external aliases.
bool overlaps_workspace(const void *workspace, std::size_t used,
                        const void *data, std::size_t bytes) noexcept {
  if (used == 0 || bytes == 0)
    return false;
  const auto scratch = reinterpret_cast<std::uintptr_t>(workspace);
  const auto port = reinterpret_cast<std::uintptr_t>(data);
  return scratch <= port ? port - scratch < used : scratch - port < bytes;
}

// Keep small attention nodes unchanged. Row-independent intermediates can
// reuse native liveness storage without changing the full KV reduction.
std::int64_t attention_query_block(const internal::Recipe &recipe,
                                   const internal::Node &node) noexcept {
  if (!node.operation.get_if<ScaledDotProductAttention>() &&
      !node.operation.custom_as<model_ops::SinkAttention>())
    return 0;
  const auto &query = recipe.values[node.inputs[0]];
  const auto q = query.dimensions();
  const auto k = recipe.values[node.inputs[1]].dimensions();
  if (q[1] <= 1)
    return 0;
  constexpr std::uint64_t threshold = 8ULL * 1024 * 1024;
  constexpr std::uint64_t budget = 64ULL * 1024 * 1024;
  // The admitted full score element count fits int32, so these products fit
  // uint64. Include scale/mask/softmax storage before liveness reuse.
  const auto original_row = 16ULL * static_cast<std::uint64_t>(q[0]) *
      static_cast<std::uint64_t>(q[2]) * static_cast<std::uint64_t>(k[1]);
  if (static_cast<std::uint64_t>(q[1]) <= threshold / original_row)
    return 0;
  const auto rows = std::max<std::uint64_t>(
      1, std::min<std::uint64_t>(128, budget / original_row));
  const auto queries = static_cast<std::uint64_t>(q[1]);
  if (rows >= queries)
    return 0;
  const auto depth = static_cast<std::uint64_t>(
      recipe.values[node.inputs[2]].dimensions()[3]);
  // Joining explicit block results retains an output frontier. Only select
  // it when even one saved F32 score plane covers two joined port payloads.
  if ((queries - rows) * static_cast<std::uint64_t>(k[1]) * sizeof(float) <=
      2 * queries * depth * data_type_size(query.data_type()))
    return 0;
  return static_cast<std::int64_t>(rows);
}

// Called only from catch blocks; native namespaces and numbers survive the
// private vendor exception boundary. No exception crosses a public callback.
Status failure() noexcept {
  try {
    throw;
  } catch (const bridge::NativeFailure<cudaError_t> &error) {
    return cuda::status(error.code);
  } catch (const bridge::NativeFailure<CUresult> &error) {
    return cuda::status(error.code);
  } catch (const bridge::NativeFailure<cublasStatus_t> &error) {
    const auto code = error.code == CUBLAS_STATUS_ALLOC_FAILED
                          ? StatusCode::ResourceExhausted
                          : StatusCode::Internal;
    return Status::FromNative(
        code,
        NativeError{NativeErrorDomain::Unknown,
                    static_cast<std::int64_t>(error.code)},
        error.operation);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("ggml CUDA host allocation failed");
  } catch (const std::exception &error) {
    return Status::Internal(error.what());
  } catch (...) {
    return Status::Internal("ggml CUDA unknown native failure");
  }
}

struct CheckedIndicesKey final {
  ggml_tensor *source = nullptr;
  std::size_t count = 0;
  std::int64_t extent = 0;
  bool operator==(const CheckedIndicesKey &) const = default;
};
struct CheckedIndicesHash final {
  std::size_t operator()(const CheckedIndicesKey &key) const noexcept {
    auto hash = std::hash<ggml_tensor *>{}(key.source);
    const auto combine = [&](std::size_t part) {
      hash ^= part + std::size_t{0x9e3779b9} + (hash << 6) + (hash >> 2);
    };
    combine(key.count);
    combine(std::hash<std::int64_t>{}(key.extent));
    return hash;
  }
};

struct Invocation final {
  CUcontext primary = nullptr;
  ggml_runtime::MetadataArena arena;
  ggml_backend *backend = nullptr;
  ggml_cgraph *graph = nullptr;
  // Only gallocr sees these dependency nodes. The encoder/capture graph and
  // its reader counts stay untouched; no additional payload is introduced.
  ggml_runtime::MetadataArena allocation_arena;
  ggml_cgraph allocation_graph{};
  using AllocationTensors = container::SmallVector<ggml_tensor *, 0, mem::AllocatorRef>;
  AllocationTensors allocation_nodes;
  AllocationTensors allocation_leafs;
  NativeWorkspace workspace;
  ggml_gallocr_t graph_allocator = nullptr;
  bool graph_bound = false;
  bool bindings_ready = false;
  std::vector<ggml_tensor *> tensors;
  std::vector<ggml_tensor *> roots;
  std::vector<ggml_backend_buffer *> buffers;
  std::vector<ggml_backend_buffer *> scratch;
  std::vector<internal::ScalarConstant> constants;
  std::vector<internal::DenseConstant> dense_constants;
  std::vector<ggml_tensor *> views;
  std::vector<void *> pointers;
  std::vector<ggml_tensor *> safe_indices;
  // A failed outer record must be ended/aborted before its private streams
  // can be observed. The original Queue Completion owns this invocation.
  cudaStream_t failed_recording_stream = nullptr;
  cuda::detail::NativeRetirement previous_retirement{};
  bool recording_guard = false;
  bool native_retirement_required = false;
  bool failed = false;
  void * packed_scratch = nullptr; // Metadata only; payload is caller workspace.

  Invocation() = default;
  Invocation(const Invocation &) = delete;
  Invocation &operator=(const Invocation &) = delete;
  ~Invocation() {
    cuda::detail::ScopedPrimaryContext scope(primary);
    if (backend)
      (void)bridge::drain(backend);
    // Completion retires every participating stream before this destruction.
    // Pool metadata must retire before the graph/backend and borrowed backing.
    internal::destroy_packed_scratch(packed_scratch);
    ggml_gallocr_free(graph_allocator);
    for (auto *buffer : buffers)
      ggml_backend_buffer_free(buffer);
    for (auto *buffer : scratch)
      ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
  }

  ggml_tensor *attention_bias(const internal::Node &node, ggml_tensor *scores,
                              std::int64_t begin = 0,
                              std::int64_t query_rows = 0) {
    const auto &operation = *node.operation.get_if<ScaledDotProductAttention>();
    if (operation.mask == MaskKind::kNone)
      return scores;
    auto *mask = tensors[node.inputs[3]];
    // Public [B,H,Q,S] broadcasting is already [S,Q,H,B] in ggml. Keep
    // singleton query rows broadcast; a tile of full Q retains its original
    // head/batch pitches, which the native binary kernel reads directly.
    if (query_rows != 0 && mask->ne[1] != 1)
      mask = ggml_view_4d(arena.get(), mask, mask->ne[0], query_rows,
          mask->ne[2], mask->ne[3], mask->nb[1], mask->nb[2], mask->nb[3],
          std::size_t(begin) * mask->nb[1]);
    // Scores are a fresh private intermediate with one local consumer.
    // The native ADD can reuse it without narrowing or copying the mask.
    return ggml_add_inplace(arena.get(), scores, mask);
  }

  ggml_tensor *attention(const internal::Node &node) {
    if (const auto sink = node.operation.custom_as<model_ops::SinkAttention>()) {
      auto *context = arena.get();
      auto *query = ggml_permute(context, tensors[node.inputs[0]], 0, 2, 1, 3);
      auto *key = ggml_permute(context, tensors[node.inputs[1]], 0, 2, 1, 3);
      auto *value = ggml_permute(context, tensors[node.inputs[2]], 0, 2, 1, 3);
      auto *scores = ggml_mul_mat(context, key, query);
      (void)ggml_prec_set_acc(scores, GGML_PREC_F32);
      (void)ggml_prec_set_src(scores, GGML_PREC_F32, 1);
      auto *probabilities = ggml_soft_max_ext(context, scores,
          ggml_runtime::sink_mask(context, tensors[node.inputs[3]]), sink->scale, 0.0F);
      ggml_soft_max_add_sinks(probabilities, tensors[node.inputs[4]]);
      value = ggml_cont(context, ggml_transpose(context, value));
      auto *result = ggml_mul_mat(context, value, probabilities);
      (void)ggml_prec_set_acc(result, GGML_PREC_F32);
      (void)ggml_prec_set_src(result, GGML_PREC_F32, 1);
      return ggml_cont(context, ggml_permute(context, result, 0, 2, 1, 3));
    }
    const auto &operation = *node.operation.get_if<ScaledDotProductAttention>();
    auto *context = arena.get();
    auto *query = tensors[node.inputs[0]];
    auto *key = tensors[node.inputs[1]];
    auto *value = tensors[node.inputs[2]];
    const auto type = query->type;
    // BSHD is [D,H,S,B] in ggml. Its attention/GQA inputs are [D,S,H,B].
    query = ggml_permute(context, ggml_cast(context, query, GGML_TYPE_F32), 0,
                         2, 1, 3);
    // This CUDA flash pin has Half PV accumulation even with PREC_F32, and
    // narrows BF16 K/V on SM89. The original F32 math path honors accumulation
    // and source precision; only the final output narrows to the port type.
    key = ggml_permute(context, ggml_cast(context, key, GGML_TYPE_F32), 0, 2, 1,
                       3);
    value = ggml_permute(context, ggml_cast(context, value, GGML_TYPE_F32), 0,
                         2, 1, 3);
    auto *scores = ggml_mul_mat(context, key, query);
    (void)ggml_prec_set_acc(scores, GGML_PREC_F32);
    (void)ggml_prec_set_src(scores, GGML_PREC_F32, 1);
    // Scale before masking: a zero or negative scale must not turn -inf
    // into NaN or +inf. Bottom-right causal offset is KV length - Q length.
    scores = ggml_scale(context, scores, operation.scale);
    scores = attention_bias(node, scores);
    if (operation.causal)
      scores = ggml_diag_mask_inf(context, scores,
                                  static_cast<int>(key->ne[1] - query->ne[1]));
    auto *probabilities = ggml_soft_max(context, scores);
    value = ggml_cont(context, ggml_transpose(context, value));
    auto *result = ggml_mul_mat(context, value, probabilities);
    (void)ggml_prec_set_acc(result, GGML_PREC_F32);
    (void)ggml_prec_set_src(result, GGML_PREC_F32, 1);
    return ggml_cast(context, ggml_permute(context, result, 0, 2, 1, 3), type);
  }

  ggml_tensor *tiled_attention(const internal::Node &node,
                               std::int64_t block_rows) {
    const auto *operation = node.operation.get_if<ScaledDotProductAttention>();
    const auto sink = node.operation.custom_as<model_ops::SinkAttention>();
    auto *context = arena.get();
    auto *source = tensors[node.inputs[0]];
    auto *key = tensors[node.inputs[1]];
    auto *value = tensors[node.inputs[2]];
    const auto type = source->type;
    const auto queries = source->ne[2];
    const auto keys = key->ne[2];
    const auto heads = source->ne[1], batch = source->ne[3];
    const auto depth = value->ne[0];
    std::vector<ggml_tensor *> parts;
    parts.reserve(static_cast<std::size_t>((queries - 1) / block_rows + 1));
    key = ggml_permute(context, ggml_cast(context, key, GGML_TYPE_F32),
                       0, 2, 1, 3);
    value = ggml_permute(context, ggml_cast(context, value, GGML_TYPE_F32),
                         0, 2, 1, 3);
    // Shared by every block in this single graph, not recopied per block.
    value = ggml_cont(context, ggml_transpose(context, value));
    for (std::int64_t begin = 0; begin < queries; begin += block_rows) {
      const auto rows = std::min(block_rows, queries - begin);
      auto *query = ggml_view_4d(context, source, source->ne[0], source->ne[1],
          rows, source->ne[3], source->nb[1], source->nb[2], source->nb[3],
          static_cast<std::size_t>(begin) * source->nb[2]);
      query = ggml_permute(context, ggml_cast(context, query, GGML_TYPE_F32),
                           0, 2, 1, 3);
      const auto past = static_cast<int>(keys - queries + begin);
      auto *scores = ggml_mul_mat(context, key, query);
      (void)ggml_prec_set_acc(scores, GGML_PREC_F32);
      (void)ggml_prec_set_src(scores, GGML_PREC_F32, 1);
      ggml_tensor *probabilities = nullptr;
      if (sink) {
        auto *mask = tensors[node.inputs[3]];
        mask = ggml_runtime::sink_mask_rows(context, mask, begin, rows);
        probabilities = ggml_soft_max_ext(context, scores, mask,
                                           sink->scale, 0.0F);
        ggml_soft_max_add_sinks(probabilities, tensors[node.inputs[4]]);
      } else {
        scores = ggml_scale(context, scores, operation->scale);
        scores = attention_bias(node, scores, begin, rows);
        if (operation->causal)
          scores = ggml_diag_mask_inf(context, scores, past);
        probabilities = ggml_soft_max(context, scores);
      }
      auto *result = ggml_mul_mat(context, value, probabilities);
      (void)ggml_prec_set_acc(result, GGML_PREC_F32);
      (void)ggml_prec_set_src(result, GGML_PREC_F32, 1);
      result = ggml_cast(context,
          ggml_permute(context, result, 0, 2, 1, 3), type);
      result = ggml_cont(context, result);
      parts.push_back(ggml_reshape_3d(context, result, depth * heads,
                                        rows, batch));
    }
    // Real native dependencies protect downstream nodes and published
    // intermediates. Balanced joins avoid retaining growing output prefixes.
    while (parts.size() > 1) {
      const auto count = parts.size();
      std::size_t next = 0;
      for (std::size_t i = 0; i + 1 < count; i += 2)
        parts[next++] = ggml_concat(context, parts[i], parts[i + 1], 1);
      if (count % 2 != 0)
        parts[next++] = parts.back();
      parts.resize(next);
    }
    return ggml_reshape_4d(context, parts.front(), depth, heads, queries, batch);
  }

  Status allocate(ggml_tensor *tensor, ggml_backend_buffer_type_t type,
                  bool metadata_only) {
    if (tensor->view_src) {
      views.push_back(tensor);
      return Status::Ok();
    }
    if (tensor->buffer)
      return Status::Ok(); // Already bound external tensor.
    const auto bytes = ggml_backend_buft_get_alloc_size(type, tensor);
    auto *buffer = metadata_only
        ? bridge::binding_buffer(workspace.ordinal, bytes)
        : ggml_backend_buft_alloc_buffer(type, bytes);
    if (!buffer)
      return Status::ResourceExhausted("ggml CUDA scratch allocation failed");
    scratch.push_back(buffer); // Capacity reserved before acquiring resources.
    if (metadata_only) {
      tensor->buffer = buffer;
      return Status::Ok();
    }
    if (ggml_backend_tensor_alloc(buffer, tensor,
                                  ggml_backend_buffer_get_base(buffer)) !=
        GGML_STATUS_SUCCESS)
      return Status::Internal("ggml CUDA scratch binding failed");
    return Status::Ok();
  }

  void bind_views() noexcept {
    for (auto *tensor : views) {
      const auto *source = tensor->view_src;
      tensor->buffer = source->buffer;
      tensor->data = source->data ? static_cast<std::byte *>(source->data) +
                                        tensor->view_offs
                                  : nullptr;
    }
  }

  void bind(std::uint32_t id, void *pointer) noexcept {
    bridge::bind_buffer(buffers[id], pointer);
    roots[id]->data = pointer;
  }

  ggml_cgraph *allocation_plan() noexcept {
    return allocation_nodes.empty() ? graph : &allocation_graph;
  }

  Status build_allocation_plan(mem::AllocatorRef allocator) {
    struct Dependency final {
      int until;
      ggml_tensor *tensor;
      std::size_t registration;
    };
    using Dependencies = std::vector<Dependency, mem::StdAllocator<Dependency>>;
    using TensorSet = std::unordered_set<ggml_tensor *, std::hash<ggml_tensor *>,
        std::equal_to<ggml_tensor *>, mem::StdAllocator<ggml_tensor *>>;
    struct Collection final {
      int node_count;
      Dependencies dependencies;
      bool invalid = false;
      Collection(int count, mem::AllocatorRef allocator)
          : node_count(count), dependencies(mem::StdAllocator<Dependency>{allocator}) {}
    };
    Collection collection{graph->n_nodes, allocator};
    bridge::collect_allocation_dependencies(
        backend, graph,
        [](void *state, ggml_tensor *tensor, int until_node) {
          auto &out = *static_cast<Collection *>(state);
          if (!tensor || tensor->view_src || until_node < 0 || until_node >= out.node_count) {
            out.invalid = true;
            return;
          }
          if (out.dependencies.size() == static_cast<std::size_t>(INT32_MAX))
            throw std::bad_alloc{};
          out.dependencies.push_back({until_node, tensor, out.dependencies.size()});
        },
        &collection);
    if (collection.invalid)
      return Status::Internal("ggml CUDA invalid allocation dependency");
    auto &deps = collection.dependencies;
    if (deps.empty())
      return Status::Ok();
    // In-place sort uses no hidden temporary allocator. Registration order
    // within an output does not depend on pointer order in the two builds.
    std::sort(deps.begin(), deps.end(), [](const auto &a, const auto &b) {
      return a.until != b.until ? a.until < b.until : a.registration < b.registration;
    });
    TensorSet seen{mem::StdAllocator<ggml_tensor *>{allocator}};
    int previous = -1;
    std::size_t unique = 0;
    for (const auto &dep : deps) {
      if (dep.until != previous) {
        seen.clear();
        previous = dep.until;
      }
      if (seen.insert(dep.tensor).second)
        deps[unique++] = dep;
    }
    deps.resize(unique);

    std::vector<ggml_tensor *, mem::StdAllocator<ggml_tensor *>> extra_roots{
        mem::StdAllocator<ggml_tensor *>{allocator}};
    seen.clear();
    // A NONE alias may hide its physical root from forward traversal. Count
    // that external leaf explicitly rather than relying on hash slack.
    for (const auto &dep : deps) {
      if (!ggml_hash_contains(&graph->visited_hash_set, dep.tensor) &&
          seen.insert(dep.tensor).second) {
        if (dep.tensor->op != GGML_OP_NONE)
          return Status::Internal("ggml CUDA allocation root outside graph");
        for (auto *source : dep.tensor->src)
          if (source)
            return Status::Internal("ggml CUDA external allocation root has sources");
        extra_roots.push_back(dep.tensor);
      }
    }
    std::size_t dep_nodes = 0;
    previous = -1;
    std::size_t group = 0;
    for (const auto &dep : deps) {
      if (dep.until != previous) {
        if (group != 0)
          dep_nodes += (group + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
        previous = dep.until;
        group = 0;
      }
      ++group;
    }
    dep_nodes += (group + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    const auto maximum = static_cast<std::size_t>(INT32_MAX);
    const auto leaf_count = static_cast<std::size_t>(graph->n_leafs) + extra_roots.size();
    if (leaf_count > maximum ||
        static_cast<std::size_t>(graph->n_nodes) > maximum - leaf_count ||
        dep_nodes > maximum - leaf_count - static_cast<std::size_t>(graph->n_nodes))
      return Status::OutOfRange("ggml CUDA allocation graph count overflows");
    const auto overhead = ggml_tensor_overhead();
    const auto byte_max = static_cast<std::size_t>(PTRDIFF_MAX) - (GGML_MEM_ALIGN - 1);
    if (overhead == 0 || dep_nodes > byte_max / overhead)
      return Status::OutOfRange("ggml CUDA allocation metadata overflows");
    HYLE_ASSIGN_OR_RETURN(auto metadata,
                          ggml_runtime::MetadataArena::create(dep_nodes * overhead, allocator));
    allocation_arena = std::move(metadata);
    allocation_nodes = AllocationTensors(allocator);
    allocation_leafs = AllocationTensors(allocator);
    if (!allocation_nodes.try_reserve(static_cast<std::size_t>(graph->n_nodes) + dep_nodes) ||
        !allocation_leafs.try_reserve(leaf_count))
      return Status::ResourceExhausted("ggml CUDA allocation graph metadata failed");
    // Initialize hidden physical roots before any original NONE aliases.
    for (auto *root : extra_roots)
      allocation_leafs.push_back(root);
    for (int i = 0; i < graph->n_leafs; ++i)
      allocation_leafs.push_back(graph->leafs[i]);
    std::size_t next = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
      allocation_nodes.push_back(graph->nodes[i]);
      while (next < deps.size() && deps[next].until == i) {
        auto *dep = ggml_view_tensor(allocation_arena.get(), deps[next].tensor);
        for (int source = 0; source < GGML_MAX_SRC && next < deps.size() && deps[next].until == i;
             ++source, ++next)
          dep->src[source] = deps[next].tensor;
        allocation_nodes.push_back(dep);
      }
    }
    // ggml-alloc only reads nodes/leafs and their counts. All other shallow
    // fields still belong to the original graph and are never mutated here.
    allocation_graph = *graph;
    allocation_graph.nodes = allocation_nodes.data();
    allocation_graph.n_nodes = static_cast<int>(allocation_nodes.size());
    allocation_graph.size = static_cast<int>(std::max(allocation_nodes.size(), allocation_leafs.size()));
    allocation_graph.leafs = allocation_leafs.data();
    allocation_graph.n_leafs = static_cast<int>(allocation_leafs.size());
    return Status::Ok();
  }

  Status finish_bind() {
    if (!graph_bound) {
      // The immutable graph was reserved cold. Real external roots are now
      // bound, so this only installs addresses in that existing allocation.
      if (!ggml_gallocr_alloc_graph(graph_allocator, allocation_plan()))
        return Status::ResourceExhausted("ggml CUDA workspace binding failed");
      graph_bound = true;
    }
    // External ports can change between replays; private addresses cannot.
    bind_views();
    return Status::Ok();
  }

  Status build(const Program &program, mem::AllocatorRef allocator,
                bool metadata_only) {
    primary = program.primary;
    workspace.initialize(program.device);
    cuda::detail::ScopedPrimaryContext scope(primary);
    HYLE_RETURN_IF_ERROR(scope.status());
    const auto &recipe = program.recipe;
    const auto count = recipe.values.size();
    constexpr auto maximum = static_cast<std::size_t>(INT32_MAX) - 16;
    if (count > maximum / 4 || recipe.nodes.size() > (maximum - count * 4) / 64)
      return Status::OutOfRange("ggml CUDA metadata count overflows");
    auto capacity = count * 4 + recipe.nodes.size() * 64;
    for (const auto &node : recipe.nodes) {
      if (const auto rows = attention_query_block(recipe, node)) {
        const auto queries = recipe.values[node.inputs[0]].dimensions()[1];
        const auto blocks = static_cast<std::size_t>((queries - 1) / rows + 1);
        // Includes per-block views/compute/Cont/reshape and balanced joins.
        if (blocks > (maximum - capacity) / 32)
          return Status::OutOfRange("ggml CUDA attention metadata count overflows");
        capacity += blocks * 32;
      }
      if (!node.operation.get_if<Concat>())
        continue;
      // A region can concatenate the same value many times. Count ports,
      // not just unique recipe values, before reserving the fixed arena.
      if (node.inputs.size() > (maximum - capacity) / 4)
        return Status::OutOfRange("ggml CUDA concat metadata count overflows");
      capacity += node.inputs.size() * 4;
    }
    capacity += 16;
    auto metadata =
        ggml_runtime::MetadataArena::for_graph(capacity, capacity, allocator);
    if (!metadata)
      return std::move(metadata).error();
    arena = std::move(*metadata);
    tensors.resize(count, nullptr);
    internal::TensorBuildFrame forms(recipe, allocator);
    using CheckedEntry = std::pair<const CheckedIndicesKey, ggml_tensor *>;
    std::unordered_map<CheckedIndicesKey, ggml_tensor *, CheckedIndicesHash,
        std::equal_to<CheckedIndicesKey>, mem::StdAllocator<CheckedEntry>> checked{
        0, CheckedIndicesHash{}, std::equal_to<CheckedIndicesKey>{},
        mem::StdAllocator<CheckedEntry>{allocator}};
    roots.resize(count, nullptr);
    buffers.resize(count, nullptr);
    pointers.resize(recipe.inputs.size() + recipe.outputs.size());
    safe_indices.reserve(program.index_checks.size());
    backend = ggml_backend_cuda_init(program.device);
    if (!backend)
      return Status::ResourceExhausted("ggml CUDA backend allocation failed");
    bridge::initialize_backend(backend);
    for (const auto id : recipe.inputs) {
      const auto &description = recipe.values[id];
      tensors[id] = internal::create_tensor_input(arena.get(), description);
      ggml_set_input(tensors[id]);
    }
    for (const auto &constant : recipe.constants) {
      tensors[constant.value] = internal::create_tensor_input(
          arena.get(), recipe.values[constant.value]);
      ggml_set_input(tensors[constant.value]);
    }
    std::size_t next_check = 0;
    for (std::size_t node_index = 0; node_index < recipe.nodes.size();
         ++node_index) {
      const auto &node = recipe.nodes[node_index];
      ggml_tensor *original_ids = nullptr;
      std::uint32_t index_id = 0;
      if (next_check < program.index_checks.size() &&
          program.index_checks[next_check].node == node_index) {
        const auto &check = program.index_checks[next_check++];
        index_id = recipe.inputs[check.input];
        original_ids = tensors[index_id];
        const CheckedIndicesKey key{original_ids, check.count, check.extent};
        auto found = checked.find(key);
        if (found == checked.end()) {
          auto *safe = internal::create_tensor_input(
              arena.get(), recipe.values[index_id]);
          ggml_set_input(safe);
          found = checked.emplace(key, safe).first;
        }
        auto *safe = found->second;
        safe_indices.push_back(safe);
        // Only this indexed consumer sees the safe IDs. Other consumers and
        // separately published IDs retain the original immutable input.
        tensors[index_id] = safe;
      }
      std::vector<ggml_tensor *> results;
      if (program.transpose_constant) {
        results.push_back(ggml_cont(arena.get(),
            ggml_transpose(arena.get(), tensors[node.inputs[0]])));
      } else if (node.operation.get_if<ScaledDotProductAttention>() ||
          node.operation.custom_as<model_ops::SinkAttention>()) {
        const auto rows = attention_query_block(recipe, node);
        if (auto *result = rows != 0
                ? tiled_attention(node, rows) : attention(node))
          results.push_back(result);
      } else {
        results = internal::lower_tensor_operation(
            arena.get(), recipe, node, tensors, forms, constants,
            dense_constants, program.hadamards[node_index],
            program.snapshot_indices.empty()
                ? std::span<const std::int32_t>{}
                : program.snapshot_indices[node_index]);
      }
      if (original_ids) {
        tensors[index_id] = original_ids;
      }
      if (results.size() != node.outputs.size())
        return Status::Unimplemented("ggml CUDA tensor lowering declined");
      for (std::size_t i = 0; i < results.size(); ++i) {
        if (!ggml_backend_supports_op(backend, results[i])) {
          char message[256];
          const auto *result = results[i];
          if (result->op == GGML_OP_ROPE) {
            const auto *data = result->src[0], *positions = result->src[1], *frequency = result->src[2];
            std::snprintf(message, sizeof(message),
                          "ggml CUDA native RoPE declined: %s value %u, X=%lld/%lld/%lld/%lld P=%lld/%lld F=%lld/%lld dense=%d/%d/%d/%d reserved=%d",
                          ggml_type_name(result->type), node.outputs[i],
                          static_cast<long long>(data->ne[0]), static_cast<long long>(data->ne[1]),
                          static_cast<long long>(data->ne[2]), static_cast<long long>(data->ne[3]),
                          static_cast<long long>(positions->ne[0]), static_cast<long long>(positions->ne[1]),
                          static_cast<long long>(frequency->ne[0]), static_cast<long long>(frequency->ne[1]),
                          ggml_is_contiguous(data), ggml_is_contiguous(positions),
                          ggml_is_contiguous(frequency), ggml_is_contiguous(result), result->op_params[1]);
          } else {
            std::snprintf(message, sizeof(message),
                          "ggml CUDA native operation declined: %s (%s), value %u",
                          ggml_op_name(result->op), ggml_type_name(result->type), node.outputs[i]);
          }
          return Status::Unimplemented(message);
        }
        tensors[node.outputs[i]] = results[i];
      }
    }
    for (std::size_t id = 0; id < count; ++id) {
      roots[id] = tensors[id];
      while (roots[id] && roots[id]->view_src)
        roots[id] = roots[id]->view_src;
    }
    graph = ggml_new_graph_custom(arena.get(), capacity, false);
    for (const auto id : recipe.outputs) {
      ggml_set_output(tensors[id]);
      // Reshape is a view: bind and protect the actual producer as well.
      // Native fusion must not consume a separately published intermediate.
      ggml_set_output(roots[id]);
      ggml_build_forward_expand(graph, tensors[id]);
    }
    for (const auto id : recipe.effect_roots)
      ggml_build_forward_expand(graph, tensors[id]);
    for (std::size_t id = 0; id < count; ++id) {
      if (!tensors[id])
        return Status::InvalidArgument("ggml CUDA recipe has an unused value");
      const auto bytes = ggml_nbytes(roots[id]);
      const bool external =
          std::ranges::find(recipe.inputs, id) != recipe.inputs.end() ||
          (program.index_checks.empty() &&
           std::ranges::find(recipe.outputs, id) != recipe.outputs.end()) ||
          std::ranges::any_of(recipe.constants, [id](const auto &constant) {
            return constant.value == id;
          });
      if (external) {
        buffers[id] = bridge::binding_buffer(program.device, bytes);
        // An idle external binding has no payload. Submission installs both
        // the buffer base and tensor address after provenance/size admission.
        roots[id]->buffer = buffers[id];
      }
    }
    for (const auto &constant : recipe.constants) {
      HYLE_ASSIGN_OR_RETURN(auto pointer, cuda::detail::const_pointer(
          constant.binding.storage(), program.alignments[constant.value], 1));
      bind(constant.value, const_cast<void *>(pointer));
    }
    const auto nodes = ggml_graph_n_nodes(graph);
    std::size_t resources = 0;
    for (auto *tensor = ggml_get_first_tensor(arena.get()); tensor;
         tensor = ggml_get_next_tensor(arena.get(), tensor))
      ++resources;
    scratch.reserve(resources);
    views.reserve(resources);
    auto *type = ggml_backend_cuda_buffer_type(program.device);
    // These inputs are written outside graph compute, or persist across
    // submissions. Keep their storage independent of temporary liveness.
    for (auto *tensor : safe_indices)
      HYLE_RETURN_IF_ERROR(allocate(tensor, type, metadata_only));
    for (const auto &constant : constants) {
      ggml_set_input(constant.tensor);
      HYLE_RETURN_IF_ERROR(allocate(constant.tensor, type, metadata_only));
    }
    for (const auto &constant : dense_constants) {
      ggml_set_input(constant.tensor);
      HYLE_RETURN_IF_ERROR(allocate(constant.tensor, type, metadata_only));
    }
    // Every native graph reuses dead temporaries, not just tiled attention.
    // External roots and published values retain their original ownership
    // and OUTPUT flags. Each invocation owns its own reserved workspace.
    for (auto *tensor = ggml_get_first_tensor(arena.get()); tensor;
         tensor = ggml_get_next_tensor(arena.get(), tensor))
      if (tensor->view_src)
        views.push_back(tensor);
    for (int i = 0; i < nodes; ++i) {
      auto *tensor = ggml_graph_node(graph, i);
      if (!ggml_backend_supports_op(backend, tensor))
        return Status::Unimplemented("ggml CUDA native graph node declined");
    }
    bind_views();
    HYLE_RETURN_IF_ERROR(build_allocation_plan(allocator));
    if (!metadata_only) {
      for (const auto &constant : constants)
        ggml_backend_tensor_set(constant.tensor, &constant.value, 0,
                                sizeof(constant.value));
      for (const auto &constant : dense_constants)
        ggml_backend_tensor_set(constant.tensor, constant.data, 0,
                                constant.bytes);
    }
    return Status::Ok();
  }

  Status reserve(ggml_backend_buffer_type_t type) {
    graph_allocator = ggml_gallocr_new(type);
    if (!graph_allocator || !ggml_gallocr_reserve(graph_allocator, allocation_plan()))
      return Status::ResourceExhausted("ggml CUDA workspace reservation failed");
    return Status::Ok();
  }

  Status measure_packed_scratch(Program &program) {
    std::vector<std::size_t> stream_bytes;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
      auto *node = ggml_graph_node(graph, i);
      const auto plan = internal::packed_mmq_node_plan(program.device, node);
      if (plan.route != internal::PackedMatmulRoute::kMmq)
        continue;
      const auto streams = internal::packed_mmq_stream_mask(backend, node);
      program.packed_nodes.push_back({static_cast<std::size_t>(i), plan, streams});
      // Native cold metadata supplies actual slots; root0 is the native
      // fallback if invalid concurrency metadata is discarded at encode.
      auto mask = streams;
      for (std::size_t slot = 0; mask; ++slot, mask >>= 1) {
        if (!(mask & 1)) continue;
        if (stream_bytes.size() <= slot) stream_bytes.resize(slot + 1);
        stream_bytes[slot] = std::max(stream_bytes[slot], plan.bytes);
      }
    }
    for (std::size_t slot = 0; slot < stream_bytes.size(); ++slot) {
      if (!stream_bytes[slot]) continue;
      auto *metadata = NativeWorkspace::allocate(&workspace, stream_bytes[slot]);
      if (!metadata)
        return Status::ResourceExhausted("ggml CUDA MMQ workspace measurement failed");
      ggml_backend_buffer_free(metadata);
      program.packed_arenas.push_back({static_cast<int>(slot), stream_bytes[slot]});
    }
    return Status::Ok();
  }

  Status bind_packed_scratch(const Program &program) {
    std::vector<internal::PackedScratchBinding> bindings;
    std::vector<internal::PackedScratchArena> arenas;
    bindings.reserve(program.packed_nodes.size());
    arenas.reserve(program.packed_arenas.size());
    for (const auto &entry : program.packed_nodes) {
      if (entry.ordinal >= static_cast<std::size_t>(ggml_graph_n_nodes(graph)))
        return Status::FailedPrecondition("ggml CUDA MMQ native node changed");
      auto *node = ggml_graph_node(graph, static_cast<int>(entry.ordinal));
      if (internal::packed_mmq_node_plan(program.device, node) != entry.plan ||
          internal::packed_mmq_stream_mask(backend, node) != entry.streams)
        return Status::FailedPrecondition("ggml CUDA MMQ native signature/streams changed");
      bindings.push_back({node, entry.plan, entry.streams});
    }
    for (const auto &entry : program.packed_arenas) {
      auto *metadata = NativeWorkspace::allocate(&workspace, entry.bytes);
      if (!metadata)
        return Status::ResourceExhausted("ggml CUDA MMQ workspace binding failed");
      ggml_backend_buffer_free(metadata);
      const auto offset = workspace.span - entry.bytes;
      arenas.push_back({entry.stream, workspace.base + offset, entry.bytes});
    }
    // Metadata only. Same-stream GPU FIFO/capture edges permit max-op arena
    // reuse; different native streams have disjoint caller-workspace claims.
    // Node pointers stay stable even if native graph ordering changes.
    packed_scratch = internal::create_packed_scratch(
        backend, bindings.data(), bindings.size(), arenas.data(), arenas.size());
    return Status::Ok();
  }

  Status initialize(const Program &program, mem::AllocatorRef allocator,
                     mem::BufferView backing) {
    cuda::detail::ScopedPrimaryContext scope(program.primary);
    HYLE_RETURN_IF_ERROR(scope.status());
    if (program.workspace.size != 0) {
      if (backing.size() < program.workspace.size)
        return Status::InvalidArgument("ggml CUDA caller workspace is too small");
      HYLE_ASSIGN_OR_RETURN(auto pointer, cuda::detail::mutable_pointer(
          backing, program.workspace.alignment, 1));
      workspace.base = static_cast<std::byte *>(pointer);
      workspace.capacity = backing.size();
    }
    // Constants can be actual readonly device bindings. Reject overlap before
    // native construction, binding or any constant upload/graph enqueue.
    for (const auto &constant : program.recipe.constants) {
      HYLE_ASSIGN_OR_RETURN(auto pointer, cuda::detail::const_pointer(
          constant.binding.storage(), program.alignments[constant.value], 1));
      HYLE_ASSIGN_OR_RETURN(auto bytes,
          program.recipe.values[constant.value].required_bytes());
      if (overlaps_workspace(workspace.base, program.workspace.size, pointer, bytes))
        return Status::InvalidArgument("ggml CUDA immutable constant overlaps caller workspace");
    }
    HYLE_RETURN_IF_ERROR(build(program, allocator, false));
    workspace.expected = program.workspace_chunks;
    auto reserved = reserve(&workspace);
    if (workspace.layout_mismatch)
      return Status::FailedPrecondition("ggml CUDA workspace layout changed");
    HYLE_RETURN_IF_ERROR(reserved);
    HYLE_RETURN_IF_ERROR(bind_packed_scratch(program));
    if (workspace.next != workspace.expected.size() ||
        workspace.span != program.workspace.size)
      return Status::FailedPrecondition("ggml CUDA workspace layout changed");
    return Status::Ok();
  }
};

Status create_context(const Provider &provider, Device device, void *backend,
                      mem::AllocatorRef allocator, void **output) noexcept {
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("ggml CUDA requires a CUDA device");
  const auto primary = cuda::detail::borrow_primary_context(backend);
  const auto *environment = cuda::detail::borrow_environment(backend);
  if (!primary || !environment)
    return Status::FailedPrecondition("ggml CUDA backend context is invalid");
  cuda::detail::ScopedPrimaryContext scope(primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  try {
#if HYLE_OS_WINDOWS
    // This DLL embeds its own ggml copy. MetadataArena initializes a separate
    // copy; native graph-cache timestamps need this copy's thread-safe timer.
    ggml_time_init();
#endif
    const int native_device = bridge::device_index(environment->device_ordinal);
    if (native_device < 0)
      return Status::Unimplemented("ggml CUDA physical device is unavailable");
    void *storage = allocator.allocate(sizeof(Context), alignof(Context));
    if (!storage)
      return Status::ResourceExhausted("ggml CUDA context allocation failed");
    *output = ::new (storage)
        Context{primary, native_device,
                default_cublas_precision(),
                detail::ProviderAccess::payload<Extension>(provider)};
    return Status::Ok();
  } catch (...) {
    return failure();
  }
}
void destroy_context(void *value, mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<Context *>(value);
  if (context) {
    context->~Context();
    allocator.deallocate(context, sizeof(Context), alignof(Context));
  }
}
void destroy_program(void *value, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<Program *>(value);
  if (program) {
    program->~Program();
    allocator.deallocate(program, sizeof(Program), alignof(Program));
  }
}
void destroy_invocation(void *value, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(value);
  if (invocation) {
    invocation->~Invocation();
    allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
  }
}
Status create_invocation(void *value, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  *output = nullptr;
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage)
    return Status::ResourceExhausted("ggml CUDA invocation allocation failed");
  auto *invocation = ::new (storage) Invocation;
  *output = invocation; // The caller owns cleanup even after partial failure.
  try {
    return invocation->initialize(*static_cast<Program *>(value), allocator,
                                   workspace);
  } catch (...) {
    return failure();
  }
}
Status observe_native_retirement(void *opaque, bool wait,
                                  bool *retired) noexcept {
  *retired = true;
  Status first;
  CUcontext ended_primary = nullptr;
  cudaStream_t ended_stream = nullptr;
  cuda::detail::NativeRetirement current{opaque, &observe_native_retirement};
  // A capture may contain hundreds of original ggml invocations. Walk their
  // retained nodes iteratively rather than growing the callback call stack.
  while (current.observe == &observe_native_retirement) {
    auto &invocation = *static_cast<Invocation *>(current.state);
    current = invocation.previous_retirement;
    // This proves the old Completion's recording ended, not that this stream
    // remains idle. A later capture belongs to another retained owner.
    const bool ended = invocation.recording_guard &&
        invocation.failed_recording_stream != nullptr &&
        invocation.primary == ended_primary &&
        invocation.failed_recording_stream == ended_stream;
    if (!ended) { ended_primary = nullptr; ended_stream = nullptr; }
    std::optional<cuda::detail::ScopedPrimaryContext> scope;
    // A same-chain positive proof skips the capture query. With no native
    // retirement left, this node only releases host-side recording guards.
    if (!ended || invocation.native_retirement_required)
      scope.emplace(invocation.primary);
    Status observed = scope ? scope->status() : Status::Ok();
    bool complete = observed.ok();
    if (complete && invocation.failed_recording_stream != nullptr) {
      const auto stream = invocation.failed_recording_stream;
      if (!ended) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        const auto result = cudaStreamIsCapturing(stream, &capture);
        observed = cuda::status(result);
        complete = result == cudaSuccess && capture == cudaStreamCaptureStatusNone;
      }
      if (complete) {
        if (invocation.recording_guard) {
          ended_primary = invocation.primary; ended_stream = stream;
        }
        invocation.failed_recording_stream = nullptr;
        if (invocation.recording_guard) {
          bridge::release_recording_guard();
          invocation.recording_guard = false;
        }
      }
    }
    if (!complete || invocation.native_retirement_required) {
      ended_primary = nullptr; ended_stream = nullptr;
    }
    if (complete && invocation.native_retirement_required) {
      const auto result = wait ? bridge::drain(invocation.backend)
                               : bridge::query(invocation.backend);
      observed = cuda::status(result);
      complete = result == cudaSuccess;
    }
    *retired = *retired && complete;
    if (first.ok() && !observed.ok())
      first = std::move(observed);
  }
  // The positive proof never leaves this original GGML retirement chain.
  ended_primary = nullptr; ended_stream = nullptr;
  if (current.observe != nullptr) {
    bool complete = false;
    auto observed = current.observe(current.state, wait, &complete);
    *retired = *retired && complete;
    if (first.ok() && !observed.ok())
      first = std::move(observed);
  }
  return first;
}

Status prepare_bindings(void *value, void *lane,
                        std::span<const ConstTensorView> inputs,
                        std::span<const TensorView> outputs) noexcept {
  const auto &program = *static_cast<Program *>(value);
  auto &invocation = *static_cast<Invocation *>(lane);
  const auto &recipe = program.recipe;
  if (invocation.failed)
    return Status::FailedPrecondition(
        "ggml CUDA invocation failed; instantiate another");
  if (inputs.size() != recipe.inputs.size() ||
      outputs.size() != recipe.outputs.size())
    return Status::InvalidArgument("ggml CUDA binding counts differ");
  bool changed = !invocation.bindings_ready;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto pointer = cuda::detail::const_pointer(
        inputs[i].storage(), program.alignments[recipe.inputs[i]], 1);
    if (!pointer)
      return std::move(pointer).error();
    HYLE_ASSIGN_OR_RETURN(auto bytes,
        recipe.values[recipe.inputs[i]].required_bytes());
    if (overlaps_workspace(invocation.workspace.base, program.workspace.size, *pointer, bytes))
      return Status::InvalidArgument("ggml CUDA input overlaps caller workspace");
    auto *address = const_cast<void *>(*pointer);
    if (invocation.pointers[i] != address) {
      changed = true;
      // A later port may reject before native binding. Never certify this
      // partially updated pointer scratch as an already bound invocation.
      invocation.bindings_ready = false;
    }
    invocation.pointers[i] = address;
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto pointer = cuda::detail::mutable_pointer(
        outputs[i].storage(), program.alignments[recipe.outputs[i]], 1);
    if (!pointer)
      return std::move(pointer).error();
    HYLE_ASSIGN_OR_RETURN(auto bytes,
        recipe.values[recipe.outputs[i]].required_bytes());
    if (overlaps_workspace(invocation.workspace.base, program.workspace.size, *pointer, bytes))
      return Status::InvalidArgument("ggml CUDA output overlaps caller workspace");
    const auto index = inputs.size() + i;
    if (invocation.pointers[index] != *pointer) {
      changed = true;
      invocation.bindings_ready = false;
    }
    invocation.pointers[index] = *pointer;
  }
  // Graph has already performed cold binding; Direct may repeat the same
  // addresses too. All current ports still passed typed resource validation.
  // The program and its bound native tensor graph belong to this slot.
  if (!changed)
    return Status::Ok();
  cuda::detail::ScopedPrimaryContext scope(program.primary);
  if (!scope)
    return scope.status();
  for (std::size_t i = 0; i < inputs.size(); ++i)
    invocation.bind(recipe.inputs[i], invocation.pointers[i]);
  // Checked regions compute every published value privately, including values
  // unrelated to the indexed node. Validation failure cannot partially publish.
  if (program.index_checks.empty())
    for (std::size_t i = 0; i < outputs.size(); ++i)
      invocation.bind(recipe.outputs[i], invocation.pointers[inputs.size() + i]);
  try {
    auto bound = invocation.finish_bind();
    if (!bound.ok())
      return bound;
  } catch (...) {
    return failure();
  }
  invocation.bindings_ready = true;
  return Status::Ok();
}

detail::EnqueueResult submit(void *value, void *lane, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &program = *static_cast<Program *>(value);
  auto &invocation = *static_cast<Invocation *>(lane);
  const auto &recipe = program.recipe;
  auto stream = cuda::stream(sequence);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  if (sequence.recording() && invocation.recording_guard)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "ggml CUDA invocation already belongs to this unfinished recording"));
  auto bound = prepare_bindings(value, lane, inputs, outputs);
  if (!bound.ok())
    return detail::EnqueueResult::rejected(std::move(bound));
  cuda::detail::ScopedPrimaryContext scope(program.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  auto capture_status = cuda::status(cudaStreamIsCapturing(*stream, &capture));
  if (!capture_status.ok())
    return detail::EnqueueResult::rejected(std::move(capture_status));
  if ((capture != cudaStreamCaptureStatusNone) != sequence.recording())
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "ggml CUDA capture requires the original recording sequence"));

  Status terminal = Status::Ok();
  unsigned *status_word = nullptr;
  if (!program.index_checks.empty()) {
    auto initialized = cuda::detail::acquire_device_status(sequence, &status_word);
    if (!initialized.status.ok())
      return initialized;
    for (std::size_t i = 0; i < program.index_checks.size(); ++i) {
      const auto &check = program.index_checks[i];
      terminal = cuda::status(internal::check_indices(
          static_cast<const std::int32_t *>(invocation.pointers[check.input]),
          static_cast<std::int32_t *>(invocation.safe_indices[i]->data),
          check.count, check.extent, status_word, *stream));
      if (!terminal.ok())
        return detail::EnqueueResult::accepted_command(std::move(terminal));
    }
  }

  cudaError_t retirement_error = cudaSuccess;
  try {
    // Shared Program plans are immutable. Invocation/backend/workspace are
    // exclusive until Completion; TLS borrows only this CPU encoder's table.
    internal::PackedScratchEncodeScope packed_scope(invocation.packed_scratch);
    if (sequence.recording()) {
      // The guard must outlive this encoder call: later providers still
      // belong to the same outer capture. This is a cold-only retained node.
      bridge::retain_recording_guard();
      invocation.recording_guard = true;
      invocation.failed_recording_stream = *stream;
      invocation.native_retirement_required = false;
      invocation.previous_retirement = cuda::detail::defer_native_retirement(
          sequence, {&invocation, &observe_native_retirement});
    }
    const auto result = sequence.recording()
        ? bridge::record_on_admitted_stream(invocation.backend, invocation.graph,
              *stream, &retirement_error)
        : bridge::compute_on_admitted_stream(invocation.backend, invocation.graph,
              *stream, &retirement_error);
    packed_scope.check();
    if (result != GGML_STATUS_SUCCESS)
      terminal = Status::Internal("ggml CUDA native graph failed");
  } catch (...) {
    terminal = failure();
  }
  if (terminal.ok() && status_word) {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto id = recipe.outputs[i];
      terminal = cuda::status(internal::publish_checked(
          invocation.tensors[id]->data, invocation.pointers[inputs.size() + i],
          recipe.values[id].required_bytes().value(), status_word, *stream));
      if (!terminal.ok())
        break;
    }
  }
  if (!terminal.ok()) {
    invocation.failed = true;
    // Commands drain before restoring the borrowed stream. Outer recording
    // must first End/abort; its same retained probe then observes the native
    // prefix. The root fence alone cannot prove unfinished auxiliary work.
    if (retirement_error != cudaSuccess) {
      invocation.native_retirement_required = true;
      if (!sequence.recording())
        invocation.previous_retirement = cuda::detail::defer_native_retirement(
            sequence, {&invocation, &observe_native_retirement});
    }
  }
  return detail::EnqueueResult::accepted_command(std::move(terminal));
}
AliasSupport alias_support(void *opaque, std::size_t input,
                           std::size_t output) noexcept {
  const auto &recipe = static_cast<Program *>(opaque)->recipe;
  if (input != 0 || output != 0 || recipe.nodes.size() != 1 ||
      !recipe.nodes[0].operation.get_if<Softmax>())
    return {};
  // Native row softmax is safe at the same address. Nontrailing axes and
  // half storage first materialize their native input before writing back.
  const auto &source = recipe.values[recipe.inputs[0]];
  return {.supported =
              source.is_dense() && source == recipe.values[recipe.outputs[0]]};
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy_program,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
    .prepare_bindings = &prepare_bindings};

Status propose(void *context, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *output) noexcept {
  *output = {};
  if (!request.shape.exact() ||
      !internal::supports(context, request.operation, request.inputs,
                          request.outputs))
    return Status::Unimplemented("ggml CUDA operation domain is not admitted");
  std::ranges::fill(inputs, internal::kBinding);
  std::ranges::fill(outputs, internal::kBinding);
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i].minimum_alignment = internal::boundary_alignment(
        request.operation, request.inputs[i].data_type());
  for (std::size_t i = 0; i < outputs.size(); ++i)
    outputs[i].minimum_alignment = internal::boundary_alignment(
        request.operation, request.outputs[i].data_type());
  *output = {.route = 1,
             .runtime = internal::kRuntime,
             .workspace_resource = internal::kBinding,
             .cost = {.workspace = std::nullopt,
                      .persistent_bytes = std::nullopt,
                      .copy_bytes = std::nullopt,
                      .repack_bytes = std::nullopt,
                      .synchronization_count = 2,
                      .dispatch_count = std::nullopt},
             .implementation = ProviderImplementationId{1}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *context,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if (request.route != 1 || !request.shape.exact() ||
      !internal::supports(context, request.operation, request.inputs,
                          request.outputs))
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("ggml CUDA prepare domain differs"));
  try {
    internal::Recipe recipe;
    recipe.values.assign(request.inputs.begin(), request.inputs.end());
    internal::Node node{request.operation, {}, {}};
    for (std::uint32_t i = 0; i < request.inputs.size(); ++i) {
      recipe.inputs.push_back(i);
      node.inputs.push_back(i);
    }
    for (const auto &description : request.outputs) {
      const auto id = static_cast<std::uint32_t>(recipe.values.size());
      recipe.values.push_back(description);
      node.outputs.push_back(id);
      recipe.outputs.push_back(id);
    }
    recipe.nodes.push_back(std::move(node));
    return internal::compile(context, std::move(recipe), allocator, output);
  } catch (...) {
    return ProviderPrepareOutcome::error(failure());
  }
}
Status drain_context(void *) noexcept { return Status::Ok(); }
Status inspect_capture_bindings(const detail::ProviderPrepared &value,
                                bool *preserves) noexcept {
  if (!preserves)
    return Status::InvalidArgument("ggml CUDA capture bindings output is null");
  *preserves = false;
  if (value.vtable != &kPrepared || !value.state)
    return Status::Unimplemented("foreign ggml CUDA prepared implementation");
  const auto &program = *static_cast<const Program *>(value.state);
  if (!program.primary || program.device < 0)
    return Status::Internal("ggml CUDA capture bindings prepared state is invalid");
  // The original slot keeps graph/workspace/scratch addresses. Its patched
  // outer-recording pools retain all returned temporaries, rejecting capacity
  // or allocation failure instead of clearing old captured addresses. Ordinary
  // Commands/private warmup can clear those pools and must invalidate first.
  *preserves = value.runtime.capture_compatible &&
      value.runtime.mode == ProviderRuntimeMode::kSharedQueue &&
      value.runtime.workspace_owner == ProviderWorkspaceOwner::kCaller;
  return Status::Ok();
}
Status inspect_projected_input(const detail::ProviderPrepared &value,
    detail::OperationPreparedAccess *access) noexcept {
  if (!access)
    return Status::InvalidArgument("ggml CUDA prepared access output is null");
  *access = {};
  if (value.vtable != &kPrepared || !value.state)
    return Status::Unimplemented("foreign ggml CUDA prepared implementation");
  const auto &program = *static_cast<const Program *>(value.state);
  const auto &recipe = program.recipe;
  // Only the original public single-operation factory qualifies. In particular
  // this is not a generic recipe analyzer, a packed table or an indexed GDN.
  if (!program.primary || program.device < 0 || program.transpose_constant ||
      recipe.nodes.size() != 1 || recipe.inputs.size() != 2 ||
      recipe.outputs.size() != 1 || recipe.values.size() != 3 ||
      !recipe.constants.empty() || !recipe.effect_roots.empty() ||
      recipe.inputs[0] != 0 || recipe.inputs[1] != 1 || recipe.outputs[0] != 2 ||
      program.index_checks.size() != 1)
    return Status::Unimplemented("ggml CUDA projection needs its original checked Gather leaf");
  const auto &node = recipe.nodes[0];
  const auto *gather = node.operation.get_if<Gather>();
  if (!gather || node.indexed_state || node.inputs.size() != 2 || node.outputs.size() != 1 ||
      node.inputs[0] != 0 || node.inputs[1] != 1 || node.outputs[0] != 2 ||
      gather->bounds != IndexBoundsPolicy::kChecked ||
      (gather->axis != 0 && gather->axis != -2))
    return Status::Unimplemented("ggml CUDA projection is not a checked axis-zero Gather");
  const auto &table = recipe.values[0], &ids = recipe.values[1], &output = recipe.values[2];
  const auto type = table.data_type();
  if (!table.is_plain() || table.rank() != 2 || table.dimensions()[0] <= 0 ||
      table.dimensions()[1] <= 0 ||
      (type != DataType::kFloat32 && type != DataType::kFloat16 && type != DataType::kBFloat16) ||
      !ids.is_dense() || ids.rank() != 1 || ids.data_type() != DataType::kInt32 ||
      ids.dimensions()[0] <= 0 || !output.is_dense() || output.rank() != 2 ||
      output.data_type() != type || output.dimensions()[0] != ids.dimensions()[0] ||
      output.dimensions()[1] != table.dimensions()[1])
    return Status::Unimplemented("ggml CUDA projection needs exact typed selected-row storage");
  const auto unit = static_cast<std::int64_t>(data_type_size(type));
  const auto &check = program.index_checks[0];
  if (table.dimensions()[1] > INT64_MAX / unit || table.byte_strides()[1] != unit ||
      table.byte_strides()[0] < table.dimensions()[1] * unit ||
      check.input != 1 || check.node != 0 || check.extent != table.dimensions()[0] ||
      !std::cmp_equal(check.count, ids.dimensions()[0]))
    return Status::Unimplemented("ggml CUDA projection differs from its actual indexed reader");
  // Native typed GET_ROWS keeps nb[1] and never converts/CONT-copies the bank.
  // The typed adapter must separately prove the private IDs match the actual
  // retained row leases; a bounds check alone does not establish that coverage.
  *access = {.projected_input = 0};
  return Status::Ok();
}
const void *query_extension(void *value, ProviderExtensionId id) noexcept {
  static constexpr detail::CaptureBindingsVTable capture{
      .inspect = &inspect_capture_bindings};
  if (id == detail::kCaptureBindingsExtensionId)
    return &capture;
  static constexpr detail::OperationAccessVTable projected{
      .inspect = &inspect_projected_input};
  if (id == detail::kProjectedInputReadExtensionId)
    return &projected;
  const auto &extension = static_cast<Context *>(value)->extension;
  return id == extension.id ? extension.table : nullptr;
}
const detail::ProviderVTable kProvider{
    .identity = {.id = kProviderId,
                 .name = "ggml-cuda",
                 .version = kProviderVersion},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context,
    .query_extension = &query_extension};
} // namespace

namespace internal {
namespace {
struct ColdPackingEpisode final {
  mem::AllocatorRef allocator;
  Program program{};
  Invocation invocation{};
};

struct ConstantRecipeEpisode final {
  mem::AllocatorRef allocator;
  std::vector<ConstTensorView> input_bindings{};
  detail::ProviderPrepared prepared{};
  mem::Buffer workspace{};
  void *invocation = nullptr;

  ~ConstantRecipeEpisode() {
    if (prepared.vtable) {
      if (invocation)
        prepared.vtable->destroy_invocation(invocation, allocator);
      // The Invocation borrows workspace and prepared metadata. Release both
      // only after its original native retirement has released this episode.
      workspace = {};
      prepared.vtable->destroy_prepared(prepared.state, allocator);
    }
  }
};

void release_constant_recipe_episode(void *value) noexcept {
  auto *episode = static_cast<ConstantRecipeEpisode *>(value);
  const auto allocator = episode->allocator;
  episode->~ConstantRecipeEpisode();
  allocator.deallocate(episode, sizeof(ConstantRecipeEpisode),
                       alignof(ConstantRecipeEpisode));
}

void release_cold_packing_episode(void *value) noexcept {
  auto *episode = static_cast<ColdPackingEpisode *>(value);
  const auto allocator = episode->allocator;
  episode->~ColdPackingEpisode();
  allocator.deallocate(episode, sizeof(ColdPackingEpisode),
                       alignof(ColdPackingEpisode));
}
} // namespace

Status pack_matrix_constant(void *value, ConstTensorView source,
                            TensorView destination, mem::AllocatorRef allocator,
                            detail::QueueSequence &sequence,
                            detail::CompletionLifetime *retained_native) noexcept {
  if (!retained_native)
    return Status::InvalidArgument("ggml CUDA constant packing owner is missing");
  *retained_native = {};
  if (!sequence.terminal().ok())
    return sequence.terminal();
  const auto &from = source.description();
  const auto &to = destination.description();
  if (!value || !from.is_plain() || !from.is_dense() ||
      !to.is_plain() || from.data_type() != to.data_type() || from.rank() < 2 ||
      from.rank() > 4 || !to.transposed() || !to.transposed()->is_dense() ||
      !std::ranges::equal(from.dimensions(), to.dimensions()))
    return Status::InvalidArgument("ggml CUDA constant packing schema differs");
  const auto &context = *static_cast<Context *>(value);
  cuda::detail::ScopedPrimaryContext scope(context.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  try {
    void *storage = allocator.allocate(sizeof(ColdPackingEpisode),
                                        alignof(ColdPackingEpisode));
    if (!storage)
      return Status::ResourceExhausted(
          "ggml CUDA cold packing owner allocation failed");
    auto *episode = ::new (storage) ColdPackingEpisode{.allocator = allocator};
    // Hand off before native preparation: every later failure may leave an
    // owned partial invocation, or commands in the original Queue prefix.
    *retained_native = {episode, +[](void *, const Status &) noexcept {},
                        &release_cold_packing_episode};
    auto &program = episode->program;
    auto &invocation = episode->invocation;
    program.primary = context.primary;
    program.device = context.device;
    program.transpose_constant = true;
    program.recipe.values = {from, to};
    program.recipe.inputs = {0};
    program.recipe.outputs = {1};
    program.recipe.nodes.push_back({Operation(Copy{}), {0}, {1}});
    program.alignments = {data_type_size(from.data_type()),
                          data_type_size(to.data_type())};
    program.hadamards.resize(1);
    HYLE_RETURN_IF_ERROR(invocation.build(program, allocator, false));
    // This factory is already an explicit cold owned operation; ordinary
    // prepared invocations always require their measured caller workspace.
    HYLE_RETURN_IF_ERROR(invocation.reserve(
        ggml_backend_cuda_buffer_type(program.device)));
    const std::array inputs{source};
    const std::array outputs{destination};
    auto result = submit(&program, &invocation, sequence, inputs, outputs);
    sequence.record(std::move(result));
    return sequence.terminal();
  } catch (...) {
    return failure();
  }
}

Status evaluate_constant_recipe(void *context, Recipe recipe,
                                std::span<const TensorView> destinations,
                                mem::AllocatorRef allocator,
                                detail::QueueSequence &sequence,
                                detail::CompletionLifetime *retained_native) noexcept {
  if (!retained_native)
    return Status::InvalidArgument("ggml CUDA constant recipe owner is missing");
  *retained_native = {};
  if (!sequence.valid())
    return Status::FailedPrecondition("ggml CUDA constant recipe needs an admitted sequence");
  if (!sequence.terminal().ok())
    return sequence.terminal();
  if (!context || !allocator.valid() || !recipe.inputs.empty() ||
      !recipe.effect_roots.empty() ||
      destinations.empty() || destinations.size() != recipe.outputs.size())
    return Status::InvalidArgument("ggml CUDA constant recipe bindings differ");
  // Ready-only regions are represented by their prepared output bindings,
  // never by an empty native graph or a dummy mathematical dispatch.
  if (recipe.nodes.empty())
    return Status::Unimplemented("ggml CUDA ready-only recipe needs no native executor");
  for (std::size_t i = 0; i < destinations.size(); ++i)
    if (recipe.outputs[i] >= recipe.values.size() || !destinations[i].valid() ||
        destinations[i].description() != recipe.values[recipe.outputs[i]])
      return Status::InvalidArgument("ggml CUDA constant recipe destination differs");
  for (const auto &constant : recipe.constants)
    if (constant.value >= recipe.values.size() || !constant.binding.valid() ||
        constant.binding.description() != recipe.values[constant.value])
      return Status::InvalidArgument("ggml CUDA constant recipe source differs");
  try {
    void *storage = allocator.allocate(sizeof(ConstantRecipeEpisode),
                                        alignof(ConstantRecipeEpisode));
    if (!storage)
      return Status::ResourceExhausted("ggml CUDA constant recipe owner allocation failed");
    auto *episode = ::new (storage) ConstantRecipeEpisode{.allocator = allocator};
    // Publish before compile/create/prepare: a later error can leave partial
    // native state or accepted commands. The original Completion owns cleanup.
    *retained_native = {episode, +[](void *, const Status &) noexcept {},
                        &release_constant_recipe_episode};
    std::vector<std::uint8_t> defined(recipe.values.size(), 0);
    for (const auto &constant : recipe.constants) {
      if (defined[constant.value])
        return Status::InvalidArgument("ggml CUDA constant recipe repeats a source");
      defined[constant.value] = 1;
    }
    std::vector<TensorDesc> inputs, outputs;
    for (const auto &node : recipe.nodes) {
      inputs.clear(); outputs.clear();
      for (const auto id : node.inputs) {
        if (id >= defined.size() || !defined[id])
          return Status::InvalidArgument("ggml CUDA constant recipe source is undefined");
        inputs.push_back(recipe.values[id]);
      }
      for (const auto id : node.outputs) {
        if (id >= defined.size() || defined[id])
          return Status::InvalidArgument("ggml CUDA constant recipe value is redefined");
        outputs.push_back(recipe.values[id]);
        defined[id] = 1;
      }
      if (!supports(context, node.operation, inputs, outputs))
        return Status::Unimplemented("ggml CUDA constant recipe operation is unsupported");
    }
    if (std::ranges::find(defined, std::uint8_t{0}) != defined.end())
      return Status::InvalidArgument("ggml CUDA constant recipe has an unused value");
    // Fixed cold roots still use the original prepared input boundary. In
    // particular, checked Gather/GroupedMatMul must see their indices here so
    // native compile retains the original validation and private-output path.
    // These ports are internal and never become externally rebindable inputs.
    for (const auto &constant : recipe.constants) {
      recipe.inputs.push_back(constant.value);
      episode->input_bindings.push_back(constant.binding);
    }
    recipe.constants.clear();
    auto outcome = compile(context, std::move(recipe), allocator, &episode->prepared);
    if (outcome.kind != ProviderPrepareOutcomeKind::kPrepared)
      return outcome.status;
    auto &prepared = episode->prepared;
    if (prepared.workspace.size != 0) {
      HYLE_ASSIGN_OR_RETURN(episode->workspace,
          mem::Buffer::create(destinations.front().storage().resource(),
                              prepared.workspace.size, prepared.workspace.alignment));
    }
    HYLE_RETURN_IF_ERROR(prepared.vtable->create_invocation(prepared.state,
        allocator, episode->workspace.view(), &episode->invocation));
    HYLE_RETURN_IF_ERROR(prepared.vtable->prepare_bindings(prepared.state,
        episode->invocation, episode->input_bindings, destinations));
    auto result = prepared.vtable->shared_submit(prepared.state,
        episode->invocation, sequence, episode->input_bindings, destinations);
    sequence.record(std::move(result));
    return sequence.terminal();
  } catch (...) {
    return failure();
  }
}

bool supports_bfloat_paired_projection(void *value, std::uint64_t inner,
                                     std::uint64_t rows) noexcept {
  if (!value)
    return false;
  try {
    return bridge::supports_bfloat_paired_projection(
        static_cast<Context *>(value)->device, inner, rows);
  } catch (...) {
    return false;
  }
}

bool supports(void *value, Operation operation,
              std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  if (!value || outputs.empty())
    return false;
  if (const auto *attention = operation.get_if<ScaledDotProductAttention>()) {
    const bool additive = attention->mask == MaskKind::kAdditive;
    if (outputs.size() != 1 || inputs.size() != (additive ? 4U : 3U) ||
        attention->layout != AttentionLayout::kBatchSequenceHeadDepth ||
        (attention->mask != MaskKind::kNone && !additive) ||
        attention->accumulation_type != DataType::kFloat32 ||
        !std::isfinite(attention->scale))
      return false;
    const auto type = inputs[0].data_type();
    if ((type != DataType::kFloat32 && type != DataType::kFloat16 &&
         type != DataType::kBFloat16) ||
        (type == DataType::kBFloat16 &&
         !static_cast<Context *>(value)->bfloat16_attention))
      return false;
    for (const auto &description : inputs.first(3))
      if (!description.valid() || !description.is_plain() ||
          !description.is_dense() || description.rank() != 4 ||
          description.data_type() != type || !description.element_count() ||
          *description.element_count() > INT32_MAX)
        return false;
    if (additive) {
      const auto &mask = inputs[3];
      const auto count = mask.element_count();
      if (!mask.valid() || !mask.is_plain() || !mask.is_dense() ||
          mask.rank() > 4 || mask.data_type() != DataType::kFloat32 ||
          !count || *count == 0 || *count > INT32_MAX / sizeof(float))
        return false;
      std::array<TensorDesc, 1> inferred;
      if (!infer_output_descriptions(operation, inputs, inferred).ok() ||
          inferred[0] != outputs[0])
        return false;
    }
    const auto q = inputs[0].dimensions();
    const auto k = inputs[1].dimensions();
    const auto v = inputs[2].dimensions();
    const std::array mla_shape{q[0], q[1], q[2], std::int64_t{512}};
    const auto mla_output = TensorDesc::dense(DataType::kFloat32, mla_shape);
    const bool absorbed_mla = type == DataType::kFloat32 &&
        (q[2] == 64 || q[2] == 96) &&
        k[2] == 1 && q[3] == 576 && v[3] == 512 && v[0] == k[0] &&
        v[1] == k[1] && v[2] == k[2] && mla_output && outputs[0] == *mla_output;
    if (q[0] <= 0 || q[0] > 65535 || q[1] <= 0 || k[1] <= 0 ||
        (attention->causal && k[1] < q[1]) ||
        q[2] <= 0 || k[2] <= 0 || q[2] % k[2] != 0 || q[3] <= 0 ||
        q[3] != k[3] || q[0] != k[0] ||
        (!(inputs[1] == inputs[2] && outputs[0] == inputs[0]) && !absorbed_mla))
      return false;
    // Bound native indexing and the BF16 score/mask allocation before any
    // upstream API asserts or buffer-size arithmetic can be reached.
    return k[1] <= INT32_MAX / q[1] / q[2] / q[0];
  }
  return tensor_operation_supports(operation, inputs, outputs,
      static_cast<Context *>(value)->device);
}

namespace {
bool supports_indexed_conv(void *value, const Node &node,
                           std::span<const TensorDesc> values) noexcept {
  if (!value || !node.indexed_state ||
      !node.operation.custom_as<model_ops::CausalConv1d>() ||
      node.inputs.size() != 5 || node.outputs.size() != 1 ||
      std::ranges::any_of(node.inputs, [&](auto id) { return id >= values.size(); }) ||
      node.outputs[0] >= values.size() ||
      ggml_hyle_cuda_conv_indexed_state_contract() != kIndexedConvContract)
    return false;
  const auto &chunk = values[node.inputs[0]], &weights = values[node.inputs[1]];
  const auto &pool = values[node.inputs[2]], &sources = values[node.inputs[3]];
  const auto &destinations = values[node.inputs[4]];
  const auto type = chunk.data_type();
  if (!chunk.valid() || !chunk.is_plain() || chunk.rank() != 3 ||
      (type != DataType::kFloat32 && type != DataType::kFloat16 &&
       type != DataType::kBFloat16) ||
      !weights.valid() || !weights.is_plain() || !weights.is_dense() ||
      weights.rank() != 2 || weights.data_type() != DataType::kFloat32 ||
      !pool.valid() || !pool.is_plain() || pool.rank() != 2 || pool.data_type() != type ||
      pool.dimensions()[0] <= 0 || pool.dimensions()[0] > INT32_MAX ||
      !sources.valid() || !sources.is_dense() || sources.rank() != 1 ||
      sources.data_type() != DataType::kInt32 ||
      !destinations.valid() || !destinations.is_dense() || destinations.rank() != 1 ||
      destinations.data_type() != DataType::kInt32)
    return false;
  const auto batches = chunk.dimensions()[0], tokens = chunk.dimensions()[1];
  const auto channels = chunk.dimensions()[2], taps = weights.dimensions()[1];
  const auto unit = static_cast<std::int64_t>(data_type_size(type));
  if (batches <= 0 || tokens <= 0 || channels <= 0 || taps <= 1 ||
      weights.dimensions()[0] != channels || sources.dimensions()[0] != batches ||
      destinations.dimensions()[0] <= 0 || destinations.dimensions()[0] % batches ||
      destinations.dimensions()[0] / batches > tokens ||
      channels > INT64_MAX / unit || channels > INT64_MAX / (taps - 1) ||
      pool.dimensions()[1] != (taps - 1) * channels ||
      pool.dimensions()[1] > INT64_MAX / unit ||
      pool.byte_strides()[1] != unit ||
      pool.byte_strides()[0] < pool.dimensions()[1] * unit || pool.byte_strides()[0] % unit ||
      chunk.byte_strides()[2] != unit || chunk.byte_strides()[1] < channels * unit ||
      chunk.byte_strides()[1] % unit || chunk.byte_strides()[0] % unit ||
      tokens - 1 > (INT64_MAX - channels * unit) / chunk.byte_strides()[1] ||
      chunk.byte_strides()[0] < (tokens - 1) * chunk.byte_strides()[1] + channels * unit ||
      tokens > INT64_MAX - (taps - 1))
    return false;
  for (const auto id : node.inputs) {
    const auto bytes = values[id].required_bytes();
    if (!bytes || *bytes > static_cast<std::uint64_t>(INT64_MAX)) return false;
  }
  // Qualify the original public F32 mathematics, not a newly invented mixed
  // precision operation. The native loader widens each original sample where
  // the old Cast did; history is separately written from its original bytes.
  const auto samples = TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 3>{batches, tokens + (taps - 1), channels});
  if (!samples) return false;
  const std::array original{*samples, weights};
  std::array<TensorDesc, 1> output;
  return infer_output_descriptions(node.operation, original, output).ok() &&
      output[0] == values[node.outputs[0]] &&
      supports(value, node.operation, original, output);
}
} // namespace

bool supports_indexed_state(void *value, const Node &node,
                             std::span<const TensorDesc> values) noexcept {
  if (node.operation.custom_as<model_ops::CausalConv1d>())
    return supports_indexed_conv(value, node, values);
  const auto *gdn = node.operation.get_if<GatedDeltaNet>();
  const auto recurrent = node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>();
  if (!value || !node.indexed_state || (!gdn && !recurrent) ||
      node.inputs.size() != (gdn ? 8U : 10U) || node.outputs.size() != 1 ||
      std::ranges::any_of(node.inputs, [&](auto id) { return id >= values.size(); }) ||
      node.outputs[0] >= values.size() ||
      ggml_hyle_cuda_gdn_indexed_state_contract() != kIndexedStateContract)
    return false;
  const auto state_port = gdn ? 5U : 7U;
  const auto &q = values[node.inputs[0]], &v = values[node.inputs[2]];
  const auto &pool = values[node.inputs[state_port]];
  const auto &sources = values[node.inputs[state_port + 1]];
  const auto &destinations = values[node.inputs[state_port + 2]];
  constexpr auto element_bytes = static_cast<std::int64_t>(sizeof(float));
  if (!q.valid() || !v.valid() || q.rank() != 4 || v.rank() != 4 || q.dimensions()[0] <= 0 ||
      q.dimensions()[0] != v.dimensions()[0] || q.dimensions()[3] <= 0 ||
      q.dimensions()[3] != v.dimensions()[3] || v.dimensions()[2] <= 0 ||
      !pool.valid() || !pool.is_plain() || pool.rank() != 2 ||
      pool.data_type() != DataType::kFloat32 || pool.dimensions()[0] <= 0 ||
      pool.dimensions()[0] > INT32_MAX || pool.dimensions()[1] <= 0 ||
      pool.byte_strides()[1] != element_bytes ||
      !sources.valid() || !sources.is_dense() || sources.rank() != 1 ||
      sources.data_type() != DataType::kInt32 ||
      sources.dimensions()[0] != q.dimensions()[0] ||
      !destinations.valid() || !destinations.is_dense() || destinations.rank() != 1 ||
      destinations.data_type() != DataType::kInt32)
    return false;
  const auto width = q.dimensions()[3], heads = v.dimensions()[2];
  const auto snapshots = gdn ? gdn->snapshot_count : recurrent->snapshot_count;
  if (!snapshots || snapshots > static_cast<std::uint64_t>(q.dimensions()[1]) ||
      width > INT64_MAX / width || width * width > INT64_MAX / heads ||
      pool.dimensions()[1] != width * width * heads ||
      pool.dimensions()[1] > INT64_MAX / element_bytes ||
      pool.byte_strides()[0] < pool.dimensions()[1] * element_bytes ||
      pool.byte_strides()[0] % element_bytes != 0 ||
      q.dimensions()[0] > INT64_MAX / snapshots ||
      destinations.dimensions()[0] != q.dimensions()[0] * snapshots)
    return false;
  const auto pool_bytes = pool.required_bytes();
  if (!pool_bytes || *pool_bytes > static_cast<std::uint64_t>(INT64_MAX)) return false;
  // Validate against the original public dense mathematics, without allocating
  // its state or changing any type, gate, head grouping or rounding boundary.
  std::array<TensorDesc, 8> original;
  for (std::size_t i = 0; i < state_port; ++i)
    original[i] = values[node.inputs[i]];
  auto initial = TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 4>{q.dimensions()[0], heads, width, width});
  if (!initial) return false;
  original[state_port] = *initial;
  std::array<TensorDesc, 2> outputs;
  const auto inputs = std::span(original).first(state_port + 1);
  return infer_output_descriptions(node.operation, inputs, outputs).ok() &&
      outputs[0] == values[node.outputs[0]] &&
      supports(value, node.operation, inputs, outputs);
}

ProviderPrepareOutcome compile(void *value, Recipe recipe,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  const auto &context = *static_cast<Context *>(value);
  std::vector<std::size_t> alignments;
  std::vector<std::vector<float>> hadamards;
  std::vector<std::vector<std::int32_t>> snapshot_indices;
  std::vector<Program::IndexCheck> index_checks;
  try {
    for (const auto id : recipe.effect_roots)
      if (id >= recipe.values.size())
        return ProviderPrepareOutcome::error(Status::InvalidArgument(
            "ggml CUDA effect root is not a recipe value"));
    alignments.resize(recipe.values.size(), 1);
    hadamards.resize(recipe.nodes.size());
    for (std::size_t node_index = 0; node_index < recipe.nodes.size();
         ++node_index) {
      const auto &node = recipe.nodes[node_index];
      if (node.indexed_state &&
          (!supports_indexed_state(value, node, recipe.values) ||
           std::ranges::find(recipe.effect_roots, node.outputs[0]) == recipe.effect_roots.end()))
        return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
            "ggml CUDA indexed state needs exact geometry and its original effect root"));
      const auto require = [&](std::uint32_t id) {
        alignments[id] = std::max(
            alignments[id],
            node.indexed_state ? data_type_size(recipe.values[id].data_type())
                : boundary_alignment(node.operation, recipe.values[id].data_type()));
      };
      for (const auto id : node.inputs)
        require(id);
      for (const auto id : node.outputs)
        require(id);
      const auto *gdn = node.operation.get_if<GatedDeltaNet>();
      const auto recurrent =
          node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>();
      const auto snapshots = gdn ? gdn->snapshot_count
                                 : recurrent ? recurrent->snapshot_count : 1;
      if (snapshots > 1 && !node.indexed_state) {
        if (snapshot_indices.empty())
          snapshot_indices.resize(recipe.nodes.size());
        const auto batches = recipe.values[node.inputs[0]].dimensions()[0];
        auto &order = snapshot_indices[node_index];
        order.reserve(static_cast<std::size_t>(batches) * snapshots);
        for (std::int64_t batch = 0; batch < batches; ++batch)
          for (std::uint32_t token = 0; token < snapshots; ++token)
            order.push_back(static_cast<std::int32_t>(
                (snapshots - 1 - token) * batches + batch));
      }
      const auto *gather = node.operation.get_if<Gather>();
      const auto elements = node.operation.custom_as<model_ops::GatherElements>();
      const bool grouped =
          node.operation.custom_as<model_ops::GroupedMatMul>().has_value();
      if (grouped || (gather && gather->bounds == IndexBoundsPolicy::kChecked) ||
          (elements && elements->bounds == IndexBoundsPolicy::kChecked)) {
        const auto index_port = grouped ? 2 : 1;
        const auto input =
            std::ranges::find(recipe.inputs, node.inputs[index_port]);
        if (input == recipe.inputs.end())
          return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
              "ggml CUDA dynamic indices require a checked boundary"));
        const auto count =
            recipe.values[node.inputs[index_port]].element_count().value();
        const auto extent = recipe.values[node.inputs[0]].dimensions()[elements ? 1 : 0];
        if (extent <= 0)
          return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
              "ggml CUDA checked indices require a nonempty table"));
        index_checks.push_back(
            {static_cast<std::uint32_t>(input - recipe.inputs.begin()),
             node_index, count,
             extent});
      }
      if (node.operation.get_if<Hadamard>()) {
        const auto width = recipe.values[node.outputs[0]].dimensions().back();
        if (width <= 0 || width > 4096 ||
            static_cast<std::uint64_t>(width) *
                    static_cast<std::uint64_t>(width) >
                SIZE_MAX / sizeof(float))
          return ProviderPrepareOutcome::unsupported(
              Status::Unimplemented("ggml CUDA Hadamard width is outside the "
                                    "admitted original backend domain"));
        auto &matrix = hadamards[node_index];
        if (width > 512) {
          const auto factors = internal::hadamard_factors(width);
          matrix.resize(factors.constant_elements());
          const auto fill = [&](std::int64_t n, std::size_t offset) {
            const auto scale = 1.0F / std::sqrt(static_cast<float>(n));
            for (std::int64_t row = 0; row < n; ++row)
              for (std::int64_t column = 0; column < n; ++column)
                matrix[offset + static_cast<std::size_t>(row * n + column)] =
                    std::popcount(static_cast<std::uint64_t>(row & column)) % 2
                        ? -scale
                        : scale;
          };
          fill(factors.low, 0);
          if (factors.high != factors.low)
            fill(factors.high,
                 static_cast<std::size_t>(factors.low * factors.low));
          continue;
        }
        matrix.resize(static_cast<std::size_t>(width * width));
        const auto scale = 1.0F / std::sqrt(static_cast<float>(width));
        for (std::int64_t row = 0; row < width; ++row)
          for (std::int64_t column = 0; column < width; ++column)
            matrix[static_cast<std::size_t>(row * width + column)] =
                std::popcount(static_cast<std::uint64_t>(row & column)) % 2
                    ? -scale
                    : scale;
      }
    }
  } catch (...) {
    return ProviderPrepareOutcome::error(failure());
  }
  void *storage = allocator.allocate(sizeof(Program), alignof(Program));
  if (!storage)
    return ProviderPrepareOutcome::error(
        Status::ResourceExhausted("ggml CUDA program allocation failed"));
  const auto cleanup = [allocator](Program *program) {
    destroy_program(program, allocator);
  };
  std::unique_ptr<Program, decltype(cleanup)> program(
      ::new (storage) Program{context.primary, context.device, std::move(recipe),
                             std::move(alignments), std::move(hadamards),
                             std::move(snapshot_indices),
                             std::move(index_checks)},
      cleanup);
  auto runtime = kRuntime;
  try {
    Invocation measurement;
    auto status = measurement.build(*program, allocator, true);
    if (!status.ok())
      return status.IsUnimplemented()
          ? ProviderPrepareOutcome::unsupported(std::move(status))
          : ProviderPrepareOutcome::error(std::move(status));
    // The actual lowered native graph, rather than a second operation list,
    // determines whether its original encoder can record without host sync.
    {
      cuda::detail::ScopedPrimaryContext scope(program->primary);
      if (!scope)
        return ProviderPrepareOutcome::error(scope.status());
      runtime.capture_compatible =
          bridge::record_compatible(measurement.backend, measurement.graph);
    }
    measurement.workspace.measurement = &program->workspace_chunks;
    status = measurement.reserve(&measurement.workspace);
    if (status.ok())
      status = measurement.measure_packed_scratch(*program);
    if (!status.ok())
      return ProviderPrepareOutcome::error(std::move(status));
    program->workspace = {
        measurement.workspace.span,
        std::max(kBinding.minimum_alignment,
                 ggml_backend_buft_get_alignment(&measurement.workspace))};
  } catch (...) {
    return ProviderPrepareOutcome::error(failure());
  }
  const auto workspace = program->workspace;
  *output = {.vtable = &kPrepared,
             .state = program.release(),
             .workspace = workspace,
             .runtime = runtime,
             .persistent_bytes = std::nullopt,
             .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
Provider with_extension(ProviderExtensionId id, const void *table) noexcept {
  return detail::ProviderAccess::make(&kProvider, Extension{id, table});
}
} // namespace internal

Provider provider() noexcept { return internal::with_extension({}, nullptr); }
} // namespace hyle::compute::ggml_cuda
