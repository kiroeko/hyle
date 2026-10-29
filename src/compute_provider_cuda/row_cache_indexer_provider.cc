#include "row_cache_indexer_provider.h"
#include "dense_bfloat_scores.h"
#include "hyle/compute/model_ops.h"
#include "native_context.h"
#include "row_cache_indexer_kernel.h"
#include "row_cache_kernel.h"
#include "tensor_kernel.h"
#include "reduction_kernel.h"
#include "../compute/operation_cold.h"
#include "../compute_batch/row_cache_map.h"
#include "../compute_batch/row_cache_planning.h"
#include "hyle/mem/std_allocator.h"
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <vector>

namespace hyle::compute::cuda_native::row_cache_indexer {
namespace {
namespace ops = cuda_native::native;
constexpr unsigned kTile = 256;
constexpr std::size_t kAlignment = 256, kLtLimit = 32u << 20;
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true};
struct Shape final {
  unsigned tokens = 0, rows = 0, width = 0, tail = 0;
  unsigned batches = 1, heads = 64, depth = 128;
  std::size_t key_batch_bytes = 0;
};
Status lt_status(cublasStatus_t code) noexcept {
  if (code == CUBLAS_STATUS_SUCCESS)
    return Status::Ok();
  const auto status =
      code == CUBLAS_STATUS_ALLOC_FAILED ? StatusCode::ResourceExhausted
      : code == CUBLAS_STATUS_NOT_SUPPORTED ||
              code == CUBLAS_STATUS_ARCH_MISMATCH
          ? StatusCode::Unimplemented
      : code == CUBLAS_STATUS_INVALID_VALUE ? StatusCode::InvalidArgument
                                            : StatusCode::Internal;
  return Status::FromNative(status,
                            NativeError{NativeErrorDomain::Unknown, code},
                            "row cache indexer cuBLASLt");
}
struct Part final {
  cublasLtMatmulDesc_t operation = nullptr;
  std::array<cublasLtMatrixLayout_t, 3> layouts{};
  cublasLtMatmulAlgo_t algorithm{};
  std::size_t workspace = 0;
  ops::TensorProgram relu{}, multiply{};
  ops::ReductionProgram sum{};
  ~Part() {
    for (auto x : layouts)
      if (x)
        (void)cublasLtMatrixLayoutDestroy(x);
    if (operation)
      (void)cublasLtMatmulDescDestroy(operation);
  }
};
struct Prepared final {
  CUcontext primary = nullptr;
  Shape shape;
  Part full, tail;
  bool paged = false;
  bool dense = false, masked = false, float_output = false;
  unsigned mask_batches = 0;
  std::array<std::size_t, 4> input_bytes{};
  std::size_t output_bytes = 0;
  row_cache::native::MapProgram mapping{};
  std::uint64_t bank_rows = 0;
  std::size_t dots = 0, sums = 0, active = 0, lt = 0, bytes = 0;
};
struct Request final {
  unsigned request = 0, begin = 0, count = 0, keys = 0, tail_width = 0;
  Part full, tail;
};
struct RequestPlans final {
  explicit RequestPlans(mem::AllocatorRef a)
      : allocator(a), items(mem::StdAllocator<Request *>(a)) {}
  ~RequestPlans() {
    for (auto *p : items)
      allocator.destroy(p);
  }
  mem::AllocatorRef allocator;
  std::vector<Request *, mem::StdAllocator<Request *>> items;
};
struct Invocation final {
  explicit Invocation(mem::AllocatorRef a) : requests(a) {}
  CUcontext primary = nullptr;
  cublasLtHandle_t handle = nullptr;
  void *scratch = nullptr;
  RequestPlans requests;
  bool configured = false;
};
struct Cold final {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  ~Cold() {
    if (preference)
      (void)cublasLtMatmulPreferenceDestroy(preference);
    if (handle)
      (void)cublasLtDestroy(handle);
  }
};
std::size_t aligned(std::size_t n) {
  return (n + kAlignment - 1) & ~(kAlignment - 1);
}
Result<Shape> select_dense(Operation operation, std::span<const TensorDesc> in,
                           std::span<const TensorDesc> out) noexcept {
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, in, inferred));
  if (out.size() != 1 || out[0] != inferred[0])
    return std::unexpected(Status::InvalidArgument("dense BF16 score output differs"));
  const auto q = in[0].dimensions(), k = in[1].dimensions();
  const auto b = std::uint64_t(q[0]), t = std::uint64_t(q[1]),
             h = std::uint64_t(q[2]), d = std::uint64_t(q[3]),
             l = std::uint64_t(k[3]);
  // An empty result has no pointer, launch-field or Lt requirement. Keep its
  // invocation on the same zero-work path without narrowing unused extents.
  if (!b || !t || !l)
    return Shape{.batches = 0};
  if (!in[0].is_dense() || !in[2].is_dense() ||
      (in.size() == 4 && !in[3].is_dense()))
    return std::unexpected(Status::Unimplemented("dense BF16 scores need dense Q, W and mask"));
  if (b > INT32_MAX || t > INT32_MAX || h > INT32_MAX ||
      d > INT32_MAX || l > INT32_MAX)
    return std::unexpected(Status::Unimplemented("dense BF16 score dimensions exceed native fields"));
  const auto strides = in[1].byte_strides();
  for (const auto stride : strides)
    if (stride < 0 || stride % 2)
      return std::unexpected(Status::Unimplemented("dense BF16 key pitch must be aligned"));
  if ((d > 1 && strides[2] != 2) ||
      (l > 1 && std::uint64_t(strides[3]) != d * 2) ||
      (b > 1 && std::uint64_t(strides[0]) < l * d * 2))
    return std::unexpected(Status::Unimplemented("dense BF16 keys need dense rows and truthful batch pitch"));
  const auto width = (std::min)(l, std::uint64_t{kTile});
  if (t > std::uint64_t(INT32_MAX) / h ||
      t * h > std::uint64_t(INT32_MAX) / width ||
      b > std::uint64_t(UINT32_MAX) / ((l / kTile + (l % kTile != 0)) * 5))
    return std::unexpected(Status::Unimplemented("dense BF16 score tile exceeds native integer fields"));
  for (const auto &input : in) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, input.required_bytes());
    if (bytes > std::size_t(INT64_MAX))
      return std::unexpected(Status::Unimplemented("dense BF16 score byte span exceeds signed offsets"));
  }
  HYLE_ASSIGN_OR_RETURN(const auto bytes, out[0].required_bytes());
  if (bytes > std::size_t(INT64_MAX))
    return std::unexpected(Status::Unimplemented("dense BF16 score output exceeds signed offsets"));
  return Shape{unsigned(t), unsigned(l), unsigned(width), unsigned(l % width),
               unsigned(b), unsigned(h), unsigned(d), std::size_t(strides[0])};
}
Result<Shape> select(Operation operation, std::span<const TensorDesc> in,
                     std::span<const TensorDesc> out) noexcept {
  if (operation.custom_as<internal::DenseBfloatScores>())
    return select_dense(operation, in, out);
  const auto paged = operation.custom_as<detail::RowCachePagedIndexerScores>();
  if (!operation.custom_as<model_ops::RowCacheIndexerScores>() && !paged)
    return std::unexpected(
        Status::Unimplemented("CUDA packed indexer operation differs"));
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, in, inferred));
  if (out.size() != 1 || out[0] != inferred[0])
    return std::unexpected(
        Status::InvalidArgument("CUDA packed indexer output differs"));
  if (paged &&
      std::uint64_t(in[1].dimensions()[0]) !=
          std::uint64_t(paged->mapping.max_pages) * paged->mapping.page_size)
    return std::unexpected(Status::InvalidArgument(
        "paged indexer needs its truthful physical row capacity"));
  const auto t = std::uint64_t(in[0].dimensions()[0]),
             l = paged ? std::uint64_t(paged->score_columns)
                       : std::uint64_t(in[1].dimensions()[0]);
  const auto width = (std::min)(l, std::uint64_t{kTile});
  // Match the original GEMM/tensor/reduction launch metadata, not a packed-bank
  // size product.
  if (t > INT32_MAX / 64 || l > INT32_MAX ||
      (width && t > std::uint64_t(INT32_MAX) / (64 * width)))
    return std::unexpected(Status::Unimplemented(
        "CUDA packed indexer tile exceeds native integer fields"));
  return Shape{unsigned(t), unsigned(l), unsigned(width),
               width ? unsigned(l % width) : 0};
}
Status configure(Part &p, Cold &cold, unsigned tokens, unsigned width,
                 unsigned heads = 64, unsigned depth = 128) {
  HYLE_RETURN_IF_ERROR(lt_status(
      cublasLtMatmulDescCreate(&p.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F)));
  const std::uint64_t m = std::uint64_t(tokens) * heads;
  HYLE_RETURN_IF_ERROR(lt_status(
      cublasLtMatrixLayoutCreate(&p.layouts[0], CUDA_R_16BF, m, depth, depth)));
  const auto row = CUBLASLT_ORDER_ROW;
  HYLE_RETURN_IF_ERROR(lt_status(cublasLtMatrixLayoutSetAttribute(
      p.layouts[0], CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row))));
  HYLE_RETURN_IF_ERROR(lt_status(
      cublasLtMatrixLayoutCreate(&p.layouts[1], CUDA_R_16BF, depth, width, depth)));
  HYLE_RETURN_IF_ERROR(lt_status(cublasLtMatrixLayoutCreate(
      &p.layouts[2], CUDA_R_16BF, m, width, std::int64_t(m))));
  cublasLtMatmulHeuristicResult_t result{};
  int count = 0;
  HYLE_RETURN_IF_ERROR(lt_status(cublasLtMatmulAlgoGetHeuristic(
      cold.handle, p.operation, p.layouts[0], p.layouts[1], p.layouts[2],
      p.layouts[2], cold.preference, 1, &result, &count)));
  if (count != 1)
    return Status::Unimplemented("cuBLASLt found no BF16 indexer tile");
  HYLE_RETURN_IF_ERROR(lt_status(result.state));
  p.algorithm = result.algo;
  p.workspace = result.workspaceSize;
  // Column-major GEMM output is physical [key, token, head]. Head reduction
  // is contiguous; no transpose or complete-history decode is introduced.
  ops::TensorLayout layout{};
  layout.elements = tokens * heads * width;
  layout.rank = 3;
  layout.dimensions[0] = width;
  layout.dimensions[1] = tokens;
  layout.dimensions[2] = heads;
  layout.strides[0][0] = std::size_t(tokens) * heads * 2;
  layout.strides[0][1] = std::size_t(heads) * 2;
  layout.strides[0][2] = 2;
  HYLE_RETURN_IF_ERROR(cuda::status(ops::tensor_prepare(
      ops::TensorKind::kRelu, ops::TensorElementType::kBFloat16,
      ops::TensorElementType::kBFloat16, 2, true, layout, &p.relu)));
  layout.strides[1][0] = 0;
  layout.strides[1][1] = std::size_t(heads) * 2;
  layout.strides[1][2] = 2;
  HYLE_RETURN_IF_ERROR(cuda::status(ops::tensor_prepare(
      ops::TensorKind::kMultiply, ops::TensorElementType::kBFloat16,
      ops::TensorElementType::kBFloat16, 2, false, layout, &p.multiply)));
  ops::ReductionLayout reduction{};
  reduction.rows = tokens * width;
  reduction.width = heads;
  reduction.rank = 3;
  reduction.axes = 1u << 2;
  reduction.dimensions[0] = width;
  reduction.dimensions[1] = tokens;
  reduction.dimensions[2] = heads;
  // Sum uses the strided reduction path, including for a dense tile. The
  // physical GEMM result is [key, token, head]; zero strides repeat dots[0].
  reduction.input_strides[0] = std::size_t(tokens) * heads * 2;
  reduction.input_strides[1] = std::size_t(heads) * 2;
  reduction.input_strides[2] = 2;
  HYLE_RETURN_IF_ERROR(cuda::status(ops::reduction_prepare(
      ops::ElementType::kBFloat16, ops::ReductionKind::kSum,
      ops::ReductionMask::kNone, true, reduction, &p.sum)));
  return Status::Ok();
}
Status setup_cold(Cold &cold, std::size_t workspace,
                  std::uint32_t alignment = 16) {
  HYLE_RETURN_IF_ERROR(lt_status(cublasLtCreate(&cold.handle)));
  HYLE_RETURN_IF_ERROR(
      lt_status(cublasLtMatmulPreferenceCreate(&cold.preference)));
  HYLE_RETURN_IF_ERROR(lt_status(cublasLtMatmulPreferenceSetAttribute(
      cold.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
      sizeof(workspace))));
  const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  HYLE_RETURN_IF_ERROR(lt_status(cublasLtMatmulPreferenceSetAttribute(
      cold.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
      sizeof(reduction))));
  for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
    HYLE_RETURN_IF_ERROR(lt_status(cublasLtMatmulPreferenceSetAttribute(
        cold.preference, attr, &alignment, sizeof(alignment))));
  return Status::Ok();
}
Status configure_requests(void *prepared, void *invocation,
                          const void *configuration) noexcept {
  const auto &p = *static_cast<const Prepared *>(prepared);
  auto &slot = *static_cast<Invocation *>(invocation);
  slot.configured = false;
  if (!p.paged)
    return Status::InvalidArgument(
        "cold request spans require the paged indexer recipe");
  const auto &c =
      *static_cast<const detail::RowCacheColdLayout *>(configuration);
  if (c.requests > p.mapping.max_requests ||
      c.query_indptr.size() <= c.requests || c.kv_indptr.size() <= c.requests ||
      c.last_page_lengths.size() < c.requests || c.query_indptr[0] != 0 ||
      c.query_indptr[c.requests] != std::int64_t(p.shape.tokens) ||
      c.kv_indptr[0] != 0 || c.kv_indptr[c.requests] < 0 ||
      std::uint32_t(c.kv_indptr[c.requests]) > p.mapping.max_refs)
    return Status::InvalidArgument("paged indexer cold layout extent differs");
  cuda::detail::ScopedPrimaryContext scope(p.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  RequestPlans candidate(slot.requests.allocator);
  Cold cold;
  if (p.shape.tokens && p.shape.rows)
    HYLE_RETURN_IF_ERROR(setup_cold(cold, kLtLimit));
  try {
    for (unsigned r = 0; r < c.requests; ++r) {
      const auto begin = c.query_indptr[r], end = c.query_indptr[r + 1],
                 first = c.kv_indptr[r], last = c.kv_indptr[r + 1],
                 tail = c.last_page_lengths[r];
      if (begin < 0 || end < begin || std::uint64_t(end) > p.shape.tokens ||
          first < 0 || last < first ||
          std::uint32_t(last) > p.mapping.max_refs ||
          !((last == first && tail == 0) ||
            (last > first && tail > 0 &&
             std::uint32_t(tail) <= p.mapping.page_size)))
        return Status::InvalidArgument(
            "paged indexer cold request span differs");
      const auto length =
          last > first ? std::uint64_t(last - first - 1) * p.mapping.page_size +
                             std::uint32_t(tail)
                       : 0;
      if (length > p.shape.rows)
        return Status::OutOfRange(
            "paged indexer cold history exceeds score width");
      if (begin == end)
        continue;
      auto *request = candidate.allocator.construct<Request>();
      if (!request)
        return Status::ResourceExhausted("paged indexer request metadata");
      try {
        candidate.items.push_back(request);
      } catch (...) {
        candidate.allocator.destroy(request);
        throw;
      }
      request->request = r;
      request->begin = unsigned(begin);
      request->count = unsigned(end - begin);
      request->keys = unsigned(length);
      if (length) {
        request->tail_width = unsigned(length % p.shape.width);
        if (length >= p.shape.width)
          HYLE_RETURN_IF_ERROR(
              configure(request->full, cold, request->count, p.shape.width));
        if (request->tail_width)
          HYLE_RETURN_IF_ERROR(configure(request->tail, cold, request->count,
                                         request->tail_width));
      }
    }
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("paged indexer request vector");
  }
  slot.requests.items.swap(candidate.items);
  slot.configured = true;
  return Status::Ok();
}
void destroy(void *state, mem::AllocatorRef a) noexcept {
  auto *p = static_cast<Prepared *>(state);
  cuda::detail::ScopedPrimaryContext scope(p->primary);
  a.destroy(p);
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef a,
                         mem::BufferView workspace, void **output) noexcept {
  *output = nullptr;
  const auto &p = *static_cast<const Prepared *>(state);
  auto *slot = a.construct<Invocation>(a);
  if (!slot)
    return Status::ResourceExhausted("CUDA packed indexer invocation metadata");
  slot->primary = p.primary;
  *output = slot;
  if (!p.shape.tokens || !p.shape.rows || (p.dense && !p.shape.batches))
    return Status::Ok();
  if (workspace.size() < p.bytes)
    return Status::InvalidArgument("CUDA packed indexer workspace too small");
  HYLE_ASSIGN_OR_RETURN(
      slot->scratch, cuda::detail::mutable_pointer(workspace, kAlignment, 1));
  cuda::detail::ScopedPrimaryContext scope(p.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  return lt_status(cublasLtCreate(&slot->handle));
}
void destroy_invocation(void *state, mem::AllocatorRef a) noexcept {
  auto *p = static_cast<Invocation *>(state);
  if (!p)
    return;
  cuda::detail::ScopedPrimaryContext scope(p->primary);
  if (p->handle)
    (void)cublasLtDestroy(p->handle);
  a.destroy(p);
}
// One tile math path serves packed, paged and dense inputs. Only the caller's
// actual K source and final publication differ; all BF16 RNE stages stay here.
Status score_tile(Invocation &slot, const Part &part, const void *q,
                   const void *keys, const void *weights, void *dots,
                   void *sums, void *lt, cudaStream_t stream) noexcept {
  const float alpha = 1, beta = 0;
  HYLE_RETURN_IF_ERROR(lt_status(cublasLtMatmul(
      slot.handle, part.operation, &alpha, q, part.layouts[0], keys,
      part.layouts[1], &beta, dots, part.layouts[2], dots, part.layouts[2],
      &part.algorithm, lt, part.workspace, stream)));
  HYLE_RETURN_IF_ERROR(cuda::status(
      ops::tensor_launch(part.relu, dots, nullptr, nullptr, dots, stream)));
  HYLE_RETURN_IF_ERROR(cuda::status(
      ops::tensor_launch(part.multiply, dots, weights, nullptr, dots, stream)));
  return cuda::status(
      ops::reduction_launch(part.sum, dots, nullptr, sums, stream));
}
detail::EnqueueResult submit_dense(const Prepared &p, Invocation &slot,
                                    detail::QueueSequence &sequence,
                                    std::span<const ConstTensorView> inputs,
                                    std::span<const TensorView> outputs) noexcept {
  if (!p.shape.batches || !p.shape.tokens || !p.shape.rows)
    return detail::EnqueueResult::no_op();
  if (inputs.size() != (p.masked ? 4u : 3u) || outputs.size() != 1)
    return detail::EnqueueResult::rejected(
        Status::InvalidArgument("dense BF16 score binding count differs"));
  auto stream = cuda::stream(sequence);
  if (!stream)
    return detail::EnqueueResult::rejected(stream.error());
  std::array<const char *, 4> source{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].storage().size() < p.input_bytes[i])
      return detail::EnqueueResult::rejected(
          Status::OutOfRange("dense BF16 score input span is too small"));
    auto pointer = cuda::detail::const_pointer(inputs[i].storage(), 2, 2);
    if (!pointer)
      return detail::EnqueueResult::rejected(pointer.error());
    source[i] = static_cast<const char *>(*pointer);
  }
  if (outputs[0].storage().size() < p.output_bytes)
    return detail::EnqueueResult::rejected(
        Status::OutOfRange("dense BF16 score output span is too small"));
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
                                               p.float_output ? 4 : 2, 2);
  if (!output)
    return detail::EnqueueResult::rejected(output.error());
  const auto bounded = [](const void *pointer, std::size_t bytes) {
    return pointer && bytes <= UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(pointer);
  };
  const auto overlap = [](const void *a, std::size_t na,
                          const void *b, std::size_t nb) {
    const auto x = reinterpret_cast<std::uintptr_t>(a),
               y = reinterpret_cast<std::uintptr_t>(b);
    return x < y + nb && y < x + na;
  };
  if (!bounded(*output, p.output_bytes) || !bounded(slot.scratch, p.bytes) ||
      overlap(*output, p.output_bytes, slot.scratch, p.bytes))
    return detail::EnqueueResult::rejected(
        Status::InvalidArgument("dense BF16 score output/workspace range differs"));
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (!bounded(source[i], p.input_bytes[i]) ||
        overlap(source[i], p.input_bytes[i], *output, p.output_bytes) ||
        overlap(source[i], p.input_bytes[i], slot.scratch, p.bytes))
      return detail::EnqueueResult::rejected(
          Status::InvalidArgument("dense BF16 score input/output/workspace overlaps"));
  cuda::detail::ScopedPrimaryContext scope(p.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  auto *base = static_cast<char *>(slot.scratch);
  const auto q_batch = std::size_t(p.shape.tokens) * p.shape.heads * p.shape.depth * 2;
  const auto w_batch = std::size_t(p.shape.tokens) * p.shape.heads * 2;
  const auto score_elements = std::size_t(p.shape.tokens) * p.shape.rows;
  for (unsigned batch = 0; batch < p.shape.batches; ++batch) {
    const auto *q = source[0] + batch * q_batch;
    const auto *keys = source[1] + batch * p.shape.key_batch_bytes;
    const auto *weights = source[2] + batch * w_batch;
    const void *mask = p.masked ? source[3] +
        (p.mask_batches == 1 ? 0 : batch * score_elements * 2) : nullptr;
    void *scores = static_cast<char *>(*output) +
        batch * score_elements * (p.float_output ? 4 : 2);
    for (unsigned begin = 0; begin < p.shape.rows; begin += p.shape.width) {
      const auto width = (std::min)(p.shape.width, p.shape.rows - begin);
      const auto &part = width == p.shape.width ? p.full : p.tail;
      auto result = score_tile(slot, part, q,
          keys + std::size_t(begin) * p.shape.depth * 2, weights,
          base + p.dots, base + p.sums, base + p.lt, *stream);
      // A vendor call may own a real Queue prefix even when it reports failure.
      if (result.ok())
        result = cuda::status(native::publish_dense(
            base + p.sums, mask, p.shape.tokens, p.shape.rows, begin, width,
            scores, p.masked, p.float_output, *stream));
      if (!result.ok())
        return detail::EnqueueResult::accepted_command(result);
    }
  }
  return detail::EnqueueResult::accepted_command(Status::Ok());
}
detail::EnqueueResult submit(void *state, void *lane, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &p = *static_cast<const Prepared *>(state);
  auto &slot = *static_cast<Invocation *>(lane);
  if (p.dense)
    return submit_dense(p, slot, sequence, inputs, outputs);
  if (p.paged && !slot.configured)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "paged indexer needs its successful cold request layout"));
  if (!p.shape.tokens)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(sequence);
  if (!stream)
    return detail::EnqueueResult::rejected(stream.error());
  auto ranges = cuda::detail::const_pointer(inputs[3].storage(), 4, 1);
  if (!ranges)
    return detail::EnqueueResult::rejected(ranges.error());
  const void *metadata = nullptr;
  if (p.paged) {
    auto x = cuda::detail::const_pointer(inputs[4].storage(), 8, 1);
    if (!x)
      return detail::EnqueueResult::rejected(x.error());
    metadata = *x;
  }
  std::array<const void *, 3> source{};
  void *output = nullptr;
  if (p.shape.rows) {
    for (unsigned i = 0; i < 3; ++i) {
      auto x = cuda::detail::const_pointer(inputs[i].storage(),
                                           i == 0   ? 16
                                           : i == 1 ? 1
                                                    : 2,
                                           1);
      if (!x)
        return detail::EnqueueResult::rejected(x.error());
      source[i] = *x;
    }
    auto x = cuda::detail::mutable_pointer(outputs[0].storage(), 4, 1);
    if (!x)
      return detail::EnqueueResult::rejected(x.error());
    output = *x;
  }
  cuda::detail::ScopedPrimaryContext scope(p.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  unsigned *status = nullptr;
  auto initialized = cuda::detail::acquire_device_status(sequence, &status);
  if (!initialized.status.ok())
    return initialized;
  // From this point even a status reset or failed vendor call may own a prefix.
  auto result = cuda::status(
      p.paged ? row_cache::native::check_paged_ranges(
                    p.mapping, p.shape.rows,
                    static_cast<const std::int32_t *>(*ranges), metadata,
                    status, *stream)
              : native::check_ranges(static_cast<const std::int32_t *>(*ranges),
                                     p.shape.tokens, p.shape.rows, status,
                                     *stream));
  if (!result.ok() || !p.shape.rows)
    return detail::EnqueueResult::accepted_command(result);
  auto *base = static_cast<char *>(slot.scratch);
  auto *keys = base;
  auto *dots = base + p.dots;
  auto *sums = base + p.sums;
  auto *active = reinterpret_cast<unsigned char *>(base + p.active);
  void *lt = base + p.lt;
  const auto *intervals = static_cast<const std::int32_t *>(*ranges);
  if (p.paged) {
    result = cuda::status(native::initialize_scores(
        static_cast<float *>(output),
        std::uint64_t(p.shape.tokens) * p.shape.rows, status, *stream));
    if (!result.ok())
      return detail::EnqueueResult::accepted_command(result);
    auto *ids = reinterpret_cast<std::int32_t *>(active);
    for (const auto *request : slot.requests.items) {
      // Each actual request shares one decoded K tile among only its own Q
      // span. Sparse ranges still do GEMM over the declared history tiles.
      for (unsigned begin = 0; begin < request->keys; begin += p.shape.width) {
        const auto width = (std::min)(p.shape.width, request->keys - begin);
        const auto &part =
            width == p.shape.width ? request->full : request->tail;
        result = cuda::status(row_cache::native::request_tile(
            p.mapping, request->request, request->begin, request->count, begin,
            width, intervals, metadata, ids, status, *stream));
        const row_cache::native::Program decode{p.bank_rows, width, false, true,
                                                true};
        if (result.ok())
          result = cuda::status(row_cache::native::launch_padded(
              decode, source[1], ids, keys, status, *stream));
        const auto *q = static_cast<const char *>(source[0]) +
                        std::size_t(request->begin) * 64 * 128 * 2;
        const auto *w = static_cast<const char *>(source[2]) +
                        std::size_t(request->begin) * 64 * 2;
        if (result.ok())
          result = score_tile(slot, part, q, keys, w, dots, sums, lt, *stream);
        if (result.ok())
          result = cuda::status(
              native::publish(sums, intervals + std::size_t(request->begin) * 2,
                              request->count, p.shape.rows, begin, width,
                              static_cast<float *>(output) +
                                  std::size_t(request->begin) * p.shape.rows,
                              status, *stream));
        if (!result.ok())
          return detail::EnqueueResult::accepted_command(result);
      }
    }
    return detail::EnqueueResult::accepted_command(Status::Ok());
  }
  for (unsigned begin = 0; begin < p.shape.rows; begin += p.shape.width) {
    const auto width = (std::min)(p.shape.width, p.shape.rows - begin);
    const auto &part = width == p.shape.width ? p.full : p.tail;
    result = cuda::status(native::active_keys(intervals, p.shape.tokens, begin,
                                              width, active, status, *stream));
    if (result.ok())
      result = cuda::status(row_cache::native::launch_float4_tile(
          static_cast<const char *>(source[1]) + std::size_t(begin) * 68,
          active, keys, width, status, *stream));
    if (result.ok())
      result = score_tile(slot, part, source[0], keys, source[2], dots, sums,
                            lt, *stream);
    if (result.ok())
      result = cuda::status(native::publish(
          sums, intervals, p.shape.tokens, p.shape.rows, begin, width,
          static_cast<float *>(output), status, *stream));
    if (!result.ok())
      return detail::EnqueueResult::accepted_command(result);
  }
  return detail::EnqueueResult::accepted_command(Status::Ok());
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace
Status propose(const detail::ProviderOperationRequest &r,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *out) noexcept {
  if (!r.shape.exact())
    return Status::Unimplemented("CUDA packed indexer needs exact shapes");
  HYLE_ASSIGN_OR_RETURN(auto shape, select(r.operation, r.inputs, r.outputs));
  const auto dense = r.operation.custom_as<internal::DenseBfloatScores>();
  const bool paged =
      r.operation.custom_as<detail::RowCachePagedIndexerScores>().has_value();
  if (dense) {
    if (!out || inputs.size() != (dense->masked ? 4u : 3u) || outputs.size() != 1)
      return Status::InvalidArgument("dense BF16 score binding count differs");
    for (auto &input : inputs)
      input = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
               .minimum_alignment = 2};
    outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                  .minimum_alignment = dense->output_type == DataType::kFloat32 ? 4u : 2u};
    // Backend execution episodes, not a claim about Lt's physical kernel count.
    // Empty shared_submit is one no-op episode (the route cost forbids zero).
    const auto episodes = shape.batches == 0 ? 1u :
        shape.batches * (shape.rows / kTile + (shape.rows % kTile != 0)) * 5;
    *out = {.route = kDenseRoute, .runtime = kRuntime,
            .workspace_resource = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                                   .minimum_alignment = kAlignment},
            .cost = {.workspace = std::nullopt, .persistent_bytes = std::nullopt,
                     .copy_bytes = 0, .repack_bytes = 0, .synchronization_count = 0,
                     .dispatch_count = episodes},
            .implementation = ProviderImplementationId{kDenseRoute}};
    return Status::Ok();
  }
  if (!out || inputs.size() != (paged ? 5u : 4u) || outputs.size() != 1)
    return Status::InvalidArgument("CUDA packed indexer binding count differs");
  for (unsigned i = 0; i < 4; ++i)
    inputs[i] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                 .minimum_alignment = i == 0   ? 16u
                                      : i == 1 ? 1u
                                      : i == 2 ? 2u
                                               : 4u};
  if (paged)
    inputs[4] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                 .minimum_alignment = 8};
  outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment = 4};
  auto runtime = kRuntime;
  if (paged)
    runtime.capture_compatible = false;
  *out = {.route = kRoute,
          .runtime = runtime,
          .workspace_resource = {.required_properties =
                                     mem::MemoryProperty::kDeviceAccessible,
                                 .minimum_alignment = kAlignment},
          .cost = {.workspace = std::nullopt,
                   .persistent_bytes = std::nullopt,
                   .copy_bytes = 0,
                   .repack_bytes = 0,
                   .synchronization_count = 0,
                   .dispatch_count = std::nullopt},
          .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &r,
                               mem::AllocatorRef a,
                               detail::ProviderPrepared *out) noexcept {
  *out = {};
  auto shape = select(r.operation, r.inputs, r.outputs);
  if (!shape)
    return shape.error().IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(shape.error())
               : ProviderPrepareOutcome::error(shape.error());
  auto *p = a.construct<Prepared>();
  if (!p)
    return ProviderPrepareOutcome::error(
        Status::ResourceExhausted("CUDA packed indexer metadata"));
  p->primary = static_cast<CUcontext>(state);
  p->shape = *shape;
  if (const auto dense = r.operation.custom_as<internal::DenseBfloatScores>()) {
    p->dense = true;
    p->masked = dense->masked;
    p->float_output = dense->output_type == DataType::kFloat32;
    p->mask_batches = p->masked && shape->batches
                          ? unsigned(r.inputs[3].dimensions()[0]) : 0;
    for (std::size_t i = 0; i < r.inputs.size(); ++i) {
      const auto bytes = r.inputs[i].required_bytes();
      if (!bytes) {
        a.destroy(p);
        return ProviderPrepareOutcome::error(bytes.error());
      }
      p->input_bytes[i] = *bytes;
    }
    const auto bytes = r.outputs[0].required_bytes();
    if (!bytes) {
      a.destroy(p);
      return ProviderPrepareOutcome::error(bytes.error());
    }
    p->output_bytes = *bytes;
  }
  if (const auto paged =
          r.operation.custom_as<detail::RowCachePagedIndexerScores>()) {
    p->paged = true;
    p->bank_rows = std::uint64_t(r.inputs[1].dimensions()[0]);
    const auto &m = paged->mapping;
    p->mapping = {shape->tokens,    0,
                  m.query_offset,   m.kv_offset,
                  m.page_offset,    m.tail_offset,
                  m.page_size,      m.max_pages,
                  m.max_requests(), m.max_refs()};
  }
  const auto setup = [&]() -> Status {
    // A zero-size dense result must not query a payload or initialize Lt/CUDA
    // functions, regardless of otherwise unused positive H/D or batch pitches.
    if (p->dense && !shape->batches)
      return Status::Ok();
    cuda::detail::ScopedPrimaryContext scope(p->primary);
    HYLE_RETURN_IF_ERROR(scope.status());
    HYLE_RETURN_IF_ERROR(cuda::status(p->dense
        ? native::prepare_dense(p->masked, p->float_output) : native::prepare()));
    if (p->paged)
      HYLE_RETURN_IF_ERROR(
          cuda::status(row_cache::native::prepare_request_tiles()));
    if (!shape->tokens || !shape->rows)
      return Status::Ok();
    if (!p->dense)
      HYLE_RETURN_IF_ERROR(cuda::status(
        p->paged ? row_cache::native::prepare_padded(
                       {p->bank_rows, shape->width, false, true, true})
                 : row_cache::native::prepare_float4_tile()));
    if (!p->paged) {
      Cold cold;
      HYLE_RETURN_IF_ERROR(setup_cold(cold, kLtLimit, p->dense ? 2u : 16u));
      HYLE_RETURN_IF_ERROR(configure(p->full, cold, shape->tokens, shape->width,
                                       shape->heads, shape->depth));
      if (shape->tail)
        HYLE_RETURN_IF_ERROR(configure(p->tail, cold, shape->tokens, shape->tail,
                                         shape->heads, shape->depth));
    }
    if (p->dense) {
      const auto tile_bytes = std::uint64_t(shape->width) * shape->tokens *
                              shape->heads * 2;
      const auto sum_bytes = std::uint64_t(shape->width) * shape->tokens * 2;
      const auto workspace = (std::max)(p->full.workspace, p->tail.workspace);
      const auto limit = (std::numeric_limits<std::size_t>::max)();
      if (tile_bytes > limit - (kAlignment - 1) ||
          sum_bytes > limit - (kAlignment - 1) || workspace > kLtLimit)
        return Status::ResourceExhausted("dense BF16 tile workspace overflow");
      p->dots = 0;
      p->sums = aligned(std::size_t(tile_bytes));
      if (sum_bytes > limit - p->sums - (kAlignment - 1))
        return Status::ResourceExhausted("dense BF16 sum workspace overflow");
      p->lt = aligned(p->sums + std::size_t(sum_bytes));
      if (workspace > limit - p->lt)
        return Status::ResourceExhausted("dense BF16 Lt workspace overflow");
      p->bytes = p->lt + workspace;
      return Status::Ok();
    }
    p->dots = aligned(std::size_t(shape->width) * 128 * 2);
    p->sums =
        aligned(p->dots + std::size_t(shape->width) * shape->tokens * 64 * 2);
    p->active =
        aligned(p->sums + std::size_t(shape->width) * shape->tokens * 2);
    p->lt = aligned(p->active + std::size_t(shape->width) * (p->paged ? 4 : 1));
    // Request qcounts become known only during the typed cold layout update.
    // The existing Lt algorithm ceiling is explicit caller workspace, never a
    // runtime private allocation. Original contiguous recipes keep exact size.
    p->bytes =
        p->lt + (p->paged ? kLtLimit
                          : (std::max)(p->full.workspace, p->tail.workspace));
    return Status::Ok();
  }();
  if (!setup.ok()) {
    a.destroy(p);
    return setup.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(setup)
                                   : ProviderPrepareOutcome::error(setup);
  }
  auto runtime = kRuntime;
  if (p->paged)
    runtime.capture_compatible = false;
  *out = {.vtable = &kPrepared,
          .state = p,
          .workspace = {.size = p->bytes, .alignment = kAlignment},
          .runtime = runtime,
          .persistent_bytes = std::nullopt,
          .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
bool inspect_access(const detail::ProviderPrepared &prepared,
                    detail::OperationPreparedAccess &access) noexcept {
  if (prepared.vtable != &kPrepared || !prepared.state ||
      !static_cast<const Prepared *>(prepared.state)->paged) return false;
  access = {.projected_input = 1, .allows_empty_footprint = true,
            .requires_configuration = true};
  return true;
}
Status configure_access(const detail::ProviderPrepared &prepared, void *invocation,
                        const void *configuration) noexcept {
  detail::OperationPreparedAccess access;
  if (!inspect_access(prepared, access))
    return Status::Unimplemented("selected CUDA implementation has no request layout");
  return configure_requests(prepared.state, invocation, configuration);
}
} // namespace hyle::compute::cuda_native::row_cache_indexer
