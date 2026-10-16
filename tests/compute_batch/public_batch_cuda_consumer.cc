#include <hyle/compute/batch_execution.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/graph/batch.h>
#include <hyle/simd/dtype.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#if !defined(HYLE_COMPUTE_FLASHINFER)
#error public CUDA batch consumer requires qualified Linux FlashInfer
#endif

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
using hyle::Status;
Status check(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
std::uint16_t encode(float value, bool bf16) {
  return bf16 ? hyle::simd::f32_to_bf16(value).bits
              : hyle::simd::f32_to_f16(value).bits;
}
float decode(std::uint16_t value, bool bf16) {
  return bf16 ? hyle::simd::bf16_to_f32({value})
              : hyle::simd::f16_to_f32({value});
}
hyle::Result<c::ReadLease> load(c::LeasePool &pool,
                              std::span<const std::uint16_t> words) {
  HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(words.size_bytes()));
  HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(words)));
  return writer.publish();
}
Status oracle(const c::BatchCapacity &cap, const c::BatchLayoutVersion &layout,
              const std::vector<std::uint16_t> &q,
              const std::vector<std::uint16_t> &k,
              const std::vector<std::uint16_t> &v,
              const c::ReadLease &output, bool bf16) {
  std::vector<std::uint16_t> result(q.size());
  HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(result))));
  const auto host = layout.host();
  double maximum = 0;
  for (unsigned r = 0; r < layout.facts().requests; ++r) {
    const int qlen = host.query_indptr[r + 1] - host.query_indptr[r];
    const int count = host.kv_indptr[r + 1] - host.kv_indptr[r];
    const int kvlen = cap.storage == c::BatchStorage::kPaged
                          ? (count - 1) * int(cap.page_size) + host.last_page_lengths[r]
                          : count;
    const auto row = [&](int i) {
      return cap.storage == c::BatchStorage::kPaged
                 ? host.page_indices[host.kv_indptr[r] + i / cap.page_size] *
                           int(cap.page_size) + i % cap.page_size
                 : host.kv_indptr[r] + i;
    };
    for (int qi = 0; qi < qlen; ++qi) {
      for (unsigned h = 0; h < cap.query_heads; ++h) {
        const int valid = kvlen - qlen + qi + 1;
        std::vector<double> scores(valid);
        double high = -1e100, sum = 0;
        const auto qbase = ((host.query_indptr[r] + qi) * cap.query_heads + h) * cap.qk_head_dim;
        const auto kvbase = [&](int i) {
          return (row(i) * cap.kv_heads + h / (cap.query_heads / cap.kv_heads)) * cap.qk_head_dim;
        };
        for (int i = 0; i < valid; ++i) {
          double dot = 0;
          for (unsigned d = 0; d < cap.qk_head_dim; ++d)
            dot += double(decode(q[qbase + d], bf16)) * decode(k[kvbase(i) + d], bf16);
          scores[i] = dot / std::sqrt(double(cap.qk_head_dim));
          high = std::max(high, scores[i]);
        }
        for (auto &value : scores) { value = std::exp(value - high); sum += value; }
        for (unsigned d = 0; d < cap.qk_head_dim; ++d) {
          double expected = 0;
          for (int i = 0; i < valid; ++i)
            expected += scores[i] / sum * decode(v[kvbase(i) + d], bf16);
          const double error = std::abs(decode(result[qbase + d], bf16) - expected);
          maximum = std::max(maximum, error);
          HYLE_RETURN_IF_ERROR(check(std::isfinite(error) && error < .007,
                                    "public CUDA independent attention oracle differs"));
        }
      }
    }
  }
  std::printf("public CUDA nonuniform oracle maxError=%.9g PASS\n", maximum);
  return Status::Ok();
}
Status exercise(bool paged, bool bf16) {
  constexpr unsigned depth = 256, page = 16;
  const c::BatchCapacity cap{
      .storage = paged ? c::BatchStorage::kPaged : c::BatchStorage::kRagged,
      .query_type=bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat16,.output_type=bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat16,.key_storage=c::CacheStorage::plain(bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat16).value(),.value_storage=c::CacheStorage::plain(bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat16).value(),
      .max_requests = 2, .max_query_rows = 4, .max_kv_rows = 32,
      .max_pages = paged ? 4u : 0u, .max_page_refs = paged ? 3u : 0u,
      .query_heads = 4, .kv_heads = 2, .qk_head_dim=depth,.value_head_dim=depth,
      .page_size = paged ? page : 1u};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(cap));
  const std::array<std::int32_t, 3> qptr{0, 1, 3}, kvptr{0, paged ? 2 : 19, paged ? 3 : 29};
  const std::array<std::int32_t, 3> ids{2, 0, 1};
  const std::array<std::int32_t, 2> tails{3, 10};
  HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({
      qptr, kvptr, paged ? std::span(ids) : std::span<const std::int32_t>{},
      paged ? std::span(tails) : std::span<const std::int32_t>{}}));
  const c::BatchProgramOptions options{
      .resource={},
      .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10,
      .max_invocations = 2, .scale = 1.f / std::sqrt(float(depth))};
  HYLE_ASSIGN_OR_RETURN(auto wrapped, gb::provider(c::flashinfer::provider(), schema, options));
  HYLE_ASSIGN_OR_RETURN(auto device_id, c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(device_id, wrapped));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto device, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
  struct Drain { c::Context &context; ~Drain() { context.drain().IgnoreError(); } } drain{context};
  auto direct_options = options;
  direct_options.resource = device.ref();
  direct_options.host_workspace = host.ref();
  HYLE_ASSIGN_OR_RETURN(auto program, c::BatchProgram::prepare(context, schema, direct_options));
  HYLE_ASSIGN_OR_RETURN(auto direct, c::BatchInvocation::create(program, queue));
  std::vector<std::uint16_t> q(cap.max_query_rows * cap.query_heads * depth);
  std::vector<std::uint16_t> k((paged ? cap.max_pages * page : cap.max_kv_rows) * cap.kv_heads * depth), v(k.size());
  for (unsigned i = 0; i < q.size(); ++i) q[i] = encode(.2f * std::sin(float(i) * .017f), bf16);
  for (unsigned i = 0; i < k.size(); ++i) {
    k[i] = encode(.3f * std::cos(float(i) * .019f), bf16);
    v[i] = encode(std::sin(float(i) * .013f) + float(i % 11) * .01f, bf16);
  }
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device.ref(), (q.size() * 3 + k.size() + v.size()) * 2 + 4096));
  HYLE_ASSIGN_OR_RETURN(auto rq, load(pool, q));
  HYLE_ASSIGN_OR_RETURN(auto rk, load(pool, k));
  HYLE_ASSIGN_OR_RETURN(auto rv, load(pool, v));
  const std::array leases{rq, rk, rv};
  HYLE_ASSIGN_OR_RETURN(auto prepared, direct.prepare_batch(layout));
  HYLE_ASSIGN_OR_RETURN(auto done, prepared.run(leases));
  HYLE_RETURN_IF_ERROR(done.wait());
  HYLE_ASSIGN_OR_RETURN(auto direct_output, direct.output());
  HYLE_RETURN_IF_ERROR(oracle(cap, layout, q, k, v, direct_output, bf16));
  HYLE_RETURN_IF_ERROR(check(!direct.prepare_batch(layout), "public Direct old output allowed overwrite"));
  HYLE_RETURN_IF_ERROR(check(direct.report().layout_generation == layout.generation(), "public Direct mirror generation differs"));

  HYLE_ASSIGN_OR_RETURN(auto qdesc, c::TensorDesc::dense(cap.query_type, std::array<std::int64_t, 3>{cap.max_query_rows, cap.query_heads, depth}));
  c::TensorDesc kvdesc;
  if (paged) { HYLE_ASSIGN_OR_RETURN(kvdesc, c::TensorDesc::dense(cap.query_type, std::array<std::int64_t, 4>{cap.max_pages, page, cap.kv_heads, depth})); }
  else { HYLE_ASSIGN_OR_RETURN(kvdesc, c::TensorDesc::dense(cap.query_type, std::array<std::int64_t, 3>{cap.max_kv_rows, cap.kv_heads, depth})); }
  HYLE_ASSIGN_OR_RETURN(auto metadata_desc, schema.metadata_description());
  const std::array descriptions{qdesc, kvdesc, kvdesc, metadata_desc};
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  std::array<g::ValueId, 4> inputs;
  for (unsigned i = 0; i < inputs.size(); ++i) { HYLE_ASSIGN_OR_RETURN(inputs[i], builder.input(descriptions[i])); }
  const auto operation = paged ? c::Operation::custom(c::PagedAttention{cap, options.scale})
                               : c::Operation::custom(c::RaggedAttention{cap, options.scale});
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(operation, inputs));
  HYLE_ASSIGN_OR_RETURN(auto output_id, builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(output_id));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array storage{g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{&context, g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(graph, {storage, targets, {}}, {.operation_fallback = g::OperationFallbackPolicy::kReject}));
  HYLE_ASSIGN_OR_RETURN(auto invocation, gb::Invocation::create(compiled.executable, std::array{gb::LaneBinding{g::ExecutionLaneId{0}, &context, &queue, host.ref()}}));
  const std::array layouts{gb::NodeLayout{node, layout}};
  HYLE_RETURN_IF_ERROR(invocation.prepare_batch(layouts));
  for (unsigned i = 0; i < leases.size(); ++i) HYLE_RETURN_IF_ERROR(invocation.bind(inputs[i], descriptions[i], leases[i]));
  HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(q.size() * 2));
  HYLE_RETURN_IF_ERROR(invocation.bind(output_id, qdesc, std::move(writer)));
  HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit());
  HYLE_RETURN_IF_ERROR(graph_done.wait());
  HYLE_ASSIGN_OR_RETURN(auto graph_output, invocation.output(output_id));
  HYLE_ASSIGN_OR_RETURN(auto result, graph_output.storage_lease());
  HYLE_RETURN_IF_ERROR(oracle(cap, layout, q, k, v, result, bf16));
  HYLE_RETURN_IF_ERROR(check(!invocation.prepare_batch(layouts).ok(), "public Graph old output allowed overwrite"));
  HYLE_ASSIGN_OR_RETURN(auto report, invocation.report(node));
  HYLE_RETURN_IF_ERROR(check(report.layout_generation == layout.generation(), "public Graph mirror generation differs"));
  std::printf("public CUDA FlashInfer %s %s D256 Direct/Graph GQA/causal/active-capacity/page-tail/old-output PASS\n",
              paged ? "paged" : "ragged", bf16 ? "BF16" : "F16");
  return Status::Ok();
}
Status run() {
  for (bool paged : {false, true}) for (bool bf16 : {false, true}) HYLE_RETURN_IF_ERROR(exercise(paged, bf16));
  return Status::Ok();
}
} // namespace
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const auto status = run();
  if (status.ok()) return 0;
  char message[768]{};
  status.ToString(message, sizeof(message));
  std::fprintf(stderr, "public CUDA batch: %s\n", message);
  return 1;
}
