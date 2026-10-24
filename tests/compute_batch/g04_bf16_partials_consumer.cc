#include <hyle/graph/cuda.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/graph/batch.h>
#include <hyle/simd/dtype.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

// A public SDK consumer for the opt-in partial policy. Its exact 11-MiB
// assertion targets the retained SM89/46-SM D256 case. It performs no timing.
namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
using hyle::Status;

Status require(bool value, const char* message) {
  return value ? Status::Ok() : Status::Internal(message);
}
double decode_bf16(std::uint16_t word) {
  return std::bit_cast<float>(std::uint32_t(word) << 16);
}
double decode_fp8(std::uint8_t code) {
  const unsigned exponent = (code & 127) >> 3, mantissa = code & 7;
  const double value = exponent ? std::ldexp(1. + double(mantissa) / 8., int(exponent) - 7)
                                : std::ldexp(double(mantissa) / 8., -6);
  return code & 128 ? -value : value;
}

struct Data {
  c::BatchCapacity capacity;
  std::array<std::int32_t, 3> qptr, kptr;
  std::array<std::int32_t, 2> tails;
  std::vector<std::int32_t> ids;
  std::vector<std::uint16_t> query;
  std::array<std::vector<std::uint8_t>, 2> cache;
  float scale;
  std::int32_t window;

  Data(unsigned depth, unsigned q1, unsigned q2, unsigned k1, unsigned k2, int left)
      : capacity{.storage = c::BatchStorage::kPaged,
          .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
          .key_storage = c::CacheStorage::float8(.5F).value(),
          .value_storage = c::CacheStorage::float8(2.F).value(),
          .max_requests = 2, .max_query_rows = q1 + q2 + 3,
          .max_kv_rows = k1 + k2 + 13,
          .max_pages = (k1 + 15) / 16 + (k2 + 15) / 16 + 2,
          .max_page_refs = (k1 + 15) / 16 + (k2 + 15) / 16,
          .query_heads = 8, .kv_heads = 2,
          .qk_head_dim = depth, .value_head_dim = depth, .page_size = 16},
        qptr{0, std::int32_t(q1), std::int32_t(q1 + q2)},
        kptr{0, std::int32_t((k1 + 15) / 16),
             std::int32_t((k1 + 15) / 16 + (k2 + 15) / 16)},
        tails{std::int32_t((k1 - 1) % 16 + 1), std::int32_t((k2 - 1) % 16 + 1)},
        scale(1.F / std::sqrt(float(depth))), window(left) {
    for (unsigned i = 0; i < capacity.max_page_refs; ++i)
      ids.push_back(std::int32_t(capacity.max_pages - 1 - i));
    ids[unsigned(kptr[1])] = ids[0];
    query.resize(std::size_t(capacity.max_query_rows) * 8 * depth);
    for (std::size_t i = 0; i < query.size(); ++i)
      query[i] = hyle::simd::f32_to_bf16(.125F + float((i / depth) % 8 % 3) * .015625F).bits;
    constexpr std::uint8_t key_codes[]{0, 56, 64, 68, 72};
    constexpr std::uint8_t value_codes[]{16, 24, 28, 32, 34};
    for (unsigned port = 0; port < 2; ++port) {
      cache[port].resize(std::size_t(capacity.max_pages) * 16 * 2 * depth);
      for (std::size_t i = 0; i < cache[port].size(); ++i) {
        const auto row = i / (2 * depth), head = (i / depth) % 2;
        const auto feature = port ? (i % depth) % 7 : 0;
        cache[port][i] = (port ? value_codes : key_codes)[(row + head + feature) % 5];
      }
      for (unsigned page = 0; page < capacity.max_pages; ++page) {
        unsigned valid = 0;
        for (unsigned r = 0; r < 2; ++r)
          for (auto ref = kptr[r]; ref < kptr[r + 1]; ++ref)
            if (ids[unsigned(ref)] == std::int32_t(page))
              valid = std::max(valid, ref + 1 == kptr[r + 1] ? unsigned(tails[r]) : 16u);
        std::fill(cache[port].begin() + std::size_t(page * 16 + valid) * 2 * depth,
                  cache[port].begin() + std::size_t((page + 1) * 16) * 2 * depth, std::uint8_t{0x78});
      }
    }
  }

  // Q/K are constant over each depth vector; V has seven interleaved feature
  // classes to detect output/depth layout errors. Compute scores once, then
  // independently decoded prefix sums for each actual causal row/window.
  std::vector<double> oracle() const {
    const auto depth = capacity.qk_head_dim;
    std::vector<double> result(std::size_t(qptr[2]) * 8 * 7);
    for (unsigned r = 0; r < 2; ++r) {
      const auto nq = unsigned(qptr[r + 1] - qptr[r]);
      const auto nk = unsigned((kptr[r + 1] - kptr[r] - 1) * 16 + tails[r]);
      for (unsigned head = 0; head < 8; ++head) {
        std::vector<double> weights(nk + 1);
        std::array<std::vector<double>, 7> moments;
        for (auto& feature : moments) feature.resize(nk + 1);
        const double q = decode_bf16(query[(std::size_t(qptr[r]) * 8 + head) * depth]);
        for (unsigned j = 0; j < nk; ++j) {
          const auto physical = unsigned(ids[unsigned(kptr[r]) + j / 16]) * 16 + j % 16;
          const auto offset = (std::size_t(physical) * 2 + head / 4) * depth;
          const double k = decode_fp8(cache[0][offset]) * .5;
          const double weight = std::exp(q * k * depth * scale);
          weights[j + 1] = weights[j] + weight;
          for (unsigned feature = 0; feature < 7; ++feature)
            moments[feature][j + 1] = moments[feature][j] +
                weight * decode_fp8(cache[1][offset + feature]) * 2.;
        }
        for (unsigned qi = 0; qi < nq; ++qi) {
          const auto end = nk - nq + qi + 1;
          const auto begin = window < 0 ? 0u : end > unsigned(window + 1) ? end - unsigned(window + 1) : 0u;
          for (unsigned feature = 0; feature < 7; ++feature)
            result[((std::size_t(qptr[r]) + qi) * 8 + head) * 7 + feature] =
                (moments[feature][end] - moments[feature][begin]) / (weights[end] - weights[begin]);
        }
      }
    }
    return result;
  }
};

struct Fixture {
  c::Context context;
  c::cuda::MemoryResource device, host;
  c::Queue queue;
  c::BatchSchema schema;
  c::BatchLayoutVersion layout;
  c::LeasePool query_pool;
  c::ReadLease query;
  c::PagedArena keys, values;
  std::vector<c::ReadLease> key_pages, value_pages;
  c::PagedArenaRead key_frame, value_frame;
  ~Fixture() { if (queue.valid()) queue.drain().IgnoreError(); }
};

c::BatchProgramOptions options(const Data& data, bool allow, std::size_t bytes) {
  c::BatchProgramOptions result{.resource = {}, .max_float_workspace_bytes = bytes,
      .max_int_workspace_bytes = 4096, .max_invocations = 2,
      .scale = data.scale, .causal = true, .window_left = data.window};
  result.allow_bf16_partials = allow;
  return result;
}

hyle::Result<std::unique_ptr<Fixture>> fixture(const Data& data) {
  auto f = std::make_unique<Fixture>();
  HYLE_ASSIGN_OR_RETURN(f->schema, c::BatchSchema::create(data.capacity));
  HYLE_ASSIGN_OR_RETURN(f->layout, f->schema.version({data.qptr, data.kptr, data.ids, data.tails}));
  const auto provider = c::flashinfer::provider();
  HYLE_ASSIGN_OR_RETURN(auto id, c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(f->context, c::Context::create(id, provider));
  HYLE_ASSIGN_OR_RETURN(f->queue, f->context.create_queue());
  HYLE_ASSIGN_OR_RETURN(f->device, c::cuda::MemoryResource::create(f->context, c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(f->host, c::cuda::MemoryResource::create(f->context, c::cuda::MemoryKind::kPinnedHost));
  const auto page_bytes = std::size_t(16) * 2 * data.capacity.qk_head_dim;
  HYLE_ASSIGN_OR_RETURN(f->keys, c::PagedArena::create(f->schema, c::BatchCacheRole::kKey,
      f->device.ref(), {.max_backing_bytes = page_bytes * data.capacity.max_pages + 1}));
  HYLE_ASSIGN_OR_RETURN(f->values, c::PagedArena::create(f->schema, c::BatchCacheRole::kValue,
      f->device.ref(), {.max_backing_bytes = page_bytes * data.capacity.max_pages + 1}));
  for (unsigned page = 0; page < data.capacity.max_pages; ++page) {
    if (std::find(data.ids.begin(), data.ids.end(), std::int32_t(page)) == data.ids.end()) continue;
    HYLE_ASSIGN_OR_RETURN(auto kw, f->keys.acquire(page));
    HYLE_ASSIGN_OR_RETURN(auto vw, f->values.acquire(page));
    HYLE_ASSIGN_OR_RETURN(auto kv, kw.lease().view());
    HYLE_ASSIGN_OR_RETURN(auto vv, vw.lease().view());
    HYLE_RETURN_IF_ERROR(kv.write(std::as_bytes(std::span(data.cache[0]).subspan(page * page_bytes, page_bytes))));
    HYLE_RETURN_IF_ERROR(vv.write(std::as_bytes(std::span(data.cache[1]).subspan(page * page_bytes, page_bytes))));
    HYLE_ASSIGN_OR_RETURN(auto kr, kw.publish(16));
    HYLE_ASSIGN_OR_RETURN(auto vr, vw.publish(16));
    f->key_pages.push_back(std::move(kr)); f->value_pages.push_back(std::move(vr));
  }
  HYLE_ASSIGN_OR_RETURN(f->key_frame, f->keys.read(f->layout, f->key_pages));
  HYLE_ASSIGN_OR_RETURN(f->value_frame, f->values.read(f->layout, f->value_pages));
  HYLE_ASSIGN_OR_RETURN(f->query_pool, c::LeasePool::create(f->device.ref(), data.query.size() * 2 + 512));
  HYLE_ASSIGN_OR_RETURN(auto writer, f->query_pool.allocate(data.query.size() * 2));
  HYLE_ASSIGN_OR_RETURN(auto view, writer.view());
  HYLE_RETURN_IF_ERROR(view.write(std::as_bytes(std::span(data.query))));
  HYLE_ASSIGN_OR_RETURN(f->query, writer.publish());
  return f;
}

Status check_output(const Data& data, const c::ReadLease& output) {
  const auto expected = data.oracle();
  std::vector<std::uint16_t> actual(std::size_t(data.capacity.max_query_rows) * 8 * data.capacity.value_head_dim);
  HYLE_ASSIGN_OR_RETURN(auto view, output.view());
  HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(actual))));
  double maximum = 0;
  for (std::size_t row_head = 0; row_head < expected.size() / 7; ++row_head)
    for (unsigned d = 0; d < data.capacity.value_head_dim; ++d) {
      const auto value = decode_bf16(actual[row_head * data.capacity.value_head_dim + d]);
      HYLE_RETURN_IF_ERROR(require(std::isfinite(value), "policy output is nonfinite"));
      maximum = std::max(maximum, std::abs(value - expected[row_head * 7 + d % 7]));
    }
  HYLE_RETURN_IF_ERROR(require(maximum < .008, "policy independent encoded-input oracle differs"));
  std::printf("POLICY_ORACLE max_abs=%.9g elements=%zu\n", maximum,
              expected.size() / 7 * data.capacity.value_head_dim);
  return Status::Ok();
}

Status check_report(const Data& data, c::BatchPlanReport report, bool narrow, std::size_t cap) {
  HYLE_RETURN_IF_ERROR(require(report.bf16_partials == narrow, "actual partial report differs from selection"));
  HYLE_RETURN_IF_ERROR(require(report.float_workspace_bytes <= cap, "planner exceeded actual reserved cap"));
  if (narrow) {
    HYLE_RETURN_IF_ERROR(require(report.split_kv && report.cta_tile_q == 64, "narrow case missed CTA64 split"));
    const auto expected = std::size_t(8) * report.padded_batch_size * 64 * (data.capacity.value_head_dim * 2 + 4);
    HYLE_RETURN_IF_ERROR(require(report.float_workspace_bytes == expected,
                                "BF16 value workspace or F32 score workspace extent differs"));
  }
  std::printf("POLICY_PLAN bf16_partials=%d workspace_bytes=%zu reserved_float_cap=%zu tile=%u padded=%u split=%d\n",
      report.bf16_partials, report.float_workspace_bytes, cap, report.cta_tile_q, report.padded_batch_size, report.split_kv);
  return Status::Ok();
}

Status direct(Fixture& f, const Data& data, bool allow, std::size_t cap, bool narrow, bool reject = false) {
  HYLE_ASSIGN_OR_RETURN(auto before, f.device.ref().resource().stats());
  auto settings = options(data, allow, cap);
  settings.resource = f.device.ref(); settings.host_workspace = f.host.ref();
  HYLE_ASSIGN_OR_RETURN(auto program, c::BatchProgram::prepare(f.context, f.schema, settings));
  std::array<c::BatchInvocation, 2> slots;
  for (auto& slot : slots) {
    HYLE_ASSIGN_OR_RETURN(slot, c::BatchInvocation::create(program, f.queue));
    auto prepared = slot.prepare_batch(f.layout);
    if (reject) {
      HYLE_RETURN_IF_ERROR(require(!prepared && prepared.error().IsResourceExhausted(),
                                  "default F32 unexpectedly fit the reduced workspace"));
      HYLE_RETURN_IF_ERROR(slot.close());
      continue;
    }
    if (!prepared) return prepared.error();
    HYLE_RETURN_IF_ERROR(check_report(data, slot.report(), narrow, cap));
  }
  HYLE_ASSIGN_OR_RETURN(auto after, f.device.ref().resource().stats());
  HYLE_RETURN_IF_ERROR(require(before.used_bytes != SIZE_MAX && after.used_bytes != SIZE_MAX &&
                              after.used_bytes >= before.used_bytes,
                              "CUDA allocation resource did not report live requested bytes"));
  std::printf("POLICY_BACKING allow=%d rejected=%d float_cap_per_slot=%zu slots=2 used_before=%zu used_after=%zu delta_used=%zu reserved_before=%zu reserved_after=%zu scope=this_Hyle_CUDA_resource_driver_private_excluded\n",
      allow, reject, cap, before.used_bytes, after.used_bytes, after.used_bytes - before.used_bytes,
      before.reserved_bytes, after.reserved_bytes);
  if (!reject) for (unsigned captured = 0; captured < 2; ++captured) {
    for (unsigned episode = 0; episode < 4; ++episode) {
      auto& slot = slots[episode % 2];
      HYLE_ASSIGN_OR_RETURN(auto prepared, slot.update_batch(f.layout));
      if (captured && episode < 2)
        HYLE_RETURN_IF_ERROR(prepared.prepare_replay(f.query, f.key_frame, f.value_frame));
      if (captured)
        HYLE_RETURN_IF_ERROR(require(slot.report().capture_compatible, "direct capture was not retained"));
      HYLE_ASSIGN_OR_RETURN(auto done, prepared.run(f.query, f.key_frame, f.value_frame));
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_ASSIGN_OR_RETURN(auto output, slot.output());
      HYLE_RETURN_IF_ERROR(check_output(data, output));
      done = {}; prepared = {};
      auto busy = slot.update_batch(f.layout);
      HYLE_RETURN_IF_ERROR(require(!busy && busy.error().IsUnavailable(), "old output allowed slot overwrite"));
      output.reset();
      HYLE_RETURN_IF_ERROR(check_report(data, slot.report(), narrow, cap));
    }
  }
  for (auto& slot : slots) HYLE_RETURN_IF_ERROR(slot.close());
  return f.queue.drain();
}

Status graph(Fixture& f, const Data& data, bool descriptor_allow, bool narrow, std::size_t cap) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto qdesc, c::batch_query_description(data.capacity));
  HYLE_ASSIGN_OR_RETURN(auto kdesc, c::batch_cache_description(data.capacity, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto vdesc, c::batch_cache_description(data.capacity, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto mdesc, f.schema.metadata_description());
  const std::array descriptions{qdesc, kdesc, vdesc, mdesc};
  std::array<g::ValueId, 4> inputs;
  for (unsigned i = 0; i < 4; ++i) { HYLE_ASSIGN_OR_RETURN(inputs[i], builder.input(descriptions[i])); }
  c::PagedAttention operation{data.capacity, data.scale, true, data.window};
  operation.allow_bf16_partials = descriptor_allow;
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(c::Operation::custom(operation), inputs));
  HYLE_ASSIGN_OR_RETURN(auto output_id, builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(output_id));
  HYLE_ASSIGN_OR_RETURN(auto graph_value, std::move(builder).finish());
  auto compiled = g::cuda::compile(graph_value, f.context, f.device.ref(),
                             {.operation_fallback = g::OperationFallbackPolicy::kReject});
  if (!compiled) return compiled.error();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &f.queue}};
  const gb::PrepareOptions preparation{.host_workspace=f.host.ref(),
      .max_float_workspace_bytes=cap,.max_int_workspace_bytes=4096};
  HYLE_ASSIGN_OR_RETURN(auto invocation, compiled->executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_ASSIGN_OR_RETURN(auto output_desc, c::batch_output_description(data.capacity, data.capacity.max_query_rows));
  HYLE_ASSIGN_OR_RETURN(auto output_bytes, output_desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto outputs, c::LeasePool::create(f.device.ref(), output_bytes + 512));
  for (unsigned captured = 0; captured < 2; ++captured) {
    for (unsigned episode = 0; episode < 2; ++episode) {
      const std::array layouts{gb::NodeLayout{node, f.layout}};
      HYLE_RETURN_IF_ERROR(gb::prepare(invocation, queues, layouts, preparation));
      HYLE_RETURN_IF_ERROR(invocation.bind(inputs[0], qdesc, f.query));
      HYLE_RETURN_IF_ERROR(gb::bind(invocation, inputs[1], f.key_frame));
      HYLE_RETURN_IF_ERROR(gb::bind(invocation, inputs[2], f.value_frame));
      HYLE_ASSIGN_OR_RETURN(auto writer, outputs.allocate(output_bytes));
      HYLE_RETURN_IF_ERROR(invocation.bind(output_id, output_desc, std::move(writer)));
      if (captured) HYLE_RETURN_IF_ERROR(gb::prepare_replay(invocation, queues));
      HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto done, invocation.submit(queues)); HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_ASSIGN_OR_RETURN(auto output, invocation.result(output_id));
      HYLE_ASSIGN_OR_RETURN(auto storage, output.storage_lease());
      HYLE_RETURN_IF_ERROR(check_output(data, storage));
      done = {};
      HYLE_RETURN_IF_ERROR(require(!gb::prepare(invocation, queues, layouts, preparation).ok(), "Graph old output allowed overwrite"));
      HYLE_ASSIGN_OR_RETURN(auto report, gb::report(invocation, node));
      HYLE_RETURN_IF_ERROR(check_report(data, report, narrow, cap));
      if (captured) HYLE_RETURN_IF_ERROR(require(report.capture_compatible, "Graph native capture was not retained"));
    }
  }
  HYLE_RETURN_IF_ERROR(invocation.close());
  return invocation.drain();
}

Status run() {
  HYLE_RETURN_IF_ERROR(require(!c::BatchProgramOptions{}.allow_bf16_partials,
                              "partial policy default changed"));
  static_assert(!c::PagedAttention{}.allow_bf16_partials);
  constexpr std::size_t reduced = 11u << 20, original = 22u << 20;
  const Data data(256, 64, 128, 1024, 768, -1);
  HYLE_ASSIGN_OR_RETURN(auto f, fixture(data));
  HYLE_ASSIGN_OR_RETURN(auto environment, c::cuda::environment(f->context));
  std::printf("POLICY_ENV device=%s sm=%d%d\n", environment.device_name.data(), environment.sm_major, environment.sm_minor);
  HYLE_RETURN_IF_ERROR(direct(*f, data, false, reduced, false, true));
  HYLE_RETURN_IF_ERROR(direct(*f, data, false, original, false));
  HYLE_RETURN_IF_ERROR(direct(*f, data, true, reduced, true));
  HYLE_RETURN_IF_ERROR(graph(*f, data, true, true, reduced));
  HYLE_RETURN_IF_ERROR(f->queue.drain()); f.reset();
  {
    HYLE_ASSIGN_OR_RETURN(auto original_graph, fixture(data));
    HYLE_RETURN_IF_ERROR(graph(*original_graph, data, false, false, original));
    HYLE_RETURN_IF_ERROR(original_graph->queue.drain());
  }
  {
    const Data tail(256, 64, 128, 1023, 767, -1);
    HYLE_ASSIGN_OR_RETURN(auto other, fixture(tail));
    HYLE_RETURN_IF_ERROR(direct(*other, tail, true, reduced, true));
    HYLE_RETURN_IF_ERROR(other->queue.drain());
  }
  for (unsigned depth : {64u, 256u}) {
    const Data fallback(depth, 1, 17, 1024, 768, 16);
    HYLE_ASSIGN_OR_RETURN(auto other, fixture(fallback));
    HYLE_RETURN_IF_ERROR(direct(*other, fallback, true, reduced, false));
    HYLE_RETURN_IF_ERROR(other->queue.drain());
  }
  {
    // The cold domain matches, but the actual query tile is 16. Planning
    // must retain four-byte partials when the runtime selects the F32 path.
    const Data small_query(256, 1, 1, 1024, 768, -1);
    HYLE_ASSIGN_OR_RETURN(auto other, fixture(small_query));
    HYLE_RETURN_IF_ERROR(direct(*other, small_query, true, reduced, false));
    HYLE_RETURN_IF_ERROR(other->queue.drain());
  }
  std::puts("BF16 partial public policy: reduced backing cap/default F32/direct/Graph/capture/reuse/domain fallback passed");
  return Status::Ok();
}
} // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const auto status = run();
  if (status.ok()) return 0;
  char text[768]{}; status.ToString(text, sizeof(text));
  std::fprintf(stderr, "BF16 partial public consumer: %s\n", text);
  return 1;
}
