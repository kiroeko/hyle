#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../../engine/inference/mtp_acceptance.h"
#include "../../engine/inference/mtp_acceptance_graph.h"
#include "../../engine/inference/program_carrier.h"

// Exercise the shared fragments through the original compiled Graph and typed
// IO. Decisions are inputs here: this is not a model/sampler quality test.
namespace inference_mtp_packet_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace g = hyle::graph;
namespace c = hyle::compute;
namespace mtp = i::mtp;
using qwen_host_image_case::require;

struct Vector final {
  std::string name;
  std::vector<std::int32_t> proposals, final_ids;
  std::vector<std::uint8_t> accepted, valid;
  std::size_t quota = 3;
  std::vector<std::int32_t> eos, extra_stop;
  bool ignore_eos = false;
};
struct Scope final { std::vector<c::ReadLease> inputs; };

inline std::size_t check_width(i::Backend &backend, std::size_t m,
    std::span<const Vector> vectors) {
  std::map<std::string, q::LoadedWeight> no_weights;
  const q::ProjectionBindings no_projections;
  const q::ModelGeometry construction_axes{.hidden = 1, .conv_kernel = 2};
  q::GraphAssembly assembly(backend);
  q::Recipe r(no_weights, no_projections, assembly, construction_axes, 1);
  const auto rows = (std::max)(m, std::size_t{1});
  std::vector<q::V> inputs;
  std::vector<c::TensorDesc> descriptions;
  const auto input = [&](q::Type type, std::size_t count) {
    const auto description = q::dense(type, {static_cast<std::int64_t>(count)});
    const auto value = r.input(description);
    descriptions.push_back(description); inputs.push_back(value);
    return value;
  };
  q::V proposals, continuation, accepts;
  if (m) {
    proposals = input(q::Type::kInt32, m);
    continuation = input(q::Type::kFloat32, m);
    accepts = input(q::Type::kBool, m);
  }
  const auto finals = input(q::Type::kInt32, rows), validity = input(q::Type::kBool, rows);
  std::vector<q::V> accepted, final_ids, valid;
  const auto element = [&](q::V source, std::size_t row) {
    return r.copy(r.slice(source, 0, q::f(static_cast<std::int64_t>(row)), q::f(1)));
  };
  for (std::size_t row = 0; row < m; ++row) accepted.push_back(element(accepts, row));
  for (std::size_t row = 0; row < rows; ++row) {
    final_ids.push_back(element(finals, row)); valid.push_back(element(validity, row));
  }
  const auto packet = mtp::sampled_prefix_packet(r, proposals, continuation,
      std::span<const q::V>(accepted), std::span<const q::V>(final_ids), std::span<const q::V>(valid));
  const auto validated = mtp::validated_ids_packet(r,
      std::span<const q::V>(final_ids), std::span<const q::V>(valid));
  const auto packet_size = m + 5, total_size = packet_size + 1 + rows;
  const std::array parts{packet, validated};
  const auto output = r.concat(parts, 0,
      r.dense_spec(q::Type::kInt32, {q::f(static_cast<std::int64_t>(total_size))}));
  q::check(r.builder.publish(output));
  const auto graph = q::take(std::move(r.builder).finish());
  const std::array shape{g::ShapeBinding{r.last_row, 0}};
  const std::array variants{g::ExactShapeVariant{shape}};
  auto compiled = q::take(backend.compile(graph, assembly.constants(),
      assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact,
          .exact_variants = variants, .constants = assembly.constants()})));
  compiled.executable = q::take(compiled.executable.exact_variant(0));
  compiled.report = q::take(compiled.report.exact_variant(0));
  i::ProgramIo io;
  q::check(io.prepare(compiled, 0, inputs.size(), 1, 0));
  std::vector<i::ProgramIo::Input> ports;
  for (const auto value : inputs) ports.push_back(q::take(io.add_input(value)));
  const auto readback = q::take(io.add_readback(output, backend)); q::check(io.seal());
  i::ProgramCarrier<Scope> carrier;
  q::check(carrier.prepare(compiled, backend.queue_bindings()));
  q::check(carrier.initialize({.io = io, .invocation = g::InvocationOptions{.resources = g::ResourceOptions{}}}));
  try {
    for (const auto &test : vectors) {
      require(test.proposals.size() == m && test.accepted.size() == m &&
          test.final_ids.size() == rows && test.valid.size() == rows,
          "packet test vector differs from its actual fixed graph width");
      const i::RequestStopPolicy policy(test.extra_stop, test.ignore_eos);
      std::vector<std::int32_t> verification{16'777'217};
      verification.insert(verification.end(), test.proposals.begin(), test.proposals.end());
      std::vector<std::int32_t> emitted;
      const auto chosen = mtp::select_sampled_prefix(verification, test.accepted,
          test.final_ids, test.quota, test.eos, (std::numeric_limits<std::int32_t>::max)(), emitted, &policy);
      std::vector<float> gates;
      for (std::size_t row = 0; row < m; ++row)
        gates.push_back(row + 1 < test.quota &&
            !i::request_token_stopped(test.proposals[row], test.eos, &policy) ? 1.F : 0.F);
      const auto all_valid = std::ranges::all_of(test.valid, [](auto value) { return value == 1; });
      std::vector<std::int32_t> expected{all_valid ? 1 : 0,
          static_cast<std::int32_t>(chosen.consumed), chosen.next_id, chosen.first_agrees ? 1 : 0};
      // Unconsumed fixed slots are local row results, never publishable outputs.
      for (std::size_t row = 0; row < m; ++row)
        expected.push_back(test.accepted[row] ? test.proposals[row] : test.final_ids[row]);
      expected.push_back(test.final_ids.back());
      expected.push_back(all_valid ? 1 : 0);
      expected.insert(expected.end(), test.final_ids.begin(), test.final_ids.end());
      auto scope = std::make_shared<Scope>();
      const auto upload = [&](const auto &values) {
        const auto bytes = std::as_bytes(std::span(values));
        auto buffer = q::take(q::mem::Buffer::create(backend.device(), bytes.size(), 256));
        q::check(backend.write(buffer.view(), bytes));
        scope->inputs.push_back(q::take(backend.adopt_buffer(std::move(buffer))));
      };
      if (m) { upload(test.proposals); upload(gates); upload(test.accepted); }
      upload(test.final_ids); upload(test.valid);
      auto *run = q::take(carrier.borrow());
      for (std::size_t port = 0; port < ports.size(); ++port)
        q::check(io.bind(*run, ports[port], descriptions[port], scope->inputs[port]));
      q::check(run->prepare_replay(carrier.queues(),
          {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
      const std::weak_ptr<Scope> lifetime = scope;
      auto completion = q::take(carrier.submit(scope)); scope.reset();
      q::check(completion.wait()); q::check(backend.collect_retired());
      require(!carrier.pending() && !run->in_flight() && lifetime.expired(),
          "packet completion did not actually retire its retained original input scope");
      std::vector<std::int32_t> actual(total_size);
      q::check(io.read(backend, *run, readback, std::as_writable_bytes(std::span(actual))));
      if (actual != expected)
        throw std::runtime_error("device MTP packet differs from exact fields/host oracle: " + test.name);
      require(std::equal(emitted.begin(), emitted.end(), actual.begin() + 4) &&
          emitted.size() == chosen.consumed &&
          i::request_token_stopped(actual[2], test.eos, &policy) == chosen.stopped,
          "packet consumed prefix or terminal token differs from original host policy");
      completion = {}; q::check(io.release_inputs(carrier));
    }
  } catch (...) {
    const auto retired = carrier.retire(backend);
    if (retired.retired) (void)carrier.reset_retired();
    throw;
  }
  const auto retired = carrier.retire(backend); q::check(retired.status);
  require(retired.retired && !carrier.pending(), "packet program did not actually retire");
  q::check(carrier.reset_retired());
  require(!carrier.initialized(), "retired packet program retained its original invocation");
  return vectors.size();
}

inline int run(q::BackendKind kind) {
  auto backend = q::take(i::Backend::create({.primary = kind}));
  constexpr std::int32_t a = 16'777'218, b = 16'777'219, eos = 16'777'220,
      extra = 16'777'221, final = 16'777'223, high = (std::numeric_limits<std::int32_t>::max)() - 2;
  const std::vector<Vector> zero{
      {.name = "target-only", .final_ids = {high}, .valid = {1}, .quota = 1},
      {.name = "target-only-invalid", .final_ids = {high}, .valid = {0}, .quota = 1}};
  const std::vector<Vector> one{
      {.name = "first-reject-terminal", .proposals = {a}, .final_ids = {eos}, .accepted = {0}, .valid = {1}, .eos = {eos}},
      {.name = "all-accept", .proposals = {a}, .final_ids = {high}, .accepted = {1}, .valid = {1}},
      {.name = "quota-one", .proposals = {a}, .final_ids = {high}, .accepted = {1}, .valid = {1}, .quota = 1}};
  const std::vector<Vector> two{
      {.name = "all-reject", .proposals = {a, b}, .final_ids = {final, high}, .accepted = {0, 0}, .valid = {1, 1}},
      {.name = "second-reject", .proposals = {a, b}, .final_ids = {final, high}, .accepted = {1, 0}, .valid = {1, 1}},
      {.name = "all-accept", .proposals = {a, b}, .final_ids = {final, high}, .accepted = {1, 1}, .valid = {1, 1}},
      {.name = "quota-one", .proposals = {a, b}, .final_ids = {final, high}, .accepted = {1, 1}, .valid = {1, 1}, .quota = 1},
      {.name = "quota-two", .proposals = {a, b}, .final_ids = {final, high}, .accepted = {1, 1}, .valid = {1, 1}, .quota = 2},
      {.name = "accepted-eos", .proposals = {eos, b}, .final_ids = {final, high}, .accepted = {1, 1}, .valid = {1, 1}, .eos = {eos}},
      {.name = "ignored-eos", .proposals = {eos, b}, .final_ids = {final, high}, .accepted = {1, 1}, .valid = {1, 1}, .eos = {eos}, .ignore_eos = true},
      {.name = "extra-stop", .proposals = {eos, extra}, .final_ids = {final, high}, .accepted = {1, 1}, .valid = {1, 1}, .eos = {eos}, .extra_stop = {extra}, .ignore_eos = true},
      {.name = "rejected-before-invalid-later", .proposals = {a, b}, .final_ids = {final, high}, .accepted = {0, 1}, .valid = {1, 0}}};
  std::size_t submissions = check_width(backend, 0, zero);
  submissions += check_width(backend, 1, one);
  submissions += check_width(backend, 2, two);
  require(submissions == 14, "packet finite case count changed");
  std::cout << "{\"status\":\"PASS\",\"case\":\"mtp-packet\",\"backend\":\"" << qwen_host_image_case::backend_name(kind)
      << "\",\"models\":0,\"graphs\":3,\"native_submissions\":" << submissions
      << ",\"packet_readbacks\":" << submissions
      << ",\"host_oracle_exact\":true,\"all_fixed_fields_checked\":true,\"validated_ids_checked\":true,"
         "\"quota_and_stop_checked\":true,\"ignore_eos_and_extra_stop_checked\":true,"
         "\"invalid_later_not_hidden\":true,\"large_i32_ids_exact\":true,\"real_retirement_checked\":true,"
         "\"quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}
} // namespace inference_mtp_packet_case
