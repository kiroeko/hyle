// Finite original Backend/Graph consumer; no Model, tokenizer or weight load.
#include "../../engine/models/qwen35/main.cc"

#include <bit>
#include <cstring>

namespace {
namespace q = qwen35;
namespace i = hyle::inference;
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using Type = c::DataType;

void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

std::vector<std::byte> encoded(Type type, std::span<const float> values) {
  const auto width = type == Type::kFloat32 ? 4u : 2u;
  std::vector<std::byte> result(values.size() * width);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (type == Type::kFloat32) std::memcpy(result.data() + index * width, &values[index], width);
    else {
      const std::uint16_t bits = type == Type::kFloat16
          ? std::bit_cast<std::uint16_t>(hyle::simd::f32_to_f16(values[index]))
          : static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(values[index]) >> 16);
      std::memcpy(result.data() + index * width, &bits, width);
    }
  }
  return result;
}

std::vector<std::byte> read(i::Backend &backend, const c::ReadLease &lease) {
  std::vector<std::byte> result(lease.size());
  q::check(backend.read(q::take(lease.view()), result)); return result;
}

std::vector<std::byte> expected_rows(std::span<const std::byte> source,
    std::span<const std::int32_t> ids, std::size_t row_bytes) {
  std::vector<std::byte> result;
  for (const auto id : ids) {
    const auto row = source.subspan(static_cast<std::size_t>(id) * row_bytes, row_bytes);
    result.insert(result.end(), row.begin(), row.end());
  }
  return result;
}

template<class T> std::vector<T> values(i::Backend &backend, const c::ReadLease &lease) {
  const auto bytes = read(backend, lease);
  require(bytes.size() % sizeof(T) == 0, "typed output byte extent differs");
  std::vector<T> result(bytes.size() / sizeof(T));
  std::memcpy(result.data(), bytes.data(), bytes.size()); return result;
}

// Inputs/outputs and constant projections retain original leases throughout
// this one real Invocation. Generated output is captured before reset too.
std::vector<c::ReadLease> execute(i::Backend &backend, q::GraphAssembly &assembly,
    const g::Graph &graph, g::ValueId input, q::Tensor &x,
    std::span<const g::ValueId> outputs, std::span<const g::ShapeBinding> shape,
    const c::ResourceBudget &budget) {
  const auto compiled = q::take(backend.compile(graph, assembly.constants(),
      assembly.options({.shape = shape, .resource_budget = budget})));
  std::vector<q::Tensor> caller;
  std::vector<i::PreparedOutputFacts> facts;
  caller.reserve(outputs.size()); facts.reserve(outputs.size());
  auto run = q::take(compiled.executable.create_invocation());
  hyle::util::ScopeGuard retirement{[&]() noexcept {
    run.close().IgnoreError(); backend.drain_retired(run).IgnoreError(); run.drain().IgnoreError();
  }};
  q::check(run.prepare_resources({.resource_budget = budget}));
  q::check(run.bind(input, x.description, x.ready));
  for (const auto output : outputs) {
    facts.push_back(q::take(i::prepare_output(compiled, output)));
    if (facts.back().caller_owned()) {
      caller.emplace_back(q::take(backend.resource_for(facts.back().storage)), facts.back().description, true, budget);
      q::check(run.bind(output, facts.back().description, std::move(caller.back().writer)));
    }
  }
  const auto queues = q::take(i::prepare_queue_bindings(backend.queue_bindings(), compiled.report));
  const auto selected = i::queue_span(queues);
  q::check(run.prepare_replay(selected, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
  auto done = q::take(run.submit(selected)); q::check(done.wait()); done = {};
  std::vector<c::ReadLease> result;
  for (const auto &fact : facts) result.push_back(q::take(i::retain_output(run, fact, budget)));
  q::check(run.reset()); retirement.dismiss(); return result;
}

void scores_and_original_ids(i::Backend &backend, q::LoadedWeight &source,
    q::LoadedWeight &selected, const i::DraftVocabulary &mapping, const c::ResourceBudget &budget) {
  q::Tensor ids(backend.device(), q::dense(Type::kInt32, {static_cast<std::int64_t>(mapping.size())}), true, budget);
  q::check(backend.write(ids.storage(), std::as_bytes(mapping.ids()))); ids.freeze(backend);
  q::Tensor x(backend.device(), q::dense(Type::kBFloat16, {2, 4}), true, budget);
  const std::array<float, 8> x_values{1, 0, 0, 0, 0, 1, 0, 0};
  const auto x_bytes = encoded(Type::kBFloat16, x_values);
  q::check(backend.write(x.storage(), x_bytes)); x.freeze(backend);
  q::ModelGeometry geometry; geometry.hidden = 4; geometry.vocabulary = 8; geometry.conv_kernel = 2;
  std::map<std::string, q::LoadedWeight> unused_bank;
  q::ProjectionBindings projections{{"lm_head.weight", q::StoredProjection{&source.tensor}}};
  q::GraphAssembly assembly(backend, {}, budget);
  q::Recipe recipe(unused_bank, projections, assembly, geometry, 2);
  const auto input = recipe.input(x.description);
  const auto full = recipe.scores(input, "lm_head.weight");
  const auto compact = recipe.emit_projection(input, q::StoredProjection{&selected.tensor}, true);
  const auto full_id = recipe.op(c::Operation(c::TopK{.count = 1}), {full})[1];
  const auto compact_id = recipe.op(c::Operation(c::TopK{.count = 1}), {compact})[1];
  const auto original_id = recipe.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
      {recipe.constant(ids), compact_id}, recipe.spec(compact_id));
  const std::array outputs{full, compact, full_id, original_id};
  for (const auto output : outputs) recipe.publish(output);
  const auto graph = q::take(std::move(recipe.builder).finish());
  // Recipe always declares its independent last-row dimension, including
  // when these component projections themselves use concrete [2,H] inputs.
  const std::array shape{g::ShapeBinding{recipe.last_row, 1}};
  const auto result = execute(backend, assembly, graph, input, x, outputs, shape, budget);
  const auto full_scores = values<float>(backend, result[0]);
  const auto compact_scores = values<float>(backend, result[1]);
  require(full_scores.size() == 16 && compact_scores.size() == 2 * mapping.size(), "projection logits lost actual batch/row width");
  for (std::size_t batch = 0; batch < 2; ++batch)
    for (std::size_t index = 0; index < mapping.size(); ++index)
      require(compact_scores[batch * mapping.size() + index] == full_scores[batch * 8 + mapping.ids()[index]],
          "selected projection changed original dense score precision or row order");
  require(values<std::int32_t>(backend, result[2]) == std::vector<std::int32_t>({1, 3}),
      "full target projection was replaced by selected rows");
  require(values<std::int32_t>(backend, result[3]) == std::vector<std::int32_t>({2, 7}),
      "device TopK/Gather leaked compact IDs or changed ascending target-ID tie order");
}

void rows(i::Backend &backend, Type type, const c::ResourceBudget &budget) {
  const auto baseline = q::take(budget.usage(backend.device())).live_bytes;
  c::ReadLease escaped;
  std::vector<std::byte> expected;
  {
    // Exactly representable finite scores: original full maxima 1/3 are
    // absent from the map; the first selected row has a genuine 2/5 tie.
    const std::array<float, 32> source_values{
        0, 0, 1, 0, 20, 1, 0, 0, 9, 2, 0, 0, 3, 30, 0, 0,
        4, 4, 0, 0, 9, 5, 0, 0, 6, 6, 0, 0, -3, 15, 0, 0};
    const auto bytes = encoded(type, source_values);
    q::LoadedWeight source;
    source.codec = q::prepared::ProjectionCodec::kDense;
    source.tensor = q::Tensor(backend.device(), q::dense(type, {8, 4}), true, budget);
    q::check(backend.write(source.tensor.storage(), bytes)); source.tensor.freeze(backend);
    const auto mapping = q::take(i::DraftVocabulary::create(8, std::array<std::int32_t, 4>{7, 2, 5, 0}));
    const auto row_bytes = bytes.size() / 8;
    auto selected = q::select_dense_weight_rows(backend, source, mapping.ids(), budget);
    expected = expected_rows(bytes, mapping.ids(), row_bytes);
    require(selected.tensor.description == q::dense(type, {4, 4}) &&
        selected.codec == q::prepared::ProjectionCodec::kDense && !selected.group_size &&
        read(backend, selected.tensor.ready) == expected && read(backend, source.tensor.ready) == bytes,
        "cold Gather changed source bytes, selected order, storage type or dense codec");
    {
      constexpr std::array<std::int32_t, 3> repeats{5, 2, 5};
      const auto repeated = q::select_dense_weight_rows(backend, source, repeats, budget);
      require(read(backend, repeated.tensor.ready) == expected_rows(bytes, repeats, row_bytes),
          "low-level row selection failed to retain caller order and repeated rows");
    }
    scores_and_original_ids(backend, source, selected, mapping, budget);
    if (type == Type::kFloat32) {
      for (const auto bad : {-1, 8}) {
        bool refused = false;
        try { (void)q::select_dense_weight_rows(backend, source, std::span(&bad, 1), budget); }
        catch (const i::EvaluationFailure &failure) {
          refused = failure.status().IsOutOfRange() || failure.status().IsInvalidArgument();
        }
        require(refused && read(backend, source.tensor.ready) == bytes, "invalid row ID was accepted or modified the original weight");
      }
      // Charge remaining requested bytes without allocating a large buffer.
      const auto usage = q::take(budget.usage(backend.device()));
      auto blocked = q::take(budget.reserve(backend.device(), usage.limit_bytes - usage.live_bytes));
      bool refused = false;
      try { (void)q::select_dense_weight_rows(backend, source, mapping.ids(), budget); }
      catch (const i::EvaluationFailure &failure) { refused = failure.status().IsResourceExhausted(); }
      require(refused, "cold selected-row allocation ignored its actual resource budget");
      blocked.reset();
      const auto retry = q::select_dense_weight_rows(backend, source, mapping.ids(), budget);
      require(read(backend, retry.tensor.ready) == expected && read(backend, source.tensor.ready) == bytes,
          "budget refusal prevented same-source retry or changed source bytes");
    }
    escaped = selected.tensor.ready;
  }
  q::check(backend.collect_retired());
  require(read(backend, escaped) == expected && q::take(budget.usage(backend.device())).live_bytes > baseline,
      "selected row lease failed to outlive cold Invocation/LoadedWeight or lost its budget charge");
  escaped.reset(); q::check(backend.collect_retired());
  require(q::take(budget.usage(backend.device())).live_bytes == baseline,
      "released original row consumer retained native resources or requested-byte charges");
}

void metadata(const std::filesystem::path &manifest, const std::filesystem::path &scratch) {
  const auto roles = q::read_weight_roles(manifest, true);
  std::filesystem::create_directories(scratch);
  const auto path = scratch / "probe-draft-vocabulary.ids";
  const auto write = [&](std::string_view identity, std::int64_t vocabulary, std::string_view ids) {
    std::ofstream output(path, std::ios::binary);
    output << "hyle-draft-vocabulary-v1\n" << identity << '\n' << vocabulary << '\n' << ids << '\n';
    require(bool(output), "cannot write finite mapping metadata");
  };
  auto wrong = roles.identity; wrong[0] = wrong[0] == '0' ? '1' : '0';
  write(wrong, roles.geometry.vocabulary, "0 2");
  bool identity_refused = false;
  try { (void)q::prepare_weight_source(manifest, true, path); }
  catch (const std::runtime_error &failure) {
    identity_refused = std::string_view(failure.what()).find("identity") != std::string_view::npos;
  }
  require(identity_refused, "mapping from another weight identity was accepted");
  for (const auto bad : {std::string("0 0"), std::string("-1"), std::to_string(roles.geometry.vocabulary)}) {
    write(roles.identity, roles.geometry.vocabulary, bad);
    bool refused = false;
    try { (void)q::prepare_weight_source(manifest, true, path); }
    catch (const i::EvaluationFailure &failure) {
      refused = failure.status().IsInvalidArgument() || failure.status().IsOutOfRange();
    }
    require(refused, "invalid explicit cold mapping token list was accepted");
  }
  write(roles.identity, roles.geometry.vocabulary, "7 2 5 0");
  const auto source = q::prepare_weight_source(manifest, true, path);
  require(source.draft_vocabulary && std::ranges::equal(source.draft_vocabulary->ids(), std::array<std::int32_t, 4>{0, 2, 5, 7}),
      "corrected mapping metadata could not retry in canonical original-ID order");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if ((argc != 2 && argc != 4) || (std::string_view(argv[1]) != "cpu" && std::string_view(argv[1]) != "cuda"))
      throw std::runtime_error("usage: qwen_draft_vocabulary_probe cpu|cuda [LOCAL_MANIFEST SCRATCH_DIR]");
    if (argc == 4) metadata(argv[2], argv[3]);
    const auto kind = std::string_view(argv[1]) == "cpu" ? i::BackendKind::kCpu : i::BackendKind::kCuda;
    auto backend = q::take(i::Backend::create({.primary = kind}));
    std::vector<c::ResourceBudgetLimit> limits;
    for (const auto resource : {backend.device(), backend.host_device(), backend.staging(), m::system_resource()})
      if (resource && std::ranges::none_of(limits, [&](const auto &limit) { return limit.resource == resource; }))
        limits.push_back({resource, 64u << 20});
    const auto budget = q::take(c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
    std::vector<std::size_t> baseline;
    for (const auto &limit : limits) baseline.push_back(q::take(budget.usage(limit.resource)).live_bytes);
    for (const auto type : {Type::kFloat32, Type::kFloat16, Type::kBFloat16}) rows(backend, type, budget);
    for (std::size_t index = 0; index < limits.size(); ++index)
      require(q::take(budget.usage(limits[index].resource)).live_bytes == baseline[index], "cold consumer retained another resource-domain budget charge");
    std::cout << "{\"status\":\"PASS\",\"backend\":\"" << argv[1]
        << "\",\"stored_types\":3,\"source_bytes_unchanged\":true,\"batched_original_ids_and_ties\":true,"
           "\"escaped_lease_budget_retired\":true,\"budget_rejection_and_retry\":true,\"invalid_ids_refused\":true,"
           "\"metadata_identity_checked\":" << (argc == 4 ? "true" : "false") << ",\"model_execution\":false}\n";
    return 0;
  } catch (const std::exception &failure) {
    std::cerr << "draft vocabulary probe: " << failure.what() << '\n'; return 1;
  }
}
