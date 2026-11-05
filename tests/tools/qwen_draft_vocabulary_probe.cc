// Finite original Backend/Graph consumer; no Model, tokenizer or weight load.
#include "../../engine/models/qwen35/graph_builder.h"
#include "../../engine/inference/backend.h"
#include "../../engine/inference/draft_vocabulary.h"
#include "../../engine/inference/execution_resources.h"
#include "../../engine/inference/prepared_output.h"
#include "hyle/simd/dtype.h"
#include "hyle/util/scope_guard.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
    auto selected = q::select_stored_weight_rows(backend, source, mapping.ids(), budget);
    expected = expected_rows(bytes, mapping.ids(), row_bytes);
    require(selected.tensor.description == q::dense(type, {4, 4}) &&
        selected.codec == q::prepared::ProjectionCodec::kDense && !selected.group_size &&
        read(backend, selected.tensor.ready) == expected && read(backend, source.tensor.ready) == bytes,
        "cold Gather changed source bytes, selected order, storage type or dense codec");
    {
      constexpr std::array<std::int32_t, 3> repeats{5, 2, 5};
      const auto repeated = q::select_stored_weight_rows(backend, source, repeats, budget);
      require(read(backend, repeated.tensor.ready) == expected_rows(bytes, repeats, row_bytes),
          "low-level row selection failed to retain caller order and repeated rows");
    }
    scores_and_original_ids(backend, source, selected, mapping, budget);
    if (type == Type::kFloat32) {
      for (const auto bad : {-1, 8}) {
        bool refused = false;
        try { (void)q::select_stored_weight_rows(backend, source, std::span(&bad, 1), budget); }
        catch (const i::EvaluationFailure &failure) {
          refused = failure.status().IsOutOfRange() || failure.status().IsInvalidArgument();
        }
        require(refused && read(backend, source.tensor.ready) == bytes, "invalid row ID was accepted or modified the original weight");
      }
      // Charge remaining requested bytes without allocating a large buffer.
      const auto usage = q::take(budget.usage(backend.device()));
      auto blocked = q::take(budget.reserve(backend.device(), usage.limit_bytes - usage.live_bytes));
      bool refused = false;
      try { (void)q::select_stored_weight_rows(backend, source, mapping.ids(), budget); }
      catch (const i::EvaluationFailure &failure) { refused = failure.status().IsResourceExhausted(); }
      require(refused, "cold selected-row allocation ignored its actual resource budget");
      blocked.reset();
      const auto retry = q::select_stored_weight_rows(backend, source, mapping.ids(), budget);
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

void encoded_rows(i::Backend&backend,c::GgufQuantizedType codec,const c::ResourceBudget&budget){
  const auto baseline=q::take(budget.usage(backend.device())).live_bytes;
  c::ReadLease escaped;std::vector<std::byte>expected;
  {
    constexpr std::int64_t rows=8,columns=256;
    q::LoadedWeight source;
    source.tensor=q::Tensor(backend.device(),q::take(q::take(c::gguf_weight_description(
        codec,Type::kFloat32,columns,rows)).transposed()),true,budget);
    const auto extent=q::take(source.tensor.description.required_bytes());
    const auto row_bytes=extent/static_cast<std::size_t>(rows);
    std::vector<std::byte>bytes(extent);
    // Exercise arbitrary physical code/scale bits, including F16 NaN payloads.
    // The exact byte oracle detects any conversion by the same-type Gather.
    for(std::size_t n=0;n<bytes.size();++n)bytes[n]=static_cast<std::byte>((n*37+n/row_bytes*19)%256);
    q::check(backend.write(source.tensor.storage(),bytes));source.tensor.freeze(backend);
    constexpr std::array<std::int32_t,4>ids{7,2,5,2};
    if(backend.kind()!=i::BackendKind::kCuda){
      bool refused=false;
      try{(void)q::select_stored_weight_rows(backend,source,ids,budget);}
      catch(const i::EvaluationFailure&failure){refused=failure.status().IsUnimplemented();}
      require(refused,"unqualified backend silently accepted encoded row selection");
      return;
    }
    auto selected=q::select_stored_weight_rows(backend,source,ids,budget);
    expected=expected_rows(bytes,ids,row_bytes);
    require(selected.tensor.description==q::take(q::take(c::gguf_weight_description(
        codec,Type::kFloat32,columns,ids.size())).transposed()) &&
        selected.codec==source.codec && !selected.group_size && read(backend,selected.tensor.ready)==expected &&
        read(backend,source.tensor.ready)==bytes,"encoded row selection changed codes/scales, format or caller order");
    for(const auto id:{-1,8}){
      bool refused=false;
      try{(void)q::select_stored_weight_rows(backend,source,std::span(&id,1),budget);}
      catch(const i::EvaluationFailure&failure){refused=failure.status().IsInvalidArgument() || failure.status().IsOutOfRange();}
      require(refused && read(backend,source.tensor.ready)==bytes,"encoded row gather accepted an invalid ID or changed its source");
    }
    const auto usage=q::take(budget.usage(backend.device()));
    auto blocked=q::take(budget.reserve(backend.device(),usage.limit_bytes-usage.live_bytes));
    bool refused=false;
    try{(void)q::select_stored_weight_rows(backend,source,ids,budget);}
    catch(const i::EvaluationFailure&failure){refused=failure.status().IsResourceExhausted();}
    require(refused,"encoded row selection ignored its original resource budget");
    blocked.reset();
    const auto retry=q::select_stored_weight_rows(backend,source,ids,budget);
    require(read(backend,retry.tensor.ready)==expected,"encoded row selection could not retry after budget refusal");
    escaped=selected.tensor.ready;
  }
  q::check(backend.collect_retired());
  require(read(backend,escaped)==expected && q::take(budget.usage(backend.device())).live_bytes>baseline,
      "encoded row lease lost its selected bytes or budget charge after source retirement");
  escaped.reset();q::check(backend.collect_retired());
  require(q::take(budget.usage(backend.device())).live_bytes==baseline,"encoded row selection retained native resources");
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

void full_head(i::Backend &backend, const std::filesystem::path &manifest,
    const std::filesystem::path &specification, const std::filesystem::path &scratch) {
  const auto source = q::prepare_weight_source(manifest, true, {}, specification);
  require(bool(source.draft_head), "explicit prepared head was not retained");
  std::filesystem::create_directories(scratch);
  const auto copy = scratch / "head.tsv";
  std::ifstream input(specification, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(input), {}};
  const auto write = [&](const std::string &value) {
    std::ofstream output(copy, std::ios::binary); output << value;
    require(bool(output), "cannot write finite head metadata");
  };
  write(text);
  const auto prepared = q::prepare_weight_source(manifest, true, {}, copy);
  write(text + "unexpected\n");
  bool refused = false;
  try { (void)q::load_prepared_draft_head(backend, prepared.roles, *prepared.draft_head); }
  catch (const std::runtime_error &) { refused = true; }
  require(refused, "changed metadata was not revalidated before streaming");
  write(text);

  std::vector<c::ResourceBudgetLimit> limits;
  for (const auto resource : {backend.device(), backend.host_device(), backend.staging(), m::system_resource()})
    if (resource && std::ranges::none_of(limits, [&](const auto &limit) { return limit.resource == resource; }))
      limits.push_back({resource, source.draft_head->role.bytes + (64u << 20)});
  const auto budget = q::take(c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
  std::vector<std::size_t> baseline;
  for (const auto &limit : limits) baseline.push_back(q::take(budget.usage(limit.resource)).live_bytes);
  auto blocked = q::take(budget.reserve(backend.device(), q::take(budget.usage(backend.device())).limit_bytes));
  refused = false;
  try { (void)q::load_prepared_draft_head(backend, prepared.roles, *prepared.draft_head, budget); }
  catch (const i::EvaluationFailure &failure) { refused = failure.status().IsResourceExhausted(); }
  require(refused, "prepared head stream ignored its device resource budget");
  blocked.reset();
  c::ReadLease escaped;
  {
    auto head = q::load_prepared_draft_head(backend, prepared.roles, *prepared.draft_head, budget);
    const auto &role = source.draft_head->role;
    const auto vocabulary = source.roles.geometry.vocabulary;
    const auto row_bytes = role.bytes / static_cast<std::size_t>(vocabulary);
    require(head.identity == source.draft_head->identity && !head.indexed_weight &&
        !head.original_ids.ready.valid() && head.extra_weight_bytes() == role.bytes &&
        head.weight.tensor.description.storage_format().id() == c::kGgufQ4K,
        "full head lost its independent encoded identity or became an indexed subset");
    const std::array<std::int32_t, 3> ids{0, static_cast<std::int32_t>(vocabulary / 2),
        static_cast<std::int32_t>(vocabulary - 1)};
    const auto selected = q::select_stored_weight_rows(backend, head.weight, ids, budget);
    std::ifstream payload(role.source, std::ios::binary);
    std::vector<std::byte> expected(row_bytes * ids.size());
    for (std::size_t n = 0; n < ids.size(); ++n) {
      payload.seekg(static_cast<std::streamoff>(ids[n] * row_bytes));
      payload.read(reinterpret_cast<char *>(expected.data() + n * row_bytes), static_cast<std::streamsize>(row_bytes));
      require(bool(payload), "cannot read original encoded head rows");
    }
    require(read(backend, selected.tensor.ready) == expected, "streamed full head changed physical row codes/scales");
    escaped = head.weight.tensor.ready;
  }
  q::check(backend.collect_retired());
  require(escaped.valid() && q::take(budget.usage(backend.device())).live_bytes >= source.draft_head->role.bytes,
      "escaping full head lease lost its original resource charge");
  escaped.reset(); q::check(backend.collect_retired());
  for (std::size_t n = 0; n < limits.size(); ++n) {
    const auto live = q::take(budget.usage(limits[n].resource)).live_bytes;
    if (live != baseline[n]) std::cerr << "head budget domain " << n << " before=" << baseline[n] << " after=" << live << '\n';
    require(live == baseline[n], "full head stream retained resource charges after retirement");
  }
  std::cout << "PASS full Q4_K head: metadata revalidation, budget refusal/retry, first/middle/last encoded rows, "
      "escaped lease and every resource domain retired\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 6 && std::string_view(argv[1]) == "cuda" && std::string_view(argv[2]) == "--full-head") {
      auto backend = q::take(i::Backend::create({.primary = i::BackendKind::kCuda}));
      full_head(backend, argv[3], argv[4], argv[5]); return 0;
    }
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
    for(const auto codec:{c::GgufQuantizedType::kQ8_0,c::GgufQuantizedType::kQ4K,c::GgufQuantizedType::kQ6K})
      encoded_rows(backend,codec,budget);
    for (std::size_t index = 0; index < limits.size(); ++index)
      require(q::take(budget.usage(limits[index].resource)).live_bytes == baseline[index], "cold consumer retained another resource-domain budget charge");
    std::cout << "{\"status\":\"PASS\",\"backend\":\"" << argv[1]
        << "\",\"stored_types\":3,\"encoded_types\":" << (kind==i::BackendKind::kCuda?3:0)
        << ",\"encoded_refusals\":" << (kind==i::BackendKind::kCuda?0:3)
        << ",\"source_bytes_unchanged\":true,\"batched_original_ids_and_ties\":true,"
           "\"escaped_lease_budget_retired\":true,\"budget_rejection_and_retry\":true,\"invalid_ids_refused\":true,"
           "\"metadata_identity_checked\":" << (argc == 4 ? "true" : "false") << ",\"model_execution\":false}\n";
    return 0;
  } catch (const std::exception &failure) {
    std::cerr << "draft vocabulary probe: " << failure.what() << '\n'; return 1;
  }
}
