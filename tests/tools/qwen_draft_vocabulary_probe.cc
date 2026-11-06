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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

void prepared_scores(i::Backend &backend, q::PreparedDraftHead &source,
    q::PreparedDraftHead &subset, const i::DraftVocabulary &mapping,
    const q::ModelGeometry &geometry, const c::ResourceBudget &budget) {
  q::Tensor x(backend.device(), q::dense(Type::kBFloat16, {1, geometry.hidden}), true, budget);
  std::vector<float> input_values(static_cast<std::size_t>(geometry.hidden));
  for (std::size_t n = 0; n < input_values.size(); ++n)
    input_values[n] = (static_cast<float>(n % 13) - 6.0f) / 64.0f;
  const auto input_bytes = encoded(Type::kBFloat16, input_values);
  q::check(backend.write(x.storage(), input_bytes)); x.freeze(backend);
  std::map<std::string, q::LoadedWeight> unused;
  q::ProjectionBindings projections{{std::string(geometry.head_role()), q::StoredProjection{&source.weight.tensor}}};
  q::GraphAssembly assembly(backend, {}, budget);
  q::Recipe recipe(unused, projections, assembly, geometry, 1);
  const auto input = recipe.input(x.description);
  const auto complete = recipe.scores(input, std::string(geometry.head_role()));
  const auto selected = recipe.emit_projection(input, q::StoredProjection{&subset.weight.tensor}, true);
  const std::array outputs{complete, selected};
  for (auto output : outputs) recipe.publish(output);
  const auto graph = q::take(std::move(recipe.builder).finish());
  const std::array bindings{g::ShapeBinding{recipe.last_row, 0}};
  const auto result = execute(backend, assembly, graph, input, x, outputs, bindings, budget);
  const auto full_values = values<float>(backend, result[0]);
  const auto subset_values = values<float>(backend, result[1]);
  require(full_values.size() == static_cast<std::size_t>(geometry.vocabulary) && subset_values.size() == mapping.size(),
      "prepared projection outputs lost full or selected vocabulary extent");
  for (std::size_t n = 0; n < mapping.size(); ++n)
    require(subset_values[n] == full_values[static_cast<std::size_t>(mapping.ids()[n])],
        "compacting prepared Q4 rows changed their computed projection values");
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

void q5_projection_composition(i::Backend &backend, const c::ResourceBudget &budget) {
  constexpr std::int64_t rows = 8, columns = 256;
  q::LoadedWeight source;
  source.tensor = q::Tensor(backend.device(), q::take(q::take(c::gguf_weight_description(
      c::GgufQuantizedType::kQ5K, Type::kFloat32, columns, rows)).transposed()), true, budget);
  const auto extent = q::take(source.tensor.description.required_bytes());
  const auto row_bytes = extent / rows;
  std::vector<std::byte> weights(extent);
  for (std::size_t n = 0; n < weights.size(); ++n)
    weights[n] = static_cast<std::byte>((n * 37 + n / row_bytes * 19) % 256);
  for (std::size_t row = 0; row < rows; ++row) {
    // The two leading F16 scales must be finite; retain varied code bits.
    weights[row * row_bytes] = std::byte{0};
    weights[row * row_bytes + 1] = std::byte{0x3c};
    weights[row * row_bytes + 2] = std::byte{0};
    weights[row * row_bytes + 3] = std::byte{0};
  }
  q::check(backend.write(source.tensor.storage(), weights)); source.tensor.freeze(backend);
  const auto mapping = q::take(i::DraftVocabulary::create(rows, std::array<std::int32_t, 4>{0, 2, 5, 7}));
  auto selected = q::select_stored_weight_rows(backend, source, mapping.ids(), budget);
  q::Tensor ids(backend.device(), q::dense(Type::kInt32, {4}), true, budget);
  q::check(backend.write(ids.storage(), std::as_bytes(mapping.ids()))); ids.freeze(backend);
  q::ModelGeometry geometry; geometry.hidden = columns; geometry.vocabulary = rows; geometry.conv_kernel = 2;
  q::prepared::Roles roles; roles.representation = q::prepared::SourceRepresentation::kGgufConverted;
  q::prepared::Role head; head.dtype = "Q5_K"; head.meaning = q::prepared::ParameterMeaning::kLinearWeight;
  roles.selected.push_back(head);
  for (const std::int64_t batch : {1, 2}) {
    q::Tensor x(backend.device(), q::dense(Type::kFloat32, {batch, columns}), true, budget);
    std::vector<float> xs(static_cast<std::size_t>(batch * columns));
    for (std::size_t n = 0; n < xs.size(); ++n) xs[n] = (static_cast<int>(n % 29) - 14) / 64.0f;
    const auto xb = encoded(Type::kFloat32, xs); q::check(backend.write(x.storage(), xb)); x.freeze(backend);
    std::map<std::string, q::LoadedWeight> unused;
    q::ProjectionBindings projections{{"lm_head.weight", q::StoredProjection{&source.tensor}}};
    q::GraphAssembly assembly(backend, q::PrecisionProfile::from_roles(roles), budget);
    q::Recipe recipe(unused, projections, assembly, geometry, static_cast<std::size_t>(batch));
    const auto input = recipe.input(x.description);
    const auto full = recipe.scores(input, "lm_head.weight");
    const auto compact = recipe.emit_projection(input, q::StoredProjection{&selected.tensor}, true);
    const auto compact_id = recipe.op(c::Operation(c::TopK{.count = 1}), {compact})[1];
    const auto original_id = recipe.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
        {recipe.constant(ids), compact_id}, recipe.spec(compact_id));
    const std::array outputs{full, compact, original_id};
    for (auto value : outputs) recipe.publish(value);
    const auto graph = q::take(std::move(recipe.builder).finish());
    const std::array shape{g::ShapeBinding{recipe.last_row, batch - 1}};
    const auto result = execute(backend, assembly, graph, input, x, outputs, shape, budget);
    const auto full_scores = values<float>(backend, result[0]);
    const auto subset = values<float>(backend, result[1]);
    const auto chosen = values<std::int32_t>(backend, result[2]);
    require(full_scores.size() == static_cast<std::size_t>(batch * rows) &&
        subset.size() == static_cast<std::size_t>(batch) * mapping.size() &&
        chosen.size() == static_cast<std::size_t>(batch), "Q5 scores shape differs");
    for (std::size_t row = 0; row < static_cast<std::size_t>(batch); ++row) {
      float best = -std::numeric_limits<float>::infinity();
      std::int32_t expected = -1;
      for (std::size_t n = 0; n < mapping.size(); ++n) {
        const auto value = subset[row * mapping.size() + n];
        require(std::isfinite(value) && value == full_scores[row * rows + mapping.ids()[n]],
            "Q5 selected projection changed score");
        if (value > best) { best = value; expected = mapping.ids()[n]; }
      }
      require(best != 0 && chosen[row] == expected, "Q5 selected projection failed original ID remapping");
    }
  }
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
  for (const auto bad : {std::string("0 0"), std::string("-1"), std::string{}, std::to_string(roles.geometry.vocabulary)}) {
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
  {
    std::ofstream output(path, std::ios::binary);
    output << "hyle-draft-vocabulary-v1\r\n" << roles.identity << "\r\n" << roles.geometry.vocabulary
        << "\r\n\t7  2\r\n5\v0";
  }
  const auto mixed_whitespace = q::prepare_weight_source(manifest, true, path);
  require(mixed_whitespace.draft_vocabulary && std::ranges::equal(mixed_whitespace.draft_vocabulary->ids(),
      std::array<std::int32_t, 4>{0, 2, 5, 7}), "bounded buffer changed CRLF, whitespace or final non-newline parsing");
  for (const auto bad : {std::string("0x2"), std::string("2147483648"), std::string(17, '1'), std::string("0\0 2", 4)}) {
    write(roles.identity, roles.geometry.vocabulary, bad);
    bool refused = false;
    try { (void)q::prepare_weight_source(manifest, true, path); }
    catch (const std::runtime_error &) { refused = true; }
    require(refused, "bounded mapping buffer accepted a malformed or overflowing token field");
  }
  {
    std::ofstream output(path, std::ios::binary);
    output << "hyle-draft-vocabulary-v1\n" << roles.identity << '\n' << roles.geometry.vocabulary << "\n0"
        << std::string(static_cast<std::size_t>(roles.geometry.vocabulary) * 32 + 257, ' ');
  }
  bool oversized = false;
  try { (void)q::prepare_weight_source(manifest, true, path); }
  catch (const std::runtime_error &) { oversized = true; }
  require(oversized, "bounded mapping buffer ignored the original file byte limit");
  write(roles.identity, roles.geometry.vocabulary, "7 2 5 0");
  require(bool(q::prepare_weight_source(manifest, true, path).draft_vocabulary),
      "mapping parser could not retry after invalid or oversized text");
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

  const auto mapping_path = scratch / "head-subset.ids";
  const auto write_mapping = [&](const std::string &identity) {
    std::ofstream output(mapping_path, std::ios::binary);
    output << "hyle-draft-vocabulary-v1\n" << identity << '\n' << source.roles.geometry.vocabulary
        << "\n0 " << source.roles.geometry.vocabulary / 2 << ' ' << source.roles.geometry.vocabulary - 1 << '\n';
    require(bool(output), "cannot write prepared-head subset metadata");
  };
  auto foreign_identity = source.roles.identity;
  foreign_identity[0] = foreign_identity[0] == '0' ? '1' : '0';
  write_mapping(foreign_identity);
  refused = false;
  try { (void)q::prepare_weight_source(manifest, true, mapping_path, copy); }
  catch (const std::runtime_error &) { refused = true; }
  require(refused, "custom head bypassed independent subset source identity validation");
  write_mapping(source.roles.identity);
  auto foreign_head = text;
  foreign_head.replace(foreign_head.find(source.roles.identity), source.roles.identity.size(), foreign_identity);
  write(foreign_head);
  refused = false;
  try { (void)q::prepare_weight_source(manifest, true, mapping_path, copy); }
  catch (const std::runtime_error &) { refused = true; }
  require(refused, "subset mapping bypassed independent custom head identity validation");
  write(text);
  const auto combined = q::prepare_weight_source(manifest, true, mapping_path, copy);
  require(combined.draft_head && combined.draft_vocabulary && combined.draft_vocabulary->size() == 3,
      "independently validated head and subset were not both retained");

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
  c::ReadLease escaped, subset_escaped;
  std::vector<std::byte> subset_expected;
  std::size_t subset_bytes = 0;
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
    const auto mapping = q::take(i::DraftVocabulary::create(vocabulary, ids));
    const auto original_identity = head.identity;
    const auto live = q::take(budget.usage(backend.device()));
    auto unavailable = q::take(budget.reserve(backend.device(), live.limit_bytes - live.live_bytes));
    refused = false;
    try { (void)q::subset_prepared_draft_head(backend, head, mapping, 1, budget); }
    catch (const i::EvaluationFailure &failure) { refused = failure.status().IsResourceExhausted(); }
    require(refused && head.identity == original_identity && head.weight.tensor.ready &&
        !head.original_ids.description.valid(), "refused subset mutated or retired the complete source head");
    unavailable.reset();
    auto compact = q::subset_prepared_draft_head(backend, head, mapping, 1, budget);
    require(!compact.indexed_weight && compact.extra_weight_bytes() == expected.size() &&
        read(backend, compact.weight.tensor.ready) == expected &&
        values<std::int32_t>(backend, compact.original_ids.ready) == std::vector<std::int32_t>(ids.begin(), ids.end()),
        "prepared head subset changed encoded rows, physical extent or original ID mapping");
    head.identity[0] = head.identity[0] == '0' ? '1' : '0';
    const auto alternative = q::subset_prepared_draft_head(backend, head, mapping, 1, budget);
    require(compact.identity != alternative.identity && compact.identity != original_identity,
        "selected head identity lost its original quantized head component");
    head.identity = original_identity;
    const auto smaller_domain = q::take(i::DraftVocabulary::create(vocabulary - 1, std::array<std::int32_t, 1>{0}));
    refused = false;
    try { (void)q::subset_prepared_draft_head(backend, head, smaller_domain, 1, budget); }
    catch (const i::EvaluationFailure &failure) { refused = failure.status().IsInvalidArgument(); }
    require(refused, "prepared subset accepted another original vocabulary domain");

    // Spread 32K IDs over the whole original matrix, including its last row.
    // Check every selected code/scale byte, not merely the three sample rows.
    std::vector<std::int32_t> all_ids(32768);
    for (std::size_t n = 0; n < all_ids.size(); ++n)
      all_ids[n] = static_cast<std::int32_t>(n * static_cast<std::uint64_t>(vocabulary - 1) / (all_ids.size() - 1));
    const auto full_mapping = q::take(i::DraftVocabulary::create(vocabulary, all_ids));
    auto subset = q::subset_prepared_draft_head(backend, head, full_mapping, 1, budget);
    subset_expected.resize(row_bytes * all_ids.size());
    for (std::size_t n = 0; n < all_ids.size(); ++n) {
      payload.clear(); payload.seekg(static_cast<std::streamoff>(all_ids[n] * row_bytes));
      payload.read(reinterpret_cast<char *>(subset_expected.data() + n * row_bytes), static_cast<std::streamsize>(row_bytes));
      require(bool(payload), "cannot read complete original 32K encoded row inventory");
    }
    subset_bytes = subset_expected.size();
    require(read(backend, subset.weight.tensor.ready) == subset_expected &&
        values<std::int32_t>(backend, subset.original_ids.ready) == all_ids && subset.extra_weight_bytes() == subset_bytes,
        "32K prepared subset differs from original codes, scales, mapping or reported bytes");
    prepared_scores(backend, head, subset, full_mapping, source.roles.geometry, budget);
    escaped = head.weight.tensor.ready;
    head = std::move(subset);
    q::check(backend.collect_retired());
    require(escaped.size() == role.bytes && read(backend, head.weight.tensor.ready) == subset_expected,
        "replacing the complete head invalidated an independently escaped source or compact projection");
    refused = false;
    try { (void)q::subset_prepared_draft_head(backend, head, full_mapping, 1, budget); }
    catch (const i::EvaluationFailure &failure) { refused = failure.status().IsInvalidArgument(); }
    require(refused, "an already selected head was reinterpreted as a complete original head");
    subset_escaped = head.weight.tensor.ready;
  }
  q::check(backend.collect_retired());
  require(escaped.valid() && q::take(budget.usage(backend.device())).live_bytes >= source.draft_head->role.bytes,
      "escaping full head lease lost its original resource charge");
  escaped.reset(); q::check(backend.collect_retired());
  const auto compact_live = q::take(budget.usage(backend.device())).live_bytes;
  require(read(backend, subset_escaped) == subset_expected && compact_live >= subset_bytes &&
      compact_live < source.draft_head->role.bytes,
      "compact lease retained the complete head allocation or lost its own selected bytes");
  subset_escaped.reset(); q::check(backend.collect_retired());
  for (std::size_t n = 0; n < limits.size(); ++n) {
    const auto live = q::take(budget.usage(limits[n].resource)).live_bytes;
    if (live != baseline[n]) std::cerr << "head budget domain " << n << " before=" << baseline[n] << " after=" << live << '\n';
    require(live == baseline[n], "full head stream retained resource charges after retirement");
  }
  std::cout << "PASS full Q4_K head and 32K subset: independent source identities, metadata revalidation, "
      "budget refusal/retry, every selected code/scale byte, combined identity, full-to-subset replacement, "
      "independent escaping leases and every resource domain retired\n";
}
void compact_head(i::Backend &backend, const std::filesystem::path &manifest,
    const std::filesystem::path &specification, const std::filesystem::path &mapping_path,
    const std::filesystem::path &full_specification, const std::filesystem::path &scratch) {
  std::filesystem::create_directories(scratch);
  const auto source=q::prepare_weight_source(manifest,true,mapping_path,specification);
  const auto full=q::prepare_weight_source(manifest,true,{},full_specification);
  require(source.draft_head && source.draft_vocabulary && full.draft_head &&
      !source.draft_head->vocabulary_identity.empty() && full.draft_head->vocabulary_identity.empty(),
      "compact fixture lost its independent full-head reference or map identity");
  const auto &mapping=*source.draft_vocabulary;
  const auto &prepared=*source.draft_head;
  require(prepared.role.shape[0]==static_cast<std::int64_t>(mapping.size()), "compact shape differs from bound map");
  bool refused=false;
  try { (void)q::prepare_weight_source(manifest,true,{},specification); }
  catch (const std::runtime_error &) { refused=true; }
  require(refused,"compact metadata was accepted without its mapping");
  std::vector<std::int32_t> changed(mapping.ids().begin(),mapping.ids().end());
  std::int32_t missing=0;
  while(std::ranges::binary_search(mapping.ids(),missing))++missing;
  changed.back()=missing;
  const auto foreign=q::take(i::DraftVocabulary::create(mapping.full_size(),changed));
  std::vector<c::ResourceBudgetLimit> limits;
  for(const auto resource:{backend.device(),backend.host_device(),backend.staging(),m::system_resource()})
    if(resource && std::ranges::none_of(limits,[&](const auto &r){return r.resource==resource;}))limits.push_back({resource,1u<<30});
  const auto budget=q::take(c::ResourceBudget::create({.limits=limits,.allow_unattributed=true}));
  std::vector<std::size_t> baseline;
  for(const auto &limit:limits)baseline.push_back(q::take(budget.usage(limit.resource)).live_bytes);
  refused=false;
  try { (void)q::load_prepared_draft_head(backend,source.roles,prepared,budget,&foreign); }
  catch(const std::runtime_error &){refused=true;}
  require(refused,"a different same-size map reinterpreted compact codes as other token IDs");
  auto blocked=q::take(budget.reserve(backend.device(),q::take(budget.usage(backend.device())).limit_bytes));
  refused=false;
  try { (void)q::load_prepared_draft_head(backend,source.roles,prepared,budget,&mapping); }
  catch(const i::EvaluationFailure &failure){refused=failure.status().IsResourceExhausted();}
  require(refused,"direct compact loading ignored its original resource budget");blocked.reset();
  c::ReadLease escaped,escaped_ids;
  std::vector<std::byte> expected(prepared.role.bytes);
  const auto row_bytes=prepared.role.bytes/mapping.size();
  std::ifstream payload(full.draft_head->role.source,std::ios::binary);
  for(std::size_t at=0;at<mapping.size();++at) {
    payload.seekg(static_cast<std::streamoff>(mapping.ids()[at]*row_bytes));
    payload.read(reinterpret_cast<char*>(expected.data()+at*row_bytes),static_cast<std::streamsize>(row_bytes));
    require(bool(payload),"cannot read original full Q4 reference row");
  }
  {
    auto head=q::load_prepared_draft_head(backend,source.roles,prepared,budget,&mapping);
    require(read(backend,head.weight.tensor.ready)==expected && head.extra_weight_bytes()==expected.size() &&
        values<std::int32_t>(backend,head.original_ids.ready)==std::vector<std::int32_t>(mapping.ids().begin(),mapping.ids().end()),
        "offline compact codes/scales or IDs differ from all original selected rows");
    auto complete=q::load_prepared_draft_head(backend,full.roles,*full.draft_head,budget);
    prepared_scores(backend,complete,head,mapping,source.roles.geometry,budget);
    escaped=head.weight.tensor.ready;escaped_ids=head.original_ids.ready;
  }
  q::check(backend.collect_retired());
  require(read(backend,escaped)==expected && q::take(budget.usage(backend.device())).live_bytes<full.draft_head->role.bytes,
      "compact escaped lease retained its full-head reference or lost its codes");
  escaped.reset();q::check(backend.collect_retired());
  require(values<std::int32_t>(backend,escaped_ids)==std::vector<std::int32_t>(mapping.ids().begin(),mapping.ids().end()),
      "mapping lease retired before its independent reader");
  escaped_ids.reset();q::check(backend.collect_retired());
  for(std::size_t at=0;at<limits.size();++at)
    require(q::take(budget.usage(limits[at].resource)).live_bytes==baseline[at],"compact source retained a resource domain after retirement");
  std::cout<<"PASS compact Q4 head: bound source/map, same-size map refusal, budget refusal/retry, every selected code/scale and projection value, independent weight/ID escaping leases and real retirement\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if(argc==8 && std::string_view(argv[1])=="cuda" && std::string_view(argv[2])=="--compact-head") {
      auto backend=q::take(i::Backend::create({.primary=i::BackendKind::kCuda}));
      compact_head(backend,argv[3],argv[4],argv[5],argv[6],argv[7]);return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--metadata-self-test") {
      metadata(argv[2], argv[3]);
      std::cout << "PASS mapping metadata bounds, whitespace, invalid fields and retry; no Backend created\n";
      return 0;
    }
    if (argc == 5 && std::string_view(argv[1]) == "--metadata-only") {
      const auto source = q::prepare_weight_source(argv[2], true, argv[3], argv[4]);
      require(source.draft_head && source.draft_vocabulary, "metadata probe requires both independent assets");
      const auto rows = static_cast<std::int64_t>(source.draft_vocabulary->size());
      const auto description = q::take(c::gguf_weight_description(c::GgufQuantizedType::kQ4K,
          Type::kFloat32, source.roles.geometry.hidden, rows));
      std::cout << "PASS metadata rows=" << rows << " encoded_bytes=" << q::take(description.required_bytes())
          << " prepare_ms=" << source.metadata_prepare_seconds * 1000 << "; no Backend created\n";
      return 0;
    }
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
    for(const auto codec:{c::GgufQuantizedType::kQ8_0,c::GgufQuantizedType::kQ4K,c::GgufQuantizedType::kQ6K,c::GgufQuantizedType::kQ5K})
      encoded_rows(backend,codec,budget);
    if (kind==i::BackendKind::kCuda) q5_projection_composition(backend,budget);
    for (std::size_t index = 0; index < limits.size(); ++index)
      require(q::take(budget.usage(limits[index].resource)).live_bytes == baseline[index], "cold consumer retained another resource-domain budget charge");
    std::cout << "{\"status\":\"PASS\",\"backend\":\"" << argv[1]
        << "\",\"stored_types\":3,\"encoded_types\":" << (kind==i::BackendKind::kCuda?4:0)
        << ",\"encoded_refusals\":" << (kind==i::BackendKind::kCuda?0:4)
        << ",\"source_bytes_unchanged\":true,\"batched_original_ids_and_ties\":true,"
           "\"escaped_lease_budget_retired\":true,\"budget_rejection_and_retry\":true,\"invalid_ids_refused\":true,"
           "\"metadata_identity_checked\":" << (argc == 4 ? "true" : "false") << ",\"model_execution\":false}\n";
    return 0;
  } catch (const std::exception &failure) {
    std::cerr << "draft vocabulary probe: " << failure.what() << '\n'; return 1;
  }
}
