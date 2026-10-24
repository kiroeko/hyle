#pragma once

#include "nvfp4_reference_fixture.h"
#include "hyle/compute/state_versions.h"
#include <cstring>
#include <numeric>

namespace ga_indexer_state_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace scalar = nvfp4_reference_test;
using scalar::take;
using scalar::check;
using scalar::dense;
using Type = c::DataType;
inline c::View reshape(std::initializer_list<std::int64_t> shape) {
  return take(c::View::reshape(take(c::TensorShape::create({shape.begin(), shape.size()}), "reshape dimensions")), "reshape operation");
}

template<class T, class Read>
std::vector<T> read_result(const g::GraphOutput &output, Read read) {
  auto view = take(output.tensor(), "actual result tensor");
  const auto &desc = view.description();
  std::vector<std::byte> raw(take(desc.required_bytes(), "actual result byte span"));
  if (!raw.empty()) check(read(view.storage(), raw), "read actual result bytes");
  std::vector<T> values(take(desc.element_count(), "logical result count"));
  for (std::size_t i = 0; i < values.size(); ++i) {
    auto coordinate = i; std::size_t offset = 0;
    for (std::size_t axis = desc.rank(); axis-- > 0;) {
      offset += coordinate % std::size_t(desc.dimensions()[axis]) * std::size_t(desc.byte_strides()[axis]);
      coordinate /= std::size_t(desc.dimensions()[axis]);
    }
    std::memcpy(&values[i], raw.data() + offset, sizeof(T));
  }
  return values;
}

inline void bind_outputs(const g::CompiledGraph &compiled, const g::Graph &graph,
    std::span<const g::ValueId> outputs, g::Invocation &slot,
    m::MemoryResourceRef resource, std::vector<m::Buffer> &external) {
  external.resize(outputs.size());
  for (std::size_t port = 0; port < outputs.size(); ++port)
    if (take(compiled.executable.output_ownership(port), "result ownership") == g::OutputOwnership::kCaller) {
      const auto desc = take(graph.value_description(outputs[port]), "result logical shape");
      external[port] = take(m::Buffer::create(resource, take(desc.required_bytes(), "result capacity"), 256), "caller result allocation");
      check(slot.bind(outputs[port], take(c::TensorView::create(desc, external[port].view()), "caller result tensor")), "caller result binding");
    }
}

template<class Write>
c::ReadLease upload(m::MemoryResourceRef resource, std::span<const std::byte> bytes,
                    Write write, c::StorageOwner owner = {}) {
  auto buffer = take(m::Buffer::create(resource, bytes.size(), 256), "immutable role allocation");
  if (!bytes.empty()) check(write(buffer.view(), bytes), "upload original role bytes");
  return owner ? take(owner.adopt_buffer(std::move(buffer)), "adopt role owner")
               : take(c::ReadLease::adopt(std::move(buffer)), "adopt role bytes");
}

template<class Compile, class Write, class Read, class Prepare>
void run_mapping(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
                 Compile compile, Write write, Read read, Prepare prepare,
                 c::StorageOwner owner = {}) {
  const auto desc = dense(Type::kInt32, {3, 5}), ids_desc = dense(Type::kInt32, {3, 3});
  auto b = take(g::GraphBuilder::create(), "integer mapping builder");
  const auto data = take(b.input(desc), "integer candidate table"), ids = take(b.input(ids_desc), "integer positions");
  const auto output = take(b.add(c::model_ops::gather_elements(), data, ids), "integer gather node");
  check(b.publish(output), "publish integer mapped candidates");
  auto graph = take(std::move(b).finish(), "integer mapping graph");
  auto compiled = take(compile(graph, g::CompileOptions{}), "compile integer mapping");
  const std::array<std::int32_t, 15> source{16777217, INT32_MAX, -1, INT32_MIN, 123456789,
      0x7f800001, 0x00000001, -16777217, -123456789, 0,
      999999937, -999999937, 42, INT32_MIN + 1, INT32_MAX - 1};
  const std::array<std::int32_t, 9> positions{0, 3, 0, 1, 0, 2, 4, 1, 3};
  std::vector<std::int32_t> expected;
  for (std::size_t i = 0; i < positions.size(); ++i) expected.push_back(source[i / 3 * 5 + positions[i]]);
  const auto data_lease = upload(resource, std::as_bytes(std::span(source)), write, owner);
  const auto ids_lease = upload(resource, std::as_bytes(std::span(positions)), write, owner);
  std::array slots{take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "map slot0"),
                   take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "map slot1")};
  std::array<std::vector<m::Buffer>, 2> external;
  for (unsigned lane = 0; lane < slots.size(); ++lane) {
    check(slots[lane].bind(data, take(data_lease.tensor(desc), "candidate integer view")), "bind integer candidates");
    check(slots[lane].bind(ids, take(ids_lease.tensor(ids_desc), "position integer view")), "bind integer positions");
    bind_outputs(compiled, graph, std::array{output}, slots[lane], resource, external[lane]);
    check(prepare(slots[lane], queues), "prepare integer mapping");
  }
  graph = {}; compiled = {};
  std::array<g::GraphOutput, 2> old;
  for (unsigned lane = 0; lane < slots.size(); ++lane) {
    auto done = take(slots[lane].submit(queues), "integer mapping submit"); check(done.wait(), "integer mapping completion");
    old[lane] = take(slots[lane].result(output), "integer mapped result");
    EXPECT_EQ(read_result<std::int32_t>(old[lane], read), expected);
    if (lane) EXPECT_EQ(read_result<std::int32_t>(old[0], read), expected);
  }
  slots = {};
  for (const auto &result : old) EXPECT_EQ(read_result<std::int32_t>(result, read), expected);
  ::testing::Test::RecordProperty("integer_casts", "none; original I32 words, including >2^24 and floating NaN/subnormal bit patterns");
  ::testing::Test::RecordProperty("successful_submissions", 2);
}

// A role consumer at the original post-QAT boundary. It deliberately does not
// simulate the F32 compressor producer or pretend that a packed FP4 bank is
// the official BF16 cache. Geometry, mask and integer mapping are one version.
template<class Compile, class Write, class Read, class Prepare>
void run_roles(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
               Compile compile, Write write, Read read, Prepare prepare,
               c::StorageOwner owner = {}) {
  constexpr std::int64_t batches = 2, tokens = 5, heads = 64, depth = 128,
                         length = 515, pitch = 520, top = 512;
  const auto cache_desc = dense(Type::kBFloat16, {batches, pitch, depth});
  const std::array input_desc{dense(Type::kBFloat16, {batches, tokens, heads, depth}),
      take(c::TensorDesc::strided_bytes(Type::kBFloat16, std::array{batches, length, depth},
          std::array{pitch * depth * 2, depth * 2, std::int64_t{2}}), "actual BF16 cache prefix"),
      dense(Type::kBFloat16, {batches, tokens, heads}),
      dense(Type::kFloat16, {batches, tokens, length}),
      dense(Type::kInt32, {batches * tokens, length})};
  auto b = take(g::GraphBuilder::create(), "typed score builder");
  std::array<g::ValueId, 5> inputs;
  for (unsigned i = 0; i < inputs.size(); ++i) inputs[i] = take(b.input(input_desc[i]), "typed score input");
  const auto key = take(b.add(reshape({batches, 1, length, depth}), inputs[1]), "view real cache prefix");
  const auto kt = take(b.add(take(c::View::permute(std::array<std::int32_t, 4>{0, 1, 3, 2}), "key permutation"), key), "transpose key metadata");
  const auto dot = take(b.add(c::MatMul{.accumulation_type = Type::kFloat32}, inputs[0], kt), "BF16 dot boundary");
  const auto positive = take(b.add(c::Unary{c::UnaryKind::kRelu}, dot), "BF16 relu");
  const auto weight = take(b.add(reshape({batches, tokens, heads, 1}), inputs[2]), "weight metadata");
  const auto product = take(b.add(c::Binary{c::BinaryKind::kMultiply}, positive, weight), "BF16 weighted boundary");
  const auto sum = take(b.add(c::Reduce{.kind = c::ReductionKind::kSum,
      .axes = take(c::Axes::from(std::array<std::int32_t, 1>{2}), "head reduction axis"),
      .keep_dimensions = false, .accumulation_type = Type::kFloat32}, product), "BF16 score boundary");
  const auto mask = take(b.add(c::Cast{Type::kBFloat16}, inputs[3]), "narrow causal mask");
  const auto masked = take(b.add(c::Binary{c::BinaryKind::kAdd}, sum, mask), "BF16 causal score");
  const auto wide = take(b.add(c::Cast{Type::kFloat32}, masked), "exact selection widening");
  const auto selection = take(b.add_node(c::Operation(c::TopK{.count = top}), std::array{wide}), "actual Flash Top512");
  const auto values = take(b.result(selection, 0), "selected scores"), ids = take(b.result(selection, 1), "local positions");
  const auto flat_ids = take(b.add(reshape({batches * tokens, top}), ids), "flatten selection rows");
  const auto mapped = take(b.add(c::model_ops::gather_elements(), inputs[4], flat_ids), "integer offset and invalid-ID map");
  const std::array outputs{masked, values, ids, mapped};
  for (auto output : outputs) check(b.publish(output), "publish typed role output");
  auto graph = take(std::move(b).finish(), "typed role graph");
  auto compiled = take(compile(graph, g::CompileOptions{}), "compile real BF16 role chain");
  std::array slots{take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "typed slot0"),
                   take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "typed slot1")};
  std::array<std::vector<m::Buffer>, 2> external;
  for (unsigned lane = 0; lane < slots.size(); ++lane)
    bind_outputs(compiled, graph, outputs, slots[lane], resource, external[lane]);

  struct Geometry { std::uint32_t length, pitch; std::int32_t offsets[2]; };
  const std::array state_specs{c::StateEntrySpec{.description = cache_desc},
      c::StateEntrySpec{.description = input_desc[3]}, c::StateEntrySpec{.description = input_desc[4]},
      c::StateEntrySpec{.kind = c::StateEntryKind::kHostMetadata, .metadata_bytes = sizeof(Geometry)}};
  auto schema = take(c::StateSchema::create(state_specs), "typed role state schema");
  auto store = take(c::StateStore::create(schema, {.max_versions = 8, .max_drafts = 2,
      .max_retained_blocks = 32, .max_backing_bytes = 8U << 20}), "typed role state store");
  std::array<c::StateEntryId, 4> state_ids;
  for (unsigned i = 0; i < state_ids.size(); ++i) state_ids[i] = take(schema.entry(i), "role state identity");
  std::array<c::StateVersion, 2> versions;
  std::array<std::array<c::ReadLease, 5>, 2> leases;
  std::array<std::array<g::GraphOutput, 4>, 2> retained;
  std::array<std::vector<std::uint16_t>, 2> expected_scores;
  c::StateHead head;
  std::size_t missing_round_differences = 0;
  constexpr std::array<std::int64_t, tokens> visible{0, 1, 512, 514, 515};
  for (unsigned epoch = 0; epoch < 2; ++epoch) {
    const Geometry geometry{std::uint32_t(length), std::uint32_t(pitch),
        {16777217 + std::int32_t(epoch * 1000), INT32_MAX - 2048 + std::int32_t(epoch * 500)}};
    std::vector<std::uint16_t> q(std::size_t(batches * tokens * heads * depth)),
        cache(std::size_t(batches * pitch * depth), 0x7fc1), w(std::size_t(batches * tokens * heads)),
        mask_bytes(std::size_t(batches * tokens * length));
    std::vector<std::int32_t> candidates(mask_bytes.size());
    const auto qi = [](auto batch, auto token, auto h, auto d) { return std::size_t(((batch * tokens + token) * heads + h) * depth + d); };
    for (std::int64_t batch = 0; batch < batches; ++batch) {
      for (std::int64_t row = 0; row < length; ++row) {
        constexpr std::array<float, 7> codes{.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
        const auto value = std::ldexp(codes[std::size_t(row % 7)], int(row / 7 - 40 + batch + epoch));
        const auto bits = scalar::bf16(value);
        std::fill_n(cache.begin() + (batch * pitch + row) * depth, depth, std::uint16_t{0});
        cache[std::size_t((batch * pitch + row) * depth)] = bits;
        // Separate group32 blocks make both nonzero terms possible original
        // FP4-QAT values; a tiny second term in the same block would round to 0.
        cache[std::size_t((batch * pitch + row) * depth + 32)] = scalar::bf16(scalar::wide(bits) / 256.F);
      }
      for (std::int64_t token = 0; token < tokens; ++token) {
        const auto wi = std::size_t((batch * tokens + token) * heads);
        q[qi(batch, token, 0, 0)] = scalar::bf16(token == 3 ? 1.5F : 1.F);
        w[wi] = scalar::bf16(token == 1 ? -1.F : token == 3 ? 1.0078125F : 1.F);
        if (token >= 2) {
          q[qi(batch, token, 1, 0)] = scalar::bf16(1.F);
          w[wi + 1] = scalar::bf16(token == 2 ? -1.F : token == 3 ? -1.515625F : 1.F / 256.F);
        }
        if (token == 2) q[qi(batch, token, 0, 32)] = scalar::bf16(1.F);
        for (std::int64_t row = 0; row < length; ++row) {
          const auto i = std::size_t((batch * tokens + token) * length + row);
          mask_bytes[i] = row < visible[std::size_t(token)] ? 0 : 0xfc00;
          candidates[i] = row < visible[std::size_t(token)] ? geometry.offsets[batch] + std::int32_t(row) : -1;
          float total = 0, without_round = 0;
          for (std::int64_t h = 0; h < heads; ++h) {
            float d = 0;
            for (const std::int64_t component : {0, 32})
              d = scalar::f32(double(d) + double(scalar::wide(q[qi(batch, token, h, component)])) *
                  scalar::wide(cache[std::size_t((batch * pitch + row) * depth + component)]));
            const auto positive_dot = std::max(scalar::rounded_bf16(d), 0.F);
            const auto ww = scalar::wide(w[wi + std::size_t(h)]);
            total = scalar::f32(double(total) + scalar::rounded_bf16(scalar::multiply(positive_dot, ww)));
            without_round = scalar::f32(double(without_round) + scalar::multiply(std::max(d, 0.F), ww));
          }
          auto score = scalar::bf16(total);
          if (row < visible[std::size_t(token)]) missing_round_differences += score != scalar::bf16(without_round);
          else score = 0xff80;
          expected_scores[epoch].push_back(score);
        }
      }
    }
    leases[epoch][0] = upload(resource, std::as_bytes(std::span(q)), write, owner);
    leases[epoch][1] = upload(resource, std::as_bytes(std::span(cache)), write, owner);
    leases[epoch][2] = upload(resource, std::as_bytes(std::span(w)), write, owner);
    leases[epoch][3] = upload(resource, std::as_bytes(std::span(mask_bytes)), write, owner);
    leases[epoch][4] = upload(resource, std::as_bytes(std::span(candidates)), write, owner);
    if (!epoch) {
      const std::array members{c::StateMember{state_ids[0], leases[0][1]},
          c::StateMember{state_ids[1], leases[0][3]}, c::StateMember{state_ids[2], leases[0][4]},
          c::StateMember{state_ids[3], {}, std::as_bytes(std::span(&geometry, 1))}};
      versions[0] = take(store.import(members), "import BF16 cache and geometry");
      head = take(c::StateHead::create(store, versions[0]), "role state head");
    } else {
      auto draft = take(store.begin(versions[0]), "next role state draft");
      check(draft.replace(state_ids[0], leases[1][1]), "replace BF16 cache");
      check(draft.replace(state_ids[1], leases[1][3]), "replace causal mask");
      check(draft.replace(state_ids[2], leases[1][4]), "replace integer map");
      check(draft.replace_metadata(state_ids[3], std::as_bytes(std::span(&geometry, 1))), "replace role geometry");
      versions[1] = take(draft.finalize(), "finalize same-generation role state");
    }
    // Bind through the actual version, retaining the full physical cache and
    // projecting only its active prefix. There is no caller-side layout Copy.
    for (unsigned port = 0; port < inputs.size(); ++port) {
      auto lease = leases[epoch][port];
      if (port == 1 || port >= 3)
        lease = take(versions[epoch].member(state_ids[port == 1 ? 0 : port - 2]), "versioned role lease");
      check(slots[epoch].bind(inputs[port], take(lease.tensor(input_desc[port]), "true role tensor")), "bind same-generation role");
    }
    check(prepare(slots[epoch], queues), "prepare typed role chain");
    auto done = take(slots[epoch].submit(queues), "typed role submit"); check(done.wait(), "typed role completion");
    for (unsigned port = 0; port < outputs.size(); ++port) retained[epoch][port] = take(slots[epoch].result(outputs[port]), "typed role result");
    EXPECT_EQ(read_result<std::uint16_t>(retained[epoch][0], read), expected_scores[epoch]);
    const auto selected_values = read_result<float>(retained[epoch][1], read);
    const auto local_ids = read_result<std::int32_t>(retained[epoch][2], read);
    const auto mapped_ids = read_result<std::int32_t>(retained[epoch][3], read);
    for (std::int64_t row = 0; row < batches * tokens; ++row) {
      std::vector<float> sorted;
      for (std::int64_t column = 0; column < length; ++column) sorted.push_back(scalar::wide(expected_scores[epoch][std::size_t(row * length + column)]));
      std::stable_sort(sorted.begin(), sorted.end(), std::greater<float>{});
      std::vector<bool> used(std::size_t(length), false);
      for (std::int64_t n = 0; n < top; ++n) {
        const auto at = std::size_t(row * top + n);
        const auto id = local_ids[at];
        ASSERT_GE(id, 0); ASSERT_LT(id, length); ASSERT_FALSE(used[std::size_t(id)]);
        used[std::size_t(id)] = true;
        EXPECT_EQ(selected_values[at], sorted[std::size_t(n)]);
        EXPECT_EQ(selected_values[at], scalar::wide(expected_scores[epoch][std::size_t(row * length + id)]));
        EXPECT_EQ(mapped_ids[at], candidates[std::size_t(row * length + id)]);
      }
    }
    if (epoch) {
      check(head.publish(versions[0].identity(), versions[1]), "publish completed role generation");
      EXPECT_FALSE(head.publish(versions[0].identity(), versions[0]).ok());
      EXPECT_TRUE(take(head.current(), "current role state").identity() == versions[1].identity());
      EXPECT_EQ(read_result<std::uint16_t>(retained[0][0], read), expected_scores[0]);
    }
  }
  EXPECT_GT(missing_round_differences, 0U);
  EXPECT_TRUE(slots[0].reset().IsUnavailable());
  slots = {}; compiled = {}; graph = {}; leases = {}; versions = {}; head = {}; store = {}; schema = {};
  for (unsigned epoch = 0; epoch < 2; ++epoch)
    EXPECT_EQ(read_result<std::uint16_t>(retained[epoch][0], read), expected_scores[epoch]);
  ::testing::Test::RecordProperty("successful_submissions", 2);
  ::testing::Test::RecordProperty("heads", int(heads));
  ::testing::Test::RecordProperty("depth", int(depth));
  ::testing::Test::RecordProperty("TopK", int(top));
  ::testing::Test::RecordProperty("cache", "BF16 B2 active515/pitch520; NaN padding, original StateVersion leases");
  ::testing::Test::RecordProperty("rounding_counterfactual_differences", int(missing_round_differences));
  ::testing::Test::RecordProperty("producer_scope", "post-QAT inputs; F32 overlapping compressor producer is not signed by this case");
}

// Official GA index_score.topk(min(profile_topk, end_pos / 4)) requests zero
// values before the first compressed row exists. This also remains an ordinary
// TopK operation for a nonempty source and for a non-last selected axis.
template<class Compile, class Write, class Read, class Prepare>
void run_empty(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
               Compile compile, Write write, Read, Prepare prepare,
               c::StorageOwner = {}) {
  struct Shape { std::array<std::int64_t, 3> dimensions; std::int32_t axis; };
  constexpr std::array cases{Shape{{2, 3, 0}, -1}, Shape{{2, 3, 7}, -1},
                             Shape{{0, 3, 7}, -1}, Shape{{2, 3, 7}, 1}};
  unsigned successful_submissions = 0;
  for (const auto &shape : cases) {
    auto input_desc = take(c::TensorDesc::dense(Type::kBFloat16, shape.dimensions), "empty score description");
    auto b = take(g::GraphBuilder::create(), "empty selection builder");
    const auto input = take(b.input(input_desc), "empty selection source");
    const auto node = take(b.add_node(c::Operation(c::TopK{.count = 0, .axis = shape.axis}),
                                      std::array{input}), "TopK0 ordinary node");
    const std::array outputs{take(b.result(node, 0), "empty values"),
                             take(b.result(node, 1), "empty indices")};
    for (auto output : outputs) check(b.publish(output), "publish empty selection");
    auto graph = take(std::move(b).finish(), "empty selection graph");
    const std::array descriptions{take(graph.value_description(outputs[0]), "empty value shape"),
                                  take(graph.value_description(outputs[1]), "empty index shape")};
    auto compiled = take(compile(graph, g::CompileOptions{}), "compile ordinary TopK0");
    std::array slots{take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "empty slot0"),
                     take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "empty slot1")};
    std::array<m::Buffer, 2> sources;
    std::array<std::array<m::Buffer, 2>, 2> external_outputs;
    for (unsigned lane = 0; lane < slots.size(); ++lane) {
      const auto bytes = take(input_desc.required_bytes(), "empty source capacity");
      sources[lane] = take(m::Buffer::create(resource, bytes, 256), "empty source allocation");
      if (bytes) {
        // NaN payload must never be compared, sorted or converted by k=0.
        std::vector<std::uint16_t> poison(bytes / 2, std::uint16_t(0x7fc1u + lane));
        check(write(sources[lane].view(), std::as_bytes(std::span(poison))), "write unused selection input");
      }
      check(slots[lane].bind(input, take(c::ConstTensorView::create(input_desc, sources[lane].view()),
                                        "empty source tensor")), "bind empty source");
      for (unsigned port = 0; port < outputs.size(); ++port) {
        ASSERT_EQ(take(descriptions[port].required_bytes(), "empty output extent"), 0U);
        if (take(compiled.executable.output_ownership(port), "empty output ownership") == g::OutputOwnership::kCaller) {
          external_outputs[lane][port] = take(m::Buffer::create(resource, 0, 256), "empty caller output");
          check(slots[lane].bind(outputs[port], take(c::TensorView::create(descriptions[port],
              external_outputs[lane][port].view()), "empty caller tensor")), "bind empty caller result");
        }
      }
      check(prepare(slots[lane], queues), "prepare empty selection");
    }
    compiled = {}; graph = {};
    std::array<std::array<g::GraphOutput, 2>, 2> retained;
    for (unsigned lane = 0; lane < slots.size(); ++lane) {
      auto done = take(slots[lane].submit(queues), "submit empty selection");
      check(done.wait(), "empty selection completion");
      ++successful_submissions;
      for (unsigned port = 0; port < outputs.size(); ++port) {
        retained[lane][port] = take(slots[lane].result(outputs[port]), "empty result owner");
        EXPECT_EQ(take(retained[lane][port].description(), "owned empty shape"), descriptions[port]);
      }
      EXPECT_TRUE(slots[lane].reset().IsUnavailable());
    }
    slots = {};
    for (const auto &lane : retained) for (unsigned port = 0; port < lane.size(); ++port)
      EXPECT_EQ(take(lane[port].description(), "empty result after invocation destruction"), descriptions[port]);
  }
  ::testing::Test::RecordProperty("successful_submissions", int(successful_submissions));
  ::testing::Test::RecordProperty("empty_geometries", int(cases.size()));
  ::testing::Test::RecordProperty("codec", "actual BF16 scores, not a packed row-cache carrier");
}

template<class Compile, class Write, class Read, class Prepare>
void run_positive(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
                  Compile compile, Write write, Read read, Prepare prepare,
                  c::StorageOwner = {}) {
  const auto desc = dense(Type::kFloat32, {2, 3});
  auto b = take(g::GraphBuilder::create(), "positive selection builder");
  const auto input = take(b.input(desc), "positive source");
  const auto node = take(b.add_node(c::Operation(c::TopK{.count = 1}), std::array{input}), "TopK1 node");
  const std::array outputs{take(b.result(node, 0), "positive values"), take(b.result(node, 1), "positive indices")};
  for (auto output : outputs) check(b.publish(output), "publish positive result");
  auto graph = take(std::move(b).finish(), "positive selection graph");
  auto compiled = take(compile(graph, g::CompileOptions{}), "compile positive selection");
  auto slot = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "positive slot");
  auto source = take(m::Buffer::create(resource, 6 * sizeof(float), 256), "positive source allocation");
  const std::array<float, 6> payload{1.F, 3.F, 2.F, 6.F, 4.F, 5.F};
  check(write(source.view(), std::as_bytes(std::span(payload))), "positive source write");
  check(slot.bind(input, take(c::ConstTensorView::create(desc, source.view()), "positive tensor")), "bind positive source");
  std::array<m::Buffer, 2> external;
  for (unsigned port = 0; port < outputs.size(); ++port)
    if (take(compiled.executable.output_ownership(port), "positive ownership") == g::OutputOwnership::kCaller) {
      const auto d = take(graph.value_description(outputs[port]), "positive output shape");
      external[port] = take(m::Buffer::create(resource, take(d.required_bytes(), "positive output bytes"), 256), "positive output allocation");
      check(slot.bind(outputs[port], take(c::TensorView::create(d, external[port].view()), "positive output tensor")), "bind positive output");
    }
  check(prepare(slot, queues), "prepare positive selection");
  auto done = take(slot.submit(queues), "positive submit"); check(done.wait(), "positive completion");
  const auto values = take(take(slot.result(outputs[0]), "positive value result").tensor(), "positive value view");
  const auto indices = take(take(slot.result(outputs[1]), "positive index result").tensor(), "positive index view");
  std::array<float, 2> actual_values{}; std::array<std::int32_t, 2> actual_indices{};
  check(read(values.storage(), std::as_writable_bytes(std::span(actual_values))), "read positive values");
  check(read(indices.storage(), std::as_writable_bytes(std::span(actual_indices))), "read positive indices");
  EXPECT_EQ(actual_values, (std::array<float, 2>{3.F, 6.F}));
  EXPECT_EQ(actual_indices, (std::array<std::int32_t, 2>{1, 0}));
}
} // namespace ga_indexer_state_test
