#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>

#include "nvfp4_reference_fixture.h"

namespace mlx_head_payload_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace scalar = nvfp4_reference_test;
using scalar::take;
using scalar::check;
using scalar::dense;
using Type = c::DataType;

template<class T> std::vector<std::byte> bytes(std::span<const T> values) {
  const auto raw = std::as_bytes(values);
  return {raw.begin(), raw.end()};
}

inline std::optional<std::filesystem::path> payload_directory() {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, "HYLE_MLX_HEAD_PAYLOAD_DIR") != 0 ||
      value == nullptr || *value == '\0') {
    std::free(value);
    return std::nullopt;
  }
  std::filesystem::path result(value);
  std::free(value);
  return result;
#else
  const auto *value = std::getenv("HYLE_MLX_HEAD_PAYLOAD_DIR");
  return value && *value ? std::optional<std::filesystem::path>(value)
                         : std::nullopt;
#endif
}

inline std::vector<std::byte> read_payload(const std::filesystem::path &path,
                                         std::size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() != static_cast<std::streamoff>(expected))
    throw std::runtime_error("MLX head payload missing or wrong size: " + path.string());
  std::vector<std::byte> result(expected);
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(result.data()),
                   static_cast<std::streamsize>(result.size())))
    throw std::runtime_error("MLX head payload read failed: " + path.string());
  return result;
}

inline std::uint16_t little_u16(std::span<const std::byte> raw, std::size_t index) {
  return static_cast<std::uint16_t>(std::to_integer<unsigned>(raw[2 * index]) |
      (std::to_integer<unsigned>(raw[2 * index + 1]) << 8));
}

struct Bank final {
  static constexpr std::int64_t heads = 64;
  std::int64_t n, k;
  std::array<c::TensorDesc, 3> descriptions;
  std::array<std::vector<std::byte>, 3> payloads;

  Bank(const std::filesystem::path &root, std::string_view name,
       std::int64_t rows, std::int64_t depth) : n(rows), k(depth) {
    descriptions = {dense(Type::kUInt32, {heads, n, k / 8}),
        dense(Type::kBFloat16, {heads, n, k / 64}),
        dense(Type::kBFloat16, {heads, n, k / 64})};
    constexpr std::array parts{"weight", "scales", "biases"};
    for (unsigned i = 0; i < parts.size(); ++i)
      payloads[i] = read_payload(root / (std::string(name) + "." + parts[i] + ".bin"),
                                take(descriptions[i].required_bytes(), "original bank bytes"));
  }

  // Read individual serialized bits, independently of the product's U32
  // one/two-word decoder. Metadata keeps its original signed BF16 value.
  float value(std::int64_t head, std::int64_t row, std::int64_t column) const {
    const auto row_index = std::size_t(head * n + row);
    const auto first_bit = row_index * std::size_t(k) * 4 + std::size_t(column) * 4;
    unsigned code = 0;
    for (unsigned bit = 0; bit < 4; ++bit) {
      const auto position = first_bit + bit;
      code |= ((std::to_integer<unsigned>(payloads[0][position / 8]) >>
               (position % 8)) & 1U) << bit;
    }
    const auto metadata = row_index * std::size_t(k / 64) + std::size_t(column / 64);
    const float scale = scalar::wide(little_u16(payloads[1], metadata));
    const float bias = scalar::wide(little_u16(payloads[2], metadata));
    return scalar::f32(double(scalar::multiply(float(code), scale)) + double(bias));
  }
};

struct Payload final {
  Bank embed, unembed;
  explicit Payload(const std::filesystem::path &root)
      : embed(root, "embed_q", 512, 192), unembed(root, "unembed_out", 256, 512) {}
};

enum class Phase { kDecode, kPrefill };

inline float pulse(unsigned epoch, std::int64_t token, std::int64_t head,
                   unsigned boundary) {
  // Each dot has one nonzero term, independent of the reduction tree. The
  // .75 pulses also distinguish prematurely rounding decoded W to BF16.
  const auto label = epoch + static_cast<unsigned>(token) +
      static_cast<unsigned>(head) + boundary;
  constexpr std::array values{1.F, .5F, .25F, .75F, -1.F, -.5F, -.25F, -.75F};
  return values[label % values.size()];
}
inline std::int64_t query_position(unsigned epoch, std::int64_t token,
                                   std::int64_t head) {
  return (std::int64_t(epoch) * 31 + token * 7 + head * 13) % 192;
}
inline std::int64_t attention_position(unsigned epoch, std::int64_t token,
                                       std::int64_t head) {
  return (std::int64_t(epoch) * 67 + token * 11 + head * 17) % 512;
}
inline std::int64_t latent_position(unsigned epoch, std::int64_t token) {
  return (std::int64_t(epoch) * 67 + token * 149 + 13) % 512;
}

struct Case final {
  Phase phase;
  std::int64_t tokens;
  g::Graph graph;
  std::vector<c::TensorDesc> input_desc;
  std::vector<g::ValueId> inputs;
  std::array<g::ValueId, 2> outputs{};
  std::array<c::TensorDesc, 2> output_desc{};
  std::array<g::ValueId, 7> constants{};
  std::vector<std::int32_t> ids;
  c::TensorDesc ids_desc;

  Case(const Payload &payload, Phase kind) : phase(kind),
      tokens(kind == Phase::kDecode ? 2 : 3), ids(std::size_t(tokens * Bank::heads)),
      ids_desc(dense(Type::kInt32, {tokens, Bank::heads})) {
    for (std::int64_t t = 0; t < tokens; ++t)
      for (std::int64_t h = 0; h < Bank::heads; ++h)
        ids[std::size_t(t * Bank::heads + h)] = static_cast<std::int32_t>(h);
    auto builder = take(g::GraphBuilder::create(), "head payload graph");
    if (phase == Phase::kDecode) {
      input_desc = {dense(Type::kBFloat16, {2, 1, Bank::heads, 256}),
                    dense(Type::kBFloat16, {2, Bank::heads, 1, 512})};
    } else {
      input_desc = {dense(Type::kBFloat16, {1, 1, 3, 512})};
    }
    for (const auto &desc : input_desc)
      inputs.push_back(take(builder.input(desc), "independent role boundary"));
    for (unsigned i = 0; i < 6; ++i)
      constants[i] = take(builder.constant((i < 3 ? payload.embed : payload.unembed)
                                              .descriptions[i % 3]), "original compressed bank");
    constants[6] = take(builder.constant(ids_desc), "identity head indices");
    const auto reshape = [&](g::ValueId value, std::initializer_list<std::int64_t> shape) {
      std::array<g::Dimension, c::kMaxTensorRank> dimensions{};
      std::size_t axis = 0;
      for (const auto extent : shape) dimensions[axis++] = g::Dimension::fixed(extent);
      return take(builder.add(take(g::ViewSpec::reshape(std::span(dimensions).first(shape.size())),
                                   "role reshape spec"), value), "role reshape");
    };
    const auto project = [&](g::ValueId a, unsigned bank, c::model_ops::MlxMatmulOrientation orientation) {
      return take(builder.add(c::model_ops::mlx_grouped_matmul(
          c::model_ops::MlxWeightFormat::kAffine, 4, 64, orientation), a,
          constants[bank], constants[bank + 1], constants[bank + 2], constants[6]),
          "original packed head projection");
    };
    using Orientation = c::model_ops::MlxMatmulOrientation;
    if (phase == Phase::kDecode) {
      auto query = take(builder.add(take(g::ViewSpec::slice(3, g::Dimension::fixed(0),
          g::Dimension::fixed(192)), "exclude PE64 slice"), inputs[0]), "query nope slice");
      query = take(builder.add(c::Copy{}, query), "small noncontiguous query Copy");
      query = reshape(query, {2, Bank::heads, 192});
      const auto embed = project(query, 0, Orientation::kTransposeWeight);
      outputs[0] = reshape(embed, {2, Bank::heads, 1, 512});
      // This supplied boundary is independent of embed, not fake attention.
      const auto attention = reshape(inputs[1], {2, Bank::heads, 512});
      const auto unembed = project(attention, 3, Orientation::kTransposeWeight);
      outputs[1] = reshape(unembed, {2, 1, 16384});
    } else {
      const auto latent = reshape(inputs[0], {3, 1, 512});
      const auto embed = project(latent, 0, Orientation::kWeightAsStored);
      const auto unembed = project(latent, 3, Orientation::kTransposeWeight);
      const std::array projected{embed, unembed};
      const std::array<std::int64_t, 2> widths{192, 256};
      for (unsigned i = 0; i < outputs.size(); ++i) {
        auto y = reshape(projected[i], {1, 3, Bank::heads, widths[i]});
        outputs[i] = take(builder.add(take(g::ViewSpec::permute(
            std::array<std::int32_t, 4>{0, 2, 1, 3}), "restore head axis spec"), y),
            "restore head axis metadata");
      }
    }
    for (auto output : outputs) check(builder.publish(output), "publish role boundary");
    graph = take(std::move(builder).finish(), "finish head payload graph");
    for (unsigned i = 0; i < outputs.size(); ++i)
      output_desc[i] = take(graph.output_description(i), "logical output layout");
  }

  std::vector<std::vector<std::byte>> data(unsigned epoch) const {
    std::vector<std::vector<std::byte>> result;
    for (unsigned boundary = 0; boundary < input_desc.size(); ++boundary) {
      std::vector<std::uint16_t> values(take(input_desc[boundary].element_count(), "input count"), 0);
      if (phase == Phase::kDecode) {
        const auto width = boundary == 0 ? 256 : 512;
        for (std::int64_t t = 0; t < tokens; ++t)
          for (std::int64_t h = 0; h < Bank::heads; ++h) {
            const auto base = std::size_t((t * Bank::heads + h) * width);
            const auto position = boundary == 0 ? query_position(epoch, t, h)
                                                : attention_position(epoch, t, h);
            values[base + std::size_t(position)] = scalar::bf16(pulse(epoch, t, h, boundary));
            if (boundary == 0)
              std::fill(values.begin() + static_cast<std::ptrdiff_t>(base + 192),
                        values.begin() + static_cast<std::ptrdiff_t>(base + 256),
                        scalar::bf16(4.F)); // Excluded PE64 must never enter the dot.
          }
      } else {
        for (std::int64_t t = 0; t < tokens; ++t)
          values[std::size_t(t * 512 + latent_position(epoch, t))] =
              scalar::bf16(pulse(epoch, t, 0, 2));
      }
      result.push_back(bytes<std::uint16_t>(values));
    }
    return result;
  }

  std::array<std::vector<std::uint16_t>, 2> oracle(const Payload &payload,
                                                unsigned epoch,
                                                bool prematurely_round_weight = false) const {
    std::array<std::vector<std::uint16_t>, 2> result;
    const auto output = [&](unsigned branch, std::int64_t t, std::int64_t h, std::int64_t column) {
      float weight, activation;
      if (phase == Phase::kDecode) {
        weight = branch == 0 ? payload.embed.value(h, column, query_position(epoch, t, h))
            : payload.unembed.value(h, column, attention_position(epoch, t, h));
        activation = pulse(epoch, t, h, branch);
      } else {
        const auto position = latent_position(epoch, t);
        weight = branch == 0 ? payload.embed.value(h, position, column)
                             : payload.unembed.value(h, column, position);
        activation = pulse(epoch, t, 0, 2);
      }
      if (prematurely_round_weight) weight = scalar::wide(scalar::bf16(weight));
      // The dot starts at +0. Named operations retain each F32 boundary.
      return scalar::bf16(scalar::f32(0.0 + double(scalar::multiply(activation, weight))));
    };
    const std::array<std::int64_t, 2> widths{phase == Phase::kDecode ? 512 : 192, 256};
    for (unsigned branch = 0; branch < result.size(); ++branch) {
      result[branch].reserve(std::size_t(tokens * Bank::heads * widths[branch]));
      if (phase == Phase::kDecode) {
        for (std::int64_t t = 0; t < tokens; ++t)
          for (std::int64_t h = 0; h < Bank::heads; ++h)
            for (std::int64_t n = 0; n < widths[branch]; ++n)
              result[branch].push_back(output(branch, t, h, n));
      } else {
        for (std::int64_t h = 0; h < Bank::heads; ++h)
          for (std::int64_t t = 0; t < tokens; ++t)
            for (std::int64_t n = 0; n < widths[branch]; ++n)
              result[branch].push_back(output(branch, t, h, n));
      }
    }
    return result;
  }
};

template<class Read>
void expect(const g::GraphOutput &output, const c::TensorDesc &logical,
            std::span<const std::uint16_t> expected, Read read) {
  auto tensor = take(output.tensor(), "retained head result");
  const auto &desc = tensor.description();
  ASSERT_EQ(desc.data_type(), Type::kBFloat16);
  ASSERT_TRUE(std::ranges::equal(desc.dimensions(), logical.dimensions()));
  ASSERT_TRUE(desc.is_plain());
  std::vector<std::byte> storage(take(desc.required_bytes(), "actual result extent"));
  check(read(take(tensor.storage().subview(0, storage.size()), "exact physical result span"), storage),
        "read original result bytes");
  ASSERT_EQ(take(desc.element_count(), "result elements"), expected.size());
  // A published permute is metadata, not a physical transpose. Read logical
  // coordinates through the actual result strides on every backend.
  for (std::size_t i = 0; i < expected.size(); ++i) {
    auto coordinate = i;
    std::size_t offset = 0;
    for (std::size_t axis = desc.rank(); axis-- > 0;) {
      const auto extent = std::size_t(desc.dimensions()[axis]);
      offset += (coordinate % extent) * std::size_t(desc.byte_strides()[axis]);
      coordinate /= extent;
    }
    const auto actual = little_u16(storage, offset / 2);
    ASSERT_EQ(actual, expected[i]) << "logical BF16 element " << i;
  }
}

template<class Compile, class Write, class Read, class Prepare>
void exercise(const Payload &payload, Phase phase, m::MemoryResourceRef resource,
              std::span<const g::QueueBinding> queues, Compile compile, Write write,
              Read read, Prepare prepare, c::StorageOwner owner,
              std::size_t &premature_rounding_differences) {
  Case shape(payload, phase);
  std::array<c::ReadLease, 7> leases;
  std::array<g::ConstantBinding, 7> constants;
  const auto identity_ids = bytes<std::int32_t>(shape.ids);
  for (unsigned i = 0; i < constants.size(); ++i) {
    const auto &desc = i == 6 ? shape.ids_desc :
        (i < 3 ? payload.embed : payload.unembed).descriptions[i % 3];
    const auto &source = i == 6 ? identity_ids :
        (i < 3 ? payload.embed : payload.unembed).payloads[i % 3];
    auto storage = take(m::Buffer::create(resource, source.size(), 256), "original head bank allocation");
    check(write(storage.view(), source), "original head bytes upload");
    leases[i] = owner ? take(owner.adopt_buffer(std::move(storage)), "owned head constant")
                      : take(c::ReadLease::adopt(std::move(storage)), "head constant lease");
    constants[i] = {shape.constants[i], take(leases[i].tensor(desc), "constant bank tensor"),
        g::ConstantPolicy::kMaterialize, owner, owner ? leases[i] : c::ReadLease{}};
  }
  auto compiled = take(compile(shape.graph, g::CompileOptions{.constants = constants}), "compile head role graph");
  std::array runs{take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "slot0"),
                  take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "slot1")};
  std::array<g::OutputOwnership, 2> ownership;
  std::size_t capacity = 0;
  for (unsigned i = 0; i < ownership.size(); ++i) {
    ownership[i] = take(compiled.executable.output_ownership(i), "role output ownership");
    if (ownership[i] == g::OutputOwnership::kCaller)
      capacity += 2 * ((take(shape.output_desc[i].required_bytes(), "role output span") + 255) / 256 * 256);
  }
  c::LeasePool pool;
  if (capacity != 0)
    pool = take(c::LeasePool::create(resource, capacity, {.max_blocks = 4}), "two-slot role outputs");
  std::array<std::vector<m::Buffer>, 2> inputs;
  for (unsigned slot = 0; slot < runs.size(); ++slot) {
    for (unsigned i = 0; i < shape.inputs.size(); ++i) {
      inputs[slot].push_back(take(m::Buffer::create(resource,
          take(shape.input_desc[i].required_bytes(), "role input bytes"), 256), "role input allocation"));
      check(runs[slot].bind(shape.inputs[i], take(c::ConstTensorView::create(
          shape.input_desc[i], inputs[slot].back().view()), "role input tensor")), "bind role input");
    }
    for (unsigned i = 0; i < shape.outputs.size(); ++i)
      if (ownership[i] == g::OutputOwnership::kCaller)
        check(runs[slot].bind(shape.outputs[i], shape.output_desc[i],
            take(pool.allocate(take(shape.output_desc[i].required_bytes(), "writer bytes")), "role writer")),
            "bind role output");
    check(prepare(runs[slot], queues), "prepare original role invocation");
  }
  compiled = {}; shape.graph = {}; constants = {}; leases = {};
  std::array<std::array<g::GraphOutput, 2>, 2> retained;
  std::array<std::array<std::vector<std::uint16_t>, 2>, 2> expected;
  for (unsigned generation = 0; generation < 2; ++generation)
    for (unsigned slot = 0; slot < runs.size(); ++slot) {
      if (generation != 0) {
        for (unsigned i = 0; i < shape.outputs.size(); ++i)
          expect(retained[slot][i], shape.output_desc[i], expected[slot][i], read);
        EXPECT_FALSE(runs[slot].submit(queues));
        retained[slot] = {};
      }
      const auto epoch = generation * 2 + slot;
      const auto source = shape.data(epoch);
      for (unsigned i = 0; i < source.size(); ++i)
        check(write(inputs[slot][i].view(), source[i]), "update independent role boundary");
      auto done = take(runs[slot].submit(queues), "head role submit");
      check(done.wait(), "head role Completion");
      expected[slot] = shape.oracle(payload, epoch);
      const auto premature = shape.oracle(payload, epoch, true);
      for (unsigned i = 0; i < shape.outputs.size(); ++i) {
        for (std::size_t j = 0; j < expected[slot][i].size(); ++j)
          premature_rounding_differences += expected[slot][i][j] != premature[i][j];
        retained[slot][i] = take(runs[slot].result(shape.outputs[i]), "head role result");
        expect(retained[slot][i], shape.output_desc[i], expected[slot][i], read);
        if (retained[1 - slot][i].valid())
          expect(retained[1 - slot][i], shape.output_desc[i], expected[1 - slot][i], read);
      }
    }
  runs = {};
  for (unsigned slot = 0; slot < retained.size(); ++slot)
    for (unsigned i = 0; i < shape.outputs.size(); ++i)
      expect(retained[slot][i], shape.output_desc[i], expected[slot][i], read);
}

template<class Compile, class Write, class Read, class Prepare>
void run(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
         Compile compile, Write write, Read read, Prepare prepare, c::StorageOwner owner = {}) {
  const auto root = payload_directory();
  if (!root) {
    GTEST_SKIP() << "Set HYLE_MLX_HEAD_PAYLOAD_DIR to the six fixed layer0 tensor files";
  }
  const Payload payload(*root);
  std::size_t premature_rounding_differences = 0;
  exercise(payload, Phase::kDecode, resource, queues, compile, write, read, prepare, owner,
           premature_rounding_differences);
  exercise(payload, Phase::kPrefill, resource, queues, compile, write, read, prepare, owner,
           premature_rounding_differences);
  EXPECT_GT(premature_rounding_differences, 0U);
  ::testing::Test::RecordProperty("successful_submissions", 8);
  ::testing::Test::RecordProperty("original_weight_banks", 6);
  ::testing::Test::RecordProperty("packed_payload_bytes", 8257536);
  ::testing::Test::RecordProperty("actual_heads", 64);
  ::testing::Test::RecordProperty("premature_bf16_weight_rounding_differences",
      static_cast<int>(premature_rounding_differences));
  ::testing::Test::RecordProperty("payload_manifest", (root.value() / "manifest.json").string());
}
} // namespace mlx_head_payload_test
