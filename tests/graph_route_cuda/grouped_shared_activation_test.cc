#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <string>
#include <cmath>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/graph/cuda.h"
#include "../compute_models/grouped_shared_activation_fixture.h"
#include "../compute_models/gguf_nvfp4_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "../compute_provider_ggml_cuda/test_support.h"
#include "../../src/compute_provider_ggml_cuda/tensor_program.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
enum class Mode { kPerSlot, kShared, kBroadcast };
enum class Storage { kFloat, kBfloat, kNvfp4 };
enum class Witness { kNone, kFloatActivation, kSubnormal };

c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
m::Buffer upload(m::MemoryResourceRef resource, std::span<const std::byte> bytes) {
  auto value = m::Buffer::create(resource, bytes.size(), 256).value();
  EXPECT_TRUE(value.view().write(bytes).ok()); return value;
}

void exercise(std::int64_t tokens, Mode mode, Storage storage = Storage::kFloat,
              std::int64_t slots = 2, bool direct_control = false,
              Witness witness = Witness::kNone) {
  grouped_shared_test::Fixture data(tokens, slots);
  const nvfp4_test::Fixture encoded(data.rows, data.columns, data.experts);
  const bool shared = mode != Mode::kPerSlot;
  const bool bfloat = storage == Storage::kBfloat, packed = storage == Storage::kNvfp4;
  const bool subnormal = witness == Witness::kSubnormal;
  if (witness != Witness::kNone) {
    ASSERT_FALSE(packed);
    if (subnormal) std::ranges::fill(data.weights, 0.F);
    for (std::int64_t expert = 0; expert < data.experts; ++expert) {
      // One independent row is exactly unit/large power-of-two. A one-hot
      // input makes the F32 activation and FTZ witnesses reduction-order free.
      const auto offset = static_cast<std::size_t>(expert * data.rows * data.columns);
      std::fill_n(data.weights.begin() + offset, data.columns, 0.F);
      data.weights[offset] = subnormal ? std::ldexp(1.F, 120) : 1.F;
    }
  }
  const auto wd = packed ? c::gguf_grouped_weight_description(c::GgufQuantizedType::kNvfp4,
      c::DataType::kFloat32, data.experts, data.rows, data.columns).value() :
      dense(bfloat ? c::DataType::kBFloat16 : c::DataType::kFloat32,
          {data.experts, data.rows, data.columns});
  const auto ad = dense(c::DataType::kFloat32, {tokens, shared ? 1 : slots, data.columns});
  const auto id = dense(c::DataType::kInt32, {tokens, slots});
  const auto yd = dense(c::DataType::kFloat32, {tokens, slots, data.rows});
  auto context = g::cuda::create_context().value();
  auto memory = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  m::Buffer weights;
  if (packed) weights = upload(memory.ref(), std::as_bytes(std::span(encoded.bytes)));
  else if (bfloat) {
    std::vector<hyle::simd::bfloat16_t> values;
    for (const auto value : data.weights) values.push_back(half_test::encode<hyle::simd::bfloat16_t>(value));
    weights = upload(memory.ref(), std::as_bytes(std::span(values)));
  } else weights = upload(memory.ref(), std::as_bytes(std::span(data.weights)));
  auto ready_weight = memory.adopt_buffer(std::move(weights)).value();
  const auto weight_view = ready_weight.tensor(wd).value();
  auto builder = g::GraphBuilder::create().value();
  const auto w = (bfloat || packed ? builder.constant(wd) : builder.input(wd)).value();
  const auto x = builder.input(ad).value(), indices = builder.input(id).value();
  auto activation = x;
  if (mode == Mode::kBroadcast) {
    const std::array shape{g::Dimension::fixed(tokens), g::Dimension::fixed(slots), g::Dimension::fixed(data.columns)};
    activation = builder.result(builder.add_node(g::ViewSpec::broadcast_to(shape).value(), x).value()).value();
  }
  const auto y = builder.result(builder.add_node(c::model_ops::grouped_matmul(),
      std::array{w, activation, indices}).value()).value();
  ASSERT_TRUE(builder.publish(y).ok());
  const std::array constants{g::ConstantBinding{w, weight_view, g::ConstantPolicy::kBorrow,
      memory, ready_weight}};
  auto compiled = g::cuda::compile(std::move(builder).finish().value(), context, memory,
      {.constants = bfloat || packed ? std::span<const g::ConstantBinding>(constants) : std::span<const g::ConstantBinding>{}});
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto last_step = compiled->report.step(compiled->report.step_count() - 1).value();
  EXPECT_EQ(last_step.provider.id, c::ggml_cuda::kProviderId);
  if (bfloat) {
    EXPECT_EQ(compiled->report.borrowed_constant_count(), 1U);
    EXPECT_EQ(compiled->report.materialized_constant_bytes(), 0U);
    EXPECT_EQ(compiled->report.packed_constant_count(), 0U);
    EXPECT_EQ(compiled->report.packed_constant_bytes(), 0U);
    EXPECT_EQ(compiled->report.constant(0)->description, wd);
  }
  std::vector<m::Buffer> retained_inputs, retained_outputs;
  std::vector<std::vector<float>> old_values;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array calls{compiled->executable.create_invocation().value(), compiled->executable.create_invocation().value()};
  const auto direct_ad = mode == Mode::kBroadcast ? c::TensorDesc::strided_bytes(c::DataType::kFloat32,
      std::array{tokens, slots, data.columns},
      std::array<std::int64_t, 3>{data.columns * 4, 0, 4}).value() : ad;
  c::PreparedOperation direct;
  if (direct_control) {
    auto prepared = c::prepare(context, c::model_ops::grouped_matmul(),
        std::array{wd, direct_ad, id}, std::array{yd}, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    direct = std::move(*prepared);
    EXPECT_EQ(direct.provider().id, c::ggml_cuda::kProviderId);
  }
  for (int round = 0; round < (subnormal ? 1 : 2); ++round)
    for (std::size_t lane = 0; lane < (subnormal ? std::size_t{1} : calls.size()); ++lane) {
      SCOPED_TRACE(::testing::Message() << tokens << '/' << slots << '/' << int(mode) << '/' << int(storage) << '/' << round << '/' << lane);
      auto values = data.activations(shared, round);
      if (witness != Witness::kNone) {
        std::ranges::fill(values, 0.F);
        const auto rows = tokens * (shared ? 1 : slots);
        for (std::int64_t row = 0; row < rows; ++row)
          values[static_cast<std::size_t>(row * data.columns)] = subnormal
              ? std::ldexp(1.F, -130)
              : 1.F + std::ldexp(1.F, -20) + float(row + round * 7) * std::ldexp(1.F, -18);
      } else if (bfloat) {
        auto first = std::ranges::find_if(values, [](float value) { return value != 0; });
        ASSERT_NE(first, values.end()); *first = std::copysign(1.00390625F, *first);
      }
      const auto ids = data.ids(lane != 0, round);
      auto expected = data.expected(shared, values, ids);
      if (packed) {
        for (std::int64_t t = 0; t < tokens; ++t)
          for (std::int64_t slot = 0; slot < slots; ++slot)
            for (std::int64_t row = 0; row < data.rows; ++row) {
              double sum = 0;
              const auto expert = ids[static_cast<std::size_t>(t * slots + slot)];
              for (std::int64_t k = 0; k < data.columns; ++k)
                sum += double(values[static_cast<std::size_t>((t * (shared ? 1 : slots) + (shared ? 0 : slot)) * data.columns + k)]) *
                    encoded.decoded[static_cast<std::size_t>((expert * data.rows + row) * data.columns + k)];
              expected[static_cast<std::size_t>((t * slots + slot) * data.rows + row)] = static_cast<float>(sum);
            }
      } else if (bfloat || witness == Witness::kFloatActivation) {
        auto narrowed = values;
        for (auto &value : narrowed) value = static_cast<float>(half_test::decode(half_test::encode<hyle::simd::bfloat16_t>(value)));
        EXPECT_NE(data.expected(shared, narrowed, ids), expected);
      }
      auto input = upload(memory.ref(), std::as_bytes(std::span(values)));
      auto selected = upload(memory.ref(), std::as_bytes(std::span(ids)));
      auto output = m::Buffer::create(memory.ref(), yd.required_bytes().value(), 256).value();
      auto direct_output = direct_control
          ? m::Buffer::create(memory.ref(), yd.required_bytes().value(), 256).value() : m::Buffer{};
      auto direct_workspace = direct_control ? ggml_cuda_test::workspace_buffer(direct, memory.ref()) : m::Buffer{};
      c::OperationInvocation direct_call;
      const std::array direct_inputs{weight_view,
          c::ConstTensorView::create(direct_ad, input.view()).value(),
          c::ConstTensorView::create(id, selected.view()).value()};
      c::TensorView direct_destination;
      if (direct_control) {
        auto call = direct.create_invocation(direct_workspace.view());
        ASSERT_TRUE(call) << call.error().message();
        direct_call = std::move(*call);
        direct_destination = c::TensorView::create(yd, direct_output.view()).value();
      }
      const auto run_direct = [&]() -> hyle::Status {
        auto done = direct_call.submit(queues[lane], direct_inputs, direct_destination);
        EXPECT_TRUE(done) << (done ? "" : done.error().message());
        if (!done) return done.error();
        return done->wait();
      };
      const auto bind = [&]() {
        if (!bfloat && !packed) {
          const auto status = calls[lane].bind(w, weight_view); EXPECT_TRUE(status.ok()) << status.message();
        }
        auto status = calls[lane].bind(x, c::ConstTensorView::create(ad, input.view()).value());
        EXPECT_TRUE(status.ok()) << status.message();
        status = calls[lane].bind(indices, c::ConstTensorView::create(id, selected.view()).value());
        EXPECT_TRUE(status.ok()) << status.message();
        status = calls[lane].bind(y, c::TensorView::create(yd, output.view()).value());
        EXPECT_TRUE(status.ok()) << status.message();
        const auto capture = g::cuda::prepare_replay(calls[lane], queues[lane],
            {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
        if (packed || tokens <= 3) {
          EXPECT_TRUE(capture.ok()) << capture.message();
          if (capture.ok()) EXPECT_EQ(calls[lane].replay_route(), g::ReplayRoute::kCaptured);
        }
        else {
          // Strict BF16/F32 uses the original F32 MMVF preference. Larger-T
          // misses keep host sorting and cannot be wrapped in outer capture.
          EXPECT_FALSE(capture.ok());
          const auto commands = g::cuda::prepare_replay(calls[lane], queues[lane]);
          EXPECT_TRUE(commands.ok()) << commands.message();
        }
      };
      bind();
      const auto run = [&]() -> hyle::Status {
        auto done = calls[lane].submit(queues[lane]);
        EXPECT_TRUE(done) << (done ? "" : done.error().message());
        if (!done) return done.error(); return done->wait();
      };
      auto status = run(); ASSERT_TRUE(status.ok()) << status.message();
      const auto actual = half_test::read<float>(output);
      EXPECT_EQ(actual, expected);
      if (witness != Witness::kNone)
        for (std::int64_t token = 0; token < tokens; ++token)
          for (std::int64_t slot = 0; slot < slots; ++slot) {
            const auto index = static_cast<std::size_t>((token * slots + slot) * data.rows);
            const auto value = values[static_cast<std::size_t>((token * (shared ? 1 : slots) +
                (shared ? 0 : slot)) * data.columns)];
            const float target = subnormal ? std::ldexp(1.F, -10) : value;
            EXPECT_EQ(std::bit_cast<std::uint32_t>(expected[index]), std::bit_cast<std::uint32_t>(target));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[index]), std::bit_cast<std::uint32_t>(target));
          }
      if (direct_control) {
        status = run_direct(); ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(half_test::read<float>(direct_output), expected);
      }
      for (const auto bad : {std::int32_t{-1}, static_cast<std::int32_t>(data.experts)}) if (!subnormal) {
        auto invalid = ids; invalid.back() = bad;
        ASSERT_TRUE(selected.view().write(std::as_bytes(std::span(invalid))).ok());
        EXPECT_TRUE(run().IsOutOfRange()); EXPECT_EQ(half_test::read<float>(output), expected);
        if (direct_control) {
          EXPECT_TRUE(run_direct().IsOutOfRange());
          EXPECT_EQ(half_test::read<float>(direct_output), expected);
          EXPECT_FALSE(direct_call.submit(queues[lane], direct_inputs, direct_destination));
        }
        EXPECT_EQ(half_test::read<std::int32_t>(selected), invalid);
        ASSERT_TRUE(selected.view().write(std::as_bytes(std::span(ids))).ok());
        EXPECT_FALSE(calls[lane].reset().ok());
        calls[lane] = compiled->executable.create_invocation().value(); bind();
        status = run(); ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(half_test::read<float>(output), expected);
        if (direct_control) {
          auto call = direct.create_invocation(direct_workspace.view());
          ASSERT_TRUE(call) << call.error().message();
          direct_call = std::move(*call);
          status = run_direct(); ASSERT_TRUE(status.ok()) << status.message();
          EXPECT_EQ(half_test::read<float>(direct_output), expected);
        }
      }
      retained_inputs.push_back(std::move(input)); retained_inputs.push_back(std::move(selected));
      retained_outputs.push_back(std::move(output)); old_values.push_back(expected);
      for (std::size_t i = 0; i < retained_outputs.size(); ++i)
        EXPECT_EQ(half_test::read<float>(retained_outputs[i]), old_values[i]);
    }
}

TEST(CudaGroupedShared, DenseAndBroadcastInputsPreserveEveryPlainSlot) {
  for (const auto tokens : {1, 9})
    for (const auto mode : {Mode::kPerSlot, Mode::kShared, Mode::kBroadcast}) exercise(tokens, mode);
  exercise(1, Mode::kBroadcast, Storage::kFloat, 1);
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto wd = dense(c::DataType::kFloat32, {3, 5, 64});
  const auto ad = c::TensorDesc::strided_bytes(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{2, 2, 64}, std::array<std::int64_t, 3>{260, 0, 4}).value();
  const auto id = dense(c::DataType::kInt32, {2, 2}), yd = dense(c::DataType::kFloat32, {2, 2, 5});
  const std::array inputs{wd, ad, id};
  const std::array outputs{yd};
  const auto unsupported = c::prepare(context, c::model_ops::grouped_matmul(), inputs, outputs, memory.ref());
  EXPECT_FALSE(unsupported); // Native sorted paths require a whole-row token pitch.
}
TEST(CudaGroupedShared, BfloatBanksRetainOriginalStorageAndFloatActivation) {
  exercise(1, Mode::kPerSlot, Storage::kBfloat);
  exercise(9, Mode::kBroadcast, Storage::kBfloat);
}
TEST(CudaGroupedShared, SmallTokenDirectAndGraphKeepFloatActivationAndSubnormalBits) {
  // Three concrete layouts, not a Cartesian product. Graph uses the real
  // slot-zero View, while Direct receives the same public strided descriptor.
  exercise(1, Mode::kPerSlot, Storage::kFloat, 3, true, Witness::kFloatActivation);
  exercise(3, Mode::kShared, Storage::kFloat, 3, true, Witness::kFloatActivation);
  exercise(3, Mode::kBroadcast, Storage::kBfloat, 3, true, Witness::kFloatActivation);
  // One tiny FTZ witness, outside the rebind/error matrix. Current-image
  // Direct is a second public route, not a physical old-fallback baseline.
  exercise(1, Mode::kShared, Storage::kFloat, 1, true, Witness::kSubnormal);
}

// Write the BF16 payload directly. In particular bits0008 must not pass
// through a host float->BF16 encoder/decoder whose caller may enable DAZ.
std::vector<hyle::simd::bfloat16_t> original_bfloat_bank(
    grouped_shared_test::Fixture &data, unsigned phase) {
  using BF16 = hyle::simd::bfloat16_t;
  std::vector<BF16> payload(data.weights.size(), BF16::from_bits(0));
  std::ranges::fill(data.weights, 0.F);
  for (std::int64_t expert = 0; expert < data.experts; ++expert) {
    const auto base = static_cast<std::size_t>(expert * data.rows * data.columns);
    const auto put = [&](std::size_t row, std::size_t column, std::uint16_t bits) {
      const auto index = base + row * static_cast<std::size_t>(data.columns) + column;
      payload[index] = BF16::from_bits(bits);
      data.weights[index] = std::bit_cast<float>(std::uint32_t{bits} << 16);
    };
    put(0, 0, 0x3f80); // W1 * A(1+2^-20).
    put(1, 1, 0x0008); // BF16 subnormal W2^-130 * A2^120.
    put(2, 2, 0x7b80); // Large BF16 W2^120 * F32 subnormal A2^-130.
    put(3, 0, static_cast<std::uint16_t>(0x3f80 + (unsigned(expert) + phase) % 32));
    put(static_cast<std::size_t>(data.rows - 1), 0,
        static_cast<std::uint16_t>(0x4000 + (unsigned(expert) + phase) % 32));
  }
  return payload;
}
std::vector<float> original_bfloat_expected(const grouped_shared_test::Fixture &data,
    std::span<const float> activation, std::span<const std::int32_t> ids) {
  auto expected = data.expected(true, activation, ids);
  for (std::int64_t slot = 0; slot < data.slots; ++slot) {
    const auto row = static_cast<std::size_t>(slot * data.rows);
    // These are independent known results, not a host subnormal dot oracle.
    expected[row] = std::bit_cast<float>(std::uint32_t{0x3f800008});
    expected[row + 1] = std::bit_cast<float>(std::uint32_t{0x3a800000});
    expected[row + 2] = std::bit_cast<float>(std::uint32_t{0x3a800000});
  }
  return expected;
}
void expect_original_bfloat_bits(std::span<const float> actual,
                                std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
        std::bit_cast<std::uint32_t>(expected[i])) << i;
}

TEST(CudaGroupedShared, OriginalBfloatBankHasNoWidenedBankAndDynamicDirectReuses64Submissions) {
  auto context = g::cuda::create_context().value();
  auto memory = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  std::array<std::size_t, 2> workspace{}, known_storage{};
  std::size_t sample = 0;
  for (const std::int64_t experts : {3, 17}) {
    grouped_shared_test::Fixture data(1, 3, 128, 17, experts);
    const auto payload = original_bfloat_bank(data, 0);
    const auto wd = dense(c::DataType::kBFloat16, {experts, data.rows, data.columns});
    const auto ad = dense(c::DataType::kFloat32, {1, 1, data.columns});
    const auto id = dense(c::DataType::kInt32, {1, data.slots});
    const auto yd = dense(c::DataType::kFloat32, {1, data.slots, data.rows});
    std::vector<float> activation(static_cast<std::size_t>(data.columns), 0.F);
    activation[0] = std::bit_cast<float>(std::uint32_t{0x3f800008});
    activation[1] = std::bit_cast<float>(std::uint32_t{0x7b800000});
    activation[2] = std::bit_cast<float>(std::uint32_t{0x00080000});
    const auto ids = data.ids(false, 0);
    const auto expected = original_bfloat_expected(data, activation, ids);
    auto packed = upload(memory.ref(), std::as_bytes(std::span(payload)));
    auto ready = memory.adopt_buffer(std::move(packed)).value();
    auto input = upload(memory.ref(), std::as_bytes(std::span(activation)));
    auto indices = upload(memory.ref(), std::as_bytes(std::span(ids)));
    auto output = m::Buffer::create(memory.ref(), yd.required_bytes().value(), 256).value();
    auto builder = g::GraphBuilder::create().value();
    const auto w = builder.constant(wd).value(), x = builder.input(ad).value(), i = builder.input(id).value();
    const std::array shape{g::Dimension::fixed(1), g::Dimension::fixed(data.slots), g::Dimension::fixed(data.columns)};
    const auto view = builder.result(builder.add_node(g::ViewSpec::broadcast_to(shape).value(), x).value()).value();
    const auto y = builder.add(c::model_ops::grouped_matmul(), w, view, i).value();
    ASSERT_TRUE(builder.publish(y).ok());
    std::vector<g::ConstantBinding> constants{{w, ready.tensor(wd).value(),
        g::ConstantPolicy::kBorrow, memory, ready}};
    auto compiled = g::cuda::compile(std::move(builder).finish().value(), context, memory, {.constants = constants});
    ASSERT_TRUE(compiled) << compiled.error().message();
    constants.clear(); ready.reset(); // Original plan retains the actual immutable bank.
    EXPECT_EQ(compiled->report.borrowed_constant_count(), 1U);
    EXPECT_EQ(compiled->report.constant(0)->description, wd);
    EXPECT_EQ(compiled->report.materialized_constant_bytes(), 0U);
    EXPECT_EQ(compiled->report.packed_constant_count(), 0U);
    EXPECT_EQ(compiled->report.packed_constant_bytes(), 0U);
    EXPECT_EQ(compiled->report.step(compiled->report.step_count() - 1)->provider.id, c::ggml_cuda::kProviderId);
    const auto storage = compiled->report.compiled_storage();
    workspace[sample] = compiled->report.workspace_bytes();
    known_storage[sample] = storage.known_bytes();
    const auto widened_bytes = data.weights.size() * sizeof(float);
    EXPECT_LT(workspace[sample], widened_bytes);
    EXPECT_LT(known_storage[sample], widened_bytes);
    const auto prefix = "E" + std::to_string(experts) + "_";
    RecordProperty(prefix + "original_bf16_bytes", std::to_string(wd.required_bytes().value()));
    RecordProperty(prefix + "caller_workspace_bytes", std::to_string(workspace[sample]));
    RecordProperty(prefix + "compiled_known_bytes", std::to_string(known_storage[sample]));
    RecordProperty(prefix + "compiled_unknown_provider_count", std::to_string(storage.unknown_provider_count));
    // Opaque native-pool storage remains unknown. These report assertions
    // concern the managed/known bank and caller workspace, not total GPU peak.
    auto queue = context.create_queue().value();
    auto slot = compiled->executable.create_invocation().value();
    ASSERT_TRUE(slot.bind(x, c::ConstTensorView::create(ad, input.view()).value()).ok());
    ASSERT_TRUE(slot.bind(i, c::ConstTensorView::create(id, indices.view()).value()).ok());
    ASSERT_TRUE(slot.bind(y, c::TensorView::create(yd, output.view()).value()).ok());
    const auto capture = g::cuda::prepare_replay(slot, queue,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(capture.ok()) << capture.message();
    EXPECT_EQ(slot.replay_route(), g::ReplayRoute::kCaptured);
    {
      auto done = slot.submit(queue); ASSERT_TRUE(done) << done.error().message();
      const auto status = done->wait(); ASSERT_TRUE(status.ok()) << status.message();
    }
    expect_original_bfloat_bits(half_test::read<float>(output), expected);
    auto old = slot.result(y).value(); // Real GraphOutput; output Buffer stays owned here.
    EXPECT_TRUE(slot.reset().IsUnavailable());
    if (experts == 17) {
      const auto direct_ad = c::TensorDesc::strided_bytes(c::DataType::kFloat32,
          std::array{std::int64_t{1}, data.slots, data.columns},
          std::array<std::int64_t, 3>{data.columns * 4, 0, 4}).value();
      auto prepared = c::prepare(context, c::model_ops::grouped_matmul(),
          std::array{wd, direct_ad, id}, std::array{yd}, memory.ref());
      ASSERT_TRUE(prepared) << prepared.error().message();
      EXPECT_EQ(prepared->provider().id, c::ggml_cuda::kProviderId);
      auto scratch = ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
      const auto scratch_bytes = prepared->workspace_requirement().size;
      EXPECT_LT(scratch_bytes, widened_bytes);
      auto created = prepared->create_invocation(scratch.view());
      ASSERT_TRUE(created) << created.error().message();
      auto call = std::move(*created);
      std::vector<m::Buffer> banks;
      banks.push_back(upload(memory.ref(), std::as_bytes(std::span(payload))));
      auto dynamic_ids = upload(memory.ref(), std::as_bytes(std::span(ids)));
      auto destination = m::Buffer::create(memory.ref(), yd.required_bytes().value(), 256).value();
      const auto out = c::TensorView::create(yd, destination.view()).value();
      for (unsigned step = 0; step < 64; ++step) {
        SCOPED_TRACE(step);
        const auto changed = original_bfloat_bank(data, step);
        if (step == 32) {
          banks.push_back(upload(memory.ref(), std::as_bytes(std::span(changed))));
          EXPECT_NE(banks[0].view().native_handle(), banks[1].view().native_handle());
        } else ASSERT_TRUE(banks.back().view().write(std::as_bytes(std::span(changed))).ok());
        const auto selected = data.ids((step & 1U) != 0, static_cast<int>(step));
        ASSERT_TRUE(dynamic_ids.view().write(std::as_bytes(std::span(selected))).ok());
        const std::array ports{c::ConstTensorView::create(wd, banks.back().view()).value(),
            c::ConstTensorView::create(direct_ad, input.view()).value(),
            c::ConstTensorView::create(id, dynamic_ids.view()).value()};
        {
          auto done = call.submit(queue, ports, out); ASSERT_TRUE(done) << done.error().message();
          const auto status = done->wait(); ASSERT_TRUE(status.ok()) << status.message();
        }
        expect_original_bfloat_bits(half_test::read<float>(destination),
            original_bfloat_expected(data, activation, selected));
        expect_original_bfloat_bits(half_test::read<float>(output), expected);
      }
      EXPECT_EQ(prepared->workspace_requirement().size, scratch_bytes);
      EXPECT_EQ(scratch.size(), scratch_bytes);
      RecordProperty("direct_submissions", "64");
      RecordProperty("direct_reused_caller_workspace_bytes", std::to_string(scratch_bytes));
      RecordProperty("direct_native_private_bytes_known", prepared->invocation_private_bytes() ? "true" : "false");
    }
    expect_original_bfloat_bits(half_test::read<float>(output), expected);
    old.reset(); ASSERT_TRUE(slot.reset().ok());
    ++sample;
  }
  // Increasing E at fixed T/S/N/K must not add an E*N*K F32 caller bank.
  const std::size_t extra_widened_bytes = (17 - 3) * 17 * 128 * sizeof(float);
  EXPECT_LT(workspace[1] > workspace[0] ? workspace[1] - workspace[0] : 0, extra_widened_bytes);
  EXPECT_LT(known_storage[1] > known_storage[0] ? known_storage[1] - known_storage[0] : 0, extra_widened_bytes);
}
TEST(CudaGroupedShared, PackedSharedInputsRetainOriginalQuantizationAndCapture) {
  for (const auto tokens : {1, 9})
    for (const auto mode : {Mode::kShared, Mode::kBroadcast}) exercise(tokens, mode, Storage::kNvfp4);
}

TEST(CudaGroupedShared, SharedCapacityUsesPhysicalRowsRatherThanBroadcastElements) {
  namespace native = c::ggml_cuda::internal;
  constexpr std::int64_t tokens = 32768, slots = 10, depth = 2048;
  const auto wd = dense(c::DataType::kFloat32, {1, 1, depth});
  const auto shared = c::TensorDesc::strided_bytes(c::DataType::kFloat32,
      std::array{tokens, slots, depth}, std::array<std::int64_t, 3>{depth * 4, 0, 4}).value();
  const auto per_slot = dense(c::DataType::kFloat32, {tokens, slots, depth});
  const auto id = dense(c::DataType::kInt32, {tokens, slots});
  const auto yd = dense(c::DataType::kFloat32, {tokens, slots, 1});
  const std::array output{yd};
  EXPECT_EQ(shared.required_bytes().value(), std::size_t{256} * 1024 * 1024);
  EXPECT_GT(shared.element_count().value(), std::size_t{(INT32_MAX - 1024) / sizeof(float)});
  EXPECT_TRUE(native::tensor_operation_supports(c::model_ops::grouped_matmul(),
      std::array{wd, shared, id}, output));
  EXPECT_FALSE(native::tensor_operation_supports(c::model_ops::grouped_matmul(),
      std::array{wd, per_slot, id}, output));
  // A folded carrier beyond the original native physical bound stays rejected.
  const auto large = c::TensorDesc::strided_bytes(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{262144, slots, depth},
      std::array<std::int64_t, 3>{depth * 4, 0, 4}).value();
  EXPECT_FALSE(native::tensor_operation_supports(c::model_ops::grouped_matmul(),
      std::array{wd, large, dense(c::DataType::kInt32, {262144, slots})},
      std::array{dense(c::DataType::kFloat32, {262144, slots, 1})}));
}

TEST(CudaGroupedShared, PrivateGateUpGluRetainsSharedAndPerSlotTopology) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto reference_context = c::Context::create(c::cuda::device(0).value(), c::ggml_cuda::provider()).value();
  double max_relative = 0;
  std::uint32_t max_ulp = 0;
  struct Case final { std::int64_t tokens; bool shared, packed; };
  constexpr std::array cases{Case{1, false, true}, Case{1, true, true},
      Case{9, false, true}, Case{9, true, true}, Case{1, true, false}};
  for (const auto [tokens, shared, packed] : cases) {
      SCOPED_TRACE(::testing::Message() << tokens << '/' << shared << '/' << packed);
      const grouped_shared_test::Fixture data(tokens, 2);
      const nvfp4_test::Fixture gate_bank(data.rows, data.columns, data.experts, 0);
      const nvfp4_test::Fixture up_bank(data.rows, data.columns, data.experts, 1);
      const auto wd = packed ? c::gguf_grouped_weight_description(c::GgufQuantizedType::kNvfp4,
          c::DataType::kFloat32, data.experts, data.rows, data.columns).value() :
          dense(c::DataType::kFloat32, {data.experts, data.rows, data.columns});
      const auto ad = dense(c::DataType::kFloat32, {tokens, shared ? 1 : data.slots, data.columns});
      const auto id = dense(c::DataType::kInt32, {tokens, data.slots});
      const auto yd = dense(c::DataType::kFloat32, {tokens, data.slots, data.rows});
      auto wg = upload(memory.ref(), packed ? std::as_bytes(std::span(gate_bank.bytes)) : std::as_bytes(std::span(data.weights)));
      auto wu = upload(memory.ref(), packed ? std::as_bytes(std::span(up_bank.bytes)) : std::as_bytes(std::span(data.weights)));
      auto builder = g::GraphBuilder::create().value();
      const auto gate_w = builder.constant(wd).value(), up_w = builder.constant(wd).value();
      const auto x = builder.input(ad).value(), ids = builder.input(id).value();
      const auto gate = builder.result(builder.add_node(c::model_ops::grouped_matmul(),
          std::array{gate_w, x, ids}).value()).value();
      const auto up = builder.result(builder.add_node(c::model_ops::grouped_matmul(),
          std::array{up_w, x, ids}).value()).value();
      const auto y = builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate, up).value();
      ASSERT_TRUE(builder.publish(y).ok());
      const std::array constants{
          g::ConstantBinding{gate_w, c::ConstTensorView::create(wd, wg.view()).value()},
          g::ConstantBinding{up_w, c::ConstTensorView::create(wd, wu.view()).value()}};
      auto compiled = g::cuda::compile(std::move(builder).finish().value(), context, memory.ref(), {.constants = constants});
      ASSERT_TRUE(compiled) << compiled.error().message();
      EXPECT_EQ(compiled->report.step_count(), 1U);
      EXPECT_EQ(compiled->report.step(0)->provider.id, c::ggml_cuda::kProviderId);
      // Independent logical dots are exact for the sparse +/-127 activation
      // fixture. The final oracle is the original standalone native GLU, not
      // the new grouped producer path or its possible fused epilogue.
      const std::array reference_inputs{yd, yd};
      const std::array reference_outputs{yd};
      auto reference = c::prepare(reference_context,
          c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}), reference_inputs, reference_outputs, memory.ref());
      ASSERT_TRUE(reference) << reference.error().message();
      auto workspace = ggml_cuda_test::workspace_buffer(*reference, memory.ref());
      auto reference_queue = reference_context.create_queue().value();
      auto reference_call = reference->create_invocation(workspace.view()).value();
      std::vector<m::Buffer> retained_inputs, retained_outputs;
      std::vector<std::vector<float>> old_values;
      std::array queues{context.create_queue().value(), context.create_queue().value()};
      std::array calls{compiled->executable.create_invocation().value(), compiled->executable.create_invocation().value()};
      for (int round = 0; round < 2; ++round)
        for (std::size_t lane = 0; lane < calls.size(); ++lane) {
          const auto values = data.activations(shared, round);
          const auto selected = data.ids(false, round + static_cast<int>(lane));
          auto logical = [&](const nvfp4_test::Fixture &bank) {
            if (!packed) return data.expected(shared, values, selected);
            std::vector<float> result(static_cast<std::size_t>(tokens * data.slots * data.rows));
            for (std::int64_t t = 0; t < tokens; ++t)
              for (std::int64_t slot = 0; slot < data.slots; ++slot)
                for (std::int64_t n = 0; n < data.rows; ++n) {
                  double sum = 0;
                  for (std::int64_t k = 0; k < data.columns; ++k)
                    sum += double(values[static_cast<std::size_t>((t * (shared ? 1 : data.slots) + (shared ? 0 : slot)) * data.columns + k)]) *
                        bank.decoded[static_cast<std::size_t>((selected[static_cast<std::size_t>(t * data.slots + slot)] * data.rows + n) * data.columns + k)];
                  result[static_cast<std::size_t>((t * data.slots + slot) * data.rows + n)] = static_cast<float>(sum);
                }
            return result;
          };
          const auto gate_values = logical(gate_bank), up_values = logical(up_bank);
          auto rg = upload(memory.ref(), std::as_bytes(std::span(gate_values)));
          auto ru = upload(memory.ref(), std::as_bytes(std::span(up_values)));
          auto expected = m::Buffer::create(memory.ref(), yd.required_bytes().value(), 256).value();
          const std::array reference_ports{c::ConstTensorView::create(yd, rg.view()).value(),
              c::ConstTensorView::create(yd, ru.view()).value()};
          auto reference_output = c::TensorView::create(yd, expected.view()).value();
          auto done = reference_call.submit(reference_queue, reference_ports, std::span(&reference_output, 1));
          ASSERT_TRUE(done); ASSERT_TRUE(done->wait().ok());
          const auto expected_values = half_test::read<float>(expected);
          auto input = upload(memory.ref(), std::as_bytes(std::span(values)));
          auto indices = upload(memory.ref(), std::as_bytes(std::span(selected)));
          auto output = m::Buffer::create(memory.ref(), yd.required_bytes().value(), 256).value();
          ASSERT_TRUE(calls[lane].bind(x, c::ConstTensorView::create(ad, input.view()).value()).ok());
          ASSERT_TRUE(calls[lane].bind(ids, c::ConstTensorView::create(id, indices.view()).value()).ok());
          ASSERT_TRUE(calls[lane].bind(y, c::TensorView::create(yd, output.view()).value()).ok());
          const auto capture = g::cuda::prepare_replay(calls[lane], queues[lane],
              {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
          if (packed || tokens <= 3) {
            ASSERT_TRUE(capture.ok()) << capture.message();
            EXPECT_EQ(calls[lane].replay_route(), g::ReplayRoute::kCaptured);
          }
          else {
            EXPECT_FALSE(capture.ok());
            ASSERT_TRUE(g::cuda::prepare_replay(calls[lane], queues[lane]).ok());
          }
          auto completed = calls[lane].submit(queues[lane]);
          ASSERT_TRUE(completed); ASSERT_TRUE(completed->wait().ok());
          const auto actual = half_test::read<float>(output);
          ASSERT_EQ(actual.size(), expected_values.size());
          for (std::size_t i = 0; i < actual.size(); ++i) {
            const double gate_value = gate_values[i], up_value = up_values[i];
            const double mathematical = gate_value / (1.0 + std::exp(-gate_value)) * up_value;
            const auto staged_exp = std::exp(-gate_values[i]);
            if (!std::isfinite(staged_exp) || mathematical == 0) {
              // Existing unmarked GLU uses expf: overflow retains its original
              // staged zero boundary, unlike ordinary stable SiLU.
              EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                  std::bit_cast<std::uint32_t>(expected_values[i]));
            } else {
              // Existing F32 CUDA nonlinear budget (stable_silu_test.cc):
              // strict standalone unary and the fast MMVQ epilogue need not
              // share exp/div bits. Both check the independent FP64 formula.
              const double budget = std::abs(mathematical) * 3e-6 +
                  256.0 * std::numeric_limits<float>::denorm_min();
              EXPECT_NEAR(expected_values[i], mathematical, budget);
              EXPECT_NEAR(actual[i], mathematical, budget);
              max_relative = std::max(max_relative, std::abs(actual[i] - mathematical) / std::abs(mathematical));
            }
            const auto a = std::bit_cast<std::uint32_t>(actual[i]);
            const auto e = std::bit_cast<std::uint32_t>(expected_values[i]);
            if ((a >> 31) == (e >> 31)) max_ulp = std::max(max_ulp, a > e ? a - e : e - a);
          }
          ::testing::Test::RecordProperty("max_glu_relative", std::to_string(max_relative));
          ::testing::Test::RecordProperty("max_glu_ulp_vs_standalone", std::to_string(max_ulp));
          retained_inputs.push_back(std::move(input)); retained_inputs.push_back(std::move(indices));
          retained_outputs.push_back(std::move(output)); old_values.push_back(actual);
          for (std::size_t i = 0; i < retained_outputs.size(); ++i)
            EXPECT_EQ(half_test::read<float>(retained_outputs[i]), old_values[i]);
        }
    }
}

} // namespace
