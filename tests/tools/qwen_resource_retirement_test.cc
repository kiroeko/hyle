// Small untrained source weights exercise the actual Model/Session host
// lifetime protocol. This is not a trained-model generation/quality test.
#include "../../engine/models/qwen35/main.cc"
#include "../../engine/inference/readback.h"
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#include <gtest/gtest.h>

namespace {
namespace q = qwen35;
namespace v = q::c::vulkan;
namespace d = v::detail;
using hyle::inference::Backend;
using hyle::inference::BackendKind;
using hyle::inference::PreparedReadback;

struct PredicateAlias final {
  q::g::Graph graph;
  q::g::ValueId input;
};
PredicateAlias predicate_alias(q::c::TensorDesc description) {
  auto builder = q::take(q::g::GraphBuilder::create());
  const auto input = q::take(builder.input(description));
  q::check(builder.publish(input));
  return {q::take(std::move(builder).finish()), input};
}
void replay_predicate_input(Backend &backend, q::g::Invocation &run,
    q::g::ValueId input, q::c::TensorDesc description, q::mem::Buffer &storage,
    std::span<const std::byte> bytes) {
  q::check(backend.write(storage.view(), bytes));
  q::check(run.bind(input, q::take(q::c::ConstTensorView::create(description, storage.view()))));
  const auto queues = backend.queue_bindings();
  q::check(run.prepare_replay(queues, {.mode = q::g::ReplayMode::kCommands}));
  auto completion = q::take(run.submit(queues));
  q::check(completion.wait());
}

q::ModelGeometry geometry() {
  q::ModelGeometry value;
  value.hidden = 64; value.intermediate = 128; value.vocabulary = 64;
  value.query_heads = value.kv_heads = 1;
  value.key_depth = value.value_depth = value.rotary = 64;
  value.gdn_key_heads = value.gdn_value_heads = 1;
  value.gdn_key_depth = value.gdn_value_depth = 64;
  value.conv_kernel = 4; value.context_limit = 32;
  value.rope_theta = 10000; value.epsilon = 1.0e-6F;
  value.tied_head = true; value.layers = "A";
  value.validate(); return value;
}

std::filesystem::path weights(const q::ModelGeometry &shape) {
  const auto folder = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()
      / ".scratch/validation/qwen-resource-retirement";
  std::filesystem::create_directories(folder);
  const auto payload_path = folder / "untrained-bf16.bin";
  std::ofstream payload(payload_path, std::ios::binary), manifest(folder / "weights.tsv");
  if (!payload || !manifest) throw std::runtime_error("cannot create small lifetime fixture");
  manifest << "HYLE_QWEN_ROLE_FILES_V2\n" << std::string(64, '1') << '\n';
  manifest << "geometry 64 128 64 1 1 64 64 64 1 1 64 64 4 32 10000 1e-6 1 0 grouped \"A\"\n";
  std::size_t offset = 0, role_index = 0;
  for (const auto &[name, parameter] : q::prepared::parameter_inventory(false, shape)) {
    std::size_t elements = 1;
    for (const auto extent : parameter.shape) elements *= static_cast<std::size_t>(extent);
    std::vector<std::uint16_t> values(elements);
    for (std::size_t i = 0; i < elements; ++i) {
      const float value = parameter.shape.size() == 1 ? 0.F
          : .03F * std::sin(float(i + role_index * 17) * .13F);
      auto bits = std::bit_cast<std::uint32_t>(value);
      bits += 0x7fffU + ((bits >> 16) & 1U);
      values[i] = static_cast<std::uint16_t>(bits >> 16);
    }
    payload.write(reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(std::uint16_t)));
    manifest << name << " BF16 " << parameter.shape.size();
    for (const auto extent : parameter.shape) manifest << ' ' << extent;
    manifest << ' ' << std::quoted(payload_path.generic_string()) << ' ' << offset
        << ' ' << values.size() * sizeof(std::uint16_t) << '\n';
    offset += values.size() * sizeof(std::uint16_t); ++role_index;
  }
  return folder / "weights.tsv";
}

struct NativeFailure final {
  // Keep only the original VkDevice/PFNs for instrumentation. This does not
  // keep the Hyle Runtime, ColdResources, Model or Session alive.
  std::shared_ptr<d::Device> device_owner;
  d::Device &device;
  d::DeviceFunctions original;
  unsigned observations = 0, drains = 0;
  bool unknown_drain = true;
  static inline NativeFailure *active = nullptr;
  explicit NativeFailure(v::Context &context)
      : device_owner(d::Access::runtime(context)->device), device(*device_owner), original(device.fn) {
    active = this;
    device.fn.GetFenceStatus = poll;
    device.fn.WaitForFences = wait;
    device.fn.QueueWaitIdle = drain;
  }
  ~NativeFailure() { device.fn = original; active = nullptr; }
  static VKAPI_ATTR VkResult VKAPI_CALL poll(VkDevice device, VkFence fence) {
    ++active->observations;
    if (active->unknown_drain) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.GetFenceStatus(device, fence);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL wait(VkDevice device, std::uint32_t count,
      const VkFence *fences, VkBool32 all, std::uint64_t timeout) {
    ++active->observations;
    if (active->unknown_drain) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.WaitForFences(device, count, fences, all, timeout);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL drain(VkQueue queue) {
    ++active->drains;
    if (active->unknown_drain) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.QueueWaitIdle(queue);
  }
};

void model_retirement(bool retain_session) {
  const auto shape = geometry();
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(shape.kv_heads), static_cast<std::size_t>(shape.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(shape.kv_heads), static_cast<std::size_t>(shape.value_depth)), 2)};
  const auto plan = q::cache::plan(shape, {.target = cache_rows}, 32, 1, 2,
      q::recurrent_bytes(q::state_schema_for(shape, 32), shape));
  auto model = std::make_unique<q::Model>(weights(shape), shape, 32, plan,
      std::filesystem::path{}, false, q::ReplayPolicy::kAuto, SIZE_MAX,
      q::Model::Sampling{false, {}, 0}, std::size_t{0},
      hyle::inference::BackendSelection{.primary=BackendKind::kVulkan});
  const std::array<std::size_t, 1> calls{2};
  model->prepare_family(calls); model->plan_execution(calls);
  auto session = model->create_session(calls);
  model->prepare_execution(calls);
  const std::array<std::int32_t, 2> tokens{1, 2};
  model->prepare_evaluation(*session, tokens);
  std::weak_ptr<q::ColdResources> scope = model->resources_;
  auto *const native_context = q::take(model->execution.vulkan_context());
  auto original_runtime = d::Access::runtime(*native_context);
  std::weak_ptr<d::Runtime> runtime = original_runtime;
  model->submit_evaluation(*session);
  ASSERT_TRUE(model->active_step);
  ASSERT_TRUE(model->active_step->run->in_flight());
  // The whole original Graph has actually been accepted. Withhold terminal
  // observations only now: Graph's internal dependency waits are not the
  // Model/destructor protocol under test, and receive no artificial fault.
  NativeFailure failure(*native_context);
  // Ensure an assertion failure does not leave the synthetic fault enabled
  // while host cleanup attempts its original native drain.
  struct AllowDrain final { NativeFailure &failure; ~AllowDrain() { failure.unknown_drain = false; } } allow{failure};
  model->active_step->completion = {};
  model.reset(); // Every host cleanup observes unknown; no infinite polling.
  EXPECT_GT(failure.drains, 0U);
  EXPECT_FALSE(scope.expired());
  EXPECT_FALSE(runtime.expired());
  EXPECT_EQ(session->pending, nullptr);
  const auto pending = session->try_retire();
  EXPECT_FALSE(pending.retired);
  EXPECT_TRUE(pending.status.IsResourceExhausted());
  const auto pending_drain = session->drain();
  EXPECT_FALSE(pending_drain.retired);
  EXPECT_EQ(pending_drain.status, pending.status);
  EXPECT_FALSE(scope.expired());
  if (!retain_session) {
    session.reset();
    EXPECT_FALSE(scope.expired()); // Only the original accepted attachment.
    failure.unknown_drain = false;
    EXPECT_TRUE(d::collect(original_runtime, true).IsResourceExhausted());
    EXPECT_TRUE(scope.expired()); // Last callback may release the whole scope.
    original_runtime.reset();
    EXPECT_TRUE(runtime.expired());
    return;
  }
  original_runtime.reset();
  failure.unknown_drain = false;
  const auto drained = session->drain();
  EXPECT_TRUE(drained.retired);
  EXPECT_EQ(drained.status, pending.status); // Original native observation diagnosis survives retirement.
  const auto retired = session->try_retire();
  EXPECT_TRUE(retired.retired);
  EXPECT_EQ(retired.status, pending.status);
  session.reset();
  EXPECT_TRUE(scope.expired());
  EXPECT_TRUE(runtime.expired()); // No application/Graph/resource-owner cycle.
}

TEST(QwenResourceRetirement, SessionRetiresOriginalWorkAfterItsModelWasDestroyed) {
  model_retirement(true);
}
TEST(QwenResourceRetirement, LastOriginalCallbackReleasesScopeWithoutApplicationOwners) {
  model_retirement(false);
}

TEST(QwenResourceRetirement, SharedReadbackReadsExactBoolAliasAndRejectsNonbinaryBytes) {
  auto backend = q::take(Backend::create({.primary=BackendKind::kVulkan}));
  const auto description = q::dense(q::Type::kBool, {4});
  // Borrowed backing precedes the Invocation and outlives every result/read.
  auto storage = q::take(q::mem::Buffer::create(backend.device(), 4, 256));
  const auto alias = predicate_alias(description);
  auto compiled = q::take(backend.compile(alias.graph, {}));
  auto port = q::take(PreparedReadback::prepare(compiled, alias.input, backend));
  ASSERT_EQ(port.facts.ownership, q::g::OutputOwnership::kSourceAlias);
  ASSERT_EQ(port.facts.description, description);
  auto run = q::take(compiled.executable.create_invocation());
  ASSERT_TRUE(port.bind(run).ok());
  constexpr std::uint8_t canary = 0xa5;
  std::array<std::uint8_t, 6> output;
  std::array<float, 6> scratch;
  for (const auto input : {std::array<std::uint8_t, 4>{0, 1, 1, 0},
                           std::array<std::uint8_t, 4>{1, 0, 0, 1}}) {
    output.fill(canary); scratch.fill(13.5F);
    replay_predicate_input(backend, run, alias.input, description, storage, std::as_bytes(std::span(input)));
    const auto status = port.predicates(backend, run, std::span(output).subspan(1, 4), scratch);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(output.front(), canary); EXPECT_EQ(output.back(), canary);
    for (std::size_t i = 0; i < input.size(); ++i) EXPECT_EQ(output[i+1], input[i]);
    for (const auto value : scratch) EXPECT_EQ(value, 13.5F);
  }
  const std::array<std::uint8_t, 4> nonbinary{0, 1, 2, 0};
  replay_predicate_input(backend, run, alias.input, description, storage, std::as_bytes(std::span(nonbinary)));
  output.fill(canary); scratch.fill(13.5F);
  EXPECT_TRUE(port.predicates(backend, run, std::span(output).subspan(1, 4), scratch).IsInvalidArgument());
  // Bool is read directly before validation; only the exact output range may change.
  EXPECT_EQ(output.front(), canary); EXPECT_EQ(output.back(), canary);
  for (const auto value : scratch) EXPECT_EQ(value, 13.5F);
}

TEST(QwenResourceRetirement, SharedReadbackReadsExactFloatAliasWithBoundedCallerScratch) {
  auto backend = q::take(Backend::create({.primary=BackendKind::kVulkan}));
  const auto description = q::dense(q::Type::kFloat32, {4});
  auto storage = q::take(q::mem::Buffer::create(backend.device(), 4*sizeof(float), 256));
  const auto alias = predicate_alias(description);
  auto compiled = q::take(backend.compile(alias.graph, {}));
  auto port = q::take(PreparedReadback::prepare(compiled, alias.input, backend));
  ASSERT_EQ(port.facts.ownership, q::g::OutputOwnership::kSourceAlias);
  ASSERT_EQ(port.facts.description, description);
  auto run = q::take(compiled.executable.create_invocation());
  ASSERT_TRUE(port.bind(run).ok());
  constexpr std::uint8_t canary = 0xa5;
  std::array<std::uint8_t, 6> output;
  std::array<float, 6> scratch;
  for (const auto input : {std::array<float, 4>{0, 1, -0.0F, 1},
                           std::array<float, 4>{1, 0, 1, 0}}) {
    replay_predicate_input(backend, run, alias.input, description, storage, std::as_bytes(std::span(input)));
    output.fill(canary); scratch.fill(13.5F);
    auto status = port.predicates(backend, run, std::span(output).subspan(1, 4), std::span(scratch).first(3));
    EXPECT_TRUE(status.IsInvalidArgument());
    for (const auto value : output) EXPECT_EQ(value, canary);
    for (const auto value : scratch) EXPECT_EQ(value, 13.5F);
    status = port.predicates(backend, run, std::span(output).subspan(1, 3), scratch);
    EXPECT_TRUE(status.IsInvalidArgument());
    for (const auto value : output) EXPECT_EQ(value, canary);
    for (const auto value : scratch) EXPECT_EQ(value, 13.5F);
    status = port.predicates(backend, run, std::span(output).subspan(1, 4), scratch);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(output.front(), canary); EXPECT_EQ(output.back(), canary);
    for (std::size_t i = 0; i < input.size(); ++i)
      EXPECT_EQ(output[i+1], static_cast<std::uint8_t>(input[i] == 1));
    EXPECT_EQ(scratch[4], 13.5F); EXPECT_EQ(scratch[5], 13.5F);
  }
  for (const auto invalid : {0.5F, std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
    const std::array<float, 4> input{0, 1, invalid, 0};
    replay_predicate_input(backend, run, alias.input, description, storage, std::as_bytes(std::span(input)));
    output.fill(canary); scratch.fill(13.5F);
    EXPECT_TRUE(port.predicates(backend, run, std::span(output).subspan(1, 4), scratch).IsInvalidArgument());
    EXPECT_EQ(output.front(), canary); EXPECT_EQ(output.back(), canary);
    EXPECT_EQ(scratch[4], 13.5F); EXPECT_EQ(scratch[5], 13.5F);
  }
}
} // namespace
