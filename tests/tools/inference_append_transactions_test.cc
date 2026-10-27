#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "../../engine/inference/append_transactions.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/custom_kernel.h"
#include "hyle/compute/providers/ggml_cpu.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace b = g::batch;
namespace i = hyle::inference;
namespace m = hyle::mem;

void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::move(*value);
}

// A real Graph effect refuses after both original append effects. It does not
// inject a CUDA/device failure or replace the episode's terminal observation.
struct Refuse final {
  static constexpr c::OperationId operation_id{0x48594c4554455354ULL, 0x494e464150504641ULL};
  static constexpr std::string_view operation_name = "inference-refuse-after-appends";
  constexpr c::OperationTraits operation_traits() const noexcept {
    return {{0, 0}, 0, c::AliasPolicy::kNone, c::EffectKind::kExternalSideEffect};
  }
  hyle::Status infer_output_descriptions(std::span<const c::TensorDesc> input,
      std::span<c::TensorDesc> output) const noexcept {
    return input.empty() && output.empty() ? hyle::Status::Ok()
        : hyle::Status::InvalidArgument("refusal effect has no tensors");
  }
};
constexpr c::ProviderRuntimeProperties runtime{.mode = c::ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = c::ProviderWorkspaceOwner::kCaller, .replay_class = c::ProviderReplayClass::kPrepared};
struct RefuseKernel final {
  struct Invocation final {};
  hyle::Result<Invocation> create_invocation(m::BufferView, m::AllocatorRef) const noexcept { return Invocation{}; }
  c::KernelCommandResult enqueue(Invocation &, c::detail::QueueSequence &,
      std::span<const c::ConstTensorView>, std::span<const c::TensorView>) const noexcept {
    return c::KernelCommandResult::rejected(hyle::Status::Internal("intentional refusal after original appends"));
  }
};
struct RefuseFactory final {
  hyle::Result<c::KernelProposal> propose(const c::KernelRequest &request,
      std::span<c::ResourceRequirement>, std::span<c::ResourceRequirement>) const noexcept {
    if (!request.operation.custom_as<Refuse>()) return std::unexpected(hyle::Status::Unimplemented("not this effect"));
    return c::KernelProposal{.runtime = runtime};
  }
  c::KernelPreparation prepare(const c::KernelPrepareRequest &request, m::AllocatorRef allocator) const noexcept {
    if (!request.operation.custom_as<Refuse>())
      return c::KernelPreparation::unsupported(hyle::Status::Unimplemented("not this effect"));
    auto kernel = c::make_custom_kernel(RefuseKernel{}, {.workspace = {0, 1}, .runtime = runtime,
        .persistent_bytes = 0, .invocation_private_bytes = 0}, allocator);
    return kernel ? c::KernelPreparation::prepared(std::move(*kernel))
        : c::KernelPreparation::error(kernel.error());
  }
};

struct Program final {
  c::Context context;
  c::Queue queue;
  c::BatchSchema schema;
  c::BatchLayoutVersion before, after;
  std::array<c::PagedArena, 2> arenas;
  std::array<c::ReadLease, 2> old_pages;
  std::array<c::PagedArenaRead, 2> old_frames;
  std::array<b::PagedAppendPages, 2> appends;
  std::array<g::NodeId, 2> nodes;
  std::array<g::ValueId, 2> inputs, cache_inputs{};
  std::array<std::uint8_t, 2> bound{};
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  c::LeasePool input_pool;
  c::ReadLease input;
  c::TensorDesc rows;
  std::array<std::uint16_t, 64> original{};

  explicit Program(bool refuse = false) {
    auto refusal = take(c::make_custom_provider({{0x48594c45, 0x49415050}, "inference-append-refusal", "1"},
        c::cpu::kBackendId, RefuseFactory{}));
    const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(), 0}, c::ProviderBinding{refusal, 0}};
    context = take(c::Context::create(c::cpu::device(), {.providers = providers}));
    queue = take(context.create_queue());
    const auto plain = take(c::CacheStorage::plain(c::DataType::kBFloat16));
    schema = take(c::BatchSchema::create({.query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
        .key_storage = plain, .value_storage = plain, .max_requests = 1, .max_query_rows = 1, .max_kv_rows = 4,
        .max_pages = 2, .max_page_refs = 2, .query_heads = 1, .kv_heads = 1, .qk_head_dim = 64,
        .value_head_dim = 64, .page_size = 2}));
    const std::array<std::int32_t, 2> query{0, 1}, ptr{0, 1};
    const std::array<std::int32_t, 1> old_id{0}, new_id{1}, old_last{1}, new_last{2};
    before = take(schema.version({query, ptr, old_id, old_last}));
    after = take(schema.version({query, ptr, new_id, new_last}));
    original.fill(0x3e80); // BF16 0.25; the uninitialized old tail stays NaN.
    for (std::size_t bank = 0; bank < 2; ++bank) {
      arenas[bank] = take(c::PagedArena::create(schema, bank ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey,
          m::system_resource(), {.max_backing_bytes = 4096}));
      auto initial = take(arenas[bank].acquire(0));
      std::array<std::uint16_t, 128> bytes; bytes.fill(0x7fc1);
      std::copy(original.begin(), original.end(), bytes.begin());
      check(initial.lease().view()->write(std::as_bytes(std::span(bytes))));
      old_pages[bank] = take(initial.publish(1));
      old_frames[bank] = take(arenas[bank].read(before, std::span(&old_pages[bank], 1)));
      std::array dirty{take(arenas[bank].acquire(1))};
      appends[bank] = take(b::PagedAppendPages::create(arenas[bank], after, old_frames[bank], dirty));
    }
    rows = take(c::TensorDesc::dense(c::DataType::kBFloat16, std::array<std::int64_t, 3>{1, 1, 64}));
    auto builder = take(g::GraphBuilder::create());
    for (std::size_t bank = 0; bank < 2; ++bank) {
      inputs[bank] = take(builder.input(rows));
      auto append = take(b::paged_append(schema, bank ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey));
      nodes[bank] = take(builder.add_node(c::Operation::custom(append), std::array{inputs[bank]}));
    }
    check(builder.depends_on(nodes[1], nodes[0]));
    if (refuse) {
      const auto stop = take(builder.add_node(c::Operation::custom(Refuse{}), {}));
      check(builder.depends_on(stop, nodes[1]));
    }
    graph = take(std::move(builder).finish());
    const std::array domains{g::StorageDomain{m::system_resource()}};
    auto source = take(g::ExecutionSource::acquire(context));
    const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
    compiled = take(b::compile(graph, {.storage_domains = domains, .targets = targets}));
    run = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    input_pool = take(c::LeasePool::create(m::system_resource(), 128, {.max_blocks = 1}));
    auto writer = take(input_pool.allocate(128, 64));
    std::array<std::uint16_t, 64> incoming; incoming.fill(0x3f40); // BF16 0.75.
    check(writer.view()->write(std::as_bytes(std::span(incoming))));
    input = take(writer.publish());
    for (const auto port : inputs) check(run.bind(port, rows, input));
    check(b::prepare(run, queues(), std::span<const b::NodeLayout>{}));
  }
  ~Program() { context.drain().IgnoreError(); }
  std::array<g::QueueBinding, 1> queues() { return {{{g::ExecutionLaneId{0}, &queue}}}; }
  i::AppendTransactions transactions() { return {nodes, appends, bound}; }
  void old_unchanged() {
    for (const auto &page : old_pages) {
      std::array<std::uint16_t, 128> actual{}, expected{};
      expected.fill(0x7fc1);
      std::copy(original.begin(), original.end(), expected.begin());
      check(page.view()->read(std::as_writable_bytes(std::span(actual))));
      EXPECT_EQ(actual, expected);
    }
  }
  void discard_and_reacquire() {
    auto discarded = transactions().discard(); ASSERT_TRUE(discarded) << discarded.error().message();
    ASSERT_TRUE(*discarded);
    appends = {};
    for (auto &arena : arenas) {
      auto writer = arena.acquire(1); ASSERT_TRUE(writer) << writer.error().message();
      EXPECT_TRUE(writer->lease().valid());
      EXPECT_EQ(writer->initialized_rows(), 0u);
    }
    old_unchanged();
  }
};

TEST(InferenceAppendTransactions, LateBindFailureRemembersActualTransferredPrefix) {
  Program p;
  auto withheld = std::move(p.appends[1]); // Deliberately empty second original transaction.
  EXPECT_FALSE(p.transactions().bind(p.run, p.graph, p.cache_inputs).ok());
  EXPECT_EQ(p.bound, (std::array<std::uint8_t, 2>{1, 0}));
  EXPECT_FALSE(p.appends[0].valid()); EXPECT_TRUE(withheld.valid());
  auto premature = p.transactions().discard(); ASSERT_TRUE(premature); EXPECT_FALSE(*premature);
  auto recovered = p.transactions().recover(p.run); ASSERT_TRUE(recovered) << recovered.error().message();
  ASSERT_TRUE(*recovered); EXPECT_EQ(p.bound, (std::array<std::uint8_t, 2>{0, 0}));
  ASSERT_TRUE(p.appends[0].valid());
  EXPECT_FALSE(p.appends[0].ready()); // No submission is no proof of success.
  auto repeated = p.transactions().recover(p.run); ASSERT_TRUE(repeated); EXPECT_TRUE(*repeated);
  p.appends[1] = std::move(withheld);
  p.discard_and_reacquire();
}

TEST(InferenceAppendTransactions, WholeFailureReturnsReservationsWithoutReadyAuthority) {
  Program p(true);
  check(p.transactions().bind(p.run, p.graph, p.cache_inputs));
  check(p.run.prepare_replay(p.queues()));
  auto completion = take(p.run.submit(p.queues()));
  // Even a fast CPU completion retains its original observation gate until wait.
  auto pending = p.transactions().recover(p.run); ASSERT_TRUE(pending); EXPECT_FALSE(*pending);
  const auto terminal = completion.wait(); completion = {};
  ASSERT_FALSE(terminal.ok());
  EXPECT_NE(terminal.message().find("intentional refusal"), std::string_view::npos);
  check(p.transactions().subspan(0, 1).take(p.run));
  EXPECT_EQ(p.bound, (std::array<std::uint8_t, 2>{0, 1}));
  auto recovered = p.transactions().recover(p.run); ASSERT_TRUE(recovered) << recovered.error().message();
  ASSERT_TRUE(*recovered); EXPECT_EQ(p.bound, (std::array<std::uint8_t, 2>{0, 0}));
  EXPECT_FALSE(p.appends[0].ready()); EXPECT_FALSE(p.appends[1].ready());
  p.discard_and_reacquire();
}
} // namespace
