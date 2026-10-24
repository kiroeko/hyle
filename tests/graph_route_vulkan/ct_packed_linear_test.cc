#include "../compute_models/ct_packed_graph_fixture.h"
#include "../compute_route_vulkan/fixture.h"
#include "hyle/graph/vulkan.h"
#include "../../src/graph/execution_source_internal.h"
#include "../../src/graph/artifact_packed_region.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"
#include "../../src/compute_packed_shared/artifact_packed_linear.h"

#include <cstring>

namespace {
namespace f = ct_packed_graph_test;
namespace v = f::c::vulkan;
namespace vn = v::detail;
namespace gd = f::g::detail;
using f::Type;
enum class Selection { kDefault, kOriginal, kPacked };

// Test-only filtering of the original source's existing region proposals.
// Its real state/context, resource callbacks, native prepare and submit remain
// unchanged; this is neither a new execution source nor a runtime switch.
struct Harness final {
  v::Context context = f::take(vulkan_test::context(), "Vulkan Context");
  v::MemoryResource device = f::take(v::MemoryResource::create(context, v::MemoryKind::kDevice), "device memory");
  v::MemoryResource staging = f::take(v::MemoryResource::create(context, v::MemoryKind::kStaging), "staging memory");
  static constexpr std::size_t capacity = 1024U * 1024U;
  v::TransferSlot transfer = f::take(v::TransferSlot::create(context, staging, capacity), "transfer slot");
  f::m::Buffer host = f::take(f::m::Buffer::create(f::m::system_resource(), capacity, 64), "host transfer bytes");
  f::g::ExecutionSource source = f::take(f::g::vulkan::execution_source(context), "native source");
  gd::ExecutionSourceState *state = gd::ExecutionSourceAccess::state(source);
  const gd::ExecutionSourceVTable *original = state->table;
  gd::ExecutionSourceVTable table = *original;
  const f::g::RegionProviderVTable *regions = original->region_provider(*state);
  f::g::RegionProviderVTable selected = *regions;
  Selection selection = Selection::kDefault;
  static inline Harness *current = nullptr;

  Harness() {
    current = this;
    selected.propose = filter;
    table.region_provider = [](const gd::ExecutionSourceState &) noexcept -> const f::g::RegionProviderVTable * { return &current->selected; };
    state->table = &table;
    ::testing::Test::RecordProperty("device", context.environment().device_name);
  }
  ~Harness() { state->table = original; current = nullptr; }
  static hyle::Status filter(void *owner, const f::g::RegionDiscoveryRequest &request,
      f::m::AllocatorRef allocator, f::g::RegionProposalWriter &writer) noexcept {
    std::array<f::g::RegionProposal, 128> proposals;
    std::array<f::g::NodeId, 1024> nodes;
    f::g::RegionProposalWriter candidates(proposals, nodes);
    HYLE_RETURN_IF_ERROR(current->regions->propose(owner, request, allocator, candidates));
    if (candidates.proposal_limit_reached() || candidates.node_limit_reached())
      return hyle::Status::ResourceExhausted("test proposal scratch exceeded");
    for (const auto &candidate : candidates.proposals()) {
      const bool packed = candidate.policy == gd::ArtifactPackedPattern::kPolicy;
      if ((current->selection == Selection::kOriginal && packed) ||
          (current->selection == Selection::kPacked && !packed)) continue;
      HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add(candidate));
      if (!more) break;
    }
    return hyle::Status::Ok();
  }
  hyle::Result<f::g::CompiledGraph> compile(const f::Logical &graph,
      Selection mode, f::g::CompileOptions options = {}) {
    selection = mode;
    const std::array domains{f::g::StorageDomain{device.ref()}};
    const std::array targets{f::g::ExecutionTarget{source, f::g::StorageDomainId{0}}};
    return f::g::compile(graph.graph, {.storage_domains = domains, .targets = targets}, options);
  }
  hyle::Status write(f::m::BufferView destination, std::span<const std::byte> bytes) {
    HYLE_ASSIGN_OR_RETURN(auto desc, f::c::TensorDesc::dense(Type::kUInt8,
        std::array{static_cast<std::int64_t>(bytes.size())}));
    HYLE_ASSIGN_OR_RETURN(auto host_bytes, host.view().subview(0, bytes.size()));
    HYLE_RETURN_IF_ERROR(host_bytes.write(bytes));
    HYLE_ASSIGN_OR_RETURN(auto input, f::c::ConstTensorView::create(desc, host_bytes));
    HYLE_ASSIGN_OR_RETURN(auto output, f::c::ConstTensorView::create(desc, destination));
    HYLE_ASSIGN_OR_RETURN(auto native, vn::MemoryAccess::tensor_view(vn::Access::runtime(context),
        output, f::m::system_allocator()));
    HYLE_ASSIGN_OR_RETURN(auto done, transfer.upload(native, input));
    return done.wait();
  }
  f::c::ReadLease upload(std::span<const std::byte> bytes) {
    auto buffer = f::take(f::m::Buffer::create(device.ref(), bytes.size(), 256), "original payload buffer");
    f::check(write(buffer.view(), bytes), "original payload upload");
    return f::take(f::c::ReadLease::adopt(std::move(buffer)), "original payload lease");
  }
  std::vector<double> read(f::c::ConstTensorView value) {
    const auto bytes = value.description().required_bytes().value();
    const auto desc = f::dense(Type::kUInt8, {static_cast<std::int64_t>(bytes)});
    const auto input = f::take(f::c::ConstTensorView::create(desc, value.storage()), "download byte view");
    auto native = f::take(vn::MemoryAccess::tensor_view(vn::Access::runtime(context), input,
        f::m::system_allocator()), "native read view");
    // Original managed outputs may expose an exact odd 16-bit logical span.
    // The existing synchronous transfer accepts that span without granting
    // TransferSlot the extra word-padding capacity it correctly requires.
    std::vector<std::byte> downloaded(bytes);
    f::check(context.download(native, downloaded), "exact output download");
    auto host_bytes = f::take(host.view().subview(0, bytes), "download host span");
    f::check(host_bytes.write(downloaded), "store exact download bytes");
    const auto type = value.description().data_type();
    const auto count = value.description().element_count().value();
    std::vector<double> result(static_cast<std::size_t>(count));
    if (type == Type::kFloat32) {
      std::vector<float> values(static_cast<std::size_t>(count));
      f::check(host_bytes.read(std::as_writable_bytes(std::span(values))), "F32 read");
      std::copy(values.begin(), values.end(), result.begin());
    } else {
      std::vector<std::uint16_t> values(static_cast<std::size_t>(count));
      f::check(host_bytes.read(std::as_writable_bytes(std::span(values))), "16-bit read");
      for (std::size_t i = 0; i < result.size(); ++i) result[i] = type == Type::kFloat16
          ? half_test::decode_bits<f::t::F16>(values[i]) : half_test::decode_bits<f::t::BF16>(values[i]);
    }
    return result;
  }
  void submit(f::g::Invocation &slot) {
    const std::span<const f::g::QueueBinding> queues;
    f::check(slot.prepare_replay(queues), "native prepare_replay");
    auto done = f::take(slot.submit(queues), "native Graph submit");
    f::check(done.wait(), "native Graph completion");
  }
};

struct Data final {
  f::t::Fixture fixture;
  std::vector<double> activation, weight;
  std::vector<f::c::ReadLease> inputs;
  Data(Harness &h, const f::Logical &graph, unsigned seed = 0, bool witness = false,
      bool special = false, bool subnormal = false)
      : fixture(graph.profile.asymmetric, seed, graph.descriptions[1].dimensions()[0],
                graph.descriptions[0].dimensions()[1]) {
    if (special) { fixture.scales[2] = std::numeric_limits<double>::infinity(); fixture.scales[4] = std::numeric_limits<double>::quiet_NaN(); }
    if (subnormal) std::ranges::fill(fixture.scales, std::ldexp(1., -130));
    activation = fixture.activation(graph.profile.m, graph.profile.operand, seed, witness);
    if (subnormal) { std::ranges::fill(activation, 0.); activation[0] = 256.; }
    weight = fixture.decoded(graph.profile.scale, graph.profile.decode, graph.profile.operand);
    inputs.push_back(h.upload(f::floating(graph.profile.operand, activation)));
    inputs.push_back(h.upload(f::bytes(fixture.codes)));
    if (graph.profile.asymmetric) inputs.push_back(h.upload(f::bytes(fixture.zeros)));
    inputs.push_back(h.upload(f::floating(graph.profile.scale, fixture.scales)));
  }
};
void selected(const f::g::CompileReport &report, bool expected) {
  unsigned count = 0;
  for (std::size_t i = 0; i < report.step_count(); ++i) {
    const auto step = report.step(i).value();
    if (step.policy != gd::ArtifactPackedPattern::kPolicy) continue;
    ++count;
    EXPECT_EQ(step.implementation, f::c::ProviderImplementationId{2});
    EXPECT_EQ(step.kind, f::g::CompileStepKind::kProviderRegion);
  }
  EXPECT_EQ(count, expected ? 1U : 0U);
}
void bind(const f::Logical &graph, const Data &data, f::g::Invocation &slot) {
  for (std::size_t i = 0; i < graph.ports.size(); ++i)
    if (!graph.constants || i == 0)
      f::check(slot.bind(graph.ports[i], graph.descriptions[i], data.inputs[i]), "bind original CT input");
}
// The same independent finite F64 sum and F32 reduction error bound as the
// shared CT graph fixture. One-hot source-round witnesses use exact equality.
void expect_dot(const f::Logical &graph, const Data &data,
    std::span<const double> actual, bool witness = false) {
  const auto n = data.fixture.rows, k = data.fixture.depth;
  ASSERT_EQ(actual.size(), static_cast<std::size_t>(graph.profile.m * n));
  constexpr double u = std::numeric_limits<float>::epsilon() / 2;
  const double terms = double(2 * k + 32), gamma = terms * u / (1 - terms * u);
  for (std::int64_t row = 0; row < graph.profile.m; ++row)
    for (std::int64_t column = 0; column < n; ++column) {
      double sum = 0, sumabs = 0;
      for (std::int64_t depth = 0; depth < k; ++depth) {
        const auto product = data.activation[static_cast<std::size_t>(row * k + depth)] *
            data.weight[static_cast<std::size_t>(column * k + depth)];
        sum += product; sumabs += std::abs(product);
      }
      const auto i = static_cast<std::size_t>(row * n + column);
      SCOPED_TRACE(i);
      if (std::isnan(sum)) { EXPECT_TRUE(std::isnan(actual[i])); continue; }
      if (std::isinf(sum)) { EXPECT_EQ(actual[i], sum); continue; }
      ASSERT_TRUE(std::isfinite(actual[i]));
      if (witness) EXPECT_EQ(actual[i], f::t::round(graph.profile.operand, sum));
      else {
        const auto bound = gamma * sumabs + double(k) * std::numeric_limits<float>::denorm_min();
        EXPECT_GE(actual[i], f::t::round(graph.profile.operand, sum - bound));
        EXPECT_LE(actual[i], f::t::round(graph.profile.operand, sum + bound));
      }
    }
}
std::vector<double> output(Harness &h, f::g::Invocation &slot, f::g::ValueId value) {
  return h.read(f::take(slot.output(value), "Graph output tensor"));
}

TEST(VulkanCtPackedLinear, PublicGraphPreservesTypedWeightBoundariesAndOriginalControl) {
  Harness h;
  constexpr std::array profiles{
      f::Profile{false, Type::kFloat32, Type::kFloat32, Type::kFloat32, 1},
      f::Profile{true, Type::kBFloat16, Type::kBFloat16, Type::kBFloat16, 1},
      f::Profile{true, Type::kBFloat16, Type::kBFloat16, Type::kBFloat16, 3},
      f::Profile{true, Type::kFloat16, Type::kFloat32, Type::kFloat32, 3},
      f::Profile{false, Type::kBFloat16, Type::kFloat16, Type::kFloat32, 3},
      f::Profile{true, Type::kFloat32, Type::kBFloat16, Type::kFloat16, 3, false},
      f::Profile{false, Type::kFloat32, Type::kFloat16, Type::kBFloat16, 3}};
  for (const auto profile : profiles) {
    SCOPED_TRACE(::testing::Message() << int(profile.scale) << '/' << int(profile.decode) << '/' << int(profile.operand));
    const auto graph = f::build(profile);
    Data data(h, graph, 0, true);
    auto default_route = f::take(h.compile(graph, Selection::kDefault), "default public Graph compile");
    selected(default_route.report, profile.m == 1 && profile.scale == Type::kBFloat16 &&
        profile.decode == Type::kBFloat16 && profile.operand == Type::kBFloat16);
    for (const auto route : {Selection::kPacked, Selection::kOriginal}) {
      auto compiled = f::take(h.compile(graph, route), "same public Graph compile");
      selected(compiled.report, route == Selection::kPacked);
      auto slot = f::invocation(compiled.executable);
      bind(graph, data, slot); h.submit(slot);
      expect_dot(graph, data, output(h, slot, graph.output), true);
    }
  }
}

TEST(VulkanCtPackedLinear, LegalNonfiniteAndAmplifiedSubnormalWeightsKeepIeeeClasses) {
  Harness h;
  for (const bool subnormal : {false, true}) {
    const auto graph = f::build({true, subnormal ? Type::kBFloat16 : Type::kFloat32,
        subnormal ? Type::kBFloat16 : Type::kFloat32, Type::kFloat32, 1});
    Data data(h, graph, 0, false, !subnormal, subnormal);
    if (subnormal) EXPECT_GT(data.weight[0] * data.activation[0], 0.);
    auto compiled = f::take(h.compile(graph, Selection::kPacked), "exceptional Graph compile");
    selected(compiled.report, true);
    auto slot = f::invocation(compiled.executable);
    bind(graph, data, slot); h.submit(slot);
    expect_dot(graph, data, output(h, slot, graph.output), subnormal);
  }
}

TEST(VulkanCtPackedLinear, IndependentSlotsRebindOriginalBytesAndRetainResults) {
  Harness h;
  const auto graph = f::build({true, Type::kBFloat16, Type::kFloat16, Type::kFloat32, 3});
  auto compiled = f::take(h.compile(graph, Selection::kPacked), "slot Graph compile");
  selected(compiled.report, true);
  std::array slots{f::invocation(compiled.executable), f::invocation(compiled.executable)};
  f::g::GraphOutput retained;
  std::vector<double> old;
  unsigned submissions = 0;
  for (unsigned round = 0; round < 2; ++round) for (const unsigned lane : {1U, 0U}) {
    if (round) {
      if (lane == 0) retained.reset();
      f::check(slots[lane].reset(), "reuse retired slot");
    }
    Data data(h, graph, round * 2 + lane);
    bind(graph, data, slots[lane]); data.inputs.clear();
    h.submit(slots[lane]); ++submissions;
    auto result = f::take(slots[lane].result(graph.output), "independent result");
    auto actual = h.read(f::take(result.tensor(), "result tensor"));
    expect_dot(graph, data, actual);
    EXPECT_TRUE(slots[lane].reset().IsUnavailable());
    if (round == 0 && lane == 0) { retained = std::move(result); old = actual; }
    if (round == 1 && lane == 1)
      f::exact_values(h.read(f::take(retained.tensor(), "old result")), old);
  }
  ::testing::Test::RecordProperty("actual_submissions", submissions);
}

TEST(VulkanCtPackedLinear, PublishedDataAndControlEscapesUseOriginalDecomposition) {
  Harness h;
  for (const auto escape : {f::Escape::kPublishedDecode, f::Escape::kExtraView, f::Escape::kControl}) {
    const auto graph = f::build({true, Type::kBFloat16, Type::kFloat32, Type::kFloat32, 3}, 9, 64, escape);
    auto compiled = f::take(h.compile(graph, Selection::kDefault), "escaped Graph compile");
    selected(compiled.report, false);
    Data data(h, graph, 2);
    auto slot = f::invocation(compiled.executable);
    bind(graph, data, slot); h.submit(slot);
    expect_dot(graph, data, output(h, slot, graph.output));
    f::exact_values(output(h, slot, graph.extra), escape == f::Escape::kControl
        ? data.activation : data.fixture.decoded(graph.profile.scale, graph.profile.decode, graph.profile.decode));
  }
}

TEST(VulkanCtPackedLinear, OriginalConstantBanksSurviveSlotsWithoutDenseNativeStorage) {
  Harness h;
  auto graph = f::build({true, Type::kBFloat16, Type::kFloat32, Type::kFloat32, 3},
      257, 1024, f::Escape::kNone, true);
  Data data(h, graph, 1);
  std::vector<f::g::ConstantBinding> constants;
  vn::Plan plan;
  for (std::size_t port = 0; port < graph.ports.size(); ++port) {
    vn::PlanValue value{graph.descriptions[port], {}};
    if (port) {
      const auto tensor = f::take(data.inputs[port].tensor(graph.descriptions[port]), "constant view");
      constants.push_back({graph.ports[port], tensor, f::g::ConstantPolicy::kMaterialize, {}, {}});
      value.constant = f::take(vn::MemoryAccess::tensor_view(vn::Access::runtime(h.context),
          tensor, f::m::system_allocator()), "constant native view");
    }
    plan.values.push_back(std::move(value));
  }
  plan.inputs = {0};
  plan.outputs = {4};
  plan.values.push_back({graph.graph.value_description(graph.output).value(), {}});
  const auto operation = f::c::Operation::custom(f::c::packed::internal::ArtifactPackedLinear{
      f::Format::kCompressedTensorsAsymmetric4, 32, Type::kFloat32, Type::kFloat32});
  plan.steps.push_back({operation, {0, 1, 2, 3}, {4}, {}, {1, 1, 1}, {}});
  vn::PlanFacts facts;
  auto native = f::take(vn::prepare_plan(h.context, plan, &facts), "bounded CT native plan");
  EXPECT_EQ(facts.native_dispatch_count, 1U);
  EXPECT_EQ(facts.managed_region_count, 0U);
  ASSERT_EQ(facts.invocation_storage_block_bytes.size(), 1U);
  EXPECT_EQ(facts.invocation_storage_block_bytes[0], 3U * 257U * sizeof(float));
  EXPECT_LT(facts.retained_constant_capacity_bytes, 257U * 1024U * sizeof(std::uint16_t));
  auto compiled = f::take(h.compile(graph, Selection::kPacked, {.constants = constants}), "original-bank Graph");
  selected(compiled.report, true);
  std::array slots{f::invocation(compiled.executable), f::invocation(compiled.executable)};
  constants.clear(); data.inputs.clear(); plan = {}; native = {}; compiled = {}; graph.graph = {};
  std::array<f::g::GraphOutput, 2> results;
  for (std::size_t lane = 0; lane < slots.size(); ++lane) {
    auto activation = h.upload(f::floating(graph.profile.operand, data.activation));
    f::check(slots[lane].bind(graph.ports[0], graph.descriptions[0], activation), "constant-bank A bind");
    activation.reset(); h.submit(slots[lane]);
    results[lane] = f::take(slots[lane].result(graph.output), "retained constant-bank result");
    expect_dot(graph, data, h.read(f::take(results[lane].tensor(), "bank result tensor")));
  }
  slots = {};
  for (auto &result : results) expect_dot(graph, data, h.read(f::take(result.tensor(), "escaped bank output")));
  ::testing::Test::RecordProperty("native_payload_storage_bytes", facts.invocation_storage_block_bytes[0]);
  ::testing::Test::RecordProperty("raw_constant_capacity_bytes", facts.retained_constant_capacity_bytes);
}
} // namespace
