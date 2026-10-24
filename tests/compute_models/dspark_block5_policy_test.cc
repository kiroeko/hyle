#include <gtest/gtest.h>

#include "dspark_block5_policy.h"
#include "hyle/compute/state_versions.h"
#include "hyle/mem/buffer.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
namespace p = dspark_original_state_consumer;
namespace c = hyle::compute;
namespace m = hyle::mem;
constexpr std::int64_t vocabulary = 129280;
constexpr std::array<std::int32_t, 6> proposals{3, 4, 5, 6, 7, 8};
constexpr std::array<std::int32_t, 6> matching{4, 5, 6, 7, 8, 9};

template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
p::Block5Limits limits(std::size_t outputs = 20, std::size_t context = 20,
    std::span<const std::int32_t> eos = {}) {
  return {.output_quota = outputs, .context_rows = context, .vocabulary = vocabulary, .eos_tokens = eos};
}

TEST(DSparkBlock5Policy, PlansOnlyActualOneThroughSixTargetRows) {
  for (std::size_t rows = 1; rows <= 6; ++rows) {
    EXPECT_EQ(take(p::block5_verification_rows(limits(rows, 20))), rows);
    EXPECT_EQ(take(p::block5_verification_rows(limits(20, rows))), rows);
  }
  EXPECT_EQ(take(p::block5_verification_rows(limits(0, 20))), 0u);
  EXPECT_EQ(take(p::block5_verification_rows(limits(20, 0))), 0u);
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  EXPECT_EQ(take(p::block5_verification_rows(limits(maximum, maximum))), 6u);
}

TEST(DSparkBlock5Policy, EveryGreedyPrefixKeepsItsActualCorrectionOrBonusRow) {
  for (std::size_t accepted = 0; accepted <= 5; ++accepted) {
    auto target = matching;
    if (accepted != 5) target[accepted] = 20 + static_cast<std::int32_t>(accepted);
    const auto result = take(p::decide_block5(proposals, target, limits()));
    EXPECT_EQ(result.accepted_drafts, accepted);
    EXPECT_EQ(result.consumed, accepted + 1);
    EXPECT_EQ(result.pending_row, accepted);
    EXPECT_EQ(result.emitted_count, accepted + 1);
    EXPECT_TRUE(result.can_continue());
  }
}

TEST(DSparkBlock5Policy, SharedGreedySpanWritesOnlySelectedPrefixAndRejectsShortOutput) {
  namespace mtp = hyle::inference::mtp;
  auto target = matching; target[2] = 99;
  std::array<std::int32_t, 8> output; output.fill(-17);
  const auto selected = mtp::select_greedy_prefix(proposals, target, 20, {},
      static_cast<std::int32_t>(vocabulary), std::span(output).first(6));
  EXPECT_EQ(selected.consumed, 3u); EXPECT_EQ(selected.consumed_drafts, 2u);
  EXPECT_EQ(selected.next_id, 99);
  EXPECT_EQ(output, (std::array<std::int32_t, 8>{4, 5, 99, -17, -17, -17, -17, -17}));

  output.fill(-17);
  // Full bounded capacity is required even when a first-row EOS could make
  // this particular execution fit. Rejection cannot partially write output.
  const std::array eos{matching[0]};
  EXPECT_THROW((void)mtp::select_greedy_prefix(proposals, matching, 20, eos,
      static_cast<std::int32_t>(vocabulary), std::span(output).first(5)), std::runtime_error);
  EXPECT_EQ(output, (std::array<std::int32_t, 8>{-17, -17, -17, -17, -17, -17, -17, -17}));
}

TEST(DSparkBlock5Policy, MatchedEosAndBonusEosAreEmittedButRemainUnconsumed) {
  for (std::size_t row = 0; row < matching.size(); ++row) {
    const std::array eos{matching[row]};
    const auto result = take(p::decide_block5(proposals, matching, limits(20, 20, eos)));
    EXPECT_EQ(result.accepted_drafts, row);
    EXPECT_EQ(result.consumed, row + 1);
    EXPECT_EQ(result.pending_row, row);
    EXPECT_EQ(result.emitted_count, row + 1);
    EXPECT_TRUE(result.stopped_on_eos);
    EXPECT_FALSE(result.can_continue());
    // Even a matching proposal EOS is not an extra consumed input. At row5
    // this is the actual bonus EOS, after all five non-EOS drafts were consumed.
    EXPECT_NE(proposals[result.consumed - 1], matching[row]);
  }
  const std::array eos{5, 9};
  EXPECT_EQ(take(p::decide_block5(proposals, matching, limits(20, 20, eos))).pending_row, 1u);
}

TEST(DSparkBlock5Policy, QuotaAndContextTailsNeverConsumeTheirFinalOutput) {
  for (std::size_t rows = 1; rows <= 6; ++rows) {
    const auto target = std::span(matching).first(rows);
    const auto quota = take(p::decide_block5(proposals, target, limits(rows, 20)));
    const auto context = take(p::decide_block5(proposals, target, limits(20, rows)));
    EXPECT_EQ(quota.consumed, rows);
    EXPECT_EQ(quota.accepted_drafts, rows - 1);
    EXPECT_EQ(quota.pending_row, rows - 1);
    EXPECT_TRUE(quota.output_quota_exhausted);
    EXPECT_FALSE(quota.context_exhausted);
    EXPECT_FALSE(quota.can_continue());
    EXPECT_EQ(context.consumed, rows);
    EXPECT_EQ(context.accepted_drafts, rows - 1);
    EXPECT_EQ(context.pending_row, rows - 1);
    EXPECT_TRUE(context.context_exhausted);
    EXPECT_FALSE(context.output_quota_exhausted);
    EXPECT_FALSE(context.can_continue());
  }
}

TEST(DSparkBlock5Policy, MaximumDraftsKeepsFiveRowTrainingAndAllowsTargetOnlyWithoutConfidence) {
  for (std::uint32_t maximum = 0; maximum <= 5; ++maximum) {
    auto bounds = limits(); bounds.draft.max_drafts = maximum;
    const auto count = take(p::block5_verification_rows(bounds));
    EXPECT_EQ(count, maximum + 1);
    const auto ids = maximum ? std::span(proposals) : std::span(proposals).first(1);
    const auto result = take(p::decide_block5(ids, std::span(matching).first(count), bounds));
    EXPECT_EQ(result.accepted_drafts, maximum);
    EXPECT_EQ(result.consumed, maximum + 1);
    EXPECT_TRUE(result.can_continue());
  }
  auto bounds = limits(); bounds.draft = {.max_drafts = 0, .minimum_confidence = 0.9F};
  EXPECT_EQ(take(p::block5_verification_rows(bounds)), 1u);
  EXPECT_TRUE(p::decide_block5(std::span(proposals).first(1), std::span(matching).first(1), bounds));
  bounds.draft.max_drafts = 5; bounds.output_quota = 1;
  EXPECT_EQ(take(p::block5_verification_rows(bounds)), 1u);
  bounds.output_quota = 0;
  EXPECT_EQ(take(p::block5_verification_rows(bounds)), 0u);
  bounds = limits(); bounds.draft.max_drafts = 6;
  EXPECT_FALSE(p::block5_verification_rows(bounds));
}

TEST(DSparkBlock5Policy, ConfidenceStopsAtTheFirstLowRowInItsExplicitDomain) {
  auto bounds = limits(); bounds.draft.minimum_confidence = 0.5F;
  for (std::size_t accepted = 0; accepted <= 5; ++accepted) {
    std::array<float, 5> raw; raw.fill(2.F);
    if (accepted < raw.size()) raw[accepted] = -0.1F;
    // Later high-confidence rows cannot bridge an earlier low-confidence row.
    const auto count = take(p::block5_verification_rows(bounds, raw));
    EXPECT_EQ(count, accepted + 1);
    const auto result = take(p::decide_block5(proposals, std::span(matching).first(count), bounds, raw));
    EXPECT_EQ(result.accepted_drafts, accepted);
    EXPECT_TRUE(result.can_continue());
  }
  const std::array<float, 5> raw{0.F, 2.F, -2.F, 4.F, 4.F};
  EXPECT_EQ(take(p::block5_verification_rows(bounds, raw)), 3u); // sigmoid(0) == .5 passes
  bounds.confidence_domain = p::ConfidenceDomain::kProbability;
  const std::array<float, 5> probability{0.8F, 0.6F, 0.4F, 0.9F, 0.9F};
  EXPECT_EQ(take(p::block5_verification_rows(bounds, probability)), 3u); // no second sigmoid
  bounds.draft.max_drafts = 1;
  EXPECT_EQ(take(p::block5_verification_rows(bounds, probability)), 2u);
  bounds.draft.max_drafts = 5; bounds.context_rows = 2;
  EXPECT_EQ(take(p::block5_verification_rows(bounds, probability)), 2u);
  bounds.context_rows = 20; bounds.draft.minimum_confidence = 0.F;
  EXPECT_EQ(take(p::block5_verification_rows(bounds)), 6u); // zero disables confidence work
  EXPECT_EQ(take(p::block5_verification_rows(bounds, std::array<float,5>{0,0,0,0,0})), 6u);
  bounds.draft.minimum_confidence = 1.F;
  EXPECT_EQ(take(p::block5_verification_rows(bounds, std::array<float,5>{1,1,1,1,1})), 6u);
  bounds.confidence_domain = p::ConfidenceDomain::kRawLogit;
  const auto inf = (std::numeric_limits<float>::infinity)();
  EXPECT_EQ(take(p::block5_verification_rows(bounds, std::array<float,5>{inf,0,inf,inf,inf})), 2u);
  bounds.draft.minimum_confidence = 0.F;
  EXPECT_EQ(take(p::block5_verification_rows(bounds, std::array<float,5>{-inf,0,0,0,0})), 6u);
}

TEST(DSparkBlock5Policy, InvalidConfidenceFailsBeforeSubmittingTargetRows) {
  auto bounds = limits(); bounds.draft.minimum_confidence = 0.5F;
  std::array<float,5> raw{};
  EXPECT_FALSE(p::block5_verification_rows(bounds));
  EXPECT_FALSE(p::block5_verification_rows(bounds, std::span(raw).first(4)));
  raw[4] = (std::numeric_limits<float>::quiet_NaN)();
  EXPECT_FALSE(p::block5_verification_rows(bounds, raw));
  raw.fill(0.F);
  for (const auto threshold : {-1.F, 2.F, (std::numeric_limits<float>::quiet_NaN)()}) {
    bounds.draft.minimum_confidence = threshold;
    EXPECT_FALSE(p::block5_verification_rows(bounds, raw));
  }
  bounds.draft.minimum_confidence = .5F;
  bounds.confidence_domain = p::ConfidenceDomain::kProbability;
  for (const auto value : {-0.01F, 1.01F, (std::numeric_limits<float>::infinity)()}) {
    raw.fill(.7F); raw[4] = value;
    EXPECT_FALSE(p::block5_verification_rows(bounds, raw));
  }
  bounds.confidence_domain = static_cast<p::ConfidenceDomain>(200);
  EXPECT_FALSE(p::block5_verification_rows(bounds, raw));
}

TEST(DSparkBlock5Policy, SampledConfidenceUsesItsActualBoundedConditionalRows) {
  auto bounds=limits();bounds.draft={.max_drafts=2,.minimum_confidence=.5F};
  EXPECT_EQ(take(p::block5_sampled_verification_rows(bounds,std::array{1.F,-1.F})),2u);
  EXPECT_EQ(take(p::block5_sampled_verification_rows(bounds,std::array{-1.F,1.F})),1u);
  EXPECT_EQ(take(p::block5_sampled_verification_rows(bounds,std::array{1.F,1.F})),3u);
  EXPECT_FALSE(p::block5_sampled_verification_rows(bounds,std::array{1.F}));
  EXPECT_FALSE(p::block5_sampled_verification_rows(bounds,std::array{1.F,1.F,1.F}));
  EXPECT_FALSE(p::block5_sampled_verification_rows(bounds,
      std::array{1.F,(std::numeric_limits<float>::quiet_NaN)()}));
  bounds.confidence_domain=p::ConfidenceDomain::kProbability;
  EXPECT_EQ(take(p::block5_sampled_verification_rows(bounds,std::array{.75F,.25F})),2u);
  EXPECT_FALSE(p::block5_sampled_verification_rows(bounds,std::array{.75F,1.25F}));
  bounds.draft.max_drafts=0;EXPECT_EQ(take(p::block5_sampled_verification_rows(bounds)),1u);
  bounds.draft.max_drafts=5;bounds.context_rows=1;
  EXPECT_EQ(take(p::block5_sampled_verification_rows(bounds)),1u);
  bounds.context_rows=0;EXPECT_EQ(take(p::block5_sampled_verification_rows(bounds)),0u);
}

TEST(DSparkBlock5Policy, CapacityValidatesPolicyBeforeRequestingTheTrainedDraft) {
  auto bounds = limits(); bounds.draft = {.max_drafts = 2, .minimum_confidence = .5F};
  EXPECT_EQ(take(p::block5_verification_capacity(bounds)), 3u); // no fabricated confidence inputs
  EXPECT_FALSE(p::block5_verification_rows(bounds)); // actual clipping still requires the producer
  for (const auto threshold : {-1.F, 1.1F, (std::numeric_limits<float>::infinity)(),
      (std::numeric_limits<float>::quiet_NaN)()}) {
    bounds.draft.minimum_confidence = threshold;
    EXPECT_FALSE(p::block5_verification_capacity(bounds));
  }
  bounds.draft.minimum_confidence = .5F;
  bounds.draft.max_drafts = 6;
  EXPECT_FALSE(p::block5_verification_capacity(bounds));
  bounds.draft.max_drafts = 0;
  EXPECT_EQ(take(p::block5_verification_capacity(bounds)), 1u);
  bounds.confidence_domain = static_cast<p::ConfidenceDomain>(200);
  EXPECT_FALSE(p::block5_verification_capacity(bounds));
}

TEST(DSparkBlock5Policy, InvalidInputsAndAlreadyFinishedRoundsAreRejected) {
  EXPECT_FALSE(p::decide_block5(proposals, matching, limits(0, 20)));
  EXPECT_FALSE(p::decide_block5(proposals, matching, limits(20, 0)));
  EXPECT_FALSE(p::decide_block5(std::span(proposals).first(5), matching, limits()));
  EXPECT_FALSE(p::decide_block5(proposals, std::span(matching).first(5), limits()));
  EXPECT_FALSE(p::decide_block5(proposals, matching, limits(2, 20)));
  auto invalid = proposals; invalid[5] = -1;
  EXPECT_FALSE(p::decide_block5(invalid, matching, limits()));
  const auto invalid_unverified_tail = p::decide_block5(invalid, std::span(matching).first(2), limits(2, 20));
  ASSERT_FALSE(invalid_unverified_tail);
  EXPECT_TRUE(invalid_unverified_tail.error().IsInvalidArgument());
  invalid = matching; invalid[0] = static_cast<std::int32_t>(vocabulary);
  EXPECT_FALSE(p::decide_block5(proposals, invalid, limits()));
  const std::array bad_eos{-1}, ended{proposals[0]};
  EXPECT_FALSE(p::block5_verification_rows(limits(20, 20, bad_eos)));
  EXPECT_FALSE(p::decide_block5(proposals, matching, limits(20, 20, ended)));
  const auto terminal_pending = p::decide_block5(proposals, matching, limits(20, 20, ended));
  ASSERT_FALSE(terminal_pending);
  EXPECT_TRUE(terminal_pending.error().IsFailedPrecondition());
  auto bad_vocabulary = limits(); bad_vocabulary.vocabulary = 0;
  EXPECT_FALSE(p::block5_verification_rows(bad_vocabulary));
}

c::ReadLease upload(std::span<const std::int32_t> values) {
  auto buffer = take(m::Buffer::create(m::system_resource(), values.size_bytes(), 64));
  check(buffer.view().write(std::as_bytes(values)));
  return take(c::ReadLease::adopt(std::move(buffer)));
}
std::vector<std::int32_t> read(const c::ReadLease &lease) {
  std::vector<std::int32_t> result(lease.size() / sizeof(std::int32_t));
  check(take(lease.view()).read(std::as_writable_bytes(std::span(result))));
  return result;
}

// Small host fixtures represent distinct completed target checkpoints and ID
// outputs. They test real StateStore/Head publication and lease selection;
// they do not execute a trained model, target projection, or large vocabulary.
struct State final {
  c::StateSchema schema;
  c::StateStore store;
  std::array<c::StateEntryId, 3> entry;
  c::StateVersion base;
  struct Rows final { std::array<c::ReadLease, 6> cursor, cache, pending; };
  State() {
    const auto scalar = take(c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{1}));
    const auto cache = take(c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{16}));
    schema = take(c::StateSchema::create(std::array{c::StateEntrySpec{.description = scalar},
        c::StateEntrySpec{.description = cache}, c::StateEntrySpec{.description = scalar}}));
    store = take(c::StateStore::create(schema, {.max_versions = 16, .max_drafts = 2,
        .max_retained_blocks = 64, .max_backing_bytes = 64u << 10}));
    for (std::size_t i = 0; i < entry.size(); ++i) entry[i] = take(schema.entry(i));
    std::array<std::int32_t, 16> empty; empty.fill(-1);
    base = take(store.import(std::array{c::StateMember{entry[0], upload(std::array<std::int32_t, 1>{0})},
        c::StateMember{entry[1], upload(empty)}, c::StateMember{entry[2], upload(std::array{proposals[0]})}}));
  }
  c::ReadLease member(const c::StateVersion &version, std::size_t index) const { return take(version.member(entry[index])); }
  Rows checkpoints(const c::StateVersion &from, std::span<const std::int32_t> proposed,
      std::span<const std::int32_t> target) const {
    Rows result;
    const auto first = read(member(from, 0))[0];
    auto cache = read(member(from, 1));
    for (std::size_t row = 0; row < target.size(); ++row) {
      cache.at(static_cast<std::size_t>(first) + row) = proposed[row];
      result.cursor[row] = upload(std::array{first + static_cast<std::int32_t>(row) + 1});
      result.cache[row] = upload(cache);
      result.pending[row] = upload(target.subspan(row, 1));
    }
    return result;
  }
  c::StateVersion select(const c::StateVersion &from, const Rows &rows, const p::VerifiedPrefix &decision) {
    auto draft = take(store.begin(from));
    check(draft.replace(entry[0], rows.cursor.at(decision.pending_row)));
    check(draft.replace(entry[1], rows.cache.at(decision.pending_row)));
    check(draft.replace(entry[2], rows.pending.at(decision.pending_row)));
    return take(draft.finalize());
  }
};

TEST(DSparkBlock5State, EosQuotaAndContextPublishOnlyTheirCompleteSelectedPrefix) {
  for (std::size_t scenario = 0; scenario < 4; ++scenario) {
    State state;
    const std::array eos{scenario == 0 ? matching[2] : scenario == 1 ? matching[5] : -1};
    const auto bounds = scenario < 2 ? limits(20, 20, eos) :
        scenario == 2 ? limits(2, 20) : limits(20, 1);
    const auto count = take(p::block5_verification_rows(bounds));
    const auto target = std::span(matching).first(count);
    const auto rows = state.checkpoints(state.base, proposals, target);
    const auto decision = take(p::decide_block5(proposals, target, bounds));
    auto head = take(c::StateHead::create(state.store, state.base));
    const auto selected = state.select(state.base, rows, decision);
    check(head.publish(state.base.identity(), selected));
    const auto current = take(head.current());
    EXPECT_EQ(read(state.member(current, 0))[0], decision.consumed);
    const auto cache = read(state.member(current, 1));
    for (std::size_t i = 0; i < cache.size(); ++i)
      EXPECT_EQ(cache[i], i < decision.consumed ? proposals[i] : -1);
    auto pending = state.member(current, 2);
    EXPECT_EQ(read(pending)[0], target[decision.pending_row]);
    EXPECT_EQ(pending.generation(), rows.pending[decision.pending_row].generation());
    EXPECT_EQ(take(pending.view()).native_handle(), take(rows.pending[decision.pending_row].view()).native_handle());
    EXPECT_FALSE(pending.try_write());
    EXPECT_EQ(read(state.member(state.base, 0))[0], 0);
    EXPECT_EQ(read(state.member(state.base, 2))[0], proposals[0]);
    EXPECT_TRUE(head.publish(state.base.identity(), state.base).IsUnavailable());
    if (decision.stopped_on_eos) {
      auto ended_proposals = proposals; ended_proposals[0] = read(pending)[0];
      EXPECT_FALSE(p::decide_block5(ended_proposals, matching, bounds));
    }
    const auto stats = state.store.stats();
    EXPECT_LE(stats.backing_bytes, 64u << 10);
  }
}

TEST(DSparkBlock5State, RejectionContinuesWithTheOriginalPendingLeaseAndProtectsOldState) {
  State state;
  auto target = matching; target[0] = 20;
  const auto first_rows = state.checkpoints(state.base, proposals, target);
  const auto rejected = take(p::decide_block5(proposals, target, limits()));
  ASSERT_EQ(rejected.accepted_drafts, 0u);
  ASSERT_TRUE(rejected.can_continue());
  auto head = take(c::StateHead::create(state.store, state.base));
  const auto first = state.select(state.base, first_rows, rejected);
  check(head.publish(state.base.identity(), first));
  const auto pending = state.member(first, 2);
  const std::array<std::int32_t, 6> next_proposed{read(pending)[0], 21, 22, 23, 24, 25};
  const std::array<std::int32_t, 3> next_target{21, 22, 23};
  const auto next_rows = state.checkpoints(first, next_proposed, next_target);
  const auto accepted = take(p::decide_block5(next_proposed, next_target, limits(3, 3)));
  ASSERT_EQ(accepted.consumed, 3u);
  const auto continued = state.select(first, next_rows, accepted);
  check(head.publish(first.identity(), continued));
  EXPECT_EQ(read(state.member(continued, 0))[0], 4);
  const auto cache = read(state.member(continued, 1));
  EXPECT_EQ((std::vector<std::int32_t>(cache.begin(), cache.begin() + 4)),
      (std::vector<std::int32_t>{3, 20, 21, 22}));
  EXPECT_EQ(read(state.member(continued, 2))[0], 23);
  EXPECT_EQ(read(state.member(first, 0))[0], 1);
  EXPECT_EQ(read(state.member(first, 2))[0], 20);
  EXPECT_EQ(read(state.member(state.base, 0))[0], 0);
  auto cancelled = take(state.store.begin(continued));
  check(cancelled.replace(state.entry[1], first_rows.cache[5]));
  check(cancelled.cancel());
  EXPECT_FALSE(cancelled.finalize());
  EXPECT_EQ(take(head.current()).identity(), continued.identity());
}

TEST(DSparkBlock5State, DisableThenReenablePublishesTheTargetOnlyHistoryAndActualPending) {
  State state;
  auto head = take(c::StateHead::create(state.store, state.base));
  auto bounds = limits(); bounds.draft = {.max_drafts = 0, .minimum_confidence = .9F};
  const std::array<std::int32_t, 1> current{read(state.member(state.base, 2))[0]}, next{20};
  const auto plain_rows = state.checkpoints(state.base, current, next);
  const auto plain = take(p::decide_block5(current, next, bounds)); // no trained draft/confidence submitted
  const auto after_plain = state.select(state.base, plain_rows, plain);
  check(head.publish(state.base.identity(), after_plain));
  EXPECT_EQ(read(state.member(after_plain, 0))[0], 1);
  EXPECT_EQ(read(state.member(after_plain, 1))[0], current[0]);
  const auto actual_pending = state.member(after_plain, 2);
  EXPECT_EQ(take(actual_pending.view()).native_handle(), take(plain_rows.pending[0].view()).native_handle());
  bounds.draft = {.max_drafts = 3, .minimum_confidence = .5F};
  const std::array<float,5> confidence{1.F, -1.F, 2.F, 2.F, 2.F};
  const auto rows = take(p::block5_verification_rows(bounds, confidence));
  ASSERT_EQ(rows, 2u);
  const std::array<std::int32_t,6> resumed{read(actual_pending)[0], 21, 22, 23, 24, 25};
  const std::array<std::int32_t,2> verified{21,22};
  const auto checkpoints = state.checkpoints(after_plain, resumed, verified);
  const auto decision = take(p::decide_block5(resumed, verified, bounds, confidence));
  const auto after_mtp = state.select(after_plain, checkpoints, decision);
  check(head.publish(after_plain.identity(), after_mtp));
  EXPECT_EQ(read(state.member(after_mtp, 0))[0], 3);
  const auto cache = read(state.member(after_mtp, 1));
  EXPECT_EQ((std::vector<std::int32_t>(cache.begin(), cache.begin()+3)), (std::vector<std::int32_t>{3,20,21}));
  EXPECT_EQ(read(state.member(after_mtp, 2))[0], 22);
  EXPECT_EQ(read(state.member(after_plain, 0))[0], 1);
  EXPECT_EQ(read(state.member(state.base, 0))[0], 0);
  EXPECT_TRUE(head.publish(state.base.identity(), state.base).IsUnavailable());
}

TEST(DSparkBlock5State, InvalidConfidenceOrDecisionCannotPublishAnyPartialRound) {
  State state;
  auto head = take(c::StateHead::create(state.store, state.base));
  auto bounds = limits(); bounds.draft.minimum_confidence = .5F;
  std::array<float,5> confidence{};
  confidence[0] = (std::numeric_limits<float>::quiet_NaN)();
  const auto before = state.store.stats();
  EXPECT_FALSE(p::block5_verification_rows(bounds, confidence));
  EXPECT_EQ(take(head.current()).identity(), state.base.identity());
  EXPECT_EQ(state.store.stats().drafts, before.drafts);
  EXPECT_EQ(state.store.stats().versions, before.versions);
  EXPECT_EQ(read(state.member(state.base, 0))[0], 0);
  confidence.fill(1.F);
  auto wrong_ids = matching; wrong_ids[0] = -1;
  EXPECT_FALSE(p::decide_block5(proposals, wrong_ids, bounds, confidence));
  EXPECT_EQ(take(head.current()).identity(), state.base.identity());
  EXPECT_EQ(read(state.member(state.base, 2))[0], proposals[0]);
  // The same complete head remains usable after either rejected round.
  const auto checkpoints = state.checkpoints(state.base, proposals, matching);
  const auto decision = take(p::decide_block5(proposals, matching, bounds, confidence));
  const auto accepted = state.select(state.base, checkpoints, decision);
  check(head.publish(state.base.identity(), accepted));
  EXPECT_EQ(read(state.member(accepted, 0))[0], 6);
  EXPECT_EQ(read(state.member(accepted, 2))[0], matching[5]);
}
} // namespace
