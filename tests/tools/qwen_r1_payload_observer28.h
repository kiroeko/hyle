#pragma once

// EXPERIMENT ONLY. Included solely by the named, unbuilt Model overlay.
// Reads add synchronous GPU transfers and may hide the original race.
// No graph outputs, math, permissions, published records or assertions change.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"

namespace hyle::experimental::r1_payload28 {
namespace c = hyle::compute;
namespace vn = hyle::compute::vulkan::detail;
using hyle::inference::SamplingCursor;
using qwen35::check;
using qwen35::take;

inline constexpr std::size_t maximum_vocabulary = 248320;
inline constexpr std::size_t maximum_float_words = 20 * maximum_vocabulary;
struct Floats final { std::vector<float> value; };
struct Sample final {
  bool armed = false, before = false, after = false;
  unsigned request = 0;
  const void *session = nullptr; // Identity only; never dereferenced or retained.
  SamplingCursor cursor{};
  std::size_t vocabulary = 0, quota = 0, planned = 0, proposals = 0, fresh_count = 0;
  std::array<Floats, 3> fresh_q, prefix_q;
  Floats target_raw_logits;
  std::array<std::uint32_t, 12> draft_keys{};
  std::array<std::int32_t, 3> draft_ids{};
  std::array<std::uint32_t, 16> prefix_keys{};
  std::array<std::int32_t, 3> prefix_ids{};
  std::array<float, 3> continuation{};
};

struct State final {
  std::array<Sample, 2> samples;
  std::array<char, 128 * 1024> text{};
  std::size_t used = 0, float_words = 0, prepared_calls = 0;
  bool incomplete = false, overflow = false, compared = false;
  void note(const char *format, ...) noexcept {
    char line[2048]; va_list args; va_start(args, format);
    const auto count = std::vsnprintf(line, sizeof(line), format, args); va_end(args);
    if (count < 0 || std::size_t(count) >= sizeof(line) ||
        std::size_t(count) > text.size() - used) { incomplete = overflow = true; return; }
    std::memcpy(text.data() + used, line, std::size_t(count)); used += std::size_t(count);
  }
  ~State() noexcept {
    note("R1_PAYLOAD28_STATUS request12=%d/%d request14=%d/%d compared=%d incomplete=%d overflow=%d float_words=%zu max_float_words=%zu timing_perturbed=1 internal_P_observed=0\n",
        int(samples[0].before), int(samples[0].after), int(samples[1].before), int(samples[1].after),
        int(compared), int(incomplete || !samples[0].after || !samples[1].after), int(overflow),
        float_words, maximum_float_words);
    if (used) (void)std::fwrite(text.data(), 1, used, stderr);
  }
};
inline State state;
inline thread_local Sample *draft_sample = nullptr;

inline void failure(std::string_view where, std::string_view message) noexcept {
  state.incomplete = true;
  state.note("R1_PAYLOAD28_UNAVAILABLE stage=%.*s message=%.*s\n", int(where.size()), where.data(),
      int((std::min)(message.size(), std::size_t{512})), message.data());
}
template<class Action> inline void observe(std::string_view where, Action &&action) noexcept {
  try { action(); }
  catch (const hyle::inference::EvaluationFailure &error) { failure(where, error.status().message()); }
  catch (const std::exception &error) { failure(where, error.what()); }
  catch (...) { failure(where, "unknown observer failure"); }
}

// A view is admitted only from the original ReadLease/WriteLease below.
// Native span lookup is backend-private observation, never Model policy.
inline void identity(const char *label, unsigned request, std::size_t row,
    mem::ConstBufferView view, const c::TensorDesc &description, std::uint64_t generation,
    hyle::inference::Backend &backend) {
  auto *context = take(backend.vulkan_context());
  const auto &runtime = vn::Access::runtime(*context);
  const auto physical = vn::MemoryAccess::copy_span(runtime, view);
  if (!physical) {
    state.incomplete = true;
    state.note("R1_STORAGE_UNAVAILABLE label=%s request=%u row=%zu bytes=%zu generation=%llu message=%.*s payload_read_still_uses_original_lease=1\n",
        label,request,row,view.size(),static_cast<unsigned long long>(generation),
        int((std::min)(physical.error().message().size(),std::size_t{256})),physical.error().message().data());
    return;
  }
  const auto dimensions=description.dimensions();
  state.note("R1_STORAGE label=%s request=%u row=%zu dtype=%u rank=%zu dim0=%lld dim1=%lld bytes=%zu view_offset=%zu generation=%llu backing=%llu physical_offset=%zu physical_bytes=%zu source=original_lease\n",
      label, request, row, unsigned(description.data_type()), dimensions.size(),
      static_cast<long long>(dimensions.empty()?-1:dimensions[0]),static_cast<long long>(dimensions.size()>1?dimensions[1]:-1),view.size(), view.offset(),
      static_cast<unsigned long long>(generation), static_cast<unsigned long long>(physical->physical.backing),
      physical->physical.origin, physical->physical.capacity);
}
inline void read_float(const char *label, Sample &sample, std::size_t row,
    const c::ReadLease &lease, const c::TensorDesc &description, Floats &destination,
    hyle::inference::Backend &backend) {
  if (description.data_type() != c::DataType::kFloat32) throw std::runtime_error("expected original F32 input");
  const auto view = take(lease.view());
  const auto bytes = take(description.required_bytes());
  if (view.size() != bytes || bytes % sizeof(float)) throw std::runtime_error("lease exact F32 extent differs");
  const auto words = bytes / sizeof(float);
  if (words > maximum_float_words - state.float_words) throw std::runtime_error("fixed host payload budget exceeded");
  destination.value.resize(words); state.float_words += words;
  identity(label, sample.request, row, view, description, lease.generation(), backend);
  check(view.read(std::as_writable_bytes(std::span(destination.value))));
}

inline void difference(const char *label, std::span<const float> before, std::span<const float> after,
    unsigned request, std::size_t row) {
  if (before.size() != after.size()) throw std::runtime_error("compared F32 shapes differ");
  std::size_t bits = 0, nonfinite = 0, shown = 0; double maximum = 0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    if (std::bit_cast<std::uint32_t>(before[i]) == std::bit_cast<std::uint32_t>(after[i])) continue;
    ++bits;
    if (std::isfinite(before[i]) && std::isfinite(after[i])) maximum = (std::max)(maximum, std::abs(double(before[i]) - double(after[i])));
    else ++nonfinite;
    if (shown++ < 8) state.note("R1_DIFF label=%s request=%u row=%zu word=%zu before_bits=%08x after_bits=%08x before=%a after=%a\n",
        label, request, row, i, std::bit_cast<std::uint32_t>(before[i]), std::bit_cast<std::uint32_t>(after[i]), double(before[i]), double(after[i]));
  }
  state.note("R1_DIFF_SUMMARY label=%s request=%u row=%zu words=%zu bit_diff=%zu max_finite_abs=%.17g nonfinite_diff=%zu\n",
      label, request, row, before.size(), bits, maximum, nonfinite);
}
// Compare a live original lease without retaining it or allocating another tensor-sized buffer.
inline void compare_live_q(const char *label, Sample &sample, std::size_t row,
    const c::ReadLease &lease, const Floats &original) {
  const auto view = take(lease.view());
  if (view.size() != original.value.size() * sizeof(float)) throw std::runtime_error("q reuse extent differs");
  std::array<float, 16384> chunk{};
  std::size_t different = 0, nonfinite = 0, shown = 0; double maximum = 0;
  for (std::size_t begin = 0; begin < original.value.size();) {
    const auto count = (std::min)(chunk.size(), original.value.size() - begin);
    check(take(view.subview(begin * sizeof(float), count * sizeof(float))).read(std::as_writable_bytes(std::span(chunk).first(count))));
    for (std::size_t i = 0; i < count; ++i) {
      const auto a = original.value[begin + i], b = chunk[i];
      if (std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b)) continue;
      ++different;
      if (std::isfinite(a) && std::isfinite(b)) maximum = (std::max)(maximum, std::abs(double(a) - double(b)));
      else ++nonfinite;
      if (shown++ < 8) state.note("R1_DIFF label=%s request=%u row=%zu word=%zu before_bits=%08x after_bits=%08x\n",
          label, sample.request, row, begin + i, std::bit_cast<std::uint32_t>(a), std::bit_cast<std::uint32_t>(b));
    }
    begin += count;
  }
  state.note("R1_DIFF_SUMMARY label=%s request=%u row=%zu words=%zu bit_diff=%zu max_finite_abs=%.17g nonfinite_diff=%zu\n",
      label, sample.request, row, original.value.size(), different, maximum, nonfinite);
}

// Read the existing exclusive input writer only after its actual invocation
// has completed; never publish/downgrade it or invent an adopted read lease.
template<class Word> inline void read_fixed(const char *label, Sample &sample,
    qwen35::Tensor &tensor, std::span<Word> output, hyle::inference::Backend &backend) {
  mem::ConstBufferView view;
  std::uint64_t generation = 0;
  if (tensor.ready.valid()) { view = take(tensor.ready.view()); generation = tensor.ready.generation(); }
  else if (tensor.writer.valid()) { view = take(tensor.writer.view()); generation = tensor.writer.generation(); }
  else throw std::runtime_error("original fixed input has no independent ready/exclusive lease; no permission workaround");
  if (view.size() != output.size_bytes()) throw std::runtime_error("fixed input exact extent differs");
  identity(label, sample.request, 0, view, tensor.description, generation, backend);
  check(view.read(std::as_writable_bytes(output)));
}

template<class Model, class Call> inline void prepared(Model &model, const Call &call, std::size_t credit) noexcept {
  if ((call.request->id != "12" && call.request->id != "14") || !call.request->session || !call.request->session->head) return;
  observe("prepared_call", [&] {
    if (++state.prepared_calls > 32) throw std::runtime_error("prepared call text budget exceeded");
    const auto cursor = call.request->session->head->sampling;
    state.note("R1_CALL request=%s episode=%zu generated=%zu remaining=%zu output_credit=%zu quota=%zu planned_K=%zu role=%u before_length=%zu pending=%d seed=%llu stream=%llu draw=%llu active_sequences=%zu\n",
        call.request->id.c_str(), call.request->rounds.size() + 1, call.request->progress.generated.size(),
        call.request->progress.remaining_outputs(), credit, call.quota, call.maximum, unsigned(call.role),
        call.request->session->head->length, call.pending, static_cast<unsigned long long>(cursor.seed),
        static_cast<unsigned long long>(cursor.stream), static_cast<unsigned long long>(cursor.draw), model.cache_plan.active_sequences);
  });
}

struct ProposalScope final {
  Sample *previous = draft_sample;
  template<class Model, class Request> ProposalScope(Model &model, const Request &request, std::size_t quota, std::size_t planned) noexcept {
    draft_sample = nullptr;
    if ((request.id != "12" && request.id != "14") || request.rounds.size() != 2) return;
    observe("arm", [&] {
      const auto index = request.id == "12" ? 0u : 1u;
      auto &sample = state.samples[index];
      if (sample.armed) throw std::runtime_error("duplicate third-round capture");
      sample.armed = true; sample.request = index ? 14 : 12; sample.session = request.session.get();
      if (!request.session || !request.session->head || !request.ledger || model.cache_plan.active_sequences != 2 || planned > 3)
        throw std::runtime_error("actual AS2 third-round request context differs");
      sample.cursor = request.session->head->sampling; sample.quota = quota; sample.planned = planned;
      sample.vocabulary = static_cast<std::size_t>(model.vocabulary);
      state.note("R1_SOURCE request=%u source_identity=%s sampler=%zu profile=%zu capacity=%zu\n",
          sample.request,model.source_identity.c_str(),request.sampling_index,request.execution_profile,model.capacity);
      const auto &sampler = request.session->sampling_configuration().configuration;
      if (!sample.vocabulary || sample.vocabulary > maximum_vocabulary || !sampler.enabled || !sampler.modifiers.empty() ||
          sampler.policy.temperature != .8F || sampler.policy.top_k != 40 || sampler.policy.top_p != .95F ||
          sample.cursor.seed != 42 || sample.cursor.stream != 0)
        throw std::runtime_error("actual sampler/seed/stream outside bounded experiment");
      const std::array<std::int32_t, 5> expected{1,2,3,2,471}; std::size_t at = 0;
      for (std::size_t i = 0; i < request.ledger->committed_count; ++i) {
        const auto &record = request.ledger->records[i];
        if (!record.committed) throw std::runtime_error("uncommitted prefix record");
        for (const auto id : record.committed_ids()) {
          if (at == expected.size() || id != expected[at++]) throw std::runtime_error("actual committed causal prefix differs");
        }
        state.note("R1_PRIOR request=%u ordinal=%zu quota=%zu planned_K=%zu verified=%zu consumed=%zu emitted=%zu before_draw=%llu after_draw=%llu\n",
            sample.request, i, record.quota, record.draft_limit, record.verified_rows, record.consumed_drafts, record.emitted_count,
            static_cast<unsigned long long>(record.before.sampling.draw), static_cast<unsigned long long>(record.after.sampling.draw));
      }
      const std::array<std::int32_t,3> delivered{2,471,220};
      if (at != expected.size() || request.session->head->length != 5 || request.session->head->next_id != 220 ||
          !std::ranges::equal(request.progress.generated, delivered)) throw std::runtime_error("actual same-prefix third round not present");
      state.note("R1_ARM request=%u prefix=1,2,3,2,471 pending=220 planned_K=%zu quota=%zu actual_before_draw=%llu EXPECTED_draw15_IS_NOT_ASSUMED=1\n",
          sample.request, planned, quota, static_cast<unsigned long long>(sample.cursor.draw));
      draft_sample = &sample;
    });
  }
  ~ProposalScope() { draft_sample = previous; }
};

inline void proposal(qwen35::DraftProgram &program, const c::ReadLease &q, std::int32_t id,
    const std::array<std::uint32_t,4> &expected, hyle::inference::Backend &backend) noexcept {
  if (!draft_sample) return;
  observe("q_fresh", [&] {
    auto &sample = *draft_sample; const auto row = sample.fresh_count;
    if (row >= 3 || !program.carrier.retired()) throw std::runtime_error("draft count/actual retirement differs");
    const auto description = program.io.output_facts(program.probabilities_output).description;
    read_float("q_just_retained", sample, row, q, description, sample.fresh_q[row], backend);
    read_fixed("draft_gpu_identity", sample, program.identity_values,
        std::span(sample.draft_keys).subspan(4 * row, 4), backend);
    sample.draft_ids[row] = id;
    state.note("R1_DRAFT request=%u row=%zu program_rows=%zu sampler=%zu steps=%zu id=%d gpu_key=%u,%u,%u,%u expected_cpu=%u,%u,%u,%u\n",
        sample.request,row,program.rows,program.sampling_index,program.steps,id,
        sample.draft_keys[4*row],sample.draft_keys[4*row+1],sample.draft_keys[4*row+2],sample.draft_keys[4*row+3],
        expected[0],expected[1],expected[2],expected[3]);
    ++sample.fresh_count;
  });
}

struct TargetScope final {
  inline static thread_local TargetScope *current = nullptr;
  TargetScope *previous = current;
  Sample *sample = nullptr;
  const c::ReadLease *generated = nullptr;
  c::ConstTensorView exact;
  TargetScope(const void *session, const c::ReadLease &lease, c::ConstTensorView view,
      std::size_t members,std::size_t physical_rows,std::size_t state_slot) noexcept
      : generated(&lease), exact(view) {
    for (auto &candidate : state.samples) if (candidate.armed && candidate.session == session && !candidate.after) sample = &candidate;
    if (sample) {
      state.note("R1_TARGET_CONTEXT request=%u members=%zu physical_query_rows=%zu state_slot=%zu\n",sample->request,members,physical_rows,state_slot);
      if (members!=1) { failure("target_scope","actual target is not R1; payload pair not observed"); sample=nullptr; }
    }
    current = this;
  }
  ~TargetScope() { current = previous; }
};

inline void prefix_before(qwen35::PrefixSamplingProgram &program, std::span<const c::ReadLease> q,
    std::span<const std::int32_t> ids, SamplingCursor cursor, std::size_t quota,
    const hyle::inference::LogitsInputs &inputs, std::span<const hyle::inference::LogitsInputs> conditional,
    hyle::inference::Backend &backend) noexcept {
  auto *scope = TargetScope::current; if (!scope || !scope->sample) return;
  observe("prefix_before", [&] {
    auto &sample = *scope->sample;
    if (sample.before || q.size() != ids.size() || q.size() > 3 || sample.fresh_count != q.size() || cursor != sample.cursor || quota != sample.quota)
      throw std::runtime_error("actual prefix identity/count/owner differs from retained draft cohort");
    sample.proposals = q.size();
    state.note("R1_PREFIX request=%u actual_K=%zu verified_rows=%zu quota=%zu seed=%llu stream=%llu draw=%llu bias=%d allowed=%d counts=%d conditional_rows=%zu\n",
        sample.request, q.size(), q.size()+1, quota, static_cast<unsigned long long>(cursor.seed),
        static_cast<unsigned long long>(cursor.stream), static_cast<unsigned long long>(cursor.draw),
        int(inputs.bias.lease.valid()),int(inputs.allowed.lease.valid()),int(inputs.counts.lease.valid()),conditional.size());
    if (inputs.bias.lease.valid() || inputs.allowed.lease.valid() || inputs.counts.lease.valid() || !conditional.empty())
      throw std::runtime_error("unexpected actual conditional inputs; not read or silently treated equal");
    for (std::size_t row=0; row<q.size(); ++row) {
      if (ids[row] != sample.draft_ids[row]) throw std::runtime_error("prefix proposal id differs from its original completed draft");
      read_float("q_before_prefix",sample,row,q[row],program.io.input_facts(program.q_inputs[row]).description,sample.prefix_q[row],backend);
      difference("q_fresh_to_prefix",sample.fresh_q[row].value,sample.prefix_q[row].value,sample.request,row);
    }
    c::ReadLease target; // Independent subview of the original ready owner, never a fabricated permission.
    if (scope->generated->valid()) {
      const auto root=take(scope->generated->view()), wanted=scope->exact.storage();
      if (root.resource()!=wanted.resource() || root.native_handle()!=wanted.native_handle() || wanted.offset()<root.offset())
        throw std::runtime_error("target original lease root differs");
      target=take(scope->generated->subview(wanted.offset()-root.offset(),take(scope->exact.description().required_bytes())));
    } else throw std::runtime_error("caller-owned target has no existing ready ReadLease; no publish/adopt/permission workaround");
    read_float("target_RAW_logits_NOT_processed_P",sample,0,target,scope->exact.description(),sample.target_raw_logits,backend);
    read_fixed("prefix_gpu_identity_before",sample,program.identity_values,std::span(sample.prefix_keys).first(4*(q.size()+1)),backend);
    if (!q.empty()) {
      read_fixed("prefix_gpu_proposals_before",sample,program.id_values,std::span(sample.prefix_ids).first(q.size()),backend);
      read_fixed("prefix_gpu_continuation_before",sample,program.continuation_values,std::span(sample.continuation).first(q.size()),backend);
    }
    for (std::size_t row=0; row<=q.size(); ++row) {
      const auto expected=cursor.identity(row==q.size()?2*q.size():2*row+1);
      state.note("R1_PREFIX_KEY request=%u row=%zu gpu=%u,%u,%u,%u cpu=%u,%u,%u,%u\n",sample.request,row,
          sample.prefix_keys[4*row],sample.prefix_keys[4*row+1],sample.prefix_keys[4*row+2],sample.prefix_keys[4*row+3],expected[0],expected[1],expected[2],expected[3]);
    }
    for (std::size_t row=0; row<q.size(); ++row) state.note("R1_PREFIX_FIXED request=%u row=%zu gpu_id=%d cpu_id=%d continuation_bits=%08x\n",
        sample.request,row,sample.prefix_ids[row],ids[row],std::bit_cast<std::uint32_t>(sample.continuation[row]));
    sample.before=true;
  });
}

inline void prefix_after(qwen35::PrefixSamplingProgram &program, std::span<const c::ReadLease> q,
    hyle::inference::Backend &backend) noexcept {
  auto *scope=TargetScope::current; if (!scope || !scope->sample) return;
  observe("prefix_after",[&] {
    auto &sample=*scope->sample;
    if (!sample.before || sample.after || !program.carrier.retired()) throw std::runtime_error("actual prefix completion/observation missing");
    std::array<std::uint32_t,16> keys{}; std::array<std::int32_t,3> ids{}; std::array<float,3> continuation{};
    read_fixed("prefix_gpu_identity_after",sample,program.identity_values,std::span(keys).first(4*(q.size()+1)),backend);
    if (!q.empty()) {
      read_fixed("prefix_gpu_proposals_after",sample,program.id_values,std::span(ids).first(q.size()),backend);
      read_fixed("prefix_gpu_continuation_after",sample,program.continuation_values,std::span(continuation).first(q.size()),backend);
    }
    state.note("R1_FIXED_STABILITY request=%u keys_equal=%d ids_equal=%d continuation_bits_equal=%d\n",sample.request,
        int(keys==sample.prefix_keys),int(ids==sample.prefix_ids),int(std::memcmp(continuation.data(),sample.continuation.data(),sizeof(continuation))==0));
    for (std::size_t row=0; row<q.size(); ++row) compare_live_q("q_prefix_consumption",sample,row,q[row],sample.prefix_q[row]);
    for (std::size_t i=0;i<program.packet_host.size();++i) state.note("R1_ORIGINAL_PACKET request=%u word=%zu value=%d\n",sample.request,i,program.packet_host[i]);
    sample.after=true;
    const auto &a=state.samples[0], &b=state.samples[1];
    if (!a.after || !b.after) return;
    if (a.vocabulary!=b.vocabulary || a.proposals!=b.proposals) throw std::runtime_error("cross-request sampler input shapes differ; no fake equality");
    difference("request12_vs14_target_RAW_logits",a.target_raw_logits.value,b.target_raw_logits.value,14,0);
    for (std::size_t row=0;row<a.proposals;++row) {
      difference("request12_vs14_q_fresh",a.fresh_q[row].value,b.fresh_q[row].value,14,row);
      difference("request12_vs14_q_before_prefix",a.prefix_q[row].value,b.prefix_q[row].value,14,row);
    }
    state.note("R1_CROSS_FIXED cpu_cursor_equal=%d quota_equal=%d planned_K_equal=%d draft_gpu_keys_equal=%d prefix_gpu_keys_equal=%d proposal_ids_equal=%d continuation_bits_equal=%d\n",
        int(a.cursor==b.cursor),int(a.quota==b.quota),int(a.planned==b.planned),int(a.draft_keys==b.draft_keys),int(a.prefix_keys==b.prefix_keys),
        int(a.prefix_ids==b.prefix_ids),int(std::memcmp(a.continuation.data(),b.continuation.data(),sizeof(a.continuation))==0));
    state.compared=true;
  });
}
template<class Model, class Call> inline void published(Model &, std::span<Call> calls) noexcept {
  for (const auto &call : calls) for (auto &sample : state.samples) {
    if (!sample.after || sample.session != call.request->session.get() || call.request->rounds.size() != 2) continue;
    observe("published_cursor",[&] {
      const auto &head=*call.request->session->head;
      const auto expected=sample.cursor.advanced(2*sample.proposals+1);
      state.note("R1_PUBLISHED request=%u actual_K=%zu quota=%zu next=%d length=%zu after_seed=%llu after_stream=%llu after_draw=%llu expected_after_draw=%llu cursor_contract_equal=%d\n",
          sample.request,sample.proposals,sample.quota,head.next_id,head.length,
          static_cast<unsigned long long>(head.sampling.seed),static_cast<unsigned long long>(head.sampling.stream),
          static_cast<unsigned long long>(head.sampling.draw),static_cast<unsigned long long>(expected.draw),int(head.sampling==expected));
    });
  }
}
} // namespace hyle::experimental::r1_payload28
