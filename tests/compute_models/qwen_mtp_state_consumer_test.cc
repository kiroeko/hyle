#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/state_versions.h"
#include "hyle/graph/batch.h"
#if defined(HYLE_TEST_QWEN_STATE_CPU) && defined(HYLE_TEST_QWEN_STATE_CUDA)
#include "hyle/graph/cpu_cuda.h"
#endif
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/graph/cuda.h"
#include "cuda_sidecar_quant_reference.h"
// This TU contains both backends. Runtime reference selection below must not
// change the legacy Bank::dots definition or turn CPU checks into CUDA work.
#undef HYLE_C4_QUANTIZED_REFERENCE
#undef HYLE_C4_GROUPED_REFERENCE
#undef HYLE_C5_GGUF_DOT_REFERENCE
#endif
#if defined(HYLE_TEST_QWEN_STATE_CPU)
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/graph/cpu.h"
#endif
#include "qwen_mtp_state_consumer.h"
#include "model_sampling_fragment.h"
#include "../../engine/inference/sampling_cursor.h"
#include "../../engine/inference/mtp_acceptance.h"
#include "qwen27_original_nextn_reference.h"
#include "gguf_nvfp4_fixture.h"
#include "qwen24_original_nextn_reference.h"
#include "qwen24_original_resident_weights.h"
#include "model_resident_fragment.h"
#include "qwen27_original_target_reference.h"
#include "qwen24_original_target_roles.h"
#include "qwen24_original_target_fragment.h"
#include "qwen24_original_target_reference.h"
#include "qwen_bfloat_original_reference.h"
#include "qwen_nvfp4_target_fragment.h"
#include "qwen_nvfp4_original_reference.h"
#include "qwen_gptq_projection_fragment.h"
#include "qwen_ct_asymmetric_projection_fragment.h"
#include "qwen_ct_asymmetric_original_reference.h"
#include "qwen_gptq_original_reference.h"
#include "qwen_bfloat_original_artifacts.h"
#include "qwen_bfloat_moe_reference.h"
#include "qwen_bfloat_moe_target_fragment.h"
#include "qwen_bfloat_selected_experts.h"
#include "qwen_bfloat_q24_artifacts.h"
#include "qwen_gguf_q24_artifacts.h"
#if defined(HYLE_TEST_QWEN_B07_ARTIFACTS_HEADER)
#include HYLE_TEST_QWEN_B07_ARTIFACTS_HEADER
#endif
// Defined only after the completed S01 acquisition generated real immutable pins.
#if defined(HYLE_TEST_QWEN_GGUF_ARTIFACTS_HEADER)
#include HYLE_TEST_QWEN_GGUF_ARTIFACTS_HEADER
#endif
// Only the complete actual Q03 acquisition may generate this include.
// Without it no Q03 test is registered; no placeholder pin/header.
#if defined(HYLE_TEST_QWEN_Q03_ARTIFACTS_HEADER)
#include HYLE_TEST_QWEN_Q03_ARTIFACTS_HEADER
#endif
// Only actual complete Q01 pins enable these entries; absent data never emits a fake header.
#if defined(HYLE_TEST_QWEN_Q01_ARTIFACTS_HEADER)
#include HYLE_TEST_QWEN_Q01_ARTIFACTS_HEADER
#endif
// Original Q10 needs completed50-role pins; no default/fake header registers it.
#if defined(HYLE_TEST_QWEN_Q10_ARTIFACTS_HEADER)
#include HYLE_TEST_QWEN_Q10_ARTIFACTS_HEADER
#endif
#if defined(HYLE_TEST_QWEN_Q09_ARTIFACTS_HEADER)
#include HYLE_TEST_QWEN_Q09_ARTIFACTS_HEADER
#endif
// Generated only after the complete original B04 loose-role acquisition.
#if defined(HYLE_TEST_QWEN_B04_ARTIFACTS_HEADER)
#include HYLE_TEST_QWEN_B04_ARTIFACTS_HEADER
#endif
#if defined(HYLE_TEST_QWEN_Q14_ARTIFACTS_HEADER)
#include HYLE_TEST_QWEN_Q14_ARTIFACTS_HEADER
#endif

namespace {
using namespace qwen_mtp_state_consumer;
namespace numeric=d5_qwen35_gguf_fragment;
namespace original_reference=qwen27_original_nextn_reference;
namespace original24_reference=qwen24_original_nextn_reference;
namespace resident24=qwen24_original_resident_weights;
namespace resident_fragment=model_resident_fragment;
namespace target_roles=qwen27_original_target_roles;
namespace target_model=qwen_full_attention_fragment;
namespace target_reference=qwen27_original_target_reference;
namespace target24_roles=qwen24_original_target_roles;
namespace target24_model=qwen24_original_target_fragment;
namespace target24_reference=qwen24_original_target_reference;
namespace moe_reference=qwen24_moe_reference;
namespace bfloat_reference=qwen_bfloat_original_reference;
namespace bfloat27=qwen_bfloat_original_artifacts::q27;
namespace bfloat_moe=qwen_bfloat_moe_reference;
namespace bfloat_moe_target=qwen_bfloat_moe_target_fragment;
namespace bfloat_experts=qwen_bfloat_selected_experts;
namespace bfloat24=qwen_bfloat_original_artifacts::q24;
namespace gguf_experts=qwen_gguf_selected_experts;
namespace gguf24=qwen_gguf_q24_artifacts;
#if defined(HYLE_TEST_QWEN_Q03_ARTIFACTS_HEADER)
namespace q03=qwen_bfloat_original_artifacts::q03;
#endif
#if defined(HYLE_TEST_QWEN_Q01_ARTIFACTS_HEADER)
namespace q01=qwen_bfloat_original_artifacts::q01;
#endif
#if defined(HYLE_TEST_QWEN_Q10_ARTIFACTS_HEADER)
namespace q10=qwen_bfloat_original_artifacts::q10;
#endif
#if defined(HYLE_TEST_QWEN_Q09_ARTIFACTS_HEADER)
namespace q09=qwen_bfloat_original_artifacts::q09;
#endif

Bytes read(const c::ReadLease &lease) {
  Bytes bytes(lease.size());check(take(lease.view(),"actual readonly result").read(bytes),"retained result bytes");return bytes;
}
template<class T> std::vector<T> typed(const c::ReadLease &lease) {
  if(lease.size()%sizeof(T))throw std::runtime_error("actual typed lease extent");
  std::vector<T> values(lease.size()/sizeof(T));
  check(take(lease.view(),"actual typed permission").read(std::as_writable_bytes(std::span(values))),"actual typed result");return values;
}
std::string directory(const char *name) {
#if defined(_WIN32)
  char *value=nullptr;std::size_t bytes=0;
  if(_dupenv_s(&value,&bytes,name))throw std::runtime_error("cannot read original-role directory");
  const std::unique_ptr<char,decltype(&std::free)> owner(value,&std::free);return value?value:"";
#else
  const auto *value=std::getenv(name);return value?value:"";
#endif
}
// Fixed receipt fields only, following the existing Pro fixture discipline.
// Mappings share DELETE on Windows so atomic external receipt replacement
// cannot race a long-lived ifstream. No product JSON parser is introduced.
std::string bfloat_receipt_text(const std::filesystem::path &path) {
  auto mapped=take(m::FileMapping::open(path.string().c_str()),"map original original expert receipt");
  if(!mapped.size()||mapped.size()>1024*1024)throw std::runtime_error("original expert receipt exceeds finite metadata bound");
  const auto raw=take(mapped.bytes(),"original original expert receipt bytes");
  return {reinterpret_cast<const char*>(raw.data()),raw.size()};
}
std::string_view bfloat_receipt_field(std::string_view text,std::string_view key) {
  const auto quoted='"'+std::string(key)+'"';auto first=text.find(quoted);
  if(first==std::string_view::npos||text.find(quoted,first+quoted.size())!=std::string_view::npos||
      (first=text.find(':',first+quoted.size()))==std::string_view::npos||
      (first=text.find_first_not_of(" \r\n\t",first+1))==std::string_view::npos)
    throw std::runtime_error("original expert receipt lacks a unique required field: "+std::string(key));
  return text.substr(first);
}
std::string bfloat_receipt_string(std::string_view text,std::string_view key) {
  const auto field=bfloat_receipt_field(text,key);
  const auto end=field.find('"',1);
  if(field.front()!='"'||end==std::string_view::npos||field.substr(1,end-1).find('\\')!=std::string_view::npos)
    throw std::runtime_error("original expert receipt has an invalid fixed string field");
  return std::string(field.substr(1,end-1));
}
template<class Contract> std::optional<std::string> original_ready_pin(const std::filesystem::path &path,
    const Contract &contract,std::string_view section,std::int32_t global) {
  const auto text=bfloat_receipt_text(path/"ready.json");
  const auto complete=bfloat_receipt_field(text,"complete");
  if(complete.starts_with("false")&&complete.size()>=6&&std::string_view(" ,}\r\n\t").find(complete[5])!=std::string_view::npos)return std::nullopt;
  if(!complete.starts_with("true")||complete.size()<5||std::string_view(" ,}\r\n\t").find(complete[4])==std::string_view::npos)
    throw std::runtime_error("original expert ready receipt has an invalid completion field");
  // The receipt's directory records acquisition history, not source identity.
  // Read the actual selected path; the source plan, requested global ID and
  // immutable catalogue/plane hashes authenticate a relocated cache as well.
  if(bfloat_receipt_string(text,"section")!=section||
      bfloat_receipt_string(text,"source_plan_sha256")!=contract.source_plan_sha256)
    throw std::runtime_error("original expert ready receipt belongs to a different source/section");
  const auto globals=bfloat_receipt_field(text,"globals");const auto end=globals.find(']');
  if(globals.front()!='['||end==std::string_view::npos)throw std::runtime_error("original expert ready globals are absent");
  const auto first=globals.find_first_not_of(" \r\n\t",1),last=globals.find_last_not_of(" \r\n\t",end-1);
  if(first==std::string_view::npos||first>=end||last<first)throw std::runtime_error("original expert ready needs exactly one original global ID");
  const auto number=globals.substr(first,last-first+1);std::int32_t recorded=-1;
  const auto parsed=std::from_chars(number.data(),number.data()+number.size(),recorded);
  if(parsed.ec!=std::errc{}||parsed.ptr!=number.data()+number.size()||recorded!=global)
    throw std::runtime_error("original expert ready global ID differs from the actual requested source");
  const auto catalogue=bfloat_receipt_field(text,"catalog");const auto close=catalogue.find('}');
  if(catalogue.front()!='{'||close==std::string_view::npos)throw std::runtime_error("original expert ready catalogue is absent");
  const auto object=catalogue.substr(0,close+1);const auto pin=bfloat_receipt_string(object,"sha256");
  if(bfloat_receipt_string(object,"file")!="catalog.tsv"||!bfloat::pinned(pin))
    throw std::runtime_error("original expert ready requires its actual immutable catalogue pin");
  return pin;
}
template<class Source,class ExpertSet> struct OriginalFrontierSupply final {
  std::shared_ptr<Source> source;
  std::filesystem::path expert_root,request;
  bool wait=false;
  std::size_t requests=0,attachments=0;
  void ensure(std::span<const std::int32_t> globals) {
    const auto section=label(source->contract().section);
    if(globals.empty()||globals.size()>full24::kResidentCapacity)throw std::runtime_error("original expert frontier must be its bounded actual Top10 selection");
    const auto started=std::chrono::steady_clock::now();bool announced=false;
    for(;;) {
      for(const auto global:source->missing(globals)) {
        auto decimal=std::to_string(global);if(decimal.size()<3)decimal.insert(0,3-decimal.size(),'0');
        const auto path=expert_root/std::string(section)/("expert"+decimal);
        if(!std::filesystem::exists(path/"ready.json"))continue;
        const auto pin=original_ready_pin(path,source->contract(),section,global);if(!pin)continue;
        const auto selected=ExpertSet::load(path,*pin,source->contract());
        if(selected->globals().size()!=1||selected->globals()[0]!=global)
          throw std::runtime_error("original immutable per-ID catalogue differs from its actual frontier");
        source->attach_experts(selected);++attachments;
      }
      const auto missing=source->missing(globals);if(missing.empty())return;
      if(!announced) {
        if(!request.empty()) {
          if(!request.parent_path().empty())std::filesystem::create_directories(request.parent_path());
          auto temporary=request;temporary+=".tmp";
          std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
          if(!out)throw std::runtime_error("cannot write actual original expert frontier request");
          const auto list=[&](std::span<const std::int32_t> ids){out<<'[';for(std::size_t i=0;i<ids.size();++i){if(i)out<<',';out<<ids[i];}out<<']';};
          out<<"{\"section\":\""<<section<<"\",\"stage\":0,\"global_ids\":";list(globals);
          out<<",\"missing_ids\":";list(missing);out<<"}\n";out.close();
          if(!out)throw std::runtime_error("cannot finish actual original expert frontier request");
#if defined(_WIN32)
          bool replaced=false;
          for(unsigned attempt=0;attempt<100;++attempt) {
            if(MoveFileExW(temporary.c_str(),request.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){replaced=true;break;}
            const auto error=GetLastError();
            if(error!=ERROR_SHARING_VIOLATION&&error!=ERROR_ACCESS_DENIED)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
          if(!replaced)throw std::runtime_error("cannot atomically publish actual original expert frontier request");
#else
          std::filesystem::rename(temporary,request);
#endif
        }
        ++requests;announced=true;
        std::printf("Qwen original actual expert frontier section=%s stage=0 missing=%zu; %s\n",section.data(),missing.size(),wait?"waiting for explicit external acquisition":"missing acquisition fails");std::fflush(stdout);
      }
      if(!wait)throw std::runtime_error("actual original original expert planes are absent; explicit frontier supply is required");
      if(std::chrono::steady_clock::now()-started>std::chrono::minutes(20))
        throw std::runtime_error("original original external expert acquisition exceeded bounded frontier wait");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
};
using BfloatFrontierSupply=OriginalFrontierSupply<bfloat_experts::Source,bfloat_experts::ExpertSet>;
using GgufFrontierSupply=OriginalFrontierSupply<gguf_experts::Source,gguf_experts::ExpertSet>;
struct State final {
  c::StateSchema schema;
  c::StateStore store;
  std::array<c::StateEntryId,kResources> entries;
  std::vector<c::StateEntryId> extra_entries;
  std::optional<c::StateEntryId> sampling_entry;
  std::size_t backing_bound=0;
  static State create(const Model &model,const c::StorageOwner &owner,std::size_t output_root_bytes,
                      std::span<const c::TensorDesc> extra_descriptions={},bool with_sampling=false) {
    std::vector<c::StateEntrySpec> specs(kResources+extra_descriptions.size());
    const auto desc=descriptions(model);std::size_t bytes=0;
    for(std::size_t i=0;i<specs.size();++i) {
      const auto &description=i<desc.size()?desc[i]:extra_descriptions[i-desc.size()];
      specs[i]={.description=description,.resource=owner.ref()};
      const auto extent=take(description.required_bytes(),"complete draft state bytes");
      if(extent>std::numeric_limits<std::size_t>::max()-255||bytes>std::numeric_limits<std::size_t>::max()-(extent+255u)/256u*256u)
        throw std::runtime_error("complete state backing bound overflow");
      bytes+=(extent+255u)/256u*256u;
    }
    if(output_root_bytes>std::numeric_limits<std::size_t>::max()-bytes)throw std::runtime_error("complete state output backing bound overflow");
    const auto sampling_slot=specs.size();
    if(with_sampling)specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(hyle::inference::SamplingCursor)});
    // One bounded metadata block per retained version; retained output roots
    // remain charged once rather than by each logical resource member.
    const std::size_t metadata_bound=with_sampling?96u*64u:0;
    if(output_root_bytes+bytes>std::numeric_limits<std::size_t>::max()-metadata_bound)
      throw std::runtime_error("complete state metadata backing bound overflow");
    State s;s.backing_bound=output_root_bytes+bytes+metadata_bound;
    s.schema=take(c::StateSchema::create(specs),"complete NextN state schema");
    // A state member retains the WHOLE output-pool root, including diagnostic
    // generations. Count it once, plus the six separate initial roots; logical
    // member extents do not measure retained physical backing.
    s.store=take(c::StateStore::create(s.schema,{.max_versions=16,.max_drafts=2,
        .max_retained_blocks=96,.max_backing_bytes=s.backing_bound}),"bounded ordinary state store");
    for(std::size_t i=0;i<s.entries.size();++i)s.entries[i]=take(s.schema.entry(i),"actual complete state member");
    for(std::size_t i=0;i<extra_descriptions.size();++i)s.extra_entries.push_back(take(s.schema.entry(kResources+i),"actual additional target state member"));
    if(with_sampling)s.sampling_entry=take(s.schema.entry(sampling_slot),"same whole-state random cursor metadata");
    return s;
  }
  c::ReadLease member(const c::StateVersion &version,std::size_t entry) const {
    return take(version.member(entries.at(entry)),"actual immutable complete-state member");
  }
  std::array<c::ReadLease,kResources> inputs(const c::StateVersion &version) const {
    std::array<c::ReadLease,kResources> values;
    for(std::size_t i=0;i<values.size();++i)values[i]=member(version,i);return values;
  }
  c::StateVersion import(std::span<const c::ReadLease> values,std::span<const c::ReadLease> extra={},
      std::optional<hyle::inference::SamplingCursor> cursor={}) {
    if(values.size()!=kResources||extra.size()!=extra_entries.size())throw std::runtime_error("complete imported NextN/target member count");
    if(sampling_entry.has_value()!=cursor.has_value())throw std::runtime_error("complete imported random cursor schema mismatch");
    std::vector<c::StateMember> members(kResources+extra.size()+std::size_t(cursor.has_value()));
    for(std::size_t i=0;i<values.size();++i)members[i]={entries[i],values[i]};
    for(std::size_t i=0;i<extra.size();++i)members[kResources+i]={extra_entries[i],extra[i]};
    if(cursor)members.back()={*sampling_entry,{},std::as_bytes(std::span(&*cursor,1))};
    return take(store.import(members),"explicit complete initial boundary");
  }
  c::StateVersion candidate(const c::StateVersion &from,const Recipe &recipe,std::span<const c::ReadLease> result,
      std::span<const c::ReadLease> extra={},const c::ReadLease &pending={}) {
    if(!extra.empty()&&extra.size()!=extra_entries.size())throw std::runtime_error("one complete target checkpoint needs every target member");
    auto draft=take(store.begin(from),"single complete NextN prefix draft");
    for(std::size_t i=0;i<entries.size();++i)
      check(draft.replace(entries[i],result[recipe.state_ports[i]]),"stage actual matching complete prefix member");
    for(std::size_t i=0;i<extra.size();++i)check(draft.replace(extra_entries[i],extra[i]),"stage actual matching target prefix member");
    if(pending)check(draft.replace(entries[kPending],pending),"stage actual verifier pending result lease");
    return take(draft.finalize(),"one complete hidden/KV/pending/position candidate");
  }
};
using Image=std::array<Bytes,kResources>;
Image image(const State &state,const c::StateVersion &version) {
  Image result;for(std::size_t i=0;i<result.size();++i)result[i]=read(state.member(version,i));return result;
}
struct NativeReferenceCount final {
  std::uint32_t submissions=0;
  ~NativeReferenceCount() {
    // Also report work if a numerical ASSERT exits the journey early.
    ::testing::Test::RecordProperty("original_native_reference_submissions",int(submissions));
  }
};
struct ExpertSource final {
  // Only immutable metadata/views are adapted. This strong owner keeps the
  // original mappings alive; there is no virtual source or payload copy.
  std::shared_ptr<const void> owner;
  std::array<c::TensorDesc,3> descriptions;
  std::array<std::span<const std::byte>,3> banks;
  std::array<m::ConstBufferView,3> views;
  std::function<std::span<const std::byte>(full24::Leg,std::int32_t)> selected_bytes;
  std::function<m::ConstBufferView(full24::Leg,std::int32_t)> selected_view;
  c::TensorDesc expert_description(full24::Leg leg) const {return descriptions.at(std::size_t(leg));}
  std::span<const std::byte> expert_bytes(full24::Leg leg,std::int32_t global) const {
    if(selected_bytes)return selected_bytes(leg,global);
    const auto index=std::size_t(leg);const auto &description=descriptions.at(index);
    if(global<0||global>=description.dimensions()[0])throw std::runtime_error("actual expert source ID out of range");
    const auto stride=std::size_t(description.byte_strides()[0]);return banks.at(index).subspan(std::size_t(global)*stride,stride);
  }
  m::ConstBufferView expert_view(full24::Leg leg,std::int32_t global) const {
    if(selected_view)return selected_view(leg,global);
    const auto bytes=expert_bytes(leg,global);const auto stride=std::size_t(descriptions.at(std::size_t(leg)).byte_strides()[0]);
    return take(views.at(std::size_t(leg)).subview(std::size_t(global)*stride,bytes.size()),"actual selected expert mapping");
  }
};
template<class Data,class Role> ExpertSource expert_source(std::shared_ptr<const Data> source,const std::array<Role,3> &roles) {
  if(!source)throw std::runtime_error("actual expert source needs its original mapping owner");
  ExpertSource result;result.owner=source;
  for(std::size_t leg=0;leg<roles.size();++leg) {
    result.descriptions[leg]=source->expert_description(static_cast<full24::Leg>(leg));
    result.banks[leg]=source->bytes(roles[leg]);result.views[leg]=source->view(roles[leg]);
    if(take(result.descriptions[leg].required_bytes(),"actual original expert extent")!=result.banks[leg].size()||
        result.views[leg].size()!=result.banks[leg].size())throw std::runtime_error("original expert bytes/view/typed extent differs");
  }
  (void)take(resident24::PackedExpertLayout::create(result.descriptions,resident24::kCapacity),"actual original expert layout");return result;
}
// Cold adapter for an authenticated selected-range source. Its immutable
// owner decides whether a real global ID is present and must throw if absent;
// lookup never fetches, synthesizes weights or pretends a sparse file has the
// full-bank SHA. Supply still validates every actual plane extent and view.
template<class Source> ExpertSource expert_source(std::shared_ptr<const Source> source) {
  if(!source)throw std::runtime_error("selected expert source needs its immutable owner");
  ExpertSource result;result.owner=source;
  for(std::size_t leg=0;leg<result.descriptions.size();++leg)
    result.descriptions[leg]=source->expert_description(static_cast<full24::Leg>(leg));
  (void)take(resident24::PackedExpertLayout::create(result.descriptions,resident24::kCapacity),"actual selected expert layout");
  result.selected_bytes=[source](full24::Leg leg,std::int32_t global){return source->expert_bytes(leg,global);};
  result.selected_view=[source](full24::Leg leg,std::int32_t global){return source->expert_view(leg,global);};
  return result;
}
struct ExpertSupply final {
  std::shared_ptr<const packed_moe::NextnRoles> roles;
  ExpertSource source;
  // Optional explicit fixture frontier hook. An external supply driver may
  // attach authenticated original slices here; raw lookup never fetches.
  std::function<void(std::span<const std::int32_t>)> before_upload;
  std::shared_ptr<resident24::CpuSupply> cpu;
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
  std::shared_ptr<resident24::CudaSupply> cuda;
#endif
  std::size_t episodes=0;
  std::vector<full24::Selection> selections;
  const resident24::PackedExpertLayout &layout() const {
    if(cpu)return cpu->layout();
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
    if(cuda)return cuda->layout();
#endif
    throw std::runtime_error("ordinary expert supply has no native backend");
  }
};
// Cold placement and actual domains shared by the existing ordinary Programs.
// CPU is used only by explicitly host-bound vocabulary readers and TopK;
// all other compute nodes remain required CUDA targets, including pending
// Copies. No callback chooses a provider or executes an extra Graph here.
struct MixedPlacement final {
  c::Context cpu;
  c::Queue cpu_queue;
  c::StorageOwner host,device,staging;
  bfloat::VocabularyNames vocabulary;
  bool vocabulary_role(std::string_view name) const {
    return name==vocabulary.embedding||name==vocabulary.head;
  }
  const c::StorageOwner &owner_for(std::string_view name) const {
    return vocabulary_role(name)?host:device;
  }
};
// Bind each ORIGINAL role once in its declared domain. Data still owns all
// original mappings; this is the existing readonly role binding operation,
// not another loader, weight conversion or model-specific source registry.
bfloat::Bound bind_mixed_bfloat(std::shared_ptr<const bfloat::Data> source,const MixedPlacement &placement) {
  if(!source||!placement.host||!placement.device||!placement.staging)
    throw std::runtime_error("mixed BF16 binding requires its actual retained storage domains");
  bfloat::Bound result;result.source=std::move(source);
  for(std::size_t slot=0;slot<result.source->size();++slot) {
    const auto name=result.source->role_name(slot);
    if(result.source->placement(name)==bfloat::Placement::kSelectedSource)continue;
    result.roles.emplace(name,orig::detail::bind(name,result.source->description(name),placement.owner_for(name),
        result.source->bytes(name),&result.source->identity()));
  }
  return result;
}
struct Program final {
  struct Phases final {
    resident_fragment::ResidentDraft<moe24::Boundary,1> plan;
    std::vector<std::shared_ptr<Program>> programs;
  };
  Recipe recipe;
  std::shared_ptr<ExpertSupply> supply; // outlives every ordinary child Invocation
  std::shared_ptr<MixedPlacement> mixed; // outlives compiled artifacts, queues and output pools
  sem::GraphCase logical;
  std::vector<c::ReadLease> parameters;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  c::LeasePool outputs;
  c::LeasePool host_outputs;
  std::vector<bool> input_on_host;
  std::vector<bool> output_on_host;
  std::vector<bool> node_on_host;
  std::vector<g::QueueBinding> required_queue_bindings;
  std::vector<std::size_t> caller_queue_ports;
  std::shared_ptr<Phases> resident;
  std::size_t output_root_bytes=0;
  std::uint32_t submissions=0;
  bool cuda_backend=false,original_full_checked=false;
  std::uint32_t original_episodes=0;
  std::uint64_t original_full_values=0,original_full_head_values=0;
  std::vector<std::int32_t> original_pending,original_predictions;
  static Program create(Recipe recipe,c::Context &context,const c::StorageOwner &owner,bool cuda,
                        bool phase=false,std::shared_ptr<ExpertSupply> supply={},std::shared_ptr<MixedPlacement> mixed={},
                        std::vector<bool> input_on_host={},std::span<const std::size_t> required_packed_matrices={},std::size_t required_packed_nodes=4) {
    if(recipe.bfloat_moe) {
      auto raw=bfloat_moe::linear_stores(recipe.fragment,recipe.bfloat_moe->moe);
      raw.visit_values([&](Value &value) {
        const auto found=std::find_if(recipe.bfloat_ports.begin(),recipe.bfloat_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
        if(found==recipe.bfloat_ports.end())recipe.bfloat_ports.push_back({value,expose(recipe,value)});
      });
    }
    Program p;p.recipe=std::move(recipe);p.supply=std::move(supply);p.mixed=std::move(mixed);
    p.logical=sem::make_graph(p.recipe.fragment);p.cuda_backend=cuda;
    if(p.mixed&&(!cuda||p.recipe.model.storage!=c::DataType::kBFloat16||p.mixed->device.ref()!=owner.ref()))
      throw std::runtime_error("mixed placement requires its actual BF16 CUDA body domain");
    p.input_on_host=std::move(input_on_host);
    if(p.mixed&&p.input_on_host.empty())p.input_on_host.assign(p.logical.inputs.size(),false);
    if(p.mixed&&p.input_on_host.size()!=p.logical.inputs.size())
      throw std::runtime_error("mixed input domains must describe every actual input lease");
    std::vector<g::ConstantBinding> constants;
    for(std::size_t i=0;i<p.logical.constants.size();++i) {
      const auto binding=std::find_if(p.recipe.external.begin(),p.recipe.external.end(),[&](const auto &item){return item.constant==i;});
      if(binding!=p.recipe.external.end()) {
        if(binding->role.description!=p.recipe.fragment.description(p.recipe.fragment.constants[i]))throw std::runtime_error("external role descriptor changed");
        constants.push_back(take(binding->role.binding(p.logical.constants[i]),"original/shared vocabulary owner binding"));
      } else {
        const auto &desc=p.recipe.fragment.description(p.recipe.fragment.constants[i]);
        if(p.recipe.fragment.constant_bytes[i].size()!=take(desc.required_bytes(),"generated constant bytes"))throw std::runtime_error("missing generated constant payload");
        p.parameters.push_back(upload(owner,p.recipe.fragment.constant_bytes[i]));
        constants.push_back({.value=p.logical.constants[i],.tensor=take(p.parameters.back().tensor(desc),"generated immutable parameter"),.owner=owner,.lease=p.parameters.back()});
      }
    }
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(*p.recipe.model.profile),"exact selected MAIN artifact"),
        c::WorkloadPhase::kDecode,{0x48594c454d4f444cULL,p.recipe.fragment.region,1}};
    const g::CompileOptions options{.constants=constants,.workload=workload};
    const auto compile=[&]() -> hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_QWEN_STATE_CPU) && defined(HYLE_TEST_QWEN_STATE_CUDA)
      if(p.mixed) {
        using namespace g::cpu_cuda;
        const auto &fragment=p.recipe.fragment;
        if(p.logical.graph.node_count()!=fragment.nodes.size())throw std::runtime_error("mixed logical node mapping differs");
        std::vector<bool> host_value(fragment.descriptions.size(),false);
        std::vector<g::NodePlacement> nodes;
        std::vector<g::ExternalResourceContract> external;
        p.node_on_host.clear();
        for(std::size_t slot=0;slot<p.logical.inputs.size();++slot) {
          const bool host=p.input_on_host.at(slot);
          host_value[fragment.inputs[slot].ordinal()]=host;
          external.push_back({p.logical.inputs[slot],host?kHostStorage:kDeviceStorage});
        }
        for(std::size_t slot=0;slot<fragment.constants.size();++slot) {
          const auto binding=std::find_if(p.recipe.external.begin(),p.recipe.external.end(),[&](const auto &item){return item.constant==slot;});
          const bool host=binding!=p.recipe.external.end()&&p.mixed->vocabulary_role(binding->role.name);
          if(binding!=p.recipe.external.end()&&binding->role.owner.ref()!=(host?p.mixed->host.ref():owner.ref()))
            throw std::runtime_error("mixed original role was bound in a different actual domain");
          host_value[fragment.constants[slot].ordinal()]=host;
          external.push_back({p.logical.constants[slot],host?kHostStorage:kDeviceStorage});
        }
        for(std::size_t index=0;index<fragment.nodes.size();++index) {
          const auto &node=fragment.nodes[index];
          const bool view=node.operation.get_if<c::View>()!=nullptr;
          const bool copy=node.operation.get_if<c::Copy>()!=nullptr;
          const bool host_reader=(node.operation.get_if<c::Gather>()&&host_value[node.inputs.at(0).ordinal()])||
              (node.operation.custom_as<c::model_ops::DenseLinearF32>()&&host_value[node.inputs.at(1).ordinal()]);
          // Only transparent vocabulary-result metadata/TopK continues on
          // CPU. Published I32 Copies deliberately return to device state.
          const bool follow=(view||copy||node.operation.get_if<c::TopK>())&&!node.inputs.empty()&&host_value[node.inputs[0].ordinal()];
          const bool integer_copy=copy&&fragment.description(node.outputs.at(0)).data_type()==c::DataType::kInt32;
          const bool host=(host_reader||follow)&&!integer_copy;
          p.node_on_host.push_back(host);
          for(const auto output:node.outputs)host_value[output.ordinal()]=host;
          if(!view)nodes.push_back({take(p.logical.graph.node_id(index),"mixed original logical node"),
              host?kCpuTarget:kCudaTarget,g::PlacementStrength::kRequired});
        }
        p.output_on_host.clear();
        for(std::size_t slot=0;slot<fragment.outputs.size();++slot) {
          const bool host=host_value[fragment.outputs[slot].ordinal()];p.output_on_host.push_back(host);
          external.push_back({p.logical.outputs[slot],host?kHostStorage:kDeviceStorage});
        }
        const std::array domains{g::StorageDomain{.resource=owner.ref(),.owner=owner},
            g::StorageDomain{.resource=p.mixed->host.ref(),.owner=p.mixed->host},
            g::StorageDomain{.resource=p.mixed->staging.ref(),.owner=p.mixed->staging}};
        const std::array targets{g::ExecutionTarget{take(g::ExecutionSource::acquire(context),"mixed CUDA source"),kDeviceStorage},
            g::ExecutionTarget{take(g::ExecutionSource::acquire(p.mixed->cpu),"mixed CPU source"),kHostStorage}};
        const std::array lanes{g::ExecutionLane{.target=kCudaTarget},g::ExecutionLane{.target=kCpuTarget}};
        auto placed=options;placed.node_placements=nodes;placed.external_resources=external;
        return g::batch::compile(p.logical.graph,{.storage_domains=domains,.targets=targets,.lanes=lanes},placed);
      }
#endif
      if(p.mixed)throw std::runtime_error("mixed Graph needs both actual CPU and CUDA sources");
#if defined(HYLE_TEST_QWEN_STATE_CPU)
      if(!cuda)return g::cpu::compile(p.logical.graph,context,options);
#endif
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
      if(cuda)return g::cuda::compile(p.logical.graph,context,owner,options);
#endif
      throw std::runtime_error("requested NextN backend is not enabled");
    };
    if(p.recipe.moe) {
      if(!p.supply||!p.supply->source.owner||(p.recipe.original24&&p.supply->roles!=p.recipe.original24))
        throw std::runtime_error("complete Q24 needs its actual original resident source");
      for(std::size_t leg=0;leg<3;++leg)
        if(p.recipe.fragment.description(p.recipe.moe->boundary.banks[leg])!=p.supply->layout().resident(leg))
          throw std::runtime_error("actual expert supply differs from the cold Graph bank description");
      p.resident=std::make_shared<Phases>();
      p.resident->plan=resident_fragment::split(p.recipe.fragment,std::array{p.recipe.moe->boundary},0x5132344d4f455000ULL);
      // Full constants were prepared exactly once above; each cold phase
      // borrows those same immutable owners, including both distinct banks.
      std::vector<orig::ReadonlyRole> ready;std::size_t generated=0;
      for(std::size_t slot=0;slot<p.logical.constants.size();++slot) {
        const auto found=std::find_if(p.recipe.external.begin(),p.recipe.external.end(),[&](const auto &entry){return entry.constant==slot;});
        if(found!=p.recipe.external.end())ready.push_back(found->role);
        else ready.push_back({"shared ready Q24 constant",p.recipe.fragment.description(p.recipe.fragment.constants[slot]),owner,p.parameters.at(generated++)});
      }
      for(const auto &part:p.resident->plan.phases) {
        Recipe local{.model=p.recipe.model,.fragment=part.fragment};
        local.state_ports.fill(std::numeric_limits<std::size_t>::max());
        for(std::size_t slot=0;slot<part.original_constants.size();++slot) {
          const auto origin=part.original_constants[slot];
          const auto found=std::find_if(p.recipe.fragment.constants.begin(),p.recipe.fragment.constants.end(),[&](Value value){return value.ordinal()==origin.ordinal();});
          if(found==p.recipe.fragment.constants.end())throw std::runtime_error("Q24 phase constant lost original identity");
          local.external.push_back({slot,ready.at(std::size_t(found-p.recipe.fragment.constants.begin()))});
        }
        std::vector<bool> child_inputs_on_host;
        if(p.mixed)for(const auto original:part.original_inputs) {
          const auto direct=std::find_if(p.recipe.fragment.inputs.begin(),p.recipe.fragment.inputs.end(),
              [&](Value value){return value.ordinal()==original.ordinal();});
          if(direct!=p.recipe.fragment.inputs.end()) {
            child_inputs_on_host.push_back(p.input_on_host.at(std::size_t(direct-p.recipe.fragment.inputs.begin())));
            continue;
          }
          // Match the very same producer port whose immutable lease the
          // resident loop places in live[original]. In particular a splitter
          // I32 Copy may already have moved a CPU TopK value back to device.
          bool found=false;
          for(std::size_t producer=0;producer<p.resident->programs.size();++producer) {
            const auto &producer_outputs=p.resident->plan.phases[producer].original_outputs;
            const auto output=std::find_if(producer_outputs.begin(),producer_outputs.end(),
                [&](Value value){return value.ordinal()==original.ordinal();});
            if(output==producer_outputs.end())continue;
            if(found)throw std::runtime_error("mixed phase input has more than one actual producer port");
            child_inputs_on_host.push_back(p.resident->programs[producer]->output_on_host.at(std::size_t(output-producer_outputs.begin())));
            found=true;
          }
          if(!found)throw std::runtime_error("mixed phase input lost its producer output domain");
        }
        p.resident->programs.push_back(std::make_shared<Program>(Program::create(std::move(local),context,owner,cuda,true,{},p.mixed,std::move(child_inputs_on_host))));
        p.output_root_bytes+=p.resident->programs.back()->output_root_bytes;
      }
      p.report_original24_roles(owner);
      p.report_bfloat_roles(owner);
      for(auto &payload:p.recipe.fragment.constant_bytes){Bytes empty;payload.swap(empty);}
      return p;
    }
    p.compiled=take(compile(),"one ordinary complete NextN Graph");
    if(!required_packed_matrices.empty()) {
      if(required_packed_matrices.size()!=7)
        throw std::runtime_error("original packed artifact target needs seven packed report proofs");
      std::array<g::NodeId,7> matrices;
      for(std::size_t slot=0;slot<matrices.size();++slot) {
        const auto copy=take(p.logical.graph.producer(p.logical.outputs.at(required_packed_matrices[slot])),"packed artifact independent result Copy");
        if(take(p.logical.graph.operation_kind(copy),"packed artifact result observer kind")!=c::OperationKind::kCopy)
          throw std::runtime_error("packed artifact actual results require independent Copy outputs");
        const auto stored=take(p.logical.graph.node_input(copy,0),"packed artifact actual BF16 result");
        matrices[slot]=take(p.logical.graph.producer(stored),"packed artifact public MatMul");
        if(take(p.logical.graph.operation_kind(matrices[slot]),"packed artifact public matrix kind")!=c::OperationKind::kMatMul)
          throw std::runtime_error("packed artifact must retain the exact original public projection recipe");
      }
      // Before Invocation/workspace allocation: no unfused whole-W decode is allowed.
      qwen_artifact_projection_fragment::require_packed(p.compiled.report,matrices,required_packed_nodes);
      ::testing::Test::RecordProperty("target_packed_contractions",int(matrices.size()));
      ::testing::Test::RecordProperty("target_packed_recipe_nodes",int(required_packed_nodes));
      ::testing::Test::RecordProperty("target_dense_weight_materializations",0);
    }
    // Cold public report facts, not the configured target count: unused
    // compute/transfer/delegated lanes must not receive a QueueBinding.
    for(std::size_t index=0;index<p.compiled.report.lane_count();++index) {
      const auto lane=take(p.compiled.report.lane(index),"actual required execution lane");
      if(!lane.queue_required)continue;
      const auto port=p.required_queue_bindings.size();
      if(lane.target==g::ExecutionTargetId{0}) {
        p.required_queue_bindings.push_back({lane.lane,nullptr});
        p.caller_queue_ports.push_back(port);
      } else if(p.mixed&&lane.target==g::ExecutionTargetId{1}) {
        p.required_queue_bindings.push_back({lane.lane,&p.mixed->cpu_queue});
      } else throw std::runtime_error("required Graph lane has no actual configured queue");
    }
    if(p.mixed)p.report_mixed_vocabulary();
    if(!phase) {
    // This existing four-case consumer is also the full packed-RHS route
    // regression. Inspect the actual MatMul behind its independent Copy;
    // smaller internal shards cannot satisfy this descriptor or report gate.
    const auto copied_logits=take(p.logical.graph.producer(p.logical.outputs[p.recipe.logits]),"full logits observer");
    if(take(p.logical.graph.operation_kind(copied_logits),"full logits observer kind")!=c::OperationKind::kCopy)
      throw std::runtime_error("full logits observer must own an independent Copy");
    const auto full_logits=take(p.logical.graph.node_input(copied_logits,0),"actual full logits value");
    const auto full_head=take(p.logical.graph.producer(full_logits),"actual full head MatMul");
    const auto full_weight=take(p.logical.graph.node_input(full_head,1),"actual complete packed head");
    const auto head_operation=take(p.logical.graph.operation(full_head),"full head actual operation");
    const bool bfloat_head=head_operation.custom_as<c::model_ops::DenseLinearF32>().has_value();
    const auto wanted_weight=bfloat_head?
        sem::dense({p.recipe.model.vocabulary,p.recipe.model.hidden},c::DataType::kBFloat16):
        take(c::gguf_weight_description(p.recipe.model.head.quant,c::DataType::kFloat32,
            p.recipe.model.hidden,p.recipe.model.vocabulary),"required complete packed head geometry");
    if((bfloat_head?(!p.recipe.original_bfloat&&!p.recipe.original):
            take(p.logical.graph.operation_kind(full_head),"full head kind")!=c::OperationKind::kMatMul)||
        take(p.logical.graph.value_description(full_weight),"full head actual weight geometry")!=wanted_weight ||
        take(p.logical.graph.value_description(full_logits),"full head actual logits geometry")!=sem::dense({1,p.recipe.model.vocabulary}))
      throw std::runtime_error("complete vocabulary head must be one actual full-bank projection");
    std::size_t selected_head_steps=0;
    for(std::size_t step_index=0;step_index<p.compiled.report.step_count();++step_index) {
      const auto selected=take(p.compiled.report.step(step_index),"full head selected plan step");
      for(std::size_t node_index=0;node_index<selected.node_count;++node_index)
        if(take(p.compiled.report.step_node(step_index,node_index),"full head selected logical node")==full_head) {
          ++selected_head_steps;
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
          if(cuda && !bfloat_head && selected.provider.id!=c::ggml_cuda::kProviderId)
            throw std::runtime_error("full packed CUDA head must retain the native ggml CUDA route");
#endif
          ::testing::Test::RecordProperty("full_head_provider",std::string(selected.provider.name));
        }
    }
    if(selected_head_steps!=1)throw std::runtime_error("full head must belong to exactly one selected plan step");
    ::testing::Test::RecordProperty("full_head_encoded_bytes",std::to_string(take(wanted_weight.required_bytes(),"full head span")));
    p.report_original_roles(owner);
    p.report_bfloat_roles(owner);
    }
    p.invocation=take(p.compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"one ordinary NextN Invocation");
    std::size_t bytes=0,host_bytes=0;
    for(std::size_t i=0;i<p.logical.outputs.size();++i) {
      if(take(p.compiled.executable.output_ownership(i),"actual NextN output ownership")!=g::OutputOwnership::kCaller)throw std::runtime_error("NextN fixture requires independent caller outputs");
      const auto extent=take(p.recipe.fragment.description(p.recipe.fragment.outputs[i]).required_bytes(),"actual NextN output span");
      auto &domain_bytes=p.mixed&&p.output_on_host.at(i)?host_bytes:bytes;
      const auto rounded=(extent+255u)/256u*256u;
      if(extent>std::numeric_limits<std::size_t>::max()-255u||domain_bytes>std::numeric_limits<std::size_t>::max()-rounded)
        throw std::runtime_error("mixed output pool extent overflows");
      domain_bytes+=rounded;
    }
    constexpr std::uint32_t generations=6;
    if(bytes>std::numeric_limits<std::size_t>::max()/generations||host_bytes>std::numeric_limits<std::size_t>::max()/generations||
        bytes>std::numeric_limits<std::size_t>::max()/generations-host_bytes)throw std::runtime_error("mixed output generations overflow");
    p.output_root_bytes=(bytes+host_bytes)*generations;
    if(bytes)p.outputs=take(c::LeasePool::create(owner.ref(),bytes*generations,
        {.max_blocks=std::uint32_t(p.logical.outputs.size()*generations),.backing_alignment=256}),"bounded retained NextN device generations");
    if(host_bytes)p.host_outputs=take(c::LeasePool::create(p.mixed->host.ref(),host_bytes*generations,
        {.max_blocks=std::uint32_t(p.logical.outputs.size()*generations),.backing_alignment=256}),"bounded retained vocabulary host generations");
    // Cold generated bytes have native immutable owners now; references retain
    // only the eight decoded prototype rows, not a second full matrix bank.
    for(auto &payload:p.recipe.fragment.constant_bytes) {Bytes empty;payload.swap(empty);}
    return p;
  }
  std::uint32_t native_submissions() const {
    if(!resident)return submissions;
    std::uint32_t count=0;for(const auto &program:resident->programs)count+=program->native_submissions();return count;
  }
  std::size_t live_output_blocks() const {
    if(!resident)return (outputs.valid()?outputs.stats().live_blocks:0)+(host_outputs.valid()?host_outputs.stats().live_blocks:0);
    std::size_t count=0;for(const auto &program:resident->programs)count+=program->live_output_blocks();return count;
  }
  std::size_t active_output_uses() const {
    if(!resident)return (outputs.valid()?outputs.stats().active_uses:0)+(host_outputs.valid()?host_outputs.stats().active_uses:0);
    std::size_t count=0;for(const auto &program:resident->programs)count+=program->active_output_uses();return count;
  }
  bool output_is_host(std::size_t port) const {
    if(!mixed)return false;
    if(resident) {
      const auto location=resident->plan.full_outputs.at(port);
      return resident->programs.at(location.phase)->output_is_host(location.port);
    }
    return output_on_host.at(port);
  }
  std::vector<c::ReadLease> execute_resident(c::Queue &queue,std::span<const c::ReadLease> inputs) {
    if(!resident||!recipe.moe||!supply||recipe.fragment.inputs.size()!=inputs.size()+4)
      throw std::runtime_error("complete original Q24 ordinary two-phase input contract");
    std::vector<c::ReadLease> live(recipe.fragment.descriptions.size());
    for(std::size_t i=0;i<inputs.size();++i)live.at(recipe.fragment.inputs[i].ordinal())=inputs[i];
    std::array<std::vector<c::ReadLease>,2> completed;
    for(std::size_t index=0;index<resident->programs.size();++index) {
      const auto &part=resident->plan.phases[index];auto &program=*resident->programs[index];
      std::optional<resident24::CpuBatch> cpu_batch;
      std::optional<full24::Selection> selected;
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
      std::optional<c::cuda::WeightBatch> cuda_batch;
      std::array<g::WeightBinding,4> weight_bindings{};
#endif
      std::array<std::size_t,4> dynamic_ports{};
      std::span<const std::size_t> externally_bound;
      std::function<void(g::Invocation&)> bind_external;
      if(part.resident_stage) {
        const auto &boundary=resident->plan.routers.at(*part.resident_stage);
        // Only these ten scheduling IDs cross to the host. Every activation,
        // router weight and state tensor remains its actual producer lease.
        selected=full24::select(typed<std::int32_t>(live.at(boundary.ids.ordinal())));
        if(supply->before_upload)supply->before_upload(selected->globals);
        const std::array bank_values{boundary.banks[0],boundary.banks[1],boundary.banks[2],boundary.map};
        for(std::size_t bank=0;bank<bank_values.size();++bank) {
          const auto found=std::find_if(part.original_inputs.begin(),part.original_inputs.end(),[&](Value value){return value.ordinal()==bank_values[bank].ordinal();});
          if(found==part.original_inputs.end())throw std::runtime_error("Q24 phase lost an expert/map generation input");
          dynamic_ports[bank]=std::size_t(found-part.original_inputs.begin());
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
          weight_bindings[bank]={program.logical.inputs.at(dynamic_ports[bank]),resident24::kBankIds[bank]};
#endif
        }
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
        if(cuda_backend) {
          if(!supply->cuda)throw std::runtime_error("original Q24 CUDA supply is missing");
          cuda_batch.emplace(take(supply->cuda->upload(supply->episodes%2,*selected,supply->source,queue),"actual original Q24 expert upload"));
          externally_bound=dynamic_ports;
          bind_external=[&](g::Invocation &active){check(g::cuda::bind(active,weight_bindings,*cuda_batch),"atomic original Q24 experts/map generation");};
        } else
#endif
        {
          if(!supply->cpu)throw std::runtime_error("original Q24 CPU supply is missing");
          cpu_batch.emplace(take(supply->cpu->upload(*selected,supply->source),"actual original Q24 CPU expert supply"));
          for(std::size_t bank=0;bank<bank_values.size();++bank)live.at(bank_values[bank].ordinal())=cpu_batch->banks[bank];
        }
      }
      std::vector<c::ReadLease> arguments;
      for(auto original:part.original_inputs)arguments.push_back(live.at(original.ordinal()));
      completed[index]=program.execute_inputs(queue,arguments,externally_bound,bind_external);
      for(std::size_t port=0;port<part.original_outputs.size();++port)
        live.at(part.original_outputs[port].ordinal())=completed[index][port];
      if(part.resident_stage) {
        supply->selections.push_back(std::move(*selected));
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
        if(cuda_batch) {
          const auto accounting=take(cuda_batch->report(),"retired Q24 expert upload accounting");
          EXPECT_EQ(accounting.logical_bytes,supply->layout().logical_bytes());EXPECT_EQ(accounting.staged_bytes,supply->layout().logical_bytes());
          EXPECT_EQ(accounting.completed_bytes,accounting.submitted_bytes);EXPECT_FALSE(accounting.copied_bytes_unknown);
          std::printf("Q24 original E10 episode=%zu slot=%zu union=%zu logical=%zu staged=%zu submitted=%zu completed=%zu unknown=%d (API upload accounting,not measured VRAM/bus traffic)\n",
              supply->episodes,supply->episodes%2,supply->selections.back().globals.size(),accounting.logical_bytes,accounting.staged_bytes,
              accounting.submitted_bytes,accounting.completed_bytes,int(accounting.copied_bytes_unknown));
        }
#endif
        const auto &boundary=resident->plan.routers.at(*part.resident_stage);
        for(auto bank:boundary.banks)live.at(bank.ordinal())={};live.at(boundary.map.ordinal())={};
      }
      // The ordinary Invocation already waited and reset. Its batch and all
      // dynamic input guards retire before the shared double slot is reused.
    }
    std::vector<c::ReadLease> result;
    for(const auto location:resident->plan.full_outputs)result.push_back(completed.at(location.phase).at(location.port));
    ++supply->episodes;++submissions;return result;
  }
  struct Actual final {const Program *program;g::ValueId value;};
  Actual observation(std::size_t port) const {
    const Program *consumer=this;std::size_t local_port=port;
    if(resident) {
      const auto location=resident->plan.full_outputs.at(port);
      consumer=resident->programs.at(location.phase).get();local_port=location.port;
    }
    const auto copy=take(consumer->logical.graph.producer(consumer->logical.outputs.at(local_port)),"actual observer producer");
    if(take(consumer->logical.graph.operation_kind(copy),"actual observer operation")!=c::OperationKind::kCopy)
      throw std::runtime_error("model observations require independent Copy backing");
    return {consumer,take(consumer->logical.graph.node_input(copy,0),"actual observed value")};
  }
  const resident_fragment::Phase &phase(const Actual &consumer) const {
    if(resident)for(std::size_t index=0;index<resident->programs.size();++index)
      if(resident->programs[index].get()==consumer.program)return resident->plan.phases[index];
    throw std::runtime_error("actual consumer is outside its ordinary resident phases");
  }
  g::ValueId phase_input(Value origin,const Actual &consumer) const {
    const auto &part=phase(consumer);
    const auto found=std::find_if(part.original_inputs.begin(),part.original_inputs.end(),[&](Value value){return value.ordinal()==origin.ordinal();});
    if(found==part.original_inputs.end())throw std::runtime_error("actual operand is absent from consumer phase");
    return consumer.program->logical.inputs.at(std::size_t(found-part.original_inputs.begin()));
  }
  g::ValueId observation_input(std::size_t port,const Actual &consumer) const {
    const auto producer=observation(port);if(producer.program==consumer.program)return producer.value;
    const auto published=recipe.fragment.outputs.at(port);
    const auto copy=std::find_if(recipe.fragment.nodes.begin(),recipe.fragment.nodes.end(),[&](const auto &node){
      return node.outputs.size()==1&&node.outputs[0].ordinal()==published.ordinal();
    });
    if(copy==recipe.fragment.nodes.end()||!copy->operation.get_if<c::Copy>()||copy->inputs.size()!=1)
      throw std::runtime_error("cross-phase observer lost its original Copy provenance");
    return phase_input(copy->inputs[0],consumer);
  }
  g::ValueId constant_operand(const orig::ReadonlyRole &expected,const Actual &consumer,const c::StorageOwner &owner) const {
    const auto found=std::find_if(recipe.external.begin(),recipe.external.end(),[&](const auto &entry){return entry.role.name==expected.name;});
    if(found==recipe.external.end()||found->role.description!=expected.description||found->role.identity!=expected.identity)
      throw std::runtime_error("original constant identity/descriptor differs");
    auto slot=found->constant;
    if(resident) {
      const auto origin=recipe.fragment.constants.at(slot);const auto &part=phase(consumer);
      const auto local=std::find_if(part.original_constants.begin(),part.original_constants.end(),[&](Value value){return value.ordinal()==origin.ordinal();});
      if(local==part.original_constants.end())throw std::runtime_error("original constant is absent in actual consumer phase");
      slot=std::size_t(local-part.original_constants.begin());
    } else if(consumer.program!=this)throw std::runtime_error("constant consumer is outside this ordinary Graph");
    const auto supplied=std::find_if(consumer.program->recipe.external.begin(),consumer.program->recipe.external.end(),[&](const auto &entry){return entry.constant==slot;});
    if(supplied==consumer.program->recipe.external.end()||supplied->role.owner.ref()!=owner.ref()||supplied->role.description!=expected.description||supplied->role.identity!=expected.identity)
      throw std::runtime_error("actual Graph lost its original readonly constant owner");
    const auto a=take(supplied->role.lease.view(),"actual readonly bank"),b=take(expected.lease.view(),"once-prepared original readonly bank");
    if(a.resource()!=b.resource()||a.native_handle()!=b.native_handle()||a.offset()!=b.offset()||a.size()!=b.size()||
        supplied->role.lease.generation()!=expected.lease.generation()||a.size()!=take(expected.description.required_bytes(),"original typed bank extent"))
      throw std::runtime_error("actual Graph copied or replaced its original immutable bank");
    return consumer.program->logical.constants.at(slot);
  }
  std::string selected_provider(const Actual &value,g::NodeId node,bool packed) const {
    std::size_t count=0;std::string name;
    for(std::size_t step=0;step<value.program->compiled.report.step_count();++step) {
      const auto selected=take(value.program->compiled.report.step(step),"actual model provider step");
      for(std::size_t index=0;index<selected.node_count;++index)
        if(take(value.program->compiled.report.step_node(step,index),"actual selected logical node")==node) {
          ++count;name=selected.provider.name;
#if defined(HYLE_TEST_QWEN_STATE_CPU)
          if(packed&&!cuda_backend&&selected.provider.id!=c::ggml_cpu::kProviderId)throw std::runtime_error("packed CPU certificate requires its actual ggml route");
#endif
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
          if(packed&&cuda_backend&&selected.provider.id!=c::ggml_cuda::kProviderId)throw std::runtime_error("packed CUDA certificate requires its actual MMVQ route");
#endif
        }
    }
    if(count!=1)throw std::runtime_error("model operation needs exactly one actual compiled provider");return name;
  }
  void report_mixed_vocabulary() const {
    if(!mixed)return;
#if defined(HYLE_TEST_QWEN_STATE_CPU) && defined(HYLE_TEST_QWEN_STATE_CUDA)
    using namespace g::cpu_cuda;
    std::size_t vocabulary_constants=0,device_vocabulary_instances=0;
    for(const auto &entry:recipe.external) {
      if(!mixed->vocabulary_role(entry.role.name))continue;
      ++vocabulary_constants;std::size_t found=0;
      const auto value=logical.constants.at(entry.constant);
      for(std::size_t index=0;index<compiled.report.physical_value_count();++index) {
        const auto physical=take(compiled.report.physical_value(index),"mixed actual vocabulary physical value");
        if(physical.value!=value)continue;
        ++found;device_vocabulary_instances+=physical.storage==kDeviceStorage;
        if(physical.storage!=kHostStorage)throw std::runtime_error("full original vocabulary must have only host physical instances");
      }
      if(!found)throw std::runtime_error("mixed vocabulary lost its actual host constant instance");
    }
    for(std::size_t step=0;step<compiled.report.step_count();++step) {
      const auto selected=take(compiled.report.step(step),"mixed actual selected step");
      // Only a Graph-inserted transfer with no logical node is exempt from
      // compute placement. Actual logical Copies, including pending I32,
      // still pass the exact same required-target guard below.
      if(selected.kind==g::CompileStepKind::kCopy&&selected.node_count==0)continue;
      for(std::size_t port=0;port<selected.node_count;++port) {
        const auto node=take(compiled.report.step_node(step,port),"mixed actual selected node");
        const auto index=std::size_t(node.ordinal());
        if(index>=recipe.fragment.nodes.size())throw std::runtime_error("mixed selected logical node differs");
        if(recipe.fragment.nodes[index].operation.get_if<c::View>())continue;
        if(selected.target!=(node_on_host.at(index)?kCpuTarget:kCudaTarget))
          throw std::runtime_error("mixed selected target violates vocabulary/body placement");
      }
    }
    std::printf("Qwen mixed compile region=%llu vocabulary_constants=%zu device_vocabulary_instances=%zu host_inputs=%zu targets=%zu copies=%zu\n",
        static_cast<unsigned long long>(recipe.fragment.region),vocabulary_constants,device_vocabulary_instances,
        std::size_t(std::count(input_on_host.begin(),input_on_host.end(),true)),compiled.report.target_count(),compiled.report.copy_bytes());
    if(device_vocabulary_instances)throw std::runtime_error("mixed vocabulary acquired a device instance");
    ::testing::Test::RecordProperty("mixed_full_vocabulary_device_instances",int(device_vocabulary_instances));
#else
    throw std::runtime_error("mixed vocabulary report requires both actual CPU and CUDA sources");
#endif
  }
  void report_bfloat_roles(const c::StorageOwner &owner) const {
    if(!recipe.original_bfloat)return;
    if(!recipe.original_bfloat->source||recipe.original||recipe.original24||recipe.embedding_bank||recipe.head_bank||
        recipe.model.storage!=c::DataType::kBFloat16)
      throw std::runtime_error("BF16 source cannot inherit generated or converted parameter identities");
    for(const auto &entry:recipe.external) {
      const auto &expected=recipe.original_bfloat->at(entry.role.name);
      const auto &expected_owner=mixed?mixed->owner_for(expected.name):owner;
      if(entry.role.description!=expected.description||entry.role.identity!=expected.identity||
          entry.role.owner.ref()!=expected_owner.ref()||expected.owner.ref()!=expected_owner.ref()||
          !recipe.fragment.constant_bytes.at(entry.constant).empty())
        throw std::runtime_error("BF16 Graph constant differs from its ready original role");
      const auto supplied=take(entry.role.lease.view(),"BF16 supplied original bank"),
          original=take(expected.lease.view(),"BF16 once-bound original bank");
      if(supplied.resource()!=original.resource()||supplied.native_handle()!=original.native_handle()||
          supplied.offset()!=original.offset()||supplied.size()!=original.size()||
          entry.role.lease.generation()!=expected.lease.generation()||
          supplied.size()!=take(expected.description.required_bytes(),"BF16 original role extent"))
        throw std::runtime_error("BF16 Graph must retain the same original readonly lease");
    }
  }
  void report_original24_roles(const c::StorageOwner &owner) const {
    if(!recipe.original24||!recipe.moe||!resident)return;
    if(recipe.external.size()!=19||recipe.embedding_bank||recipe.head_bank)
      throw std::runtime_error("complete original Q24 needs exactly19 static original banks and no generated vocabulary");
    const auto actual=[&](std::size_t port){return observation(port);};
    const auto local_input=[&](Value origin,const Actual &consumer){return phase_input(origin,consumer);};
    const auto operand=[&](std::size_t port,const Actual &consumer){return observation_input(port,consumer);};
    const auto bank=[&](const orig::ReadonlyRole &expected,const Actual &consumer){return constant_operand(expected,consumer,owner);};
    const auto provider=[&](const Actual &value,g::NodeId node,const char *label,bool packed) {
      auto name=selected_provider(value,node,packed);::testing::Test::RecordProperty(std::string("original24_provider_")+label,name);return name;
    };
    const auto probe=[&](Value origin) {
      const auto found=std::find_if(recipe.moe_ports.begin(),recipe.moe_ports.end(),[&](const auto &entry){return entry.first.ordinal()==origin.ordinal();});
      if(found==recipe.moe_ports.end())throw std::runtime_error("original Q24 MoE observer missing");return found->second;
    };
    const auto &ports=recipe.moe->ports;const auto &roles=*recipe.original24;
    struct Projection final {std::size_t port,input;const orig::ReadonlyRole *role;bool packed;};
    const auto old=[&](std::string_view name)->const orig::ReadonlyRole* {
      const auto found=std::find_if(recipe.external.begin(),recipe.external.end(),[&](const auto &entry){return entry.role.name==name;});
      if(found==recipe.external.end())throw std::runtime_error("original Q24 old K/V/norm supply missing");return &found->role;
    };
    const std::array projections{
      Projection{recipe.eh,recipe.joined,&roles.at(full24::Role::kProjection),true},
      Projection{recipe.qg,recipe.nx,&roles.at(full24::Role::kQuery),true},
      Projection{recipe.key,recipe.nx,old("blk.92.attn_k.weight"),true},Projection{recipe.value,recipe.nx,old("blk.92.attn_v.weight"),true},
      Projection{recipe.oproj,recipe.attended,&roles.at(full24::Role::kOutput),true},
      Projection{recipe.logits,recipe.state_ports[kHidden],&roles.at(full24::Role::kHead),true},
      Projection{probe(ports.logits),recipe.ffn_input,&roles.at(full24::Role::kRouter),false},
      Projection{probe(ports.shared_gate_raw),recipe.ffn_input,&roles.at(full24::Role::kSharedRouter),false},
      Projection{probe(ports.shared.gate),recipe.ffn_input,&roles.at(full24::Role::kSharedGate),true},
      Projection{probe(ports.shared.up),recipe.ffn_input,&roles.at(full24::Role::kSharedUp),true},
      Projection{probe(ports.shared.down),probe(ports.shared.activation),&roles.at(full24::Role::kSharedDown),true}};
    for(const auto &projection:projections) {
      const auto value=actual(projection.port);const auto &graph=value.program->logical.graph;
      const auto node=take(graph.producer(value.value),"original Q24 actual MatMul producer");
      const auto operation=take(graph.operation(node),"original Q24 actual MatMul operation");const auto *matmul=operation.get_if<c::MatMul>();
      auto input=take(graph.node_input(node,0),"original Q24 actual MatMul activation");
      if(projection.port==recipe.oproj) {
        const auto reshape=take(graph.producer(input),"Q24 output head flatten");
        const auto view=take(graph.operation(reshape),"Q24 output flatten operation");
        if(!view.get_if<c::View>()||view.get_if<c::View>()->kind()!=c::ViewKind::kReshape)throw std::runtime_error("Q24 output projection lost its actual head flatten");
        input=take(graph.node_input(reshape,0),"Q24 actual attended heads");
      }
      if(!matmul||matmul->accumulation_type!=c::DataType::kFloat32||input!=operand(projection.input,value)||
          take(graph.node_input(node,1),"original Q24 actual MatMul bank")!=bank(*projection.role,value)||
          take(graph.value_description(value.value),"Q24 projection actual dtype").data_type()!=c::DataType::kFloat32)
        throw std::runtime_error("Q24 original MatMul operands/accumulation/type differ");
      const auto selected_provider=provider(value,node,std::string(projection.role->name).c_str(),projection.packed);
      if(projection.port==recipe.logits)::testing::Test::RecordProperty("full_head_provider",selected_provider);
    }
    const auto head=actual(recipe.logits);
    if(take(head.program->logical.graph.value_description(head.value),"Q24 full head output extent")!=sem::dense({1,kVocabulary})||
        roles.at(full24::Role::kHead).description!=take(c::gguf_weight_description(recipe.model.head.quant,c::DataType::kFloat32,recipe.model.hidden,recipe.model.vocabulary),"Q24 actual full original head"))
      throw std::runtime_error("Q24 head must be a single complete original vocabulary MatMul");
    const auto embedding=actual(recipe.embedding);const auto &embedding_graph=embedding.program->logical.graph;
    const auto lookup=take(embedding_graph.producer(embedding.value),"Q24 original embedding Gather");
    if(take(embedding_graph.operation_kind(lookup),"Q24 embedding operation")!=c::OperationKind::kGather||
        take(embedding_graph.node_input(lookup,0),"Q24 actual embedding table")!=bank(roles.at(full24::Role::kEmbedding),embedding)||
        take(embedding_graph.node_input(lookup,1),"Q24 actual pending ID")!=local_input(recipe.fragment.inputs[kPending],embedding))
      throw std::runtime_error("Q24 full embedding must consume its actual pending lease and independent original bank");
    const auto gamma=[&](std::size_t port,const orig::ReadonlyRole &expected,bool rotary=false) {
      const auto value=actual(port);const auto &graph=value.program->logical.graph;
      auto normalized=value.value;
      if(rotary) {
        const auto rope=take(graph.producer(normalized),"Q24 per-head RoPE producer");
        if(take(graph.operation_kind(rope),"Q24 actual RoPE operation")!=c::OperationKind::kRoPE)throw std::runtime_error("Q24 per-head norm lost actual RoPE boundary");
        normalized=take(graph.node_input(rope,0),"Q24 actual per-head norm");
      }
      const auto node=take(graph.producer(normalized),"Q24 actual original RMS producer");
      const auto operation=take(graph.operation(node),"Q24 actual original RMS operation");const auto *rms=operation.get_if<c::RmsNorm>();
      if(!rms||rms->epsilon!=1e-6F||take(graph.node_input(node,1),"Q24 actual original gamma")!=bank(expected,value))
        throw std::runtime_error("Q24 RMS must directly borrow its original GGUF gamma without adding1");
    };
    gamma(recipe.hn,roles.at(full24::Role::kHiddenNorm));gamma(recipe.en,roles.at(full24::Role::kEmbeddingNorm));
    gamma(recipe.nx,roles.at(full24::Role::kAttentionNorm));gamma(recipe.ffn_input,roles.at(full24::Role::kPostNorm));
    gamma(recipe.state_ports[kHidden],roles.at(full24::Role::kHeadNorm));
    gamma(recipe.query,roles.at(full24::Role::kQueryNorm),true);gamma(recipe.rotated_key,*old("blk.92.attn_k_norm.weight"),true);
    const auto local_ids=actual(probe(ports.local_ids));const auto &mapped_graph=local_ids.program->logical.graph;
    const auto gather=take(mapped_graph.producer(local_ids.value),"Q24 global-to-local mapping producer");
    if(take(mapped_graph.operation_kind(gather),"Q24 mapping operation")!=c::OperationKind::kGather||
        take(mapped_graph.node_input(gather,0),"Q24 actual generation map")!=local_input(recipe.moe->boundary.map,local_ids)||
        take(mapped_graph.node_input(gather,1),"Q24 actual full512 expert IDs")!=operand(probe(ports.ids),local_ids))
      throw std::runtime_error("Q24 mapping must consume actual global IDs without modulo or domain filtering");
    const std::array grouped_ports{ports.routed.gate,ports.routed.up,ports.routed.down};
    for(std::size_t leg=0;leg<3;++leg) {
      const auto value=actual(probe(grouped_ports[leg]));const auto &graph=value.program->logical.graph;
      const auto node=take(graph.producer(value.value),"Q24 original compact grouped producer");
      const auto operation=take(graph.operation(node),"Q24 original grouped operation");
      const auto grouped=operation.custom_as<c::model_ops::GroupedMatMul>();
      auto input=take(graph.node_input(node,1),"Q24 actual grouped activation");
      if(leg<2) {
        const auto reshape=take(graph.producer(input),"Q24 routed input slot reshape");
        const auto view=take(graph.operation(reshape),"Q24 routed input slot operation");
        if(!view.get_if<c::View>()||view.get_if<c::View>()->kind()!=c::ViewKind::kReshape)throw std::runtime_error("Q24 original routed input requires its actual shared activation");
        input=take(graph.node_input(reshape,0),"Q24 original routed source activation");
      }
      if(!grouped||take(graph.node_input(node,0),"Q24 actual compact original bank")!=local_input(recipe.moe->boundary.banks[leg],value)||
          input!=operand(probe(leg<2?ports.input:ports.routed.activation),value)||
          take(graph.node_input(node,2),"Q24 actual checked local IDs")!=operand(probe(ports.local_ids),value)||
          take(graph.value_description(take(graph.node_input(node,0),"Q24 compact descriptor operand")),"Q24 compact descriptor")!=roles.source->resident_description(static_cast<full24::Leg>(leg)))
        throw std::runtime_error("Q24 original experts must use canonical E10 inputs with actual checked mapping");
      provider(value,node,(std::string("expert_")+std::to_string(leg)).c_str(),true);
    }
    ::testing::Test::RecordProperty("full_head_encoded_bytes",std::to_string(take(roles.at(full24::Role::kHead).description.required_bytes(),"Q24 full packed head bytes")));
  }
  void report_original_roles(const c::StorageOwner &owner) const {
    if(!recipe.original)return;
    const auto &source=*recipe.original;
    if(recipe.external.size()!=17 || recipe.embedding_bank || recipe.head_bank)
      throw std::runtime_error("complete original Q27 must have exactly17 original banks and no generated vocabulary");
    const auto actual=[&](std::size_t port) {
      const auto copy=take(logical.graph.producer(logical.outputs.at(port)),"original Q27 independent observation");
      if(take(logical.graph.operation_kind(copy),"original Q27 observer kind")!=c::OperationKind::kCopy)
        throw std::runtime_error("original Q27 observations must own independent Copy outputs");
      return take(logical.graph.node_input(copy,0),"original Q27 actual observed value");
    };
    const auto ready=[&](const orig::ReadonlyRole &expected) {
      const auto found=std::find_if(recipe.external.begin(),recipe.external.end(),[&](const auto &entry){return entry.role.name==expected.name;});
      if(found==recipe.external.end()||found->role.description!=expected.description||found->role.owner.ref()!=owner.ref()||
          found->role.identity!=&source.source->identity()||!recipe.fragment.constant_bytes.at(found->constant).empty())
        throw std::runtime_error("original Q27 constant must borrow its one authenticated ready owner");
      const auto supplied=take(found->role.lease.view(),"original Q27 supplied readonly bank");
      const auto original=take(expected.lease.view(),"original Q27 once-bound readonly bank");
      if(supplied.resource()!=original.resource()||supplied.native_handle()!=original.native_handle()||
          supplied.offset()!=original.offset()||supplied.size()!=original.size()||
          found->role.lease.generation()!=expected.lease.generation()||
          supplied.size()!=take(expected.description.required_bytes(),"original Q27 compressed bank bytes"))
        throw std::runtime_error("original Q27 ready bank identity or compressed extent changed");
      return logical.constants.at(found->constant);
    };
    for(std::size_t slot=0;slot<full::kRoles;++slot)(void)ready(source.at(full::role_name(static_cast<full::Role>(slot))));
    for(const auto name:full::kInputRoleNames)(void)ready(source.at(name));
    const auto norm=[&](std::size_t port,full::Role role) {
      const auto node=take(logical.graph.producer(actual(port)),"original Q27 RMS producer");
      const auto op=take(logical.graph.operation(node),"original Q27 RMS operation");
      const auto *rms=op.get_if<c::RmsNorm>();
      if(!rms||rms->epsilon!=1e-6F||take(logical.graph.node_input(node,1),"original Q27 RMS gamma")!=ready(source.at(full::role_name(role))))
        throw std::runtime_error("original Q27 RMS must use its actual GGUF gamma without an offset");
    };
    norm(recipe.nx,full::Role::kAttentionNorm);norm(recipe.ffn_input,full::Role::kPostNorm);
    norm(recipe.state_ports[kHidden],full::Role::kHeadNorm);
    for(const auto &[port,role]:std::array{std::pair{recipe.query,full::Role::kQueryNorm},std::pair{recipe.rotated_key,full::Role::kKeyNorm}}) {
      const auto rope=take(logical.graph.producer(actual(port)),"original Q27 RoPE producer");
      const auto normalized=take(logical.graph.node_input(rope,0),"original Q27 actual per-head RMS value");
      const auto producer=take(logical.graph.producer(normalized),"original Q27 per-head RMS producer");
      const auto op=take(logical.graph.operation(producer),"original Q27 per-head RMS operation");
      const auto *rms=op.get_if<c::RmsNorm>();
      if(!rms||rms->epsilon!=1e-6F||take(logical.graph.node_input(producer,1),"original Q27 per-head original gamma")!=ready(source.at(full::role_name(role))))
        throw std::runtime_error("original Q27 Q/K RMS gamma source changed");
    }
    const auto lookup=take(logical.graph.producer(actual(recipe.embedding)),"original Q27 full embedding Gather");
    if(take(logical.graph.operation_kind(lookup),"original Q27 embedding operation")!=c::OperationKind::kGather||
        take(logical.graph.node_input(lookup,0),"original Q27 full embedding bank")!=ready(source.at(full::role_name(full::Role::kEmbedding)))||
        take(logical.graph.node_input(lookup,1),"original Q27 actual pending input")!=logical.inputs[kPending])
      throw std::runtime_error("original Q27 Gather must consume the actual complete-vocabulary pending lease");
    const auto original_eh=std::find_if(recipe.external.begin(),recipe.external.end(),[](const auto &entry){
      return entry.role.name=="blk.64.nextn.eh_proj.weight";
    });
    if(original_eh==recipe.external.end())throw std::runtime_error("complete Q27 must keep its actual original EH");
    struct Projection final {std::size_t port,input;const orig::ReadonlyRole *role;};
    const std::array projections{
      Projection{recipe.eh,recipe.joined,&original_eh->role},
      Projection{recipe.qg,recipe.nx,&source.at(full::role_name(full::Role::kQuery))},Projection{recipe.key,recipe.nx,&source.at(full::role_name(full::Role::kKey))},
      Projection{recipe.value,recipe.nx,&source.at(full::role_name(full::Role::kValue))},Projection{recipe.oproj,recipe.attended,&source.at(full::role_name(full::Role::kOutput))},
      Projection{recipe.dense_ffn_ports[0],recipe.ffn_input,&source.at(full::role_name(full::Role::kFfnGate))},
      Projection{recipe.dense_ffn_ports[1],recipe.ffn_input,&source.at(full::role_name(full::Role::kFfnUp))},
      Projection{recipe.ffn_output,recipe.dense_ffn_ports[2],&source.at(full::role_name(full::Role::kFfnDown))},
      Projection{recipe.logits,recipe.state_ports[kHidden],&source.at(full::role_name(full::Role::kHead))}};
    for(const auto &projection:projections) {
      const auto node=take(logical.graph.producer(actual(projection.port)),"original Q27 actual packed MatMul");
      auto input=take(logical.graph.node_input(node,0),"original Q27 projection actual input");
      const auto operation=take(logical.graph.operation(node),"original Q27 typed projection operation");
      const bool bfloat=attention::bfloat_weight(projection.role->description);
      if(bfloat) {
        if(!operation.custom_as<c::model_ops::DenseLinearF32>())throw std::runtime_error("original BF16 bank needs its declared F32-accumulating projection");
        const auto cast=take(logical.graph.producer(input),"original BF16 actual activation Cast");
        const auto cast_operation=take(logical.graph.operation(cast),"original BF16 activation Cast operation");
        const auto *conversion=cast_operation.get_if<c::Cast>();
        if(!conversion||conversion->output_type!=c::DataType::kBFloat16)
          throw std::runtime_error("original BF16-W/F32-state policy must retain its actual BF16 activation store");
        input=take(logical.graph.node_input(cast,0),"original BF16 activation actual predecessor");
      }
      if(projection.port==recipe.oproj) {
        const auto reshape=take(logical.graph.producer(input),"original Q27 head flatten producer");
        const auto op=take(logical.graph.operation(reshape),"original Q27 head flatten operation");
        const auto *view=op.get_if<c::View>();
        if(!view||view->kind()!=c::ViewKind::kReshape)throw std::runtime_error("original Q27 output projection must flatten actual heads");
        input=take(logical.graph.node_input(reshape,0),"original Q27 actual attended heads");
      }
      if((!bfloat&&take(logical.graph.operation_kind(node),"original Q27 projection operation")!=c::OperationKind::kMatMul)||
          input!=actual(projection.input)||take(logical.graph.node_input(node,1),"original Q27 actual original RHS")!=ready(*projection.role))
        throw std::runtime_error("original Q27 projection actual operand/bank wiring changed");
      std::size_t count=0;std::string provider;
      for(std::size_t step=0;step<compiled.report.step_count();++step) {
        const auto selected=take(compiled.report.step(step),"original Q27 selected projection provider");
        for(std::size_t index=0;index<selected.node_count;++index)
          if(take(compiled.report.step_node(step,index),"original Q27 selected projection node")==node) {
            ++count;provider=selected.provider.name;
#if defined(HYLE_TEST_QWEN_STATE_CPU)
            if(!cuda_backend&&!bfloat&&selected.provider.id!=c::ggml_cpu::kProviderId)
              throw std::runtime_error("original Q27 CPU certificate requires its actual native ggml AVX2 projection route");
#endif
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
            if(cuda_backend&&!bfloat&&selected.provider.id!=c::ggml_cuda::kProviderId)
              throw std::runtime_error("original Q27 CUDA certificate requires its actual native ggml MMVQ projection route");
#endif
          }
      }
      if(count!=1)throw std::runtime_error("original Q27 projection must have one actual selected provider");
      ::testing::Test::RecordProperty(std::string("original_provider_")+std::string(projection.role->name),provider);
    }
  }
  std::vector<c::ReadLease> execute(c::Queue &queue,const State &state,const c::StateVersion &from) {
    const auto inputs=state.inputs(from);
    const auto cursor=typed<std::int32_t>(inputs[kCursor]).at(0),position=typed<std::int32_t>(inputs[kPosition]).at(0);
    const auto pending=typed<std::int32_t>(inputs[kPending]).at(0);
    if(cursor<0 || cursor>=kCapacity || position<0 || position==std::numeric_limits<std::int32_t>::max() || pending<0 || pending>=recipe.model.vocabulary)
      throw std::runtime_error("complete NextN boundary exceeds prepared capacity/position/vocabulary");
    if(resident)return execute_resident(queue,inputs);
    return execute_inputs(queue,inputs);
  }
  std::vector<c::ReadLease> execute_inputs(c::Queue &queue,std::span<const c::ReadLease> inputs,
      std::span<const std::size_t> externally_bound={},const std::function<void(g::Invocation&)> &bind_external={}) {
    if(resident) {
      if(!externally_bound.empty()||bind_external)throw std::runtime_error("resident scheduling owns its actual batch binding");
      return execute_resident(queue,inputs);
    }
    if(inputs.size()!=logical.inputs.size())throw std::runtime_error("ordinary Qwen Graph input count");
    std::vector<bool> external_ports(inputs.size(),false);
    for(const auto index:externally_bound) {
      if(index>=inputs.size()||external_ports[index])throw std::runtime_error("resident Q24 input index");
      external_ports[index]=true;
    }
    if(!externally_bound.empty()&&!bind_external)throw std::runtime_error("resident Q24 generation binder missing");
    struct ResetOnFailure final {g::Invocation &active;bool done=false;~ResetOnFailure(){if(!done)active.reset().IgnoreError();}} reset{invocation};
    for(std::size_t i=0;i<inputs.size();++i)if(!external_ports[i])
      check(invocation.bind(logical.inputs[i],recipe.fragment.description(recipe.fragment.inputs[i]),inputs[i]),"bind actual state lease without host reconstruction");
    if(bind_external)bind_external(invocation);
    for(std::size_t i=0;i<logical.outputs.size();++i) {
      const auto &desc=recipe.fragment.description(recipe.fragment.outputs[i]);
      auto &pool=mixed&&output_on_host.at(i)?host_outputs:outputs;
      check(invocation.bind(logical.outputs[i],desc,take(pool.allocate(take(desc.required_bytes(),"NextN output extent")),"new result generation writer")),"bind actual NextN output");
    }
    for(const auto port:caller_queue_ports)required_queue_bindings[port].queue=&queue;
    const std::span<const g::QueueBinding> queues(required_queue_bindings);
    check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),"prepare connected NextN replay");
    {auto done=take(invocation.submit(queues),"complete NextN Graph submit");++submissions;check(done.wait(),"actual NextN retirement");}
    std::vector<c::ReadLease> values;
    for(const auto output:logical.outputs)values.push_back(take(take(invocation.result(output),"successful NextN Graph result").storage_lease(),"independent actual result backing"));
    check(invocation.reset(),"withdraw completed invocation permissions");reset.done=true;
    if(invocation.submit(queues))throw std::runtime_error("reset NextN invocation still prepared");
    return values;
  }
};

void quantized(std::span<const float> actual,std::span<const float> expected,const char *label) {
  SCOPED_TRACE(label);ASSERT_EQ(actual.size(),expected.size());
  float peak=0;for(const auto value:expected){ASSERT_TRUE(std::isfinite(value));peak=std::max(peak,std::abs(value));}
  for(const auto value:actual)ASSERT_TRUE(std::isfinite(value));
  if(peak==0){for(const auto value:actual)EXPECT_EQ(value,0.F);return;}
  std::vector<float> a(actual.size()),e(expected.size());
  for(std::size_t i=0;i<a.size();++i){a[i]=actual[i]/peak;e[i]=expected[i]/peak;}
  // Same peak-normalized quantized-chain contract as the original Q27/Q24
  // components and C5 consumers. Strict local F32 operations stay separate.
  d5_gguf_test::expect_quantized_chain_matmul_close(a,e);
}
std::vector<float> attention_reference(const Model &model,std::span<const float> query,
    std::span<const float> key,std::span<const float> value,std::span<const float> gate,int rows) {
  std::vector<float> result(std::size_t(model.heads*kDepth));
  for(std::int64_t head=0;head<model.heads;++head) {
    const auto kv=head/(model.heads/kKvHeads);std::vector<double> logits(std::size_t(rows),0);
    for(int row=0;row<rows;++row)for(std::int64_t d=0;d<kDepth;++d)
      logits[std::size_t(row)]+=double(query[std::size_t(head*kDepth+d)])*key[std::size_t((row*kKvHeads+kv)*kDepth+d)]/16.;
    const auto maximum=*std::max_element(logits.begin(),logits.end());double sum=0;
    for(auto &score:logits)sum+=score=std::exp(score-maximum);
    for(std::int64_t d=0;d<kDepth;++d) {
      double total=0;for(int row=0;row<rows;++row)total+=logits[std::size_t(row)]/sum*value[std::size_t((row*kKvHeads+kv)*kDepth+d)];
      const auto i=std::size_t(head*kDepth+d);result[i]=float(total/(1+std::exp(-double(gate[i]))));
    }
  }
  return result;
}
void observe_original24_moe(Program &program,std::span<const c::ReadLease> output,
    original24_reference::ActivationCodec codec,bool full_check) {
  const auto &r=program.recipe;const auto &ports=r.moe->ports;const auto &source=*r.original24->source;
  const auto port=[&](Value value) {
    const auto found=std::find_if(r.moe_ports.begin(),r.moe_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found==r.moe_ports.end())throw std::runtime_error("actual Q24 MoE observation missing");return found->second;
  };
  const auto get=[&](Value value){return typed<float>(output[port(value)]);};
  const auto matrix=[&](full24::Role role) {
    const auto &spec=source.spec(role);return original24_reference::PackedMatrix{source.bytes(role),std::size_t(spec.n),std::size_t(spec.k),spec.native};
  };
  const moe_reference::Parameters parameters{source.bytes(full24::Role::kRouter),source.bytes(full24::Role::kSharedRouter),
      {matrix(full24::Role::kSharedGate),matrix(full24::Role::kSharedUp),matrix(full24::Role::kSharedDown)}};
  const auto integers=[&](Value value){return typed<std::int32_t>(output[port(value)]);};
  const auto expert=[&](std::size_t leg,std::int32_t global) {
    const auto value=static_cast<full24::Leg>(leg);const auto &spec=source.spec(full24::expert_role(value));
    return original24_reference::PackedMatrix{source.expert_bytes(value,global),std::size_t(spec.n),std::size_t(spec.k),spec.native};
  };
  moe_reference::observe(parameters,ports,get,integers,expert,codec,full_check,program.original_full_values);
  ASSERT_FALSE(::testing::Test::HasFailure());
  const auto ids=integers(ports.ids),mapping=integers(ports.map);
  ASSERT_TRUE(program.supply);ASSERT_EQ(program.supply->episodes,std::size_t(program.original_episodes)+1);
  ASSERT_FALSE(program.supply->selections.empty());
  EXPECT_EQ(mapping,std::vector<std::int32_t>(program.supply->selections.back().map.begin(),program.supply->selections.back().map.end()));
  for(const auto global:ids)EXPECT_TRUE(std::binary_search(program.supply->selections.back().globals.begin(),program.supply->selections.back().globals.end(),global));
}
std::size_t bfloat_port(const Recipe &recipe,Value value) {
  for(const auto *ports:{&recipe.bfloat_ports,&recipe.moe_ports}) {
    const auto found=std::find_if(ports->begin(),ports->end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found!=ports->end())return found->second;
  }
  throw std::runtime_error("BF16 observation is absent from actual output leases");
}
const orig::ReadonlyRole &bfloat_constant(const Recipe &recipe,Value value) {
  const auto found=std::find_if(recipe.external.begin(),recipe.external.end(),[&](const auto &entry){
    return recipe.fragment.constants.at(entry.constant).ordinal()==value.ordinal();
  });
  if(found==recipe.external.end())throw std::runtime_error("BF16 observer lost its original constant binding");return found->role;
}
std::pair<const orig::ReadonlyRole*,bool> bfloat_gamma_source(const Program &program,Value value) {
  const auto &recipe=program.recipe;
  const auto producer=[&](Value output)->const d5_qwen38_24t_full_router::DirectNode& {
    const auto found=std::find_if(recipe.fragment.nodes.begin(),recipe.fragment.nodes.end(),[&](const auto &node){
      return node.outputs.size()==1&&node.outputs[0].ordinal()==output.ordinal();
    });
    if(found==recipe.fragment.nodes.end())throw std::runtime_error("BF16 gamma has no actual producer");return *found;
  };
  const auto *node=&producer(value);bool delta=false;
  if(const auto *binary=node->operation.get_if<c::Binary>()) {
    if(binary->kind!=c::BinaryKind::kAdd||node->inputs.size()!=2)
      throw std::runtime_error("BF16 gamma is not the declared source conversion");
    const auto one=std::find_if(recipe.fragment.constants.begin(),recipe.fragment.constants.end(),[&](Value constant){return constant.ordinal()==node->inputs[1].ordinal();});
    if(one==recipe.fragment.constants.end()||recipe.fragment.description(*one)!=sem::dense({1}))
      throw std::runtime_error("BF16 delta norm must add the exact F32 one");
    const auto slot=std::size_t(one-recipe.fragment.constants.begin());std::size_t generated=0;
    for(std::size_t i=0;i<=slot;++i) {
      const bool external=std::any_of(recipe.external.begin(),recipe.external.end(),[&](const auto &entry){return entry.constant==i;});
      if(i==slot) {
        // Cold payloads were released after binding. Inspect the same live
        // immutable scalar lease used by this Program, not the empty recipe.
        if(external||typed<float>(program.parameters.at(generated))!=std::vector<float>{1.F})
          throw std::runtime_error("BF16 delta norm must add the exact resident F32 one");
      } else if(!external)++generated;
    }
    delta=true;node=&producer(node->inputs[0]);
  }
  const auto *convert=node->operation.get_if<c::Cast>();
  if(!convert||convert->output_type!=c::DataType::kFloat32||node->inputs.size()!=1)
    throw std::runtime_error("BF16 gamma must exactly widen its original BF16 coefficient");
  return {&bfloat_constant(recipe,node->inputs[0]),delta};
}
bool bfloat_rule(const Recipe &recipe,std::string_view role) {
  if(!recipe.bfloat_rules)throw std::runtime_error("BF16 source norm conventions are absent");
  const auto &rules=*recipe.bfloat_rules;
  const auto rule=role.ends_with("pre_fc_norm_hidden.weight")?rules.hidden:
      role.ends_with("pre_fc_norm_embedding.weight")?rules.embedding:
      role.ends_with("input_layernorm.weight")?rules.input:
      role.ends_with("q_norm.weight")?rules.query:role.ends_with("k_norm.weight")?rules.key:
      role.ends_with("post_attention_layernorm.weight")?rules.post:rules.head;
  return rule==bfloat::NormConvention::kDeltaPlusOne;
}
void observe_bfloat_trace(Program &program,std::span<const c::ReadLease> output,const attention::LocalTrace &trace,bool full) {
  const auto &recipe=program.recipe;const auto &source=*recipe.original_bfloat->source;
  const auto bits=[&](Value value){return typed<std::uint16_t>(output[bfloat_port(recipe,value)]);};
  const auto floats=[&](Value value){return typed<float>(output[bfloat_port(recipe,value)]);};
  for(const auto &projection:trace.projections) {
    const auto &role=bfloat_constant(recipe,projection.weight);const auto &description=role.description;
    ASSERT_TRUE(bfloat_reference::expect_projection(floats(projection.raw),bits(projection.input),source.bytes(role.name),
        std::size_t(description.dimensions()[0]),std::size_t(description.dimensions()[1]),role.name,full));
    ASSERT_TRUE(bfloat_reference::expect_store(bits(projection.stored),floats(projection.raw),role.name));
    if(full)program.original_full_values+=floats(projection.raw).size();
  }
  for(const auto &norm:trace.norms) {
    const auto [role,delta]=bfloat_gamma_source(program,norm.gamma);ASSERT_EQ(delta,bfloat_rule(recipe,role->name));
    const auto gamma=floats(norm.gamma);
    ASSERT_TRUE(bfloat_reference::expect_gamma(gamma,source.norm(role->name),delta,role->name));
    ASSERT_TRUE(bfloat_reference::expect_norm(floats(norm.raw),bits(norm.stored),bits(norm.input),gamma,role->name));
  }
}
void observe_bfloat_rotary(const Model &model,std::span<const std::uint16_t> actual,
    std::span<const std::uint16_t> input,std::int32_t position) {
  ASSERT_EQ(actual.size(),input.size());ASSERT_EQ(input.size()%std::size_t(model.depth),0u);
  const auto frequencies=text_frequencies(model);const auto pairs=std::size_t(model.rotary/2);
  for(std::size_t row=0;row<input.size()/std::size_t(model.depth);++row) {
    const auto first=row*std::size_t(model.depth);
    for(std::size_t pair=0;pair<pairs;++pair) {
      const double angle=double(position)*frequencies[pair],cosine=std::cos(angle),sine=std::sin(angle);
      const double a=bfloat_reference::widen(input[first+pair]),b=bfloat_reference::widen(input[first+pairs+pair]);
      ASSERT_TRUE(bfloat_reference::expect_local(actual[first+pair],a*cosine-b*sine,"actual BF16 F32-fused RoPE"));
      ASSERT_TRUE(bfloat_reference::expect_local(actual[first+pairs+pair],a*sine+b*cosine,"actual BF16 F32-fused RoPE"));
    }
    for(std::size_t column=std::size_t(model.rotary);column<std::size_t(model.depth);++column)EXPECT_EQ(actual[first+column],input[first+column]);
  }
}
void observe_bfloat_attention(Program &program,std::span<const c::ReadLease> output,
    const attention::TargetAttentionPorts &ports,const attention::LocalTrace &trace,
    const c::ReadLease &old_key,const c::ReadLease &old_value,std::int32_t cursor,std::int32_t position) {
  const auto &r=program.recipe;const auto &model=r.model;
  const auto bits=[&](Value value){return typed<std::uint16_t>(output[bfloat_port(r,value)]);};
  const auto get=[&](std::size_t port){return typed<std::uint16_t>(output[port]);};
  ASSERT_TRUE(trace.query_rotary_input);ASSERT_TRUE(trace.key_rotary_input);ASSERT_TRUE(trace.sdpa);ASSERT_TRUE(trace.gate_sigmoid);
  observe_bfloat_rotary(model,get(r.query),bits(*trace.query_rotary_input),position);
  observe_bfloat_rotary(model,get(r.rotated_key),bits(*trace.key_rotary_input),position);
  const auto qg=get(r.qg),gate=get(r.gate);
  for(const auto &norm:trace.norms) {
    const auto [role,delta]=bfloat_gamma_source(program,norm.gamma);(void)delta;
    if(role->name.ends_with("q_norm.weight")) {
      std::vector<std::uint16_t> expected;expected.reserve(std::size_t(model.heads*model.depth));
      for(std::int64_t head=0;head<model.heads;++head)
        expected.insert(expected.end(),qg.begin()+std::ptrdiff_t(head*2*model.depth),qg.begin()+std::ptrdiff_t(head*2*model.depth+model.depth));
      EXPECT_EQ(bits(norm.input),expected);
    } else if(role->name.ends_with("k_norm.weight"))EXPECT_EQ(bits(norm.input),get(r.key));
  }
  for(std::size_t head=0;head<std::size_t(model.heads);++head)
    for(std::size_t column=0;column<std::size_t(model.depth);++column)
      EXPECT_EQ(gate[head*std::size_t(model.depth)+column],qg[head*std::size_t(2*model.depth)+std::size_t(model.depth)+column]);
  const std::array old{old_key,old_value};
  for(std::size_t entry=0;entry<old.size();++entry) {
    auto expected=read(old[entry]);const auto fresh=read(output[entry==0?r.rotated_key:r.value]);
    const auto row_bytes=std::size_t(model.kv_heads*model.depth)*2;
    ASSERT_GE(cursor,0);ASSERT_LT(cursor,kCapacity);ASSERT_EQ(fresh.size(),row_bytes);
    std::copy(fresh.begin(),fresh.end(),expected.begin()+std::ptrdiff_t(std::size_t(cursor)*row_bytes));
    EXPECT_EQ(read(output[r.state_ports[entry]]),expected);
  }
  const auto query=get(r.query),keys=get(r.state_ports[kKey]),values=get(r.state_ports[kValue]);
  const auto context=bits(*trace.sdpa);const auto scale=model.score_scale==0?1/std::sqrt(double(model.depth)):double(model.score_scale);
  for(std::int64_t head=0;head<model.heads;++head) {
    const auto kv=head/(model.heads/model.kv_heads);std::vector<double> scores(std::size_t(cursor)+1,0.);
    for(std::int32_t row=0;row<=cursor;++row)for(std::int64_t column=0;column<model.depth;++column)
      scores[std::size_t(row)]+=double(bfloat_reference::widen(query[std::size_t(head*model.depth+column)]))*
          bfloat_reference::widen(keys[std::size_t((row*model.kv_heads+kv)*model.depth+column)])*scale;
    const auto maximum=*std::max_element(scores.begin(),scores.end());double total=0.;
    for(auto &score:scores)total+=score=std::exp(score-maximum);
    for(std::int64_t column=0;column<model.depth;++column) {
      double center=0.;for(std::int32_t row=0;row<=cursor;++row)
        center+=scores[std::size_t(row)]/total*bfloat_reference::widen(values[std::size_t((row*model.kv_heads+kv)*model.depth+column)]);
      ASSERT_TRUE(bfloat_reference::expect_local(context[std::size_t(head*model.depth+column)],center,"BF16 finite GQA SDPA"));
    }
  }
  const auto sigmoid=bits(*trace.gate_sigmoid);
  ASSERT_TRUE(bfloat_reference::expect_unary(sigmoid,gate,false,"BF16 Q gate sigmoid"));
  ASSERT_TRUE(bfloat_reference::expect_binary(get(r.attended),context,sigmoid,true,"BF16 gated SDPA"));
  ASSERT_TRUE(bfloat_reference::expect_binary(get(r.residual),bits(ports.input),get(r.oproj),false,"BF16 attention residual"));
}
void observe_bfloat(Program &program,const State &state,const c::StateVersion &from,std::span<const c::ReadLease> output) {
  const auto &r=program.recipe;const auto &model=r.model;
  if(!r.original_bfloat||(!r.bfloat_dense&&!r.bfloat_moe))
    throw std::runtime_error("BF16 observation needs its actual original dense or selected-MoE result");
  const bool full=!program.original_full_checked;const auto &source=*r.original_bfloat->source;
  const auto bits=[&](Value value){return typed<std::uint16_t>(output[bfloat_port(r,value)]);};
  const auto floats=[&](Value value){return typed<float>(output[bfloat_port(r,value)]);};
  const auto get=[&](std::size_t port){return typed<std::uint16_t>(output[port]);};
  const auto token=typed<std::int32_t>(state.member(from,kPending));
  const auto &prefix=r.bfloat_dense?r.bfloat_dense->prefix.ports:r.bfloat_moe->prefix.ports;
  ASSERT_TRUE(bfloat_reference::expect_embedding(get(r.embedding),source.bytes(r.bfloat_vocabulary.embedding),token,
      std::size_t(model.vocabulary),std::size_t(model.hidden)));
  const std::array input{typed<std::uint16_t>(state.member(from,kHidden)),get(r.embedding)};
  const std::array norm_names{bfloat::kPrefixRoleNames[0],bfloat::kPrefixRoleNames[1]};
  const std::array stored{prefix.input.hidden,prefix.input.embedding};
  for(std::size_t slot=0;slot<2;++slot) {
    const auto delta=bfloat_rule(r,norm_names[slot]);const auto gamma=floats(prefix.pre_gamma[slot]);
    ASSERT_TRUE(bfloat_reference::expect_gamma(gamma,source.norm(norm_names[slot]),delta,norm_names[slot]));
    ASSERT_TRUE(bfloat_reference::expect_norm(floats(prefix.pre_norm_raw[slot]),bits(stored[slot]),input[slot],gamma,norm_names[slot]));
  }
  auto joined=bits(prefix.input.embedding);const auto hn=bits(prefix.input.hidden);joined.insert(joined.end(),hn.begin(),hn.end());
  EXPECT_EQ(bits(prefix.input.joined),joined);
  ASSERT_TRUE(bfloat_reference::expect_projection(floats(prefix.projection_raw),joined,source.bytes(bfloat::kPrefixRoleNames[2]),
      std::size_t(model.hidden),std::size_t(2*model.hidden),"original BF16 EH",full));
  ASSERT_TRUE(bfloat_reference::expect_store(bits(prefix.projection_stored),floats(prefix.projection_raw),"actual BF16 EH store"));
  const auto gamma=floats(prefix.input_gamma);
  ASSERT_TRUE(bfloat_reference::expect_gamma(gamma,source.norm(bfloat::kPrefixRoleNames[3]),bfloat_rule(r,bfloat::kPrefixRoleNames[3]),"BF16 input norm gamma"));
  ASSERT_TRUE(bfloat_reference::expect_norm(floats(prefix.input_norm_raw),bits(prefix.normalized),bits(prefix.projection_stored),gamma,"BF16 input norm"));
  const auto &maybe_trace=r.bfloat_dense?r.bfloat_dense->block.trace:r.bfloat_moe->attention.trace;
  ASSERT_TRUE(maybe_trace);const auto &trace=*maybe_trace;
  const auto &attention_ports=r.bfloat_dense?static_cast<const attention::TargetAttentionPorts&>(r.bfloat_dense->block.ports):r.bfloat_moe->attention.ports;
  observe_bfloat_trace(program,output,trace,full);ASSERT_FALSE(::testing::Test::HasFailure());
  const auto cursor=typed<std::int32_t>(state.member(from,kCursor)).at(0),position=typed<std::int32_t>(state.member(from,kPosition)).at(0);
  observe_bfloat_attention(program,output,attention_ports,trace,state.member(from,kKey),state.member(from,kValue),cursor,position);
  ASSERT_FALSE(::testing::Test::HasFailure());
  if(r.bfloat_moe) {
    ASSERT_TRUE(program.supply);ASSERT_TRUE(program.supply->source.owner);
    const auto parameters=bfloat_moe::from_source(source,"mtp.layers.0.mlp.gate.weight","mtp.layers.0.mlp.shared_expert_gate.weight",
        std::array<std::string_view,3>{"mtp.layers.0.mlp.shared_expert.gate_proj.weight","mtp.layers.0.mlp.shared_expert.up_proj.weight","mtp.layers.0.mlp.shared_expert.down_proj.weight"});
    const auto actual=[&](Value value){return read(output[bfloat_port(r,value)]);};
    bfloat_moe::observe(r.fragment,parameters,r.bfloat_moe->moe,bfloat_moe::linear_stores(r.fragment,r.bfloat_moe->moe),
        program.supply->source,actual,full,program.original_full_values);
    ASSERT_FALSE(::testing::Test::HasFailure());
    observe_bfloat_trace(program,output,r.bfloat_moe->tail_trace,full);ASSERT_FALSE(::testing::Test::HasFailure());
    ASSERT_EQ(program.supply->episodes,std::size_t(program.submissions));ASSERT_FALSE(program.supply->selections.empty());
    EXPECT_EQ(typed<std::int32_t>(output[bfloat_port(r,r.bfloat_moe->moe.ports.values.map)]),
        std::vector<std::int32_t>(program.supply->selections.back().map.begin(),program.supply->selections.back().map.end()));
  } else {
    ASSERT_TRUE(trace.ffn_silu);
    ASSERT_TRUE(bfloat_reference::expect_unary(bits(*trace.ffn_silu),get(r.dense_ffn_ports[0]),true,"BF16 dense SiLU"));
    ASSERT_TRUE(bfloat_reference::expect_binary(get(r.dense_ffn_ports[2]),bits(*trace.ffn_silu),get(r.dense_ffn_ports[1]),true,"BF16 dense SwiGLU"));
  }
  ASSERT_TRUE(bfloat_reference::expect_binary(get(r.post_ffn),get(r.residual),get(r.ffn_output),false,"BF16 FFN residual"));
  const auto logits=typed<float>(output[r.logits]),scores=typed<float>(output[r.head_score]);
  const auto predicted=typed<std::int32_t>(output[r.state_ports[kPending]]);
  ASSERT_TRUE(bfloat_reference::expect_projection(logits,get(r.state_ports[kHidden]),source.bytes(r.bfloat_vocabulary.head),
      std::size_t(model.vocabulary),std::size_t(model.hidden),"original BF16 full-V head",full,predicted));
  ASSERT_TRUE(bfloat_reference::expect_greedy(logits,predicted,scores,std::size_t(model.vocabulary)));
  EXPECT_EQ(typed<std::int32_t>(output[r.state_ports[kCursor]]),std::vector<std::int32_t>{cursor+1});
  EXPECT_EQ(typed<std::int32_t>(output[r.state_ports[kPosition]]),std::vector<std::int32_t>{position+1});
  program.original_pending.push_back(token.at(0));program.original_predictions.push_back(predicted.at(0));
  if(full){program.original_full_values+=std::size_t(model.hidden);program.original_full_head_values+=logits.size();}
  program.original_full_checked=true;++program.original_episodes;
}
void observe(Program &program,const State &state,const c::StateVersion &from,
             std::span<const c::ReadLease> output) {
  if(program.recipe.original_bfloat){observe_bfloat(program,state,from,output);return;}
  const auto &r=program.recipe;const auto &model=r.model;
  const bool full_check=(r.original||r.original24)&&!program.original_full_checked;
  const auto codec=program.cuda_backend?original_reference::ActivationCodec::kCudaMmvqQ8_1:original_reference::ActivationCodec::kCpuAvx2;
  const auto get=[&](std::size_t port){return typed<float>(output[port]);};
  const auto original_projection=[&](auto role,std::span<const float> actual,std::span<const float> input,
                                    std::span<const std::size_t> extra=std::span<const std::size_t>{}) {
    const auto name=[&]()->std::string_view {
      if constexpr(std::is_same_v<decltype(role),full::Role>)return full::role_name(role);
      else return role;
    }();
    const auto &spec=r.original->source->spec(name);std::vector<std::uint16_t> stored;
    if(spec.native==GGML_TYPE_BF16) {
      const auto external=std::find_if(r.external.begin(),r.external.end(),[&](const auto &entry){return entry.role.name==name;});
      if(external==r.external.end())throw std::runtime_error("original BF16 observation lost its actual weight binding");
      const auto weight=r.fragment.constants.at(external->constant);
      const auto projection=std::find_if(r.original_trace.projections.begin(),r.original_trace.projections.end(),
          [&](const auto &entry){return entry.weight.ordinal()==weight.ordinal();});
      if(projection==r.original_trace.projections.end())throw std::runtime_error("original BF16 observation lost its actual activation store");
      const auto found=std::find_if(r.original_ports.begin(),r.original_ports.end(),[&](const auto &entry){return entry.first.ordinal()==projection->input.ordinal();});
      if(found==r.original_ports.end())throw std::runtime_error("original BF16 activation was not explicitly published");
      stored=typed<std::uint16_t>(output[found->second]);
    }
    const original24_reference::PackedMatrix matrix{r.original->source->bytes(name),std::size_t(spec.n),std::size_t(spec.k),spec.native};
    const auto passed=qwen_full_attention_reference::expect_projection(actual,matrix,input,codec,spec.name.c_str(),full_check,extra,stored);
    if(passed&&full_check)program.original_full_values+=actual.size();return passed;
  };
  const auto original24_projection=[&](full24::Role role,std::span<const float> actual,std::span<const float> input,
                                      std::span<const std::size_t> extra=std::span<const std::size_t>{}) {
    const auto &spec=r.original24->source->spec(role);
    const auto passed=original24_reference::expect_projection(actual,r.original24->source->bytes(role),std::size_t(spec.n),std::size_t(spec.k),
        spec.native,input,codec,std::string(spec.name).c_str(),full_check,extra);
    if(passed&&full_check)program.original_full_values+=actual.size();return passed;
  };
  const auto cursor=typed<std::int32_t>(state.member(from,kCursor)).at(0),position=typed<std::int32_t>(state.member(from,kPosition)).at(0);
  const auto token=typed<std::int32_t>(state.member(from,kPending)).at(0);
  const auto old_hidden=typed<float>(state.member(from,kHidden)),embedding=get(r.embedding);
  if(r.original24) {
    ASSERT_TRUE(original24_reference::expect_embedding(embedding,r.original24->source->bytes(full24::Role::kEmbedding),
        std::size_t(model.hidden),std::size_t(kVocabulary),token,r.original24->source->spec(full24::Role::kEmbedding).native));
    program.original_pending.push_back(token);
  } else if(r.original) {
    ASSERT_TRUE(original_reference::expect_embedding(embedding,r.original->source->bytes(full::role_name(full::Role::kEmbedding)),
        std::size_t(model.hidden),std::size_t(kVocabulary),token,r.original->source->spec(full::role_name(full::Role::kEmbedding)).native));
    program.original_pending.push_back(token);
  } else {
    const auto embedding_first=r.embedding_bank->decoded.begin()+std::ptrdiff_t((token%fixture::prototypes)*model.hidden);
    EXPECT_EQ(embedding,std::vector<float>(embedding_first,embedding_first+model.hidden));
  }
  numeric::expect_close(get(r.hn),sem::oracle::rms_norm_rows(old_hidden,r.hidden_scale,model.hidden,1e-6F),3e-5F,2e-5F);
  numeric::expect_close(get(r.en),sem::oracle::rms_norm_rows(embedding,r.embedding_scale,model.hidden,1e-6F),3e-5F,2e-5F);
  auto joined=get(r.en);const auto hn=get(r.hn);joined.insert(joined.end(),hn.begin(),hn.end());EXPECT_EQ(get(r.joined),joined);
  if(r.original24)ASSERT_TRUE(original24_projection(full24::Role::kProjection,get(r.eh),joined));
  else if(r.original) {
    ASSERT_TRUE(original_projection(full::kInputRoleNames[2],get(r.eh),joined));
  } else quantized(get(r.eh),r.eh_reference(joined),"actual input boundary -> original/generated EH");
  numeric::expect_close(get(r.nx),sem::oracle::rms_norm_rows(get(r.eh),r.attention_scale,model.hidden,1e-6F),3e-5F,2e-5F);
  const auto nx=get(r.nx);
  if(r.original24) {
    ASSERT_TRUE(original24_projection(full24::Role::kQuery,get(r.qg),nx));
    ASSERT_TRUE(original24_reference::expect_projection(get(r.key),r.original_kv->key_bytes(),1024,8192,model.kv.native,nx,codec,"original Q24 key",full_check));
    ASSERT_TRUE(original24_reference::expect_projection(get(r.value),r.original_kv->value_bytes(),1024,8192,model.kv.native,nx,codec,"original Q24 value",full_check));
    if(full_check)program.original_full_values+=2048;
  } else if(r.original) {
    ASSERT_TRUE(original_projection(full::Role::kQuery,get(r.qg),nx));
    ASSERT_TRUE(original_projection(full::Role::kKey,get(r.key),nx));
    ASSERT_TRUE(original_projection(full::Role::kValue,get(r.value),nx));
  } else {
    quantized(get(r.qg),r.qg_reference(nx),"query/gate projection");
    quantized(get(r.key),r.key_reference(nx),"original/generated K projection");
    quantized(get(r.value),r.value_reference(nx),"original/generated V projection");
  }
  const auto qg=get(r.qg);std::vector<float> query(std::size_t(model.heads*kDepth)),gate(query.size());
  for(std::int64_t head=0;head<model.heads;++head)for(std::int64_t d=0;d<kDepth;++d) {
    const auto out=std::size_t(head*kDepth+d),in=std::size_t(head*2*kDepth+d);query[out]=qg[in];gate[out]=qg[in+std::size_t(kDepth)];
  }
  EXPECT_EQ(get(r.gate),gate);
  query=sem::oracle::rms_norm_rows(query,r.query_scale,kDepth,1e-6F);
  numeric::expect_close(get(r.query),cpu_c5_qwen_attention::rotate(query,position),8e-5F,4e-5F);
  const auto normalized_key=sem::oracle::rms_norm_rows(get(r.key),r.key_scale,kDepth,1e-6F);
  numeric::expect_close(get(r.rotated_key),cpu_c5_qwen_attention::rotate(normalized_key,position),8e-5F,4e-5F);
  for(std::size_t entry=0;entry<2;++entry) {
    auto expected=read(state.member(from,entry));
    const auto fresh=read(output[entry==kKey?r.rotated_key:r.value]);
    const auto row_bytes=std::size_t(kKvHeads*kDepth)*sizeof(float);
    std::copy(fresh.begin(),fresh.end(),expected.begin()+std::ptrdiff_t(std::size_t(cursor)*row_bytes));
    EXPECT_EQ(read(output[r.state_ports[entry]]),expected)<<"complete KV, exact prior rows and NaN tail";
  }
  const auto keys=get(r.state_ports[kKey]),values=get(r.state_ports[kValue]);
  if(r.original24) {
    numeric::expect_close(get(r.attended),attention_reference(model,get(r.query),keys,values,gate,cursor+1),8e-5F,4e-5F);
    ASSERT_TRUE(original24_projection(full24::Role::kOutput,get(r.oproj),get(r.attended)));
  } else if(r.original) {
    numeric::expect_close(get(r.attended),attention_reference(model,get(r.query),keys,values,gate,cursor+1),8e-5F,4e-5F);
    ASSERT_TRUE(original_projection(full::Role::kOutput,get(r.oproj),get(r.attended)));
  } else {
    quantized(get(r.attended),attention_reference(model,get(r.query),keys,values,gate,cursor+1),"finite valid-only full-history GQA and q gate");
    quantized(get(r.oproj),r.output_reference(get(r.attended)),"attention output projection");
  }
  auto residual=get(r.eh);const auto out=get(r.oproj);
  for(std::size_t i=0;i<residual.size();++i)residual[i]+=out[i];
  numeric::expect_close(get(r.residual),residual,1e-6F,1e-6F);
  numeric::expect_close(get(r.ffn_input),sem::oracle::rms_norm_rows(get(r.residual),r.ffn_scale,model.hidden,1e-6F),3e-5F,2e-5F);
  if(r.original24) {observe_original24_moe(program,output,codec,full_check);ASSERT_FALSE(::testing::Test::HasFailure());}
  else if(r.original) {
    const auto actual_gate=get(r.dense_ffn_ports[0]),actual_up=get(r.dense_ffn_ports[1]),activation=get(r.dense_ffn_ports[2]);
    ASSERT_TRUE(original_projection(full::Role::kFfnGate,actual_gate,get(r.ffn_input)));
    ASSERT_TRUE(original_projection(full::Role::kFfnUp,actual_up,get(r.ffn_input)));
    numeric::expect_close(activation,nextn::gated(actual_gate,actual_up),8e-5F,4e-5F);
    ASSERT_TRUE(original_projection(full::Role::kFfnDown,get(r.ffn_output),activation));
  } else quantized(get(r.ffn_output),r.ffn_reference(get(r.ffn_input)),model.moe?"top10 routed plus gated shared MoE":"Q27 dense SwiGLU");
  auto post=get(r.residual);const auto ffn=get(r.ffn_output);for(std::size_t i=0;i<post.size();++i)post[i]+=ffn[i];
  numeric::expect_close(get(r.post_ffn),post,1e-6F,1e-6F);
  const auto hidden=get(r.state_ports[kHidden]);
  numeric::expect_close(hidden,sem::oracle::rms_norm_rows(get(r.post_ffn),r.head_scale,model.hidden,1e-6F),3e-5F,2e-5F);
  EXPECT_EQ(typed<std::int32_t>(output[r.state_ports[kCursor]]),std::vector<std::int32_t>{cursor+1});
  EXPECT_EQ(typed<std::int32_t>(output[r.state_ports[kPosition]]),std::vector<std::int32_t>{position+1});
  if(r.original||r.original24) {
    const auto logits=get(r.logits),scores=get(r.head_score);
    const auto predicted=typed<std::int32_t>(output[r.state_ports[kPending]]);
    ASSERT_EQ(logits.size(),std::size_t(kVocabulary));ASSERT_EQ(scores.size(),1u);ASSERT_EQ(predicted.size(),1u);
    ASSERT_TRUE(original_reference::expect_greedy(logits,scores[0],predicted[0]));
    // Always include the actual stable top4 rows in the later bounded dot
    // checks. Selection still comes exclusively from the Graph's full-V TopK.
    std::array<std::size_t,4> candidates{};
    std::vector<std::size_t> order(logits.size());std::iota(order.begin(),order.end(),0);
    std::partial_sort(order.begin(),order.begin()+std::ptrdiff_t(candidates.size()),order.end(),[&](auto a,auto b){
      return logits[a]>logits[b]||(logits[a]==logits[b]&&a<b);
    });
    std::copy_n(order.begin(),candidates.size(),candidates.begin());
    if(r.original24)ASSERT_TRUE(original24_projection(full24::Role::kHead,logits,hidden,candidates));
    else ASSERT_TRUE(original_projection(full::Role::kHead,logits,hidden,candidates));
    if(full_check)program.original_full_head_values=logits.size();
    program.original_predictions.push_back(predicted[0]);++program.original_episodes;
    if(full_check)program.original_full_checked=true;
    return;
  }
  // Check every full-bank logit against the same matching-carrier reference,
  // and the actual TopK against the actual full vector. These observations
  // never replace the pending-ID lease consumed by the next Graph.
  float best=-std::numeric_limits<float>::infinity();std::int32_t best_id=-1;
  const auto reference=r.dot_reference?r.dot_reference(*r.head_bank,hidden):r.head_bank->dots(hidden);
  const auto peak=*std::max_element(reference.begin(),reference.end());
  const auto logits=get(r.logits),scores=get(r.head_score);
  ASSERT_EQ(logits.size(),std::size_t(kVocabulary));ASSERT_EQ(scores.size(),1u);
  std::vector<float> expected_logits(logits.size(),0.F);
  for(std::size_t vocabulary_token=0;vocabulary_token<logits.size();++vocabulary_token) {
    ASSERT_TRUE(std::isfinite(logits[vocabulary_token]));
    expected_logits[vocabulary_token]=reference[vocabulary_token%fixture::prototypes];
    if(logits[vocabulary_token]>best) {best=logits[vocabulary_token];best_id=std::int32_t(vocabulary_token);}
  }
  quantized(logits,expected_logits,"matching-carrier full-V generated head logits");
  const std::array<float,1> wanted{peak};
  quantized(scores,wanted,"matching-carrier full-V generated head maximum");
  EXPECT_EQ(scores[0],best);
  EXPECT_EQ(typed<std::int32_t>(output[r.state_ports[kPending]]),std::vector<std::int32_t>{best_id});
  EXPECT_GE(best_id,0);EXPECT_LT(best_id,kVocabulary);
}

struct VocabularyHead final {
  Program program;
  std::size_t logits,scores;
  std::vector<std::size_t> pending;
  std::optional<std::size_t> alternate;
  std::size_t matmuls;
  std::string provider;
  std::optional<std::size_t> stored_input;
};
VocabularyHead prepare_vocabulary_head(const Model &model,const orig::ReadonlyRole &head,
    c::Context &context,const c::StorageOwner &owner,bool cuda,bool with_alternate=false,
    std::shared_ptr<MixedPlacement> mixed={},std::size_t rows=2) {
  if(rows<2||rows>3)throw std::runtime_error("finite vocabulary verifier supports two or three actual rows");
  Recipe recipe{.model=model,.fragment={.profile=model.profile,.region=0x5157454e484d3201ULL}};
  auto &fragment=recipe.fragment;
  const bool bfloat_head=head.description.is_plain()&&head.description.data_type()==c::DataType::kBFloat16;
  if((bfloat_head&&head.description!=sem::dense({model.vocabulary,model.hidden},c::DataType::kBFloat16))||
      (!bfloat_head&&model.storage!=c::DataType::kFloat32))
    throw std::runtime_error("full vocabulary head storage differs from its actual hidden rows");
  const auto first=fragment.input(sem::dense({1,model.hidden},model.storage));
  const auto second=fragment.input(sem::dense({1,model.hidden},model.storage));
  auto joined=fragment.one(c::Operation(c::Concat{.axis=0}),{first,second});
  if(rows==3) {
    const auto third=fragment.input(sem::dense({1,model.hidden},model.storage));
    joined=fragment.one(c::Operation(c::Concat{.axis=0}),{joined,third});
  }
  const auto head_value=external(recipe,head);
  const auto head_input=bfloat_head?attention::cast(fragment,joined,c::DataType::kBFloat16):joined;
  const auto logits=fragment.one(bfloat_head?c::model_ops::dense_linear_f32():
      c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{head_input,head_value});
  const auto logits_port=expose(recipe,logits);
  const auto winner=fragment.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);
  const auto scores_port=expose(recipe,winner[0]);
  std::vector<std::size_t> pending_ports(rows);
  for(std::size_t row=0;row<pending_ports.size();++row) {
    const auto sliced=fragment.one(c::Operation(take(c::View::slice(0,std::int64_t(row),1),"M2 pending row slice")),{winner[1]});
    pending_ports[row]=expose(recipe,core::reshape(fragment,sliced,{1}));
  }
  std::optional<std::size_t> alternate_port;
  if(with_alternate) {
    const auto two=fragment.append(c::Operation(c::TopK{.count=2,.stable=true}),{logits},2);
    const auto row=fragment.one(c::Operation(take(c::View::slice(0,0,1),"K1 alternate first row")),{two[1]});
    const auto alternate=fragment.one(c::Operation(take(c::View::slice(1,1,1),"K1 alternate second candidate")),{row});
    alternate_port=expose(recipe,core::reshape(fragment,alternate,{1}));
  }
  const auto stored_input=bfloat_head&&model.storage==c::DataType::kFloat32?std::optional<std::size_t>{expose(recipe,head_input)}:std::nullopt;
  auto batched=Program::create(std::move(recipe),context,owner,cuda,true,{},mixed);
  const auto &graph=batched.logical.graph;
  const auto observer=take(graph.producer(batched.logical.outputs[logits_port]),"M2 full logits Copy");
  if(take(graph.operation_kind(observer),"M2 logits observer kind")!=c::OperationKind::kCopy)
    throw std::runtime_error("M2 full logits must have independent caller storage");
  const auto full_logits=take(graph.node_input(observer,0),"M2 actual full logits");
  const auto projection=take(graph.producer(full_logits),"M2 actual full head");
  const auto projection_input=take(graph.node_input(projection,0),"M2 actual two-row head input");
  auto concatenated=projection_input;
  if(bfloat_head&&model.storage==c::DataType::kFloat32) {
    const auto conversion=take(graph.producer(projection_input),"M2 actual BF16 activation store");
    const auto operation=take(graph.operation(conversion),"M2 BF16 activation conversion operation");
    const auto *cast=operation.get_if<c::Cast>();
    if(!cast||cast->output_type!=c::DataType::kBFloat16)throw std::runtime_error("M2 BF16-W/F32-state head requires an actual BF16 activation Cast");
    concatenated=take(graph.node_input(conversion,0),"M2 actual F32 tap concatenation");
  }
  const auto concatenate=take(graph.producer(concatenated),"M2 actual hidden-row concatenation");
  std::size_t matmuls=0;
  for(std::size_t node_index=0;node_index<graph.node_count();++node_index) {
    const auto node_operation=take(graph.operation(take(graph.node_id(node_index),"M2 logical node")),"M2 logical node operation");
    matmuls+=node_operation.get_if<c::MatMul>()!=nullptr||node_operation.custom_as<c::model_ops::DenseLinearF32>().has_value();
  }
  const auto operation=take(graph.operation(projection),"M2 actual projection operation");
  const auto *matmul=operation.get_if<c::MatMul>();
  if(matmuls!=1||(bfloat_head?!operation.custom_as<c::model_ops::DenseLinearF32>():
          (!matmul||matmul->accumulation_type!=c::DataType::kFloat32))||
      take(graph.value_description(full_logits),"M2 logits description")!=sem::dense({std::int64_t(rows),model.vocabulary})||
      take(graph.value_description(projection_input),"M2 activation description")!=sem::dense({std::int64_t(rows),model.hidden},bfloat_head?c::DataType::kBFloat16:model.storage)||
      take(graph.node_input(projection,1),"M2 exact original head operand")!=batched.logical.constants.at(0)||
      take(graph.constant_description(0),"M2 original head description")!=head.description||
      take(graph.operation_kind(concatenate),"M2 concatenate operation")!=c::OperationKind::kConcat||
      (rows==2&&take(graph.node_input(concatenate,0),"M2 actual first tap")!=batched.logical.inputs.at(0))||
      take(graph.node_input(concatenate,1),"finite verifier final actual tap")!=batched.logical.inputs.at(rows-1))
    throw std::runtime_error("M2 head must be one complete original-bank projection on two actual hidden rows");
  if(rows==3) {
    const auto left=take(graph.node_input(concatenate,0),"M3 first two actual taps");
    const auto pair=take(graph.producer(left),"M3 first pair concatenation");
    if(take(graph.operation_kind(pair),"M3 first pair kind")!=c::OperationKind::kConcat||
        take(graph.node_input(pair,0),"M3 actual first tap")!=batched.logical.inputs.at(0)||
        take(graph.node_input(pair,1),"M3 actual second tap")!=batched.logical.inputs.at(1))
      throw std::runtime_error("M3 head must preserve the three actual ordered target taps");
  }
  const auto &bound_head=batched.recipe.external.at(0).role;
  const auto original_view=take(head.lease.view(),"M2 once-bound original head view");
  const auto bound_view=take(bound_head.lease.view(),"M2 actual bound head view");
  const auto &head_owner=mixed?mixed->host:owner;
  if(bound_head.name!=head.name||head.owner.ref()!=head_owner.ref()||bound_head.owner.ref()!=head_owner.ref()||
      original_view.resource()!=bound_view.resource()||original_view.native_handle()!=bound_view.native_handle()||
      original_view.offset()!=bound_view.offset()||original_view.size()!=bound_view.size()||head.lease.generation()!=bound_head.lease.generation())
    throw std::runtime_error("M2 head must retain the same original readonly bank and owner");
  std::size_t selected_steps=0;std::string selected_provider;
  for(std::size_t step=0;step<batched.compiled.report.step_count();++step) {
    const auto selected=take(batched.compiled.report.step(step),"M2 actual compiled step");
    for(std::size_t index=0;index<selected.node_count;++index)
      if(take(batched.compiled.report.step_node(step,index),"M2 selected logical node")==projection) {
        ++selected_steps;selected_provider=selected.provider.name;
#if defined(HYLE_TEST_QWEN_STATE_CPU)
        if(!cuda&&!bfloat_head&&selected.provider.id!=c::ggml_cpu::kProviderId)
          throw std::runtime_error("M2 CPU numeric certificate requires its actual ggml CPU route");
#endif
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
        if(cuda&&!bfloat_head&&selected.provider.id!=c::ggml_cuda::kProviderId)
          throw std::runtime_error("M2 CUDA numeric certificate requires its actual native MMVQ route");
#endif
      }
  }
  if(selected_steps!=1)throw std::runtime_error("M2 full head needs exactly one selected provider step");

  return {std::move(batched),logits_port,scores_port,pending_ports,alternate_port,matmuls,std::move(selected_provider),stored_input};
}

void check_original_head_m2(const Program &single,c::Context &context,const c::StorageOwner &owner,
    c::Queue &queue,const std::array<c::ReadLease,2> &hidden_rows,std::size_t &additional_submissions) {
  const auto &source=single.recipe;const auto &model=source.model;
  if(bool(source.original)==bool(source.original24))throw std::runtime_error("M2 head needs exactly one authenticated original vocabulary");
  const auto &head=source.original24?source.original24->at(full24::Role::kHead):source.original->at(full::role_name(full::Role::kHead));
  const auto &embedding=source.original24?source.original24->at(full24::Role::kEmbedding):source.original->at(full::role_name(full::Role::kEmbedding));
  const auto packed_head=source.original24?source.original24->source->bytes(full24::Role::kHead):source.original->source->bytes(full::role_name(full::Role::kHead));
  const auto packed_embedding=source.original24?source.original24->source->bytes(full24::Role::kEmbedding):source.original->source->bytes(full::role_name(full::Role::kEmbedding));
  const auto codec=single.cuda_backend?original_reference::ActivationCodec::kCudaMmvqQ8_1:original_reference::ActivationCodec::kCpuAvx2;
  const std::array actual_hidden{typed<float>(hidden_rows[0]),typed<float>(hidden_rows[1])};
  for(const auto &row:actual_hidden)ASSERT_EQ(row.size(),std::size_t(model.hidden));
  ASSERT_NE(actual_hidden[0],actual_hidden[1]);

  // Both rows are genuine normalized hidden result leases from the M1 NextN
  // journey. Only their small activation matrix is concatenated; the original
  // complete packed head remains the same once-bound immutable bank.
  auto prepared=prepare_vocabulary_head(model,head,context,owner,single.cuda_backend);
  auto &batched=prepared.program;const auto logits_port=prepared.logits,scores_port=prepared.scores;
  const auto pending_ports=prepared.pending;

  Recipe feedback_recipe{.model=model,.fragment={.profile=model.profile,.region=0x5157454e484d3202ULL}};
  auto &feedback_fragment=feedback_recipe.fragment;
  const auto embedding_value=external(feedback_recipe,embedding);
  for(std::size_t row=0;row<pending_ports.size();++row) {
    const auto pending=feedback_fragment.input(sem::dense({1},c::DataType::kInt32));
    expose(feedback_recipe,feedback_fragment.one(c::Operation(c::Gather{}),{embedding_value,pending}));
  }
  auto feedback=Program::create(std::move(feedback_recipe),context,owner,single.cuda_backend,true);
  {
    const auto result=batched.execute_inputs(queue,hidden_rows);
    const auto actual_logits=typed<float>(result[logits_port]);
    const auto actual_scores=typed<float>(result[scores_port]);
    ASSERT_EQ(actual_logits.size(),2*std::size_t(kVocabulary));ASSERT_EQ(actual_scores.size(),2u);
    std::array<c::ReadLease,2> actual_pending;
    std::array<std::int32_t,2> predicted;
    for(std::size_t row=0;row<pending_ports.size();++row) {
      actual_pending[row]=result[pending_ports[row]];
      const auto ids=typed<std::int32_t>(actual_pending[row]);ASSERT_EQ(ids.size(),1u);predicted[row]=ids[0];
      const auto values=std::span(actual_logits).subspan(row*std::size_t(kVocabulary),std::size_t(kVocabulary));
      ASSERT_TRUE(original_reference::expect_greedy(values,actual_scores[row],predicted[row]));
      // Two host row references observe one native M2 MatMul, not two M1
      // executions. The pinned native M2 MMVQ uses the same per-row AQ8_1
      // quantizer/dot as M1 on this validation GPU; CPU uses AQ8_K.
      if(source.original24)ASSERT_TRUE(original24_reference::expect_projection(values,packed_head,std::size_t(kVocabulary),
          std::size_t(model.hidden),model.head.native,actual_hidden[row],codec,"actual Q24 M2 full head",true));
      else if(head.description.is_plain()) {
        if(!prepared.stored_input)throw std::runtime_error("F32-state M2 original BF16 head lost its stored activation output");
        const auto stored=typed<std::uint16_t>(result[*prepared.stored_input]);
        const original24_reference::PackedMatrix matrix{packed_head,std::size_t(model.vocabulary),std::size_t(model.hidden),GGML_TYPE_BF16};
        ASSERT_TRUE(qwen_full_attention_reference::expect_projection(values,matrix,actual_hidden[row],codec,"actual Q27 BF16 M2 full head",true,{},
            std::span(stored).subspan(row*std::size_t(model.hidden),std::size_t(model.hidden))));
      } else ASSERT_TRUE(original_reference::expect_projection(values,packed_head,std::size_t(kVocabulary),
          std::size_t(model.hidden),source.original->source->spec(full::role_name(full::Role::kHead)).native,actual_hidden[row],codec,"actual Q27 M2 full head",true));
    }
    const auto retained_logits=read(result[logits_port]);
    const auto embedded=feedback.execute_inputs(queue,actual_pending);
    for(std::size_t row=0;row<embedded.size();++row) {
      ASSERT_TRUE(original_reference::expect_embedding(typed<float>(embedded[row]),packed_embedding,
          std::size_t(model.hidden),std::size_t(kVocabulary),predicted[row],
          source.original24?source.original24->source->spec(full24::Role::kEmbedding).native:source.original->source->spec(full::role_name(full::Role::kEmbedding)).native));
      auto pending_reader=actual_pending[row];EXPECT_FALSE(pending_reader.try_write());
    }
    EXPECT_EQ(read(result[logits_port]),retained_logits);
    for(std::size_t row=0;row<hidden_rows.size();++row)EXPECT_EQ(typed<float>(hidden_rows[row]),actual_hidden[row]);
  }
  EXPECT_EQ(batched.live_output_blocks(),0u);EXPECT_EQ(batched.active_output_uses(),0u);
  EXPECT_EQ(feedback.live_output_blocks(),0u);EXPECT_EQ(feedback.active_output_uses(),0u);
  additional_submissions+=batched.native_submissions()+feedback.native_submissions();
  EXPECT_EQ(batched.native_submissions(),1u);EXPECT_EQ(feedback.native_submissions(),1u);
  ::testing::Test::RecordProperty("packed_head_m2_rows",2);
  ::testing::Test::RecordProperty("packed_head_m2_matmuls",int(prepared.matmuls));
  ::testing::Test::RecordProperty("packed_head_m2_full_reference_values",std::to_string(2*std::size_t(kVocabulary)));
  ::testing::Test::RecordProperty("packed_head_m2_pending_feedback_rows",2);
  ::testing::Test::RecordProperty("packed_head_m2_submissions",int(batched.native_submissions()+feedback.native_submissions()));
  ::testing::Test::RecordProperty("packed_head_m2_provider",prepared.provider);
  ::testing::Test::RecordProperty("packed_head_m2_scope","one full packed M2 projection of two actual M1 normalized hidden leases; both actual row-ID leases feed original embedding Gather; no two-token NextN/target equivalence or TPS claim");
}

Bytes storage_bytes(const Model &model,std::span<const float> values) {
  if(model.storage==c::DataType::kFloat32)return sem::bytes<float>(values);
  if(model.storage!=c::DataType::kBFloat16)throw std::runtime_error("unsupported actual Qwen state storage");
  std::vector<std::uint16_t> bits(values.size());
  for(std::size_t i=0;i<values.size();++i) {
    // Unwritten finite-history tails are explicit poisoned storage, not
    // arithmetic operands. Keep a quiet BF16 NaN instead of silently zeroing.
    bits[i]=std::isnan(values[i])?std::uint16_t((std::bit_cast<std::uint32_t>(values[i])>>16U)|0x40U):bfloat_reference::round(values[i]);
  }
  return sem::bytes(bits);
}
c::StateVersion initial(State &state,const Model &model,const c::StorageOwner &owner) {
  std::array<c::ReadLease,kResources> values;
  for(std::size_t entry=0;entry<2;++entry) {
    std::vector<float> cache(std::size_t(kCapacity*model.kv_heads*model.depth));
    for(std::size_t i=0;i<cache.size();++i)cache[i]=i<2*std::size_t(model.kv_heads*model.depth)
        ?float(int((i*7+entry*13)%31)-15)/32.F:std::bit_cast<float>(std::uint32_t(0x7fc00001u+entry));
    values[entry]=upload(owner,storage_bytes(model,cache));
  }
  values[kHidden]=upload(owner,storage_bytes(model,d5_gguf_test::activation(model.hidden,3)));
  values[kPending]=upload(owner,sem::bytes(std::vector<std::int32_t>{3}));
  values[kCursor]=upload(owner,sem::bytes(std::vector<std::int32_t>{2}));
  values[kPosition]=upload(owner,sem::bytes(std::vector<std::int32_t>{37}));
  return state.import(values);
}
void run(const Model &model,std::span<const orig::ReadonlyRole> originals,
    const orig::q27::Data *q27_data,const orig::q24::Data *q24_data,
    c::Context &context,const c::StorageOwner &owner,bool cuda,
    std::shared_ptr<const full::Roles> original={},std::shared_ptr<const packed_moe::NextnRoles> original24={},
    std::shared_ptr<ExpertSupply> expert_supply={}) {
  const bool complete=bool(original)||bool(original24);
  std::size_t mapped_bytes=0;if(original)for(std::size_t slot=0;slot<original->source->size();++slot)
    mapped_bytes+=original->source->spec(original->source->name(slot)).bytes;
  if(model.moe) {
    ASSERT_EQ(model.profile->artifact_repository,orig::q24::kIdentity.repository);
    ASSERT_EQ(model.profile->artifact_revision,orig::q24::kIdentity.revision);ASSERT_EQ(model.profile->variant,orig::q24::kIdentity.variant);
  } else {
    ASSERT_EQ(model.profile->artifact_repository,orig::q27::kIdentity.repository);
    ASSERT_EQ(model.profile->artifact_revision,orig::q27::kIdentity.revision);ASSERT_EQ(model.profile->variant,orig::q27::kIdentity.variant);
  }
  const auto native_reference_submissions=std::make_shared<NativeReferenceCount>();
  DotReference dot_reference;
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
  if(cuda&&!complete)dot_reference=[native_reference_submissions](const fixture::Bank &bank,std::span<const float> input) {
    const auto values=cuda_sidecar_quant_reference::catalogue_dots(
        bank.type,bank.k,bank.packed,bank.decoded,input,false);
    if(ggml_is_quantized(bank.type))++native_reference_submissions->submissions;
    return values;
  };
#endif
  auto program=Program::create(build(model,owner,originals,q27_data,q24_data,std::move(dot_reference),original,original24),context,owner,cuda,false,expert_supply);
  auto state=State::create(model,owner,program.output_root_bytes);auto queue=take(context.create_queue(),"one original complete NextN queue");
  std::size_t additional_submissions=0;
  {
    const auto base=initial(state,model,owner);
    const auto base_image=image(state,base);
    const auto first_outputs=program.execute(queue,state,base);observe(program,state,base,first_outputs);ASSERT_FALSE(::testing::Test::HasFailure());
    const auto first=state.candidate(base,program.recipe,first_outputs);
    const auto first_image=image(state,first);
    // Real pending ID, normalized hidden, entire KV and absolute position all
    // come from the first Graph. No observer vector is uploaded for step two.
    const auto second_outputs=program.execute(queue,state,first);observe(program,state,first,second_outputs);ASSERT_FALSE(::testing::Test::HasFailure());
    const auto second=state.candidate(first,program.recipe,second_outputs);
    const auto second_image=image(state,second);
    std::array<c::StateHead,3> heads;
    for(auto &head:heads)head=take(c::StateHead::create(state.store,base),"independent actual acceptance head");
    const std::array<c::StateVersion,3> prefixes{base,first,second};
    // Failure after staging a genuine K result cannot publish a mismatched V,
    // hidden, pending token or cursor. Every head still names complete base.
    {
      auto failed=take(state.store.begin(base),"failed whole-prefix draft");
      check(failed.replace(state.entries[kKey],first_outputs[program.recipe.state_ports[kKey]]),"stage genuine K candidate");
      const auto bad=take(first_outputs[program.recipe.state_ports[kValue]].subview(0,sizeof(float)),"genuine wrong-extent V reader");
      EXPECT_TRUE(failed.replace(state.entries[kValue],bad).IsInvalidArgument());
      check(failed.cancel(),"cancel unpublishable complete prefix");
      for(const auto &head:heads)EXPECT_TRUE(take(head.current(),"unchanged complete head").identity()==base.identity());
    }
    // accept0 preserves base exactly, accept1 excludes the second complete
    // generation, accept2 publishes both consumed inputs. This finite selector
    // is explicit; it does not claim original target/natural acceptance.
    for(std::size_t accepted=1;accepted<heads.size();++accepted)
      check(heads[accepted].publish(base.identity(),prefixes[accepted]),"one atomic complete accepted-prefix publication");
    EXPECT_TRUE(heads[1].publish(base.identity(),second).IsUnavailable());
    EXPECT_TRUE(image(state,take(heads[0].current(),"rejected-all head"))==base_image);
    EXPECT_TRUE(image(state,take(heads[1].current(),"accepted1 head"))==first_image);
    EXPECT_TRUE(image(state,take(heads[2].current(),"accepted2 head"))==second_image);
    std::array<c::StateVersion,3> continued;
    std::array<Image,3> continued_images;
    for(std::size_t accepted=0;accepted<heads.size();++accepted) {
      SCOPED_TRACE(accepted);
      const auto from=take(heads[accepted].current(),"ACTUAL accepted head for next step");
      const auto values=program.execute(queue,state,from);observe(program,state,from,values);ASSERT_FALSE(::testing::Test::HasFailure());
      continued[accepted]=state.candidate(from,program.recipe,values);
      check(heads[accepted].publish(from.identity(),continued[accepted]),"publish actual branch continuation");
      continued_images[accepted]=image(state,continued[accepted]);
      EXPECT_TRUE(image(state,base)==base_image);EXPECT_TRUE(image(state,first)==first_image);EXPECT_TRUE(image(state,second)==second_image);
      for(std::size_t i=0;i<=accepted;++i)EXPECT_TRUE(image(state,continued[i])==continued_images[i]);
    }
    // Same immutable plans and actual old heads replay exactly, while the
    // accepted2 branch alone advances its complete cursor/position to5/40.
    EXPECT_TRUE(continued_images[0]==first_image);EXPECT_TRUE(continued_images[1]==second_image);
    EXPECT_EQ(typed<std::int32_t>(state.member(continued[2],kCursor)),std::vector<std::int32_t>{5});
    EXPECT_EQ(typed<std::int32_t>(state.member(continued[2],kPosition)),std::vector<std::int32_t>{40});
    if(complete) {
      ASSERT_EQ(program.original_pending.size(),5u);ASSERT_EQ(program.original_predictions.size(),5u);
      const std::vector<std::int32_t> actual_feedback{3,program.original_predictions[0],3,program.original_predictions[0],program.original_predictions[1]};
      EXPECT_EQ(program.original_pending,actual_feedback);
      EXPECT_EQ(program.original_predictions[2],program.original_predictions[0]);
      EXPECT_EQ(program.original_predictions[3],program.original_predictions[1]);
      ASSERT_TRUE(program.original_full_checked);EXPECT_EQ(program.original_episodes,5u);
      check_original_head_m2(program,context,owner,queue,
          {first_outputs[program.recipe.state_ports[kHidden]],second_outputs[program.recipe.state_ports[kHidden]]},additional_submissions);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
    for(const auto &version:{base,first,second,continued[0],continued[1],continued[2]})
      for(std::size_t i=0;i<kResources;++i)EXPECT_FALSE(state.member(version,i).try_write());
  }
  check(state.store.collect(),"retire all complete NextN versions");
  EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);
  EXPECT_EQ(program.live_output_blocks(),0u);EXPECT_EQ(program.active_output_uses(),0u);
  EXPECT_EQ(program.submissions,5u);
  for(const auto &binding:program.recipe.external) {
    auto reader=binding.role.lease;EXPECT_FALSE(reader.try_write());
  }
  ::testing::Test::RecordProperty("backend",cuda?"ordinary CUDA Graph":"ordinary CPU Graph");
  ::testing::Test::RecordProperty("actual_submissions",int(program.native_submissions()+additional_submissions));
  ::testing::Test::RecordProperty("logical_nextn_steps",int(program.submissions));
  ::testing::Test::RecordProperty("quantized_reference",cuda?
      "existing original-native CUDA integration oracle with matching activation codec; independent original-role F64/decoded and strict F32 local gates retained":
      "unchanged CPU C5 native-codec dots; independent original-role F64/decoded and strict F32 local gates retained");
  if(complete) {
    ::testing::Test::RecordProperty("quantized_reference",original24?(cuda?
        "independent original Q4_K/Q5_K/Q6_K/Q8_0 bytes with actual F32 input and fixed native MMVQ Q8_1 integer-minimum correction; no MMQ or native reference submissions":
        "independent original Q4_K/Q5_K/Q6_K/Q8_0 bytes with actual F32 input and fixed native AVX2 activation codecs; no native reference submissions"):cuda?
        "independent original Q4_K/Q6_K/Q8_0 bytes with actual F32 input and bounded native MMVQ Q8_1 codec/strict-fast division envelope; no native reference submissions":
        "independent original Q4_K/Q6_K/Q8_0 bytes with actual F32 input and fixed native AVX2 Q8_K/Q8_0 codec; no native reference submissions");
    ::testing::Test::RecordProperty("original_role_count",original24?22:int(original->source->size()));
    ::testing::Test::RecordProperty("original_role_bytes",std::to_string(original24?full24::kOriginalBytes+11535360ULL:mapped_bytes));
    ::testing::Test::RecordProperty("original_first_full_projection_values",std::to_string(program.original_full_values));
    ::testing::Test::RecordProperty("original_first_full_head_values",std::to_string(program.original_full_head_values));
    ::testing::Test::RecordProperty("original_prediction_embedding_observations",int(program.original_episodes));
    ::testing::Test::RecordProperty("original_shared_bank_capacity_bytes",std::to_string(original24?full24::kStaticBytes+11535360ULL:mapped_bytes));
    ::testing::Test::RecordProperty("original_feedback",std::to_string(program.original_pending.at(0))+","+
        std::to_string(program.original_pending.at(1))+","+std::to_string(program.original_pending.at(2))+","+
        std::to_string(program.original_pending.at(3))+","+std::to_string(program.original_pending.at(4)));
  }
  ::testing::Test::RecordProperty("complete_draft_resources",6);
  ::testing::Test::RecordProperty("full_vocabulary",int(kVocabulary));
  ::testing::Test::RecordProperty("head_matmuls",1);
  if(original24) {
    ASSERT_TRUE(expert_supply);EXPECT_EQ(expert_supply->episodes,5u);EXPECT_EQ(expert_supply->selections.size(),5u);
    EXPECT_EQ(program.native_submissions(),10u);
    ::testing::Test::RecordProperty("resident_expert_capacity",10);
    ::testing::Test::RecordProperty("expert_domain",512);
    ::testing::Test::RecordProperty("resident_slots",2);
    ::testing::Test::RecordProperty("resident_slot_bytes",std::to_string(expert_supply->layout().slot_bytes()));
    ::testing::Test::RecordProperty("expert_supply_scope","original compressed bytes plus I32 generation map; slot capacities/API upload accounting are not measured physical VRAM or bus traffic");
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
    if(cuda)check(expert_supply->cuda->close(),"close retired Q24 expert upload slots after every Invocation reset");
#endif
  }
  if(!complete)::testing::Test::RecordProperty("original_role_bytes",model.moe?11535360:43048960);
  ::testing::Test::RecordProperty("scope",original24?
      "Q24 MAIN UD-IQ4_XS all20 original blk92 NextN roles plus independent original fullV embedding/head; full512 expert domain/actual Top10 original selected experts supplied as compressed E10 banks; two ordinary Graph phases keep actual activations/state leases; complete prefix0/1/2 and eachhead continuation; fixture initial target hidden/history and explicit acceptance remain external, no92-layer target/natural acceptance/quality/TPS claim":original?
      "Q27 MAIN UD-Q4_K_M all15 original blk64 NextN roles plus original untied fullV embedding/head; actual pending/normalized hidden/fullKV/cursor/position prefix0/1/2 and eachhead continuation; fixture initial target hidden/history and explicit acceptance lengths remain external; no complete64-layer target, natural acceptance, quality or TPS claim":model.moe?
      "Q24 MAIN UD-IQ4_XS original blk92 K/V/knorm inside complete generated-parameter MoE NextN; generated8-resident expert mapping,shared embedding/head; actual pending/hidden/fullKV/cursor/position prefix0/1/2 and eachhead continuation; no original512-expert target or natural acceptance claim":
      "Q27 MAIN UD-Q4_K_M original blk64 EH/enorm/hnorm inside complete generated-parameter dense NextN; generated embedding and distinct-codec fullVhead; actual pending/hidden/fullKV/cursor/position prefix0/1/2 and eachhead continuation; no complete64-layer target or natural acceptance claim");
}

enum TargetEntry:std::size_t {kTargetKey,kTargetValue,kTargetCursor,kTargetPosition,kTargetRaw,kTargetTap,kTargetResources};
// The target vocabulary is explicit even when current artifacts share it
// with the draft. The source retains authenticated mapped bytes; head retains
// its actual immutable native lease. No new allocation or executor is added.
struct TargetVocabulary final {
  std::shared_ptr<const void> source;
  orig::ReadonlyRole head;
  std::span<const std::byte> bytes;
  ggml_type native;
};
template<class Source> TargetVocabulary target_vocabulary(std::shared_ptr<const Source> source,
    const orig::ReadonlyRole &head,std::span<const std::byte> bytes,ggml_type native) {
  if(!source||!head.owner||!head.lease||!head.identity||bytes.empty()||
      bytes.size()!=take(head.description.required_bytes(),"actual target vocabulary extent"))
    throw std::runtime_error("target verifier needs explicit authenticated vocabulary owner/bytes/readonly bank");
  return {std::move(source),head,bytes,native};
}
struct TargetPlan final {
  TargetVocabulary vocabulary; // outlives program and its byte references
  Program program;
  std::shared_ptr<const target_roles::Bound> parameters;
  std::shared_ptr<const full::Roles> named_parameters;
  std::shared_ptr<const packed_moe::TargetRoles> parameters24;
  std::shared_ptr<const bfloat::Bound> bfloat_parameters;
  target_model::TargetResult result;
  std::optional<target24_model::Result> result24;
  std::optional<bfloat_moe_target::Result> result_bfloat_moe;
  std::optional<qwen_nvfp4_target_fragment::Result> result_nvfp4;
  std::optional<qwen_gptq_projection_fragment::Result> result_gptq;
  std::optional<qwen_ct_asymmetric_projection_fragment::Result> result_ct;
  bfloat_moe_target::Names bfloat_moe_names;
  std::vector<std::pair<Value,std::size_t>> observed;
  std::array<std::size_t,kTargetResources> members;
  bool full_checked=false;
  std::size_t full_projection_values=0;
  std::size_t role_count=target_roles::kRoles,source_bytes=target_roles::kOriginalBytes;
  std::size_t port(Value value) const {
    const auto found=std::find_if(observed.begin(),observed.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found==observed.end())throw std::runtime_error("original target observation missing");return found->second;
  }
  std::array<c::ReadLease,kTargetResources> state(std::span<const c::ReadLease> output) const {
    std::array<c::ReadLease,kTargetResources> result_values;
    for(std::size_t entry=0;entry<result_values.size();++entry)result_values[entry]=output[members[entry]];return result_values;
  }
};
TargetPlan prepare_bfloat_target(const Model &model,std::shared_ptr<const bfloat::Bound> parameters,
    const bfloat::DenseNormConventions &rules,std::string_view layer_prefix,std::string_view final_norm,
    TargetVocabulary vocabulary,c::Context &context,const c::StorageOwner &owner,bool cuda,
    std::span<const qwen_nvfp4_target_fragment::ProjectionNames> nvfp4={},
    std::span<const qwen_gptq_projection_fragment::Names> gptq={},
    std::span<const qwen_ct_asymmetric_projection_fragment::Names> ct={}) {
  if(model.storage!=c::DataType::kBFloat16||model.moe||!parameters||!parameters->source||layer_prefix.empty()||final_norm.empty())
    throw std::runtime_error("BF16 dense target needs distinct original layer/norm roles; selected-MoE target is not implicitly substituted");
  Recipe recipe{.model=model,.fragment={.profile=model.profile,.region=0x5142463136544754ULL}};
  recipe.original_bfloat=parameters;recipe.bfloat_rules=rules;auto &f=recipe.fragment;
  const auto input=f.input(sem::dense({1,model.hidden},model.storage));
  const auto old_key=f.input(descriptions(model)[kKey]),old_value=f.input(descriptions(model)[kValue]);
  const auto cursor=f.input(sem::dense({1},c::DataType::kInt32)),position=f.input(sem::dense({1},c::DataType::kInt32));
  const auto parameter=[&](std::string_view suffix) {return external(recipe,parameters->at(std::string(layer_prefix)+std::string(suffix)));};
  const auto gamma=[&](std::string_view suffix,bfloat::NormConvention convention,std::int64_t width) {
    return bfloat::effective_gamma(f,{parameter(suffix),convention},width);
  };
  if(int(!nvfp4.empty())+int(!gptq.empty())+int(!ct.empty())>1)
    throw std::runtime_error("target must select one explicit original projection source contract");
  if(!gptq.empty()&&gptq.size()!=qwen_gptq_projection_fragment::kProjections)
    throw std::runtime_error("GPTQ target needs seven explicit original quartet names");
  if(!ct.empty()&&ct.size()!=qwen_ct_asymmetric_projection_fragment::kProjections)
    throw std::runtime_error("CT target needs seven explicit original codes/Z/BF16SF/shape names");
  const auto projection_weight=[&](std::size_t slot,std::string_view suffix) {
    return !ct.empty()?external(recipe,parameters->at(ct[slot].codes)):
        gptq.empty()?parameter(suffix):external(recipe,parameters->at(gptq[slot].qweight));
  };
  const attention::TargetParameters supplied{
      gamma(".input_layernorm.weight",rules.input,model.hidden),projection_weight(0,".self_attn.q_proj.weight"),
      projection_weight(1,".self_attn.k_proj.weight"),projection_weight(2,".self_attn.v_proj.weight"),
      gamma(".self_attn.q_norm.weight",rules.query,model.depth),gamma(".self_attn.k_norm.weight",rules.key,model.depth),
      projection_weight(3,".self_attn.o_proj.weight"),gamma(".post_attention_layernorm.weight",rules.post,model.hidden),
      projection_weight(4,".mlp.gate_proj.weight"),projection_weight(5,".mlp.up_proj.weight"),projection_weight(6,".mlp.down_proj.weight"),
      bfloat::effective_gamma(f,{external(recipe,parameters->at(final_norm)),rules.head},model.hidden)};
  const auto frequency=f.constant(sem::dense({model.rotary/2}),sem::bytes(text_frequencies(model)));
  TargetPlan plan;plan.vocabulary=std::move(vocabulary);plan.bfloat_parameters=std::move(parameters);
  const attention::Geometry geometry{model.hidden,model.heads,model.kv_heads,model.depth,model.rotary,1.e-6F,model.score_scale,model.storage};
  if(nvfp4.empty()&&gptq.empty()&&ct.empty())plan.result=attention::append_target(f,input,old_key,old_value,cursor,position,frequency,supplied,geometry);
  else if(!nvfp4.empty()) {
    namespace nv=qwen_nvfp4_target_fragment;
    if(nvfp4.size()!=nv::kProjections)throw std::runtime_error("NVFP4 dense target needs all seven explicit calibration quartets");
    nv::Parameters calibrated{supplied.input_norm,supplied.query_norm,supplied.key_norm,supplied.post_norm,supplied.target_norm,{}};
    const std::array codes{supplied.query,supplied.key,supplied.value,supplied.output,supplied.ffn_gate,supplied.ffn_up,supplied.ffn_down};
    for(std::size_t index=0;index<codes.size();++index) {
      if(bfloat_constant(recipe,codes[index]).name!=nvfp4[index].codes)
        throw std::runtime_error("NVFP4 target slot differs from its actual original projection name");
      calibrated.projections[index]={codes[index],external(recipe,plan.bfloat_parameters->at(nvfp4[index].scales)),
          external(recipe,plan.bfloat_parameters->at(nvfp4[index].weight_global)),
          external(recipe,plan.bfloat_parameters->at(nvfp4[index].input_global))};
    }
    plan.result_nvfp4=nv::append_target(f,input,old_key,old_value,cursor,position,frequency,calibrated,geometry);
    plan.result=plan.result_nvfp4->block;
  } else if(!gptq.empty()) {
    namespace gp=qwen_gptq_projection_fragment;
    std::array<gp::Parameters,gp::kProjections> packed;
    const std::array codes{supplied.query,supplied.key,supplied.value,supplied.output,supplied.ffn_gate,supplied.ffn_up,supplied.ffn_down};
    for(std::size_t index=0;index<codes.size();++index) {
      if(bfloat_constant(recipe,codes[index]).name!=gptq[index].qweight)
        throw std::runtime_error("GPTQ target slot differs from actual original qweight name");
      packed[index]={codes[index],external(recipe,plan.bfloat_parameters->at(gptq[index].qzeros)),
          external(recipe,plan.bfloat_parameters->at(gptq[index].scales)),
          external(recipe,plan.bfloat_parameters->at(gptq[index].group_index)),gptq[index].group_size};
    }
    plan.result_gptq=gp::append_target(f,input,old_key,old_value,cursor,position,frequency,supplied,geometry,packed);
    plan.result=plan.result_gptq->block;
  } else {
    namespace cp=qwen_ct_asymmetric_projection_fragment;
    std::array<cp::Parameters,cp::kProjections> packed;
    const std::array codes{supplied.query,supplied.key,supplied.value,supplied.output,supplied.ffn_gate,supplied.ffn_up,supplied.ffn_down};
    for(std::size_t index=0;index<codes.size();++index) {
      const auto &bank=bfloat_constant(recipe,codes[index]);
      if(bank.name!=ct[index].codes||bank.description.data_type()!=c::DataType::kInt32||bank.description.rank()!=2||
          bank.description.dimensions()[1]>std::numeric_limits<std::int64_t>::max()/8)
        throw std::runtime_error("CT target slot differs from actual original packed code role");
      const auto &shape=plan.bfloat_parameters->at(ct[index].shape);
      if(shape.description.data_type()!=c::DataType::kInt64||shape.description.rank()!=1||shape.description.dimensions()[0]!=2)
        throw std::runtime_error("CT shape must remain original I64[2] metadata");
      // Authenticated metadata is checked here before compile, never made a
      // fourth decode input or passed through a numeric/gamma observation.
      bfloat::require_ct_shape(plan.bfloat_parameters->source->bytes(shape.name),bank.description.dimensions()[0],bank.description.dimensions()[1]*8);
      packed[index]={codes[index],external(recipe,plan.bfloat_parameters->at(ct[index].zeros)),
          external(recipe,plan.bfloat_parameters->at(ct[index].scales)),ct[index].group_size};
    }
    plan.result_ct=cp::append_target(f,input,old_key,old_value,cursor,position,frequency,supplied,geometry,packed);
    plan.result=plan.result_ct->block;
  }
  const auto expose_value=[&](Value value) {
    const auto found=std::find_if(plan.observed.begin(),plan.observed.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found!=plan.observed.end())return found->second;
    const auto port=expose(recipe,value);plan.observed.push_back({value,port});return port;
  };
  plan.result.ports.visit_values([&](Value &value){(void)expose_value(value);});
  if(!plan.result.trace)throw std::runtime_error("BF16 target must retain its actual local storage boundaries");
  plan.result.trace->visit_values([&](Value &value){(void)expose_value(value);});
  if(plan.result_nvfp4)for(auto &projection:plan.result_nvfp4->projections)
    projection.visit_values([&](Value &value){(void)expose_value(value);});
  std::vector<std::size_t> packed_matrices;
  if(plan.result_gptq)for(auto &projection:plan.result_gptq->projections) {
    projection.visit_values([&](Value &value){(void)expose_value(value);});
    packed_matrices.push_back(plan.port(projection.stored));
  }
  if(plan.result_ct)for(auto &projection:plan.result_ct->projections) {
    projection.visit_values([&](Value &value){(void)expose_value(value);});
    packed_matrices.push_back(plan.port(projection.stored));
  }
  plan.members={expose_value(plan.result.key),expose_value(plan.result.value),expose_value(increment(f,cursor)),
      expose_value(increment(f,position)),expose_value(plan.result.raw_post_ffn),expose_value(plan.result.normalized_tap)};
  recipe.bfloat_ports=plan.observed;
  recipe.state_ports[kKey]=plan.members[kTargetKey];recipe.state_ports[kValue]=plan.members[kTargetValue];
  recipe.state_ports[kHidden]=plan.members[kTargetTap];recipe.qg=plan.port(plan.result.ports.qg);
  recipe.key=plan.port(plan.result.ports.key);recipe.value=plan.port(plan.result.ports.value);
  recipe.query=plan.port(plan.result.ports.query);recipe.gate=plan.port(plan.result.ports.gate);
  recipe.rotated_key=plan.port(plan.result.ports.rotated_key);recipe.attended=plan.port(plan.result.ports.attended);
  recipe.oproj=plan.port(plan.result.ports.oproj);recipe.residual=plan.port(plan.result.ports.residual);
  recipe.ffn_output=plan.port(plan.result.ports.ffn_down);recipe.post_ffn=plan.port(plan.result.ports.post_ffn);
  recipe.dense_ffn_ports={plan.port(plan.result.ports.ffn_gate),plan.port(plan.result.ports.ffn_up),plan.port(plan.result.ports.ffn_activation)};
  plan.role_count=recipe.external.size();plan.source_bytes=0;
  for(const auto &binding:recipe.external)plan.source_bytes+=take(binding.role.description.required_bytes(),"actual BF16 target source extent");
  plan.program=Program::create(std::move(recipe),context,owner,cuda,true,{},{},{},packed_matrices,plan.result_ct?3:4);plan.program.report_bfloat_roles(owner);
  return plan;
}
// Same ordinary two-phase Program and target six-member state. The actual
// target static roles/selected source remain distinct from NextN's roles.
TargetPlan prepare_bfloat_moe_target(const Model &model,std::shared_ptr<const bfloat::Bound> parameters,
    const bfloat::DenseNormConventions &rules,const bfloat_moe_target::Names &names,
    TargetVocabulary vocabulary,std::shared_ptr<ExpertSupply> supply,c::Context &context,const c::StorageOwner &owner,bool cuda,
    std::shared_ptr<MixedPlacement> mixed={}) {
  if(model.storage!=c::DataType::kBFloat16||!model.moe||!parameters||!parameters->source||!supply||!supply->source.owner)
    throw std::runtime_error("BF16 MoE target requires actual static roles and a bounded selected original source");
  Recipe recipe{.model=model,.fragment={.profile=model.profile,.region=0x51424631364d5447ULL}};
  recipe.original_bfloat=parameters;recipe.bfloat_rules=rules;auto &f=recipe.fragment;
  const auto input=f.input(sem::dense({1,model.hidden},model.storage));
  const auto old_key=f.input(descriptions(model)[kKey]),old_value=f.input(descriptions(model)[kValue]);
  const auto cursor=f.input(sem::dense({1},c::DataType::kInt32)),position=f.input(sem::dense({1},c::DataType::kInt32));
  const std::array banks{f.input(supply->layout().resident(0)),f.input(supply->layout().resident(1)),f.input(supply->layout().resident(2))};
  const auto map=f.input(supply->layout().map_description());
  const auto frequency=f.constant(sem::dense({model.rotary/2}),sem::bytes(text_frequencies(model)));
  TargetPlan plan;plan.vocabulary=std::move(vocabulary);plan.bfloat_parameters=parameters;
  for(std::size_t slot=0;slot<names.size();++slot)plan.bfloat_moe_names[slot]=parameters->at(names[slot]).name;
  plan.result_bfloat_moe=bfloat_moe_target::append(f,input,old_key,old_value,cursor,position,frequency,*parameters,names,rules,
      {model.hidden,model.heads,model.kv_heads,model.depth,model.rotary,1.e-6F,model.score_scale,model.storage},banks,map);
  const auto &actual=*plan.result_bfloat_moe;
  for(const auto &constant:actual.constants) {
    const auto found=std::find_if(f.constants.begin(),f.constants.end(),[&](Value value){return value.ordinal()==constant.value.ordinal();});
    if(found==f.constants.end())throw std::runtime_error("BF16 target parameter lost its original constant identity");
    recipe.external.push_back({std::size_t(found-f.constants.begin()),constant.role});
  }
  recipe.moe=moe24::Result{actual.feed_forward.output,actual.feed_forward.ports.values,actual.feed_forward.boundary};
  const auto expose_value=[&](Value value) {
    const auto found=std::find_if(plan.observed.begin(),plan.observed.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found!=plan.observed.end())return found->second;
    const auto port=expose(recipe,value);plan.observed.push_back({value,port});return port;
  };
  plan.result_bfloat_moe->visit_values([&](Value &value){(void)expose_value(value);});
  auto linear=bfloat_moe::linear_stores(f,actual.feed_forward);
  linear.visit_values([&](Value &value){(void)expose_value(value);});
  plan.members={expose_value(actual.attention.key),expose_value(actual.attention.value),expose_value(increment(f,cursor)),
      expose_value(increment(f,position)),expose_value(actual.post_ffn),expose_value(actual.normalized_tap)};
  recipe.bfloat_ports=plan.observed;
  recipe.state_ports[kKey]=plan.members[kTargetKey];recipe.state_ports[kValue]=plan.members[kTargetValue];recipe.state_ports[kHidden]=plan.members[kTargetTap];
  const auto &ap=actual.attention.ports;
  recipe.qg=plan.port(ap.qg);recipe.key=plan.port(ap.key);recipe.value=plan.port(ap.value);recipe.query=plan.port(ap.query);recipe.gate=plan.port(ap.gate);
  recipe.rotated_key=plan.port(ap.rotated_key);recipe.attended=plan.port(ap.attended);recipe.oproj=plan.port(ap.oproj);recipe.residual=plan.port(ap.residual);
  recipe.ffn_input=plan.port(ap.ffn_input);recipe.ffn_output=plan.port(actual.feed_forward.output);recipe.post_ffn=plan.port(actual.post_ffn);
  plan.role_count=recipe.external.size();plan.source_bytes=0;
  for(const auto &binding:recipe.external)plan.source_bytes+=take(binding.role.description.required_bytes(),"actual BF16 MoE target source extent");
  plan.program=Program::create(std::move(recipe),context,owner,cuda,true,std::move(supply),std::move(mixed));plan.program.report_bfloat_roles(owner);return plan;
}
TargetPlan prepare_target(std::shared_ptr<const target_roles::Bound> parameters,TargetVocabulary vocabulary,c::Context &context,
    const c::StorageOwner &owner,bool cuda,std::shared_ptr<const full::Roles> named_parameters={},const Model &model=q27) {
  if(bool(parameters)==bool(named_parameters)||(parameters&&!parameters->source)||(named_parameters&&!named_parameters->source))
    throw std::runtime_error("original target63 needs exactly one authenticated parameter source");
  if(model.storage!=c::DataType::kFloat32)throw std::runtime_error("GGUF target63 preserves F32 state");
  Recipe recipe{.model=model,.fragment={.profile=model.profile,.region=0x5157454e54363301ULL}};auto &f=recipe.fragment;
  const auto input=f.input(sem::dense({1,model.hidden}));
  const auto old_key=f.input(descriptions(model)[kKey]),old_value=f.input(descriptions(model)[kValue]);
  const auto cursor=f.input(sem::dense({1},c::DataType::kInt32)),position=f.input(sem::dense({1},c::DataType::kInt32));
  const auto role=[&](target_roles::Role value){return external(recipe,named_parameters?named_parameters->at(target_roles::spec(value).name):parameters->at(value));};
  const target_model::TargetParameters supplied{role(target_roles::Role::kAttentionNorm),role(target_roles::Role::kQuery),
      role(target_roles::Role::kKey),role(target_roles::Role::kValue),role(target_roles::Role::kQueryNorm),role(target_roles::Role::kKeyNorm),
      role(target_roles::Role::kOutput),role(target_roles::Role::kPostNorm),role(target_roles::Role::kFfnGate),role(target_roles::Role::kFfnUp),
      role(target_roles::Role::kFfnDown),role(target_roles::Role::kTargetNorm)};
  const auto frequency=f.constant(sem::dense({32}),sem::bytes(text_frequencies(model)));
  TargetPlan plan;plan.vocabulary=std::move(vocabulary);plan.parameters=std::move(parameters);plan.named_parameters=std::move(named_parameters);
  plan.result=target_model::append_target(f,input,old_key,old_value,cursor,position,frequency,supplied,
      {model.hidden,model.heads,kKvHeads,kDepth,64,1e-6F,0.0625F});
  const auto expose_value=[&](Value value) {
    const auto found=std::find_if(plan.observed.begin(),plan.observed.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found!=plan.observed.end())return found->second;
    const auto port=expose(recipe,value);plan.observed.push_back({value,port});return port;
  };
  plan.result.ports.visit_values([&](Value &value){(void)expose_value(value);});
  if(plan.result.trace)plan.result.trace->visit_values([&](Value &value){(void)expose_value(value);});
  plan.members={expose_value(plan.result.key),expose_value(plan.result.value),expose_value(increment(f,cursor)),
      expose_value(increment(f,position)),expose_value(plan.result.raw_post_ffn),expose_value(plan.result.normalized_tap)};
  plan.program=Program::create(std::move(recipe),context,owner,cuda,true);
  const auto &graph=plan.program.logical.graph;
  const auto actual=[&](Value value) {
    const auto copy=take(graph.producer(plan.program.logical.outputs[plan.port(value)]),"target actual observer");
    if(take(graph.operation_kind(copy),"target observer kind")!=c::OperationKind::kCopy)throw std::runtime_error("target outputs need independent caller storage");
    return take(graph.node_input(copy,0),"target actual observed predecessor");
  };
  const auto constant=[&](target_roles::Role value) {
    const auto &expected=plan.named_parameters?plan.named_parameters->at(target_roles::spec(value).name):plan.parameters->at(value);
    const auto found=std::find_if(plan.program.recipe.external.begin(),plan.program.recipe.external.end(),[&](const auto &entry){return entry.role.name==expected.name;});
    if(found==plan.program.recipe.external.end()||found->role.description!=expected.description||found->role.owner.ref()!=owner.ref())
      throw std::runtime_error("target lost original constant identity/owner");
    const auto a=take(found->role.lease.view(),"target actual constant"),b=take(expected.lease.view(),"target original constant");
    if(a.resource()!=b.resource()||a.native_handle()!=b.native_handle()||a.offset()!=b.offset()||a.size()!=b.size()||
        found->role.lease.generation()!=expected.lease.generation())throw std::runtime_error("target copied or replaced an original readonly bank");
    return plan.program.logical.constants[found->constant];
  };
  const auto &ports=plan.result.ports;
  struct Projection final {target_roles::Role role;Value output,input;};
  const std::array projections{Projection{target_roles::Role::kQuery,ports.qg,ports.nx},
      Projection{target_roles::Role::kKey,ports.key,ports.nx},Projection{target_roles::Role::kValue,ports.value,ports.nx},
      Projection{target_roles::Role::kOutput,ports.oproj,ports.attended},Projection{target_roles::Role::kFfnGate,ports.ffn_gate,ports.ffn_input},
      Projection{target_roles::Role::kFfnUp,ports.ffn_up,ports.ffn_input},Projection{target_roles::Role::kFfnDown,ports.ffn_down,ports.ffn_activation}};
  for(const auto &expected:projections) {
    const auto node=take(graph.producer(actual(expected.output)),"target original projection");
    const auto operation=take(graph.operation(node),"target actual projection operation");const auto *matmul=operation.get_if<c::MatMul>();
    auto input_value=take(graph.node_input(node,0),"target projection actual input");
    const auto &bank=plan.named_parameters?plan.named_parameters->at(target_roles::spec(expected.role).name):plan.parameters->at(expected.role);
    const bool bfloat=attention::bfloat_weight(bank.description);
    if(bfloat) {
      if(!operation.custom_as<c::model_ops::DenseLinearF32>())throw std::runtime_error("target BF16-W needs its declared F32 accumulation");
      const auto conversion=take(graph.producer(input_value),"target actual BF16 activation store");
      const auto converted=take(graph.operation(conversion),"target BF16 activation operation");const auto *store=converted.get_if<c::Cast>();
      if(!store||store->output_type!=c::DataType::kBFloat16)throw std::runtime_error("target BF16-W/F32-state requires actual activation Cast");
      input_value=take(graph.node_input(conversion,0),"target actual F32 activation predecessor");
    }
    if(expected.role==target_roles::Role::kOutput) {
      const auto flatten=take(graph.producer(input_value),"target attended flatten");
      const auto view=take(graph.operation(flatten),"target flatten operation");const auto *reshape=view.get_if<c::View>();
      if(!reshape||reshape->kind()!=c::ViewKind::kReshape)throw std::runtime_error("target O must flatten its actual attended heads");
      input_value=take(graph.node_input(flatten,0),"target actual attended heads");
    }
    if((!bfloat&&(!matmul||matmul->accumulation_type!=c::DataType::kFloat32))||input_value!=actual(expected.input)||
        take(graph.node_input(node,1),"target actual original RHS")!=constant(expected.role))
      throw std::runtime_error("target projection must consume its actual predecessor and distinct original codec bank");
    std::size_t selected_count=0;std::string provider;
    for(std::size_t step=0;step<plan.program.compiled.report.step_count();++step) {
      const auto selected=take(plan.program.compiled.report.step(step),"target actual projection route");
      for(std::size_t index=0;index<selected.node_count;++index)
        if(take(plan.program.compiled.report.step_node(step,index),"target selected logical projection")==node) {
          ++selected_count;provider=selected.provider.name;
#if defined(HYLE_TEST_QWEN_STATE_CPU)
          if(!cuda&&!bfloat&&selected.provider.id!=c::ggml_cpu::kProviderId)throw std::runtime_error("target AQ reference requires its fixed CPU ggml route");
#endif
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
          if(cuda&&!bfloat&&selected.provider.id!=c::ggml_cuda::kProviderId)throw std::runtime_error("target AQ reference requires its fixed CUDA MMVQ route");
#endif
        }
    }
    if(selected_count!=1)throw std::runtime_error("target projection needs exactly one actual compiled route");
    ::testing::Test::RecordProperty(std::string("target_provider_")+std::string(target_roles::spec(expected.role).name),provider);
  }
  const auto target_norm=take(graph.producer(actual(ports.tap)),"target global output norm");
  const auto norm_operation=take(graph.operation(target_norm),"target final norm operation");const auto *rms=norm_operation.get_if<c::RmsNorm>();
  if(!rms||rms->epsilon!=1e-6F||take(graph.node_input(target_norm,0),"target final norm input")!=actual(ports.post_ffn)||
      take(graph.node_input(target_norm,1),"target original global norm gamma")!=constant(target_roles::Role::kTargetNorm))
    throw std::runtime_error("Qwen target h_nextn must follow its distinct original global output_norm");
  if(plan.named_parameters) {
    plan.role_count=plan.program.recipe.external.size();plan.source_bytes=0;
    for(const auto &binding:plan.program.recipe.external)plan.source_bytes+=take(binding.role.description.required_bytes(),"actual named target role bytes");
  }
  return plan;
}
TargetPlan prepare_target24(std::shared_ptr<const packed_moe::TargetRoles> parameters,TargetVocabulary vocabulary,std::shared_ptr<ExpertSupply> supply,
    c::Context &context,const c::StorageOwner &owner,bool cuda,const Model &model=q24) {
  if(!parameters||!parameters->source||!supply||supply->source.owner.get()!=parameters->source.get())
    throw std::runtime_error("target91 needs its own original readonly expert source");
  for(std::size_t leg=0;leg<3;++leg) {
    const auto kind=static_cast<full24::Leg>(leg);
    if(supply->source.expert_description(kind)!=parameters->source->expert_description(kind)||
        supply->layout().source(leg)!=parameters->source->expert_description(kind))
      throw std::runtime_error("target91 supply must retain its exact actual original per-leg descriptor");
  }
  Recipe recipe{.model=model,.fragment={.profile=model.profile,.region=0x5157454e54393101ULL}};auto &f=recipe.fragment;
  const auto input=f.input(sem::dense({1,model.hidden}));
  const auto old_key=f.input(descriptions(model)[kKey]),old_value=f.input(descriptions(model)[kValue]);
  const auto cursor=f.input(sem::dense({1},c::DataType::kInt32)),position=f.input(sem::dense({1},c::DataType::kInt32));
  const auto role=[&](target24_roles::Role value){return external(recipe,parameters->at(value));};
  const target_model::TargetAttentionParameters attention{role(target24_roles::Role::kAttentionNorm),role(target24_roles::Role::kQuery),
      role(target24_roles::Role::kKey),role(target24_roles::Role::kValue),role(target24_roles::Role::kQueryNorm),role(target24_roles::Role::kKeyNorm),
      role(target24_roles::Role::kOutput),role(target24_roles::Role::kPostNorm)};
  const moe24::Parameters feed_forward{role(target24_roles::Role::kRouter),role(target24_roles::Role::kSharedRouter),
      {role(target24_roles::Role::kSharedGate),role(target24_roles::Role::kSharedUp),role(target24_roles::Role::kSharedDown)},
      {f.input(supply->layout().resident(0)),f.input(supply->layout().resident(1)),f.input(supply->layout().resident(2))},
      f.input(supply->layout().map_description())};
  const auto gamma=role(target24_roles::Role::kTargetNorm);
  const auto frequency=f.constant(sem::dense({32}),sem::bytes(text_frequencies(model)));
  TargetPlan plan;plan.vocabulary=std::move(vocabulary);plan.parameters24=std::move(parameters);plan.role_count=target24_roles::kRoles;plan.source_bytes=0;for(const auto payload:plan.parameters24->source->payloads)plan.source_bytes+=payload.size();
  plan.result24=target24_model::append(f,input,old_key,old_value,cursor,position,frequency,{attention,feed_forward,gamma},
      {model.hidden,model.heads,model.kv_heads,model.depth,model.rotary,1e-6F,0.0625F});
  recipe.moe=plan.result24->feed_forward;
  const auto expose_value=[&](Value value) {
    const auto found=std::find_if(plan.observed.begin(),plan.observed.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found!=plan.observed.end())return found->second;
    const auto port=expose(recipe,value);plan.observed.push_back({value,port});return port;
  };
  plan.result24->visit_values([&](Value &value){(void)expose_value(value);});
  plan.members={expose_value(plan.result24->attention.key),expose_value(plan.result24->attention.value),expose_value(increment(f,cursor)),
      expose_value(increment(f,position)),expose_value(plan.result24->raw_post_ffn),expose_value(plan.result24->normalized_tap)};
  plan.program=Program::create(std::move(recipe),context,owner,cuda,true,std::move(supply));
  // Resolve each observation in its ACTUAL ordinary phase. The target never
  // attaches the distinct NextN92 Roles just to reuse a report or executor.
  const auto actual=[&](Value value){return plan.program.observation(plan.port(value));};
  const auto &ap=plan.result24->attention.ports;const auto &mp=plan.result24->feed_forward.ports;
  struct Projection final {target24_roles::Role role;Value output,input;};
  const std::array projections{Projection{target24_roles::Role::kQuery,ap.qg,ap.nx},Projection{target24_roles::Role::kKey,ap.key,ap.nx},
      Projection{target24_roles::Role::kValue,ap.value,ap.nx},Projection{target24_roles::Role::kOutput,ap.oproj,ap.attended},
      Projection{target24_roles::Role::kRouter,mp.logits,mp.input},Projection{target24_roles::Role::kSharedRouter,mp.shared_gate_raw,mp.input},
      Projection{target24_roles::Role::kSharedGate,mp.shared.gate,mp.input},Projection{target24_roles::Role::kSharedUp,mp.shared.up,mp.input},
      Projection{target24_roles::Role::kSharedDown,mp.shared.down,mp.shared.activation}};
  for(const auto &expected:projections) {
    const auto observed=actual(expected.output);const auto &graph=observed.program->logical.graph;
    const auto node=take(graph.producer(observed.value),"target91 actual static projection");
    const auto operation=take(graph.operation(node),"target91 projection operation");const auto *matmul=operation.get_if<c::MatMul>();
    auto input_value=take(graph.node_input(node,0),"target91 projection activation");
    if(expected.role==target24_roles::Role::kOutput) {
      const auto flatten=take(graph.producer(input_value),"target91 attended flatten");
      const auto view=take(graph.operation(flatten),"target91 attended flatten operation");const auto *reshape=view.get_if<c::View>();
      if(!reshape||reshape->kind()!=c::ViewKind::kReshape)throw std::runtime_error("target91 output must flatten its actual attended heads");
      input_value=take(graph.node_input(flatten,0),"target91 actual attended heads");
    }
    if(!matmul||matmul->accumulation_type!=c::DataType::kFloat32||input_value!=plan.program.observation_input(plan.port(expected.input),observed)||
        take(graph.node_input(node,1),"target91 original static RHS")!=plan.program.constant_operand(plan.parameters24->at(expected.role),observed,owner))
      throw std::runtime_error("target91 projection lost its actual predecessor or distinct original static bank");
    const auto provider=plan.program.selected_provider(observed,node,plan.parameters24->source->spec(expected.role).codec.has_value());
    ::testing::Test::RecordProperty(std::string("target_provider_")+std::string(plan.parameters24->source->spec(expected.role).name),provider);
  }
  const auto &boundary=plan.result24->feed_forward.boundary;
  for(std::size_t leg=0;leg<3;++leg) {
    const auto output=leg==0?mp.routed.gate:leg==1?mp.routed.up:mp.routed.down;
    const auto observed=actual(output);const auto &graph=observed.program->logical.graph;
    const auto node=take(graph.producer(observed.value),"target91 actual selected expert projection");
    const auto operation=take(graph.operation(node),"target91 actual grouped operation");
    const auto grouped=operation.custom_as<c::model_ops::GroupedMatMul>();
    auto input_value=take(graph.node_input(node,1),"target91 actual expert activation");
    if(leg<2) {
      const auto flatten=take(graph.producer(input_value),"target91 expert input reshape");
      const auto view=take(graph.operation(flatten),"target91 expert input view");const auto *reshape=view.get_if<c::View>();
      if(!reshape||reshape->kind()!=c::ViewKind::kReshape)throw std::runtime_error("target91 expert must use its actual shared activation");
      input_value=take(graph.node_input(flatten,0),"target91 shared expert activation");
    }
    const auto bank=take(graph.node_input(node,0),"target91 actual packed resident bank");
    if(!grouped||bank!=plan.program.phase_input(boundary.banks[leg],observed)||
        take(graph.value_description(bank),"target91 packed actual descriptor")!=plan.program.supply->layout().resident(leg)||
        input_value!=plan.program.observation_input(plan.port(leg<2?mp.input:mp.routed.activation),observed)||
        take(graph.node_input(node,2),"target91 actual local expert IDs")!=plan.program.observation_input(plan.port(mp.local_ids),observed))
      throw std::runtime_error("target91 experts require actual original packed bytes and exact generation mapping");
    ::testing::Test::RecordProperty(std::string("target_expert_provider_")+std::to_string(leg),plan.program.selected_provider(observed,node,true));
  }
  const auto normalized=actual(plan.result24->normalized_tap);const auto &graph=normalized.program->logical.graph;
  const auto node=take(graph.producer(normalized.value),"target91 original global norm");
  const auto operation=take(graph.operation(node),"target91 global norm operation");const auto *rms=operation.get_if<c::RmsNorm>();
  if(!rms||rms->epsilon!=1e-6F||take(graph.node_input(node,0),"target91 global norm input")!=plan.program.observation_input(plan.port(plan.result24->raw_post_ffn),normalized)||
      take(graph.node_input(node,1),"target91 original global norm gamma")!=plan.program.constant_operand(plan.parameters24->at(target24_roles::Role::kTargetNorm),normalized,owner))
    throw std::runtime_error("target91 h_nextn must follow the distinct original global output_norm");
  return plan;
}
void observe_target(TargetPlan &plan,std::span<const c::ReadLease> input,std::span<const c::ReadLease> output) {
  ASSERT_EQ(input.size(),5u);
  const auto cursor=typed<std::int32_t>(input[3]).at(0),position=typed<std::int32_t>(input[4]).at(0);
  if(plan.result_bfloat_moe) {
    ASSERT_TRUE(plan.bfloat_parameters);const auto &actual=*plan.result_bfloat_moe;
    ASSERT_TRUE(actual.attention.trace);const auto &trace=*actual.attention.trace;
    const auto bits=[&](Value value){return typed<std::uint16_t>(output[plan.port(value)]);};
    const auto observed=[&](Value value){return read(output[plan.port(value)]);};
    observe_bfloat_trace(plan.program,output,trace,!plan.full_checked);ASSERT_FALSE(::testing::Test::HasFailure());
    observe_bfloat_attention(plan.program,output,actual.attention.ports,trace,input[1],input[2],cursor,position);
    ASSERT_FALSE(::testing::Test::HasFailure());ASSERT_TRUE(plan.program.supply);
    const auto role_name=[&](bfloat_moe_target::Role role){return plan.bfloat_moe_names[std::size_t(role)];};
    const auto parameters=bfloat_moe::from_source(*plan.bfloat_parameters->source,role_name(bfloat_moe_target::Role::kRouter),
        role_name(bfloat_moe_target::Role::kSharedGate),std::array<std::string_view,3>{role_name(bfloat_moe_target::Role::kSharedProjectionGate),
        role_name(bfloat_moe_target::Role::kSharedUp),role_name(bfloat_moe_target::Role::kSharedDown)});
    bfloat_moe::observe(plan.program.recipe.fragment,parameters,actual.feed_forward,
        bfloat_moe::linear_stores(plan.program.recipe.fragment,actual.feed_forward),plan.program.supply->source,
        observed,!plan.full_checked,plan.program.original_full_values);
    ASSERT_FALSE(::testing::Test::HasFailure());
    ASSERT_TRUE(bfloat_reference::expect_binary(bits(actual.post_ffn),bits(actual.attention.ports.residual),bits(actual.feed_forward.output),false,"BF16 target MoE residual"));
    observe_bfloat_trace(plan.program,output,actual.tail_trace,!plan.full_checked);ASSERT_FALSE(::testing::Test::HasFailure());
    ASSERT_EQ(plan.program.supply->episodes,std::size_t(plan.program.submissions));ASSERT_FALSE(plan.program.supply->selections.empty());
    EXPECT_EQ(typed<std::int32_t>(output[plan.port(actual.feed_forward.ports.values.map)]),
        std::vector<std::int32_t>(plan.program.supply->selections.back().map.begin(),plan.program.supply->selections.back().map.end()));
    EXPECT_EQ(typed<std::int32_t>(output[plan.members[kTargetCursor]]),std::vector<std::int32_t>{cursor+1});
    EXPECT_EQ(typed<std::int32_t>(output[plan.members[kTargetPosition]]),std::vector<std::int32_t>{position+1});
    if(!plan.full_checked)plan.full_projection_values=plan.program.original_full_values;
    if(!::testing::Test::HasFailure())plan.full_checked=true;return;
  }
  if(plan.bfloat_parameters) {
    ASSERT_TRUE(plan.result.trace);const auto &trace=*plan.result.trace;
    const auto bits=[&](Value value){return typed<std::uint16_t>(output[plan.port(value)]);};
    observe_bfloat_trace(plan.program,output,trace,!plan.full_checked);ASSERT_FALSE(::testing::Test::HasFailure());
    if(plan.result_gptq)for(const auto &projection:plan.result_gptq->projections) {
      const auto &parameters=projection.parameters;const auto &recipe=plan.program.recipe;
      const auto &source=*plan.bfloat_parameters->source;
      const auto &codes=bfloat_constant(recipe,parameters.qweight),&zeros=bfloat_constant(recipe,parameters.qzeros),
          &scales=bfloat_constant(recipe,parameters.scales),&groups=bfloat_constant(recipe,parameters.group_index);
      const auto &shape=codes.description;
      ASSERT_EQ(shape.data_type(),c::DataType::kInt32);ASSERT_EQ(shape.rank(),2u);
      ASSERT_TRUE(qwen_gptq_original_reference::expect_projection(bits(projection.stored),bits(projection.input),
          source.bytes(codes.name),source.bytes(zeros.name),source.bytes(scales.name),source.bytes(groups.name),
          std::size_t(shape.dimensions()[1]),std::size_t(shape.dimensions()[0])*8,parameters.group_size,codes.name,!plan.full_checked));
      ASSERT_FALSE(::testing::Test::HasFailure());
      if(!plan.full_checked)plan.program.original_full_values+=bits(projection.stored).size();
    }
    if(plan.result_ct)for(const auto &projection:plan.result_ct->projections) {
      const auto &parameters=projection.parameters;const auto &recipe=plan.program.recipe;
      const auto &source=*plan.bfloat_parameters->source;
      const auto &codes=bfloat_constant(recipe,parameters.codes),&zeros=bfloat_constant(recipe,parameters.zeros),
          &scales=bfloat_constant(recipe,parameters.scales);
      const auto &shape=codes.description;
      ASSERT_EQ(shape.data_type(),c::DataType::kInt32);ASSERT_EQ(shape.rank(),2u);
      ASSERT_TRUE(qwen_ct_asymmetric_original_reference::expect_projection(bits(projection.stored),bits(projection.input),
          source.bytes(codes.name),source.bytes(zeros.name),source.bytes(scales.name),
          std::size_t(shape.dimensions()[0]),std::size_t(shape.dimensions()[1])*8,parameters.group_size,codes.name,!plan.full_checked));
      ASSERT_FALSE(::testing::Test::HasFailure());
      if(!plan.full_checked)plan.program.original_full_values+=bits(projection.stored).size();
    }
    if(plan.result_nvfp4)for(const auto &projection:plan.result_nvfp4->projections) {
      const auto &parameters=projection.parameters;const auto &recipe=plan.program.recipe;
      const auto &source=*plan.bfloat_parameters->source;
      const auto &codes=bfloat_constant(recipe,parameters.codes),&scales=bfloat_constant(recipe,parameters.scales),
          &wg=bfloat_constant(recipe,parameters.weight_global),&ig=bfloat_constant(recipe,parameters.input_global);
      const auto &shape=codes.description;
      ASSERT_EQ(shape.data_type(),c::DataType::kUInt8);ASSERT_EQ(shape.rank(),2u);
      ASSERT_TRUE(qwen_nvfp4_original_reference::expect_projection(bits(projection.stored),bits(projection.input),
          source.bytes(codes.name),source.bytes(scales.name),source.bytes(wg.name),source.bytes(ig.name),
          std::size_t(shape.dimensions()[0]),std::size_t(shape.dimensions()[1])*2,codes.name,!plan.full_checked));
      ASSERT_FALSE(::testing::Test::HasFailure());
      if(!plan.full_checked)plan.program.original_full_values+=bits(projection.stored).size();
    }
    observe_bfloat_attention(plan.program,output,plan.result.ports,trace,input[1],input[2],cursor,position);
    ASSERT_FALSE(::testing::Test::HasFailure());ASSERT_TRUE(trace.ffn_silu);
    ASSERT_TRUE(bfloat_reference::expect_unary(bits(*trace.ffn_silu),bits(plan.result.ports.ffn_gate),true,"BF16 target SiLU"));
    ASSERT_TRUE(bfloat_reference::expect_binary(bits(plan.result.ports.ffn_activation),bits(*trace.ffn_silu),bits(plan.result.ports.ffn_up),true,"BF16 target SwiGLU"));
    ASSERT_TRUE(bfloat_reference::expect_binary(bits(plan.result.ports.post_ffn),bits(plan.result.ports.residual),bits(plan.result.ports.ffn_down),false,"BF16 target FFN residual"));
    EXPECT_EQ(typed<std::int32_t>(output[plan.members[kTargetCursor]]),std::vector<std::int32_t>{cursor+1});
    EXPECT_EQ(typed<std::int32_t>(output[plan.members[kTargetPosition]]),std::vector<std::int32_t>{position+1});
    if(!plan.full_checked)plan.full_projection_values=plan.program.original_full_values;
    if(!::testing::Test::HasFailure())plan.full_checked=true;return;
  }
  const auto read_value=[&](Value value) {
    if(plan.program.recipe.fragment.description(value).data_type()==c::DataType::kBFloat16) {
      const auto bits=typed<std::uint16_t>(output[plan.port(value)]);std::vector<float> values;values.reserve(bits.size());
      for(const auto bit:bits)values.push_back(bfloat_reference::widen(bit));return values;
    }
    return typed<float>(output[plan.port(value)]);
  };
  const auto attention=[&](auto query,auto keys,auto values,auto gate,int rows){return attention_reference(plan.program.recipe.model,query,keys,values,gate,rows);};
  const auto codec=plan.program.cuda_backend?original_reference::ActivationCodec::kCudaMmvqQ8_1:original_reference::ActivationCodec::kCpuAvx2;
  if(plan.parameters24) {
    const auto integers=[&](Value value){return typed<std::int32_t>(output[plan.port(value)]);};
    target24_reference::observe(*plan.parameters24->source,*plan.result24,read_value,integers,typed<float>(input[1]),typed<float>(input[2]),cursor,position,
        codec,!plan.full_checked,attention,plan.full_projection_values);
    const auto ids=integers(plan.result24->feed_forward.ports.ids),mapping=integers(plan.result24->feed_forward.ports.map);
    ASSERT_TRUE(plan.program.supply);ASSERT_EQ(plan.program.supply->episodes,std::size_t(plan.program.submissions));
    ASSERT_FALSE(plan.program.supply->selections.empty());
    EXPECT_EQ(mapping,std::vector<std::int32_t>(plan.program.supply->selections.back().map.begin(),plan.program.supply->selections.back().map.end()));
    for(const auto global:ids)EXPECT_TRUE(std::binary_search(plan.program.supply->selections.back().globals.begin(),plan.program.supply->selections.back().globals.end(),global));
  } else if(plan.named_parameters)target_reference::observe(*plan.named_parameters->source,plan.result,read_value,typed<float>(input[1]),typed<float>(input[2]),cursor,position,
      codec,!plan.full_checked,attention,plan.full_projection_values);
  else target_reference::observe(*plan.parameters->source,plan.result,read_value,typed<float>(input[1]),typed<float>(input[2]),cursor,position,
      codec,!plan.full_checked,attention,plan.full_projection_values);
  EXPECT_EQ(typed<std::int32_t>(output[plan.members[kTargetCursor]]),std::vector<std::int32_t>{cursor+1});
  EXPECT_EQ(typed<std::int32_t>(output[plan.members[kTargetPosition]]),std::vector<std::int32_t>{position+1});
  if(!::testing::Test::HasFailure())plan.full_checked=true;
}
void same_state_storage(const c::ReadLease &actual,const c::ReadLease &expected) {
  const auto a=take(actual.view(),"actual whole-state lease"),b=take(expected.view(),"expected whole-state producer lease");
  EXPECT_EQ(a.resource(),b.resource());EXPECT_EQ(a.native_handle(),b.native_handle());EXPECT_EQ(a.offset(),b.offset());
  EXPECT_EQ(a.size(),b.size());EXPECT_EQ(actual.generation(),expected.generation());
}
std::vector<Bytes> whole_image(const State &state,const c::StateVersion &version) {
  std::vector<Bytes> result;for(std::size_t entry=0;entry<kResources;++entry)result.push_back(read(state.member(version,entry)));
  for(const auto entry:state.extra_entries)result.push_back(read(take(version.member(entry),"whole target state member")));
  if(state.sampling_entry)result.push_back(read(take(version.member(*state.sampling_entry),"same whole-prefix random cursor")));
  return result;
}
std::vector<float> upstream_fixture(const Model &model,std::int32_t token,std::int32_t position) {
  // This remains an external input from the preceding layer (62 or90).
  // Choosing a row from the consumed ID does not compute that trained trunk.
  if(token<0||token>=model.vocabulary||position<0)throw std::runtime_error("synthetic upstream boundary token/position");
  return d5_gguf_test::activation(model.hidden,101+int((std::int64_t(token)*13+position*7)%997));
}
std::vector<float> target_history(const Model &model,std::size_t valid,std::size_t variant) {
  std::vector<float> values(std::size_t(kCapacity*model.kv_heads*model.depth));
  for(std::size_t index=0;index<values.size();++index)values[index]=index<valid*std::size_t(model.kv_heads*model.depth)?
      float(int((index*7+variant*11)%31)-15)/64.F:std::bit_cast<float>(std::uint32_t(0x7fc00011u+variant));
  return values;
}
void run_original_target_verifier(Program draft,TargetPlan target,
    c::Context &context,const c::StorageOwner &owner,bool cuda,bool stochastic=false,
    hyle::inference::mtp::SamplingLimits limits={},std::size_t max_drafts=1) {
  const auto &model=draft.recipe.model;
  if(!max_drafts||max_drafts>2||(!stochastic&&max_drafts!=1))throw std::runtime_error("finite original Qwen component supports stochastic K1/K2");
  const auto planned_drafts=stochastic&&max_drafts==2&&limits.quota>2&&limits.remaining_context>2?2u:1u;
  if(stochastic)limits.validate(model.vocabulary);
  const auto &target_model=target.program.recipe.model;
  if(model.hidden!=target_model.hidden||model.vocabulary!=target_model.vocabulary||model.storage!=target_model.storage||
      model.kv_heads!=target_model.kv_heads||model.depth!=target_model.depth)
    throw std::runtime_error("target/draft K1 needs an explicit compatible tap/state/vocabulary contract");
  const auto &head_role=target.vocabulary.head;
  const auto packed_head=target.vocabulary.bytes;
  if(!target.vocabulary.source||!head_role.identity||packed_head.empty())
    throw std::runtime_error("target verifier vocabulary is explicit, never inferred from draft identity");
  auto verifier=prepare_vocabulary_head(target.program.recipe.model,head_role,context,owner,cuda,true,draft.mixed,planned_drafts+1);
  std::optional<Program> random_q,random_decision,next_target_logits,next_target_sampler,short_decision;
  std::optional<VocabularyHead> short_verifier;
  if(planned_drafts==2&&!limits.eos.empty())short_verifier=prepare_vocabulary_head(target.program.recipe.model,head_role,context,owner,cuda,false,draft.mixed,2);
  if(stochastic) {
    const auto prepare_sampler=[&](bool input_host,std::uint64_t region) {
      Recipe qr{.model=model,.fragment={.profile=model.profile,.region=region}};
      auto &qf=qr.fragment;
      const auto q=qf.input(sem::dense({1,model.vocabulary})),key=qf.input(sem::dense({1,4},c::DataType::kUInt32));
      const auto proposal=model_sampling_fragment::append_proposal(qf,q,key);
      qf.outputs={proposal.log_probabilities,proposal.proposal,proposal.valid};
      const auto domains=draft.mixed?std::vector<bool>{input_host,false}:std::vector<bool>{};
      return Program::create(std::move(qr),context,owner,cuda,true,{},draft.mixed,domains);
    };
    const bool q_input_host=draft.output_is_host(draft.recipe.logits);
    random_q=prepare_sampler(q_input_host,0x51574d545051524eULL);
    const auto prepare_decision=[&](std::size_t count) {
    Recipe pr{.model=model,.fragment={.profile=model.profile,.region=0x51574d545044524eULL}};
    auto &pf=pr.fragment;
    std::vector<bool> decision_domains;
    if(count==1) {
      const auto p=pf.input(sem::dense({2,model.vocabulary})),logq=pf.input(sem::dense({1,model.vocabulary}));
      const auto proposed=pf.input(sem::dense({1},c::DataType::kInt32));
      const auto accept=pf.input(sem::dense({1,4},c::DataType::kUInt32)),final=pf.input(sem::dense({1,4},c::DataType::kUInt32));
      const auto decision=model_sampling_fragment::append_decision(pf,p,logq,proposed,accept,final);
      pf.outputs={decision.accepted,decision.pending,decision.valid};
      if(draft.mixed)decision_domains={verifier.program.output_is_host(verifier.logits),random_q->output_is_host(0),random_q->output_is_host(1),false,false};
    } else {
      const auto p=pf.input(sem::dense({3,model.vocabulary}));
      const auto identities=pf.input(sem::dense({3,4},c::DataType::kUInt32));
      const auto first=pf.input(sem::dense({1},c::DataType::kInt32)),second=pf.input(sem::dense({1},c::DataType::kInt32));
      const auto proposed=pf.one(c::Operation(c::Concat{.axis=0}),{first,second});
      model_sampling_fragment::Builder r(pf);
      const auto decision=hyle::inference::sampling::speculative_prefix(r,p,identities,proposed);
      for(const auto value:decision.accepted)pf.outputs.push_back(r.copy(value));
      for(const auto value:decision.final_ids)pf.outputs.push_back(r.copy(value));
      for(const auto value:decision.valid)pf.outputs.push_back(r.copy(value));
      if(draft.mixed)decision_domains={verifier.program.output_is_host(verifier.logits),false,
          random_q->output_is_host(1),random_q->output_is_host(1),random_q->output_is_host(0),random_q->output_is_host(0)};
    }
    return Program::create(std::move(pr),context,owner,cuda,true,{},draft.mixed,decision_domains);
    };
    random_decision=prepare_decision(planned_drafts);
    if(short_verifier)short_decision=prepare_decision(1);
    // Ordinary one-row target continuation borrows the SAME authenticated
    // target head. It does not reinterpret NextN q as target probability.
    Recipe nr{.model=target_model,.fragment={.profile=target_model.profile,.region=0x51574d54504e5854ULL}};
    auto &nf=nr.fragment;const auto tap=nf.input(sem::dense({1,model.hidden},model.storage));
    const bool bf16=head_role.description.is_plain()&&head_role.description.data_type()==c::DataType::kBFloat16;
    const auto matrix=external(nr,head_role),input=bf16?attention::cast(nf,tap,c::DataType::kBFloat16):tap;
    const auto logits=nf.one(bf16?c::model_ops::dense_linear_f32():c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{input,matrix});
    (void)expose(nr,logits);
    const auto target_domains=draft.mixed?std::vector<bool>{target.program.output_is_host(target.members[kTargetTap])}:std::vector<bool>{};
    next_target_logits=Program::create(std::move(nr),context,owner,cuda,true,{},draft.mixed,target_domains);
    const bool next_logits_host=next_target_logits->output_is_host(0);
    if(next_logits_host!=q_input_host)next_target_sampler=prepare_sampler(next_logits_host,0x51574d545051524fULL);
    ::testing::Test::RecordProperty("sampling_q_input_host",int(q_input_host));
    ::testing::Test::RecordProperty("sampling_target_logits_input_host",int(verifier.program.output_is_host(verifier.logits)));
    ::testing::Test::RecordProperty("sampling_q_log_output_host",int(random_q->output_is_host(0)));
    ::testing::Test::RecordProperty("sampling_pending_output_host",int(random_decision->output_is_host(planned_drafts==2?2:1)));
    ::testing::Test::RecordProperty("sampling_next_target_logits_host",int(next_logits_host));
  }
  auto queue=take(context.create_queue(),"one target/NextN whole-state queue");
  std::array<c::ReadLease,5> warm_input{upload(owner,storage_bytes(model,upstream_fixture(model,2,36))),upload(owner,storage_bytes(model,target_history(model,1,2))),
      upload(owner,storage_bytes(model,target_history(model,1,3))),upload(owner,sem::bytes(std::vector<std::int32_t>{1})),
      upload(owner,sem::bytes(std::vector<std::int32_t>{36}))};
  auto warmup=target.program.execute_inputs(queue,warm_input);observe_target(target,warm_input,warmup);ASSERT_FALSE(::testing::Test::HasFailure());
  warm_input={};
  std::array<c::TensorDesc,kTargetResources> target_descriptions;
  for(std::size_t entry=0;entry<target_descriptions.size();++entry)
    target_descriptions[entry]=target.program.recipe.fragment.description(target.program.recipe.fragment.outputs[target.members[entry]]);
  std::size_t pool_bound=0;
  for(const auto *program:{&draft,&target.program,&verifier.program}) {
    if(pool_bound>std::numeric_limits<std::size_t>::max()-program->output_root_bytes)throw std::runtime_error("joint target output bound overflow");
    pool_bound+=program->output_root_bytes;
  }
  if(stochastic)for(const auto *program:{&*random_q,&*random_decision,&*next_target_logits}) {
    if(pool_bound>std::numeric_limits<std::size_t>::max()-program->output_root_bytes)throw std::runtime_error("joint random target output bound overflow");
    pool_bound+=program->output_root_bytes;
  }
  for(const auto *program:{short_verifier?&short_verifier->program:nullptr,short_decision?&*short_decision:nullptr})if(program) {
    if(pool_bound>std::numeric_limits<std::size_t>::max()-program->output_root_bytes)throw std::runtime_error("short verifier output bound overflow");
    pool_bound+=program->output_root_bytes;
  }
  if(next_target_sampler) {
    if(pool_bound>std::numeric_limits<std::size_t>::max()-next_target_sampler->output_root_bytes)throw std::runtime_error("separate target sampler output bound overflow");
    pool_bound+=next_target_sampler->output_root_bytes;
  }
  const auto codec=cuda?original_reference::ActivationCodec::kCudaMmvqQ8_1:original_reference::ActivationCodec::kCpuAvx2;
  std::size_t repairs=0,feedbacks=0,full_verifier_values=0,peak_backing=0,declared_budget=0;
  std::size_t repair_submissions=0;
  std::size_t publications=0,cancellations=0,stale_rejections=0,reseed_checks=0,bounded_tails=0;
  std::array<std::size_t,2> accepted_counts{};
  bool full_verifier_checked=false,multi_executed=false;
  std::size_t multi_accepted=0,multi_repaired_prefixes=0,multi_actual_drafts=0;
  std::array<c::ReadLease,2> controlled_proposals;
  std::int32_t natural_proposal=-1;std::uint32_t natural_accepted=0;
  for(int scenario=0;scenario<(stochastic?1:3);++scenario) {
    SCOPED_TRACE(scenario); // natural, independently controlled match/reject.
    auto state=State::create(model,owner,pool_bound,target_descriptions,stochastic);
    using SamplingCursor=hyle::inference::SamplingCursor;
    const SamplingCursor initial_cursor{0x5157454e32344dULL,19,0};
    const auto cursor=[&](const c::StateVersion &from) {
      if(!state.sampling_entry)throw std::runtime_error("Qwen stochastic state needs its declared cursor");
      const auto data=read(take(from.member(*state.sampling_entry),"actual Qwen random cursor"));SamplingCursor result;
      if(data.size()!=sizeof(result))throw std::runtime_error("Qwen cursor metadata extent");
      std::memcpy(&result,data.data(),sizeof(result));return result;
    };
    const auto identity=[&](SamplingCursor at,std::uint64_t offset) {
      const auto words=at.identity(offset);return upload(owner,std::as_bytes(std::span(words)));
    };
    const auto flag=[&](const c::ReadLease &lease) {
      const auto values=typed<std::uint8_t>(lease);
      if(values.size()!=1||values[0]>1)throw std::runtime_error("Qwen stochastic flag must be Bool[1]");return values[0]!=0;
    };
    const auto sample=[&](Program &sampler,const c::ReadLease &logits,SamplingCursor at) {
      const std::array inputs{logits,identity(at,0)};auto result=sampler.execute_inputs(queue,inputs);
      if(!flag(result[2]))throw std::runtime_error("actual Qwen distribution is invalid");
      const auto id=typed<std::int32_t>(result[1]);
      if(id.size()!=1||id[0]<0||id[0]>=model.vocabulary)throw std::runtime_error("actual Qwen sampled ID is outside vocabulary");
      return result;
    };
    const auto target_member=[&](const c::StateVersion &version,TargetEntry entry) {
      return take(version.member(state.extra_entries.at(std::size_t(entry))),"actual target checkpoint member");
    };
    const auto target_snapshot=[&](const c::StateVersion &version) {
      std::array<c::ReadLease,kTargetResources> result;
      for(std::size_t entry=0;entry<result.size();++entry)result[entry]=target_member(version,static_cast<TargetEntry>(entry));return result;
    };
    {
      const auto warm_state=target.state(warmup);
      const std::array<c::ReadLease,kResources> initial_values{upload(owner,storage_bytes(model,target_history(model,2,0))),
          upload(owner,storage_bytes(model,target_history(model,2,1))),warm_state[kTargetTap],upload(owner,sem::bytes(std::vector<std::int32_t>{3})),
          warm_state[kTargetCursor],warm_state[kTargetPosition]};
      const auto base=state.import(initial_values,warm_state,stochastic?std::optional{initial_cursor}:std::nullopt);const auto base_image=whole_image(state,base);
      auto bounded=limits;
      const auto current_id=typed<std::int32_t>(state.member(base,kPending)).at(0);
      if(stochastic) {
        const auto occupied=typed<std::int32_t>(state.member(base,kCursor)).at(0);
        if(occupied<0||occupied>=kCapacity)throw std::runtime_error("Qwen original target has no remaining context row");
        bounded.remaining_context=(std::min)(bounded.remaining_context,std::size_t(kCapacity-occupied));
        if(std::ranges::find(bounded.eos,current_id)!=bounded.eos.end())
          throw std::runtime_error("Qwen cannot consume an EOS pending token");
      }
      auto head=take(c::StateHead::create(state.store,base),"one12-member target/NextN accepted Head");
      const auto reseed=[&](const c::StateVersion &from) {
        auto pending=take(state.store.begin(from),"fresh actual target-tap seed");
        check(pending.replace(state.entries[kHidden],target_member(from,kTargetTap)),"bind previous-position normalized target tap");
        ++reseed_checks;return take(pending.finalize(),"unpublished complete target-tap seed");
      };
      const auto combined=[&](const c::StateVersion &from,c::ReadLease *retained_logits=nullptr) {
        const auto cursor=typed<std::int32_t>(state.member(from,kCursor)).at(0),position=typed<std::int32_t>(state.member(from,kPosition)).at(0);
        const auto token=typed<std::int32_t>(state.member(from,kPending)).at(0);
        if(cursor!=typed<std::int32_t>(target_member(from,kTargetCursor)).at(0)||position!=typed<std::int32_t>(target_member(from,kTargetPosition)).at(0))
          throw std::runtime_error("target/NextN token frontiers differ before execution");
        const auto before=whole_image(state,from);
        const std::array target_input{upload(owner,storage_bytes(model,upstream_fixture(model,token,position))),target_member(from,kTargetKey),
            target_member(from,kTargetValue),target_member(from,kTargetCursor),target_member(from,kTargetPosition)};
        const auto target_output=target.program.execute_inputs(queue,target_input);observe_target(target,target_input,target_output);
        if(::testing::Test::HasFailure())throw std::runtime_error("original target numerical gate failed before staging");
        const auto draft_output=draft.execute(queue,state,from);observe(draft,state,from,draft_output);
        if(::testing::Test::HasFailure())throw std::runtime_error("original NextN numerical gate failed before staging");
        if(retained_logits)*retained_logits=draft_output[draft.recipe.logits];
        const auto next=state.candidate(from,draft.recipe,draft_output,target.state(target_output));
        for(std::size_t entry=0;entry<kTargetResources;++entry)same_state_storage(target_member(next,static_cast<TargetEntry>(entry)),target_output[target.members[entry]]);
        EXPECT_EQ(whole_image(state,from),before);++feedbacks;
        peak_backing=std::max(peak_backing,state.store.stats().backing_bytes);
        return next;
      };
      const auto seeded=reseed(base);same_state_storage(state.member(seeded,kHidden),target_member(base,kTargetTap));
      c::ReadLease draft_logits;
      const auto first=combined(seeded,stochastic?&draft_logits:nullptr);const auto first_image=whole_image(state,first);
      const auto single=[&](hyle::inference::mtp::SamplingLimits boundary,bool terminal_control) {
        boundary.validate(model.vocabulary);
        const auto before_target=target.program.native_submissions(),before_draft=draft.native_submissions();
        const std::array input{target_member(first,kTargetTap)};
        const auto logits=next_target_logits->execute_inputs(queue,input);
        auto &sampler=next_target_sampler?*next_target_sampler:*random_q;
        const auto sampled=sample(sampler,logits[0],initial_cursor);
        const auto id=typed<std::int32_t>(sampled[1]).at(0);const std::array terminal{id};
        if(terminal_control)boundary.eos=terminal;
        const std::array inputs{current_id},finals{id};std::vector<std::int32_t> emitted;
        const auto choice=hyle::inference::mtp::select_sampled_prefix(inputs,{},finals,boundary.quota,boundary.eos,
            std::int32_t(model.vocabulary),emitted);
        auto branch=take(c::StateHead::create(state.store,base),"Qwen single-row bounded Head");
        auto publication=take(state.store.begin(first),"complete mandatory-current Qwen state");
        check(publication.replace(state.entries[kPending],sampled[1]),"actual single target categorical pending lease");
        const auto rng=initial_cursor.advanced(1);
        check(publication.replace_metadata(*state.sampling_entry,std::as_bytes(std::span(&rng,1))),"one target draw commits with whole Qwen state");
        const auto selected=take(publication.finalize(),"Qwen single-row complete publication");
        EXPECT_FALSE(branch.publish(first.identity(),selected).ok());EXPECT_EQ(cursor(take(branch.current(),"single stale Head retains RNG")),initial_cursor);
        check(branch.publish(base.identity(),selected),"one target row publishes same-prefix state and RNG");
        EXPECT_EQ(choice.consumed,1u);EXPECT_EQ(choice.consumed_drafts,0u);EXPECT_EQ(cursor(selected),rng);
        same_state_storage(state.member(selected,kPending),sampled[1]);
        for(std::size_t entry=0;entry<kResources;++entry)if(entry!=kPending)same_state_storage(state.member(selected,entry),state.member(first,entry));
        for(std::size_t entry=0;entry<kTargetResources;++entry)same_state_storage(target_member(selected,static_cast<TargetEntry>(entry)),target_member(first,static_cast<TargetEntry>(entry)));
        EXPECT_EQ(whole_image(state,base),base_image);EXPECT_EQ(whole_image(state,first),first_image);
        EXPECT_EQ(target.program.native_submissions(),before_target);EXPECT_EQ(draft.native_submissions(),before_draft);
        EXPECT_EQ(typed<std::int32_t>(state.member(selected,kCursor)).at(0),3);
        EXPECT_EQ(typed<std::int32_t>(target_member(selected,kTargetCursor)).at(0),3);
        std::printf("QWEN RANDOM TAIL context=%zu quota=%zu pending=%d consumed=1 draw=1 no_second=1\n",boundary.remaining_context,boundary.quota,id);++bounded_tails;
      };
      if(stochastic&&planned_drafts==2&&!bounded.single_row()) {
        // This reuses the same real NextN and selected-MoE target programs.
        // The target body advances true rows; only its original full-V head
        // is evaluated as M3. No claim of a parallel T3 MoE body is made.
        std::array<c::StateVersion,3> prefixes{first,{},{}},repaired{first,{},{}};
        std::array<std::vector<c::ReadLease>,2> sampled;
        std::array<std::int32_t,3> inputs{current_id,-1,-1};
        sampled[0]=sample(*random_q,draft_logits,initial_cursor);
        inputs[1]=typed<std::int32_t>(sampled[0][1]).at(0);
        const auto terminal=[&](auto id){return std::ranges::find(bounded.eos,id)!=bounded.eos.end();};
        const std::size_t count=terminal(inputs[1])?1u:2u;
        multi_actual_drafts=count;
        c::ReadLease next_logits;
        for(std::size_t row=1;row<=count;++row) {
          auto source=take(state.store.begin(prefixes[row-1]),"actual recursive Qwen sampled proposal");
          check(source.replace(state.entries[kPending],sampled[row-1][1]),"actual original q ID drives target and NextN");
          const auto proposed=take(source.finalize(),"unpublished complete recursive proposal");
          prefixes[row]=combined(proposed,row<count?&next_logits:nullptr);
          if(row<count) {
            sampled[row]=sample(*random_q,next_logits,initial_cursor.advanced(row));
            inputs[row+1]=typed<std::int32_t>(sampled[row][1]).at(0);
          }
        }
        auto &head_program=count==2?verifier:*short_verifier;
        auto &decision_program=count==2?*random_decision:*short_decision;
        std::vector<c::ReadLease> taps;
        for(std::size_t row=0;row<=count;++row)taps.push_back(target_member(prefixes[row],kTargetTap));
        const auto verified=head_program.program.execute_inputs(queue,taps);
        const auto logits=typed<float>(verified[head_program.logits]);
        ASSERT_EQ(logits.size(),(count+1)*std::size_t(model.vocabulary));
        for(std::size_t row=0;row<=count;++row) {
          const auto values=std::span(logits).subspan(row*std::size_t(model.vocabulary),std::size_t(model.vocabulary));
          const auto id=typed<std::int32_t>(verified[head_program.pending[row]]).at(0);
          const auto scores=typed<float>(verified[head_program.scores]);
          ASSERT_TRUE(original_reference::expect_greedy(values,scores[row],id));
          const std::array<std::size_t,1> selected{std::size_t(id)};
          if(head_role.description.is_plain()&&head_role.description.data_type()==c::DataType::kBFloat16) {
            auto hidden=typed<std::uint16_t>(taps[row]);
            if(model.storage==c::DataType::kFloat32) {
              if(!head_program.stored_input)throw std::runtime_error("finite BF16 head lost its activation rounding");
              const auto stored=typed<std::uint16_t>(verified[*head_program.stored_input]);
              hidden.assign(stored.begin()+std::ptrdiff_t(row*std::size_t(model.hidden)),stored.begin()+std::ptrdiff_t((row+1)*std::size_t(model.hidden)));
            }
            const std::array<std::int32_t,1> predicted{id};
            ASSERT_TRUE(bfloat_reference::expect_projection(values,hidden,packed_head,std::size_t(model.vocabulary),std::size_t(model.hidden),"actual finite Qwen BF16 full-V verifier",false,predicted));
          } else if(model.moe)ASSERT_TRUE(original24_reference::expect_projection(values,packed_head,std::size_t(kVocabulary),std::size_t(model.hidden),target.vocabulary.native,typed<float>(taps[row]),codec,"actual M3 target full-V head",false,selected));
          else ASSERT_TRUE(original_reference::expect_projection(values,packed_head,std::size_t(kVocabulary),std::size_t(model.hidden),target.vocabulary.native,typed<float>(taps[row]),codec,"actual finite target full-V head",false,selected));
        }
        std::vector<c::ReadLease> decision_input;
        if(count==2) {
          std::vector<std::uint32_t> keys;
          for(std::size_t draw=count;draw<=2*count;++draw) {
            const auto key=initial_cursor.identity(draw);keys.insert(keys.end(),key.begin(),key.end());
          }
          decision_input={verified[head_program.logits],upload(owner,std::as_bytes(std::span(keys))),sampled[0][1],sampled[1][1],sampled[0][0],sampled[1][0]};
        } else decision_input={verified[head_program.logits],sampled[0][0],sampled[0][1],identity(initial_cursor,1),identity(initial_cursor,2)};
        const auto decision=decision_program.execute_inputs(queue,decision_input);
        std::vector<std::uint8_t> flags;std::vector<std::int32_t> finals;
        for(std::size_t row=0;row<count;++row) {
          if(!flag(decision[count==2?4+row:2]))throw std::runtime_error("actual finite Qwen p/q prefix distribution invalid");
          flags.push_back(std::uint8_t(flag(decision[row])));
          finals.push_back(typed<std::int32_t>(decision[count==2?2+row:1]).at(0));
        }
        std::vector<std::int32_t> emitted;
        const auto selected=hyle::inference::mtp::select_sampled_prefix(std::span(inputs).first(count+1),flags,finals,bounded.quota,bounded.eos,std::int32_t(model.vocabulary),emitted);
        const auto draws=2*count+1;
        const auto committed_cursor=initial_cursor.advanced(draws);
        // Repair all possible accepted prefixes once, in true target-tap
        // order. These controls test state selection, never claim additional
        // natural stochastic acceptance trials or choose new random addresses.
        for(std::size_t row=1;row<=count;++row) {
          auto seed=take(state.store.begin(repaired[row-1]),"teacher-forced complete accepted NextN prefix");
          check(seed.replace(state.entries[kPending],sampled[row-1][1]),"original accepted proposal ID retained");
          check(seed.replace(state.entries[kHidden],target_member(prefixes[row-1],kTargetTap)),"true target tap replaces recursive hidden before repair");
          const auto actual=take(seed.finalize(),"unpublished true-target repair seed");
          const auto output=draft.execute(queue,state,actual);observe(draft,state,actual,output);ASSERT_FALSE(::testing::Test::HasFailure());
          repaired[row]=state.candidate(prefixes[row],draft.recipe,output);
          for(std::size_t member=0;member<kResources;++member)same_state_storage(state.member(repaired[row],member),output[draft.recipe.state_ports[member]]);
          for(std::size_t member=0;member<kTargetResources;++member)same_state_storage(target_member(repaired[row],static_cast<TargetEntry>(member)),target_member(prefixes[row],static_cast<TargetEntry>(member)));
          ++multi_repaired_prefixes;
        }
        const auto publication=[&](std::size_t prefix,const c::ReadLease &pending) {
          auto candidate=take(state.store.begin(repaired.at(prefix)),"selected complete finite Qwen prefix");
          check(candidate.replace(state.entries[kPending],pending),"actual selected pending ID lease");
          check(candidate.replace_metadata(*state.sampling_entry,std::as_bytes(std::span(&committed_cursor,1))),"whole selected prefix and actual reserved draws publish together");
          return take(candidate.finalize(),"atomic finite Qwen state/pending/RNG candidate");
        };
        for(std::size_t prefix=0;prefix<=count;++prefix) {
          // A request may retain an emitted EOS as pending, but cannot commit
          // a controlled state prefix that already consumed that EOS.
          if(std::ranges::any_of(std::span(inputs).subspan(1,prefix),terminal))continue;
          // Actual target Top1 is an explicitly controlled pending token for
          // this state-only branch, distinct from the natural p/q decision.
          const auto pending=verified[head_program.pending[prefix]];
          const auto candidate=publication(prefix,pending);
          auto branch=take(c::StateHead::create(state.store,base),"independent legal accepted-prefix state control");
          EXPECT_FALSE(branch.publish(prefixes[count].identity(),candidate).ok());
          EXPECT_EQ(cursor(take(branch.current(),"stale prefix control retains original RNG")),initial_cursor);
          check(branch.publish(base.identity(),candidate),"whole0/1/2 repaired prefix publication");
          EXPECT_EQ(cursor(candidate),committed_cursor);
          EXPECT_EQ(typed<std::int32_t>(state.member(candidate,kCursor)).at(0),3+int(prefix));
          EXPECT_EQ(typed<std::int32_t>(target_member(candidate,kTargetCursor)).at(0),3+int(prefix));
          for(std::size_t member=0;member<kResources;++member)if(member!=kPending)same_state_storage(state.member(candidate,member),state.member(repaired[prefix],member));
          for(std::size_t member=0;member<kTargetResources;++member)same_state_storage(target_member(candidate,static_cast<TargetEntry>(member)),target_member(prefixes[prefix],static_cast<TargetEntry>(member)));
          same_state_storage(state.member(candidate,kPending),pending);
          std::printf("QWEN MULTI PREFIX prefix=%zu committed_cursor=%d actual_draws=%zu controlled_state_only=1\n",prefix,3+int(prefix),draws);
        }
        const auto pending_for=[&](const hyle::inference::mtp::AcceptedPrefix &choice) {
          if(choice.consumed<=count&&flags[choice.consumed-1]&&choice.next_id==inputs[choice.consumed])return sampled[choice.consumed-1][1];
          return decision[count==2?2+std::min(choice.consumed_drafts,count-1):1];
        };
        const auto pending=pending_for(selected);
        EXPECT_EQ(typed<std::int32_t>(pending).at(0),selected.next_id);
        const auto accepted_version=publication(selected.consumed_drafts,pending);
        auto failed=take(state.store.begin(accepted_version),"cancel wrong typed multi-prefix splice");
        EXPECT_FALSE(failed.replace(state.entries[kPending],target_member(first,kTargetTap)).ok());check(failed.cancel(),"cancel cannot publish state/RNG");
        EXPECT_EQ(cursor(take(head.current(),"cancelled K2 retains initial cursor")),initial_cursor);
        EXPECT_FALSE(head.publish(prefixes[count].identity(),accepted_version).ok());
        check(head.publish(base.identity(),accepted_version),"natural finite accepted prefix with pending and RNG");
        EXPECT_EQ(cursor(take(head.current(),"successful K2 committed cursor")),committed_cursor);
        EXPECT_EQ(typed<std::int32_t>(state.member(accepted_version,kCursor)).at(0),2+int(selected.consumed));
        EXPECT_EQ(typed<std::int32_t>(target_member(accepted_version,kTargetCursor)).at(0),2+int(selected.consumed));
        const auto boundary=[&](std::size_t quota,std::span<const std::int32_t> eos,const char *name) {
          std::vector<std::int32_t> tokens;const auto limited=hyle::inference::mtp::select_sampled_prefix(std::span(inputs).first(count+1),flags,finals,quota,eos,std::int32_t(model.vocabulary),tokens);
          const auto candidate=publication(limited.consumed_drafts,pending_for(limited));
          auto branch=take(c::StateHead::create(state.store,base),"bounded K2 whole-state Head");
          check(branch.publish(base.identity(),candidate),"bounded K2 prefix publishes atomically");
          EXPECT_EQ(typed<std::int32_t>(state.member(candidate,kCursor)).at(0),2+int(limited.consumed));
          EXPECT_EQ(typed<std::int32_t>(target_member(candidate,kTargetCursor)).at(0),2+int(limited.consumed));
          EXPECT_EQ(typed<std::int32_t>(state.member(candidate,kPending)).at(0),limited.next_id);EXPECT_EQ(cursor(candidate),committed_cursor);
          for(std::size_t member=0;member<kResources;++member)if(member!=kPending)same_state_storage(state.member(candidate,member),state.member(repaired[limited.consumed_drafts],member));
          for(std::size_t member=0;member<kTargetResources;++member)same_state_storage(target_member(candidate,static_cast<TargetEntry>(member)),target_member(prefixes[limited.consumed_drafts],static_cast<TargetEntry>(member)));
          std::printf("QWEN MULTI BOUNDARY name=%s emitted=%zu consumed=%zu pending=%d stopped=%d actual_draws=%zu\n",name,tokens.size(),limited.consumed,limited.next_id,int(limited.stopped),draws);
        };
        for(std::size_t quota=1;quota<=count+1;++quota)boundary(quota,bounded.eos,"quota");
        for(std::size_t row=0;row<count;++row)if(inputs[row+1]!=current_id) {
          const std::array eos{inputs[row+1]};boundary(count+1,eos,"proposal_EOS");
        }
        if(selected.next_id!=current_id) {const std::array eos{selected.next_id};boundary(count+1,eos,"final_EOS");}
        const auto accepted_image=whole_image(state,accepted_version);
        if(!selected.stopped&&emitted.size()<bounded.quota&&selected.consumed<bounded.remaining_context) {
          const auto seed=reseed(take(head.current(),"actual K2 accepted state for pending continuation"));
          auto continued=combined(seed);
          const std::array input{target_member(continued,kTargetTap)};const auto logits1=next_target_logits->execute_inputs(queue,input);
          auto &sampler=next_target_sampler?*next_target_sampler:*random_q;
          const auto sampled_next=sample(sampler,logits1[0],committed_cursor);
          auto next=take(state.store.begin(continued),"actual K2 residual/bonus target-only continuation");
          check(next.replace(state.entries[kPending],sampled_next[1]),"real next target sampled pending lease");
          const auto advanced=committed_cursor.advanced(1);
          check(next.replace_metadata(*state.sampling_entry,std::as_bytes(std::span(&advanced,1))),"one actual continuation draw");
          continued=take(next.finalize(),"complete actual post-K2 continuation");
          check(head.publish(accepted_version.identity(),continued),"next pending consumption publishes with target/draft state");
          EXPECT_EQ(cursor(continued),advanced);
          EXPECT_EQ(typed<std::int32_t>(state.member(continued,kCursor)).at(0),3+int(selected.consumed));
          std::printf("QWEN RANDOM MULTI K=%zu q1=%d q2=%d accepted=%zu pending=%d continued=%d draw=%llu head_rows=%zu target_body=sequential_true_rows\n",count,inputs[1],inputs[2],selected.consumed_drafts,selected.next_id,typed<std::int32_t>(state.member(continued,kPending)).at(0),static_cast<unsigned long long>(advanced.draw),count+1);
        }
        EXPECT_EQ(whole_image(state,base),base_image);EXPECT_EQ(whole_image(state,first),first_image);EXPECT_EQ(whole_image(state,accepted_version),accepted_image);
        multi_accepted=selected.consumed_drafts;multi_executed=true;
      } else {
      if(stochastic&&bounded.single_row()) {
        single(bounded,false);
        ::testing::Test::RecordProperty("random_tail_target_rows",1);::testing::Test::RecordProperty("random_tail_draws",1);
      } else {
      if(stochastic) {
        // Both request-tail shapes reuse the retired mandatory-current state,
        // before sampling q or submitting a proposed second target/NextN row.
        single({std::numeric_limits<std::size_t>::max(),1,{}},false);check(state.store.collect(),"retire context-one Qwen Head");
        single({1,std::numeric_limits<std::size_t>::max(),{}},false);check(state.store.collect(),"retire quota-one Qwen Head");
      }
      c::StateVersion proposal_source=first;
      std::vector<c::ReadLease> actual_q;
      if(stochastic) {
        actual_q=sample(*random_q,draft_logits,initial_cursor);
        auto selected=take(state.store.begin(first),"actual Qwen q categorical proposal");
        check(selected.replace(state.entries[kPending],actual_q[1]),"actual sampled I32 proposal feeds target row1");
        proposal_source=take(selected.finalize(),"same complete Qwen proposal checkpoint");
      }
      if(scenario!=0) {
        ASSERT_TRUE(controlled_proposals[std::size_t(scenario-1)]);
        auto policy=take(state.store.begin(first),"controlled actual Top1/Top2 proposal");
        check(policy.replace(state.entries[kPending],controlled_proposals[std::size_t(scenario-1)]),"actual target ID lease controls consumed proposal");
        proposal_source=take(policy.finalize(),"unpublished controlled proposal checkpoint");
        for(std::size_t entry=0;entry<kResources;++entry)if(entry!=kPending)same_state_storage(state.member(proposal_source,entry),state.member(first,entry));
      }
      const auto proposal=typed<std::int32_t>(state.member(proposal_source,kPending)).at(0);
      const auto second=combined(proposal_source);const auto second_image=whole_image(state,second);
      const std::array verifier_inputs{target_member(first,kTargetTap),target_member(second,kTargetTap)};
      const auto verified=verifier.program.execute_inputs(queue,verifier_inputs);
      const auto logits=typed<float>(verified[verifier.logits]),scores=typed<float>(verified[verifier.scores]);
      ASSERT_EQ(logits.size(),2*std::size_t(model.vocabulary));ASSERT_EQ(scores.size(),2u);
      const auto alternate=typed<std::int32_t>(verified[*verifier.alternate]).at(0);
      ASSERT_GE(alternate,0);ASSERT_LT(alternate,model.vocabulary);
      std::array<std::int32_t,2> target_ids;
      for(std::size_t row=0;row<2;++row) {
        const auto ids=typed<std::int32_t>(verified[verifier.pending[row]]);ASSERT_EQ(ids.size(),1u);target_ids[row]=ids[0];
        const auto values=std::span(logits).subspan(row*std::size_t(model.vocabulary),std::size_t(model.vocabulary));
        ASSERT_TRUE(original_reference::expect_greedy(values,scores[row],target_ids[row]));
        const std::array<std::size_t,2> selected{std::size_t(target_ids[row]),row==0?std::size_t(alternate):std::size_t(target_ids[row])};
        if(head_role.description.is_plain()&&head_role.description.data_type()==c::DataType::kBFloat16) {
          auto actual_hidden=typed<std::uint16_t>(verifier_inputs[row]);
          if(model.storage==c::DataType::kFloat32) {
            if(!verifier.stored_input)throw std::runtime_error("F32-state verifier BF16 head lost its actual activation store");
            const auto all=typed<std::uint16_t>(verified[*verifier.stored_input]);
            actual_hidden.assign(all.begin()+std::ptrdiff_t(row*std::size_t(model.hidden)),all.begin()+std::ptrdiff_t((row+1)*std::size_t(model.hidden)));
            ASSERT_TRUE(bfloat_reference::expect_store(actual_hidden,typed<float>(verifier_inputs[row]),"actual verifier BF16 activation store"));
          }
          const std::array<std::int32_t,1> predicted{target_ids[row]};
          ASSERT_TRUE(bfloat_reference::expect_projection(values,actual_hidden,packed_head,std::size_t(model.vocabulary),std::size_t(model.hidden),
              "actual BF16 target M2 full-V verifier",!full_verifier_checked,predicted));
        } else if(model.moe)ASSERT_TRUE(original24_reference::expect_projection(values,packed_head,std::size_t(kVocabulary),std::size_t(model.hidden),
            target.vocabulary.native,typed<float>(verifier_inputs[row]),codec,"actual target M2 full vocabulary verifier",!full_verifier_checked,selected));
        else ASSERT_TRUE(original_reference::expect_projection(values,packed_head,std::size_t(kVocabulary),std::size_t(model.hidden),
            target.vocabulary.native,typed<float>(verifier_inputs[row]),codec,"actual target M2 full vocabulary verifier",!full_verifier_checked,selected));
      }
      if(!full_verifier_checked){full_verifier_values=logits.size();full_verifier_checked=true;}
      std::array<std::size_t,2> top2{0,1};if(logits[1]>logits[0])std::swap(top2[0],top2[1]);
      for(std::size_t index=2;index<std::size_t(model.vocabulary);++index) {
        if(logits[index]>logits[top2[0]]){top2[1]=top2[0];top2[0]=index;}
        else if(logits[index]>logits[top2[1]])top2[1]=index;
      }
      EXPECT_EQ(alternate,std::int32_t(top2[1]));EXPECT_NE(alternate,target_ids[0]);
      std::vector<c::ReadLease> random_result;
      if(stochastic) {
        const std::array input{verified[verifier.logits],actual_q[0],actual_q[1],identity(initial_cursor,1),identity(initial_cursor,2)};
        random_result=random_decision->execute_inputs(queue,input);
        if(!flag(random_result[2]))throw std::runtime_error("actual Qwen p/q/bonus decision is invalid");
      }
      const bool sampled_accept=stochastic&&flag(random_result[0]);
      hyle::inference::mtp::AcceptedPrefix sampled_choice;std::vector<std::int32_t> emitted;
      if(stochastic) {
        const std::array inputs{current_id,proposal};const std::array finals{typed<std::int32_t>(random_result[1]).at(0)};
        const std::array flags{std::uint8_t(sampled_accept)};
        sampled_choice=hyle::inference::mtp::select_sampled_prefix(inputs,flags,finals,bounded.quota,bounded.eos,
            std::int32_t(model.vocabulary),emitted);
      }
      const std::uint32_t accepted=stochastic?std::uint32_t(sampled_choice.consumed_drafts):std::uint32_t(proposal==target_ids[0]);++accepted_counts[accepted];
      if(scenario==0) {
        controlled_proposals={verified[verifier.pending[0]],verified[*verifier.alternate]};
        natural_proposal=proposal;natural_accepted=accepted;
      } else EXPECT_EQ(accepted,scenario==1?1u:0u);
      const auto pending=stochastic?(sampled_accept&&sampled_choice.consumed==1?actual_q[1]:random_result[1]):verified[verifier.pending[accepted]];
      if(stochastic)EXPECT_EQ(typed<std::int32_t>(pending).at(0),sampled_choice.next_id);
      c::StateVersion accepted_version;
      if(accepted) {
        // Recursive second used draft hidden to speculate. Rebuild its row
        // with the TRUE target h[current] paired with proposal at current+1,
        // exactly the shifted-target process() cache repair, before publish.
        const auto repair_seed=reseed(proposal_source);
        same_state_storage(state.member(repair_seed,kHidden),target_member(first,kTargetTap));
        const auto repair_begin=draft.native_submissions();
        const auto repair_output=draft.execute(queue,state,repair_seed);observe(draft,state,repair_seed,repair_output);
        repair_submissions+=draft.native_submissions()-repair_begin;
        ASSERT_FALSE(::testing::Test::HasFailure());
        accepted_version=state.candidate(second,draft.recipe,repair_output,{},pending);++repairs;
        for(std::size_t entry=0;entry<kResources;++entry)
          same_state_storage(state.member(accepted_version,entry),entry==kPending?pending:repair_output[draft.recipe.state_ports[entry]]);
        for(std::size_t entry=0;entry<kTargetResources;++entry)
          same_state_storage(target_member(accepted_version,static_cast<TargetEntry>(entry)),target_member(second,static_cast<TargetEntry>(entry)));
      } else {
        auto policy=take(state.store.begin(first),"zero-draft correction keeps after-current checkpoint");
        check(policy.replace(state.entries[kPending],pending),"actual target correction pending lease");
        accepted_version=take(policy.finalize(),"whole after-current checkpoint with target correction");
        for(std::size_t entry=0;entry<kResources;++entry)if(entry!=kPending)same_state_storage(state.member(accepted_version,entry),state.member(first,entry));
        EXPECT_NE(accepted_version.identity(),base.identity());
      }
      if(stochastic) {
        auto publication=take(state.store.begin(accepted_version),"same whole-prefix Qwen stochastic publication");
        const auto next=initial_cursor.advanced(3);
        check(publication.replace_metadata(*state.sampling_entry,std::as_bytes(std::span(&next,1))),"pending/state and RNG commit together");
        accepted_version=take(publication.finalize(),"complete Qwen stochastic accepted checkpoint");
        auto failure=take(state.store.begin(accepted_version),"cancel incomplete Qwen stochastic publication");
        EXPECT_FALSE(failure.replace(state.entries[kPending],target_member(first,kTargetTap)).ok());
        check(failure.cancel(),"wrong typed result cannot publish state or RNG");++cancellations;
        EXPECT_EQ(cursor(take(head.current(),"cancelled Qwen decision retains original cursor")),initial_cursor);
        EXPECT_FALSE(head.publish(second.identity(),accepted_version).ok());++stale_rejections;
        EXPECT_EQ(cursor(take(head.current(),"stale Qwen publish retains original cursor")),initial_cursor);
      }
      same_state_storage(state.member(accepted_version,kPending),pending);
      EXPECT_EQ(typed<std::int32_t>(state.member(accepted_version,kCursor)).at(0),3+int(accepted));
      EXPECT_TRUE(take(head.current(),"unpublished repaired checkpoint").identity()==base.identity());
      check(head.publish(base.identity(),accepted_version),"atomic target and repaired NextN same-prefix publication");
      if(stochastic)EXPECT_EQ(cursor(take(head.current(),"successful Qwen K1 cursor")),initial_cursor.advanced(3));
      ++publications;
      EXPECT_TRUE(head.publish(base.identity(),second).IsUnavailable());
      ++stale_rejections;
      const auto accepted_image=whole_image(state,accepted_version);
      if(stochastic&&limits.quota==std::numeric_limits<std::size_t>::max()&&
          limits.remaining_context==std::numeric_limits<std::size_t>::max()&&limits.eos.empty()) {
        const std::array inputs{current_id,proposal};const std::array finals{typed<std::int32_t>(random_result[1]).at(0)};
        const std::array flags{std::uint8_t(sampled_accept)};
        const auto boundary=[&](std::size_t quota,std::span<const std::int32_t> eos,const char *name) {
          std::vector<std::int32_t> tokens;
          const auto selected=hyle::inference::mtp::select_sampled_prefix(inputs,flags,finals,quota,eos,
              std::int32_t(model.vocabulary),tokens);
          const auto lease=sampled_accept&&selected.consumed==1?actual_q[1]:random_result[1];
          const auto from=selected.consumed_drafts?accepted_version:first;
          auto branch=take(c::StateHead::create(state.store,base),"Qwen independent bounded policy Head");
          auto publication=take(state.store.begin(from),"Qwen same complete selected prefix");
          check(publication.replace(state.entries[kPending],lease),"bounded actual Qwen pending lease");
          const auto rng=initial_cursor.advanced(3);
          check(publication.replace_metadata(*state.sampling_entry,std::as_bytes(std::span(&rng,1))),"bounded Qwen reserved random addresses");
          const auto version=take(publication.finalize(),"complete bounded Qwen state and RNG");
          check(branch.publish(base.identity(),version),"one bounded Qwen publication");
          EXPECT_EQ(typed<std::int32_t>(state.member(version,kPending)).at(0),selected.next_id);same_state_storage(state.member(version,kPending),lease);
          EXPECT_EQ(cursor(version),rng);EXPECT_EQ(typed<std::int32_t>(state.member(version,kCursor)).at(0),2+int(selected.consumed));
          EXPECT_EQ(typed<std::int32_t>(target_member(version,kTargetCursor)).at(0),2+int(selected.consumed));
          for(std::size_t entry=0;entry<kResources;++entry)if(entry!=kPending)same_state_storage(state.member(version,entry),state.member(from,entry));
          for(std::size_t entry=0;entry<kTargetResources;++entry)same_state_storage(target_member(version,static_cast<TargetEntry>(entry)),target_member(from,static_cast<TargetEntry>(entry)));
          std::printf("QWEN RANDOM BOUNDARY name=%s accepted=%d emitted=%zu consumed=%zu pending=%d stopped=%d draw=3\n",name,int(sampled_accept),tokens.size(),selected.consumed,selected.next_id,int(selected.stopped));++bounded_tails;
        };
        boundary(1,{},"quota1");check(state.store.collect(),"retire independent Qwen quota Head");
        if(sampled_accept&&proposal!=current_id){const std::array eos{proposal};boundary(2,eos,"accepted_proposal_EOS");check(state.store.collect(),"retire independent Qwen accepted EOS Head");}
        if(finals[0]!=current_id&&(!sampled_accept||finals[0]!=proposal)){const std::array eos{finals[0]};boundary(2,eos,sampled_accept?"bonus_EOS":"residual_EOS");check(state.store.collect(),"retire independent Qwen final EOS Head");}
      }
      {
        auto cancelled=take(state.store.begin(accepted_version),"failed target-only splice");
        check(cancelled.replace(state.extra_entries[kTargetTap],target_member(second,kTargetTap)),"stage target-only producer");
        EXPECT_TRUE(cancelled.replace(state.extra_entries[kTargetKey],state.member(first,kPending)).IsInvalidArgument());
        check(cancelled.cancel(),"cancel incomplete target/NextN splice");
        ++cancellations;
      }
      EXPECT_EQ(whole_image(state,take(head.current(),"unchanged head after failed splice")),accepted_image);
      // Explicit limits describe one component episode, not a request loop.
      if(!stochastic||(limits.quota==std::numeric_limits<std::size_t>::max()&&
          limits.remaining_context==std::numeric_limits<std::size_t>::max()&&limits.eos.empty())) {
      const auto continuing_seed=reseed(take(head.current(),"accepted whole prefix for continuation"));
      same_state_storage(state.member(continuing_seed,kHidden),target_member(accepted_version,kTargetTap));
      auto continued=combined(continuing_seed);
      if(stochastic) {
        const std::array input{target_member(continued,kTargetTap)};
        const auto target_logits=next_target_logits->execute_inputs(queue,input);
        auto &sampler=next_target_sampler?*next_target_sampler:*random_q;
        const auto target_sample=sample(sampler,target_logits[0],initial_cursor.advanced(3));
        auto next=take(state.store.begin(continued),"ordinary Qwen target consumes actual residual/bonus");
        check(next.replace(state.entries[kPending],target_sample[1]),"next target categorical pending result lease");
        const auto rng=initial_cursor.advanced(4);
        check(next.replace_metadata(*state.sampling_entry,std::as_bytes(std::span(&rng,1))),"one target-only draw commits with full prefix");
        continued=take(next.finalize(),"real Qwen target continuation and cursor");
      }
      check(head.publish(accepted_version.identity(),continued),"publish actual target/draft continuation");
      if(stochastic) {
        EXPECT_EQ(cursor(take(head.current(),"continued Qwen target cursor")),initial_cursor.advanced(4));
        std::printf("QWEN RANDOM K1 moe=%d proposal=%d accepted=%u pending=%d continued_pending=%d cursor=%llu\n",int(model.moe),proposal,accepted,
            typed<std::int32_t>(pending).at(0),typed<std::int32_t>(state.member(continued,kPending)).at(0),static_cast<unsigned long long>(cursor(continued).draw));
      }
      ++publications;
      EXPECT_EQ(typed<std::int32_t>(state.member(continued,kPending)).size(),1u);
      EXPECT_EQ(whole_image(state,base),base_image);EXPECT_EQ(whole_image(state,first),first_image);EXPECT_EQ(whole_image(state,second),second_image);
      EXPECT_EQ(whole_image(state,accepted_version),accepted_image);
      for(const auto &version:{base,first,second,accepted_version,continued}) {
        for(std::size_t entry=0;entry<kResources;++entry){auto reader=state.member(version,entry);EXPECT_FALSE(reader.try_write());}
        for(const auto entry:state.extra_entries){auto reader=take(version.member(entry),"retained target reader");EXPECT_FALSE(reader.try_write());}
      }
      } else {
        ::testing::Test::RecordProperty("random_finite_emitted",int(emitted.size()));
        ::testing::Test::RecordProperty("random_finite_consumed",int(sampled_choice.consumed));
        ::testing::Test::RecordProperty("random_finite_stopped",int(sampled_choice.stopped));
      }
      }
      } // existing bounded K1 / target-only path
      peak_backing=std::max(peak_backing,state.store.stats().backing_bytes);
      std::size_t initial_bound=0;
      for(const auto &desc:descriptions(model))initial_bound+=(take(desc.required_bytes(),"joint draft initial bound")+255u)/256u*256u;
      for(const auto &desc:target_descriptions)initial_bound+=(take(desc.required_bytes(),"joint target initial bound")+255u)/256u*256u;
      declared_budget=pool_bound+initial_bound+(stochastic?96u*64u:0);EXPECT_EQ(declared_budget,state.backing_bound);EXPECT_LE(state.store.stats().backing_bytes,declared_budget);
    }
    check(state.store.collect(),"retire complete12-member verifier checkpoints");
    EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);
  }
  controlled_proposals={};warmup.clear();
  for(const auto *program:{&draft,&target.program,&verifier.program}) {
    EXPECT_EQ(program->live_output_blocks(),0u);EXPECT_EQ(program->active_output_uses(),0u);
    if(program->supply) {
      if(program->supply->cpu) {
        EXPECT_EQ(program->supply->cpu->stats().live_blocks,0u);EXPECT_EQ(program->supply->cpu->stats().active_uses,0u);
      }
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
      if(program->supply->cuda)check(program->supply->cuda->close(),"close actual model expert slots after every ordinary phase reset");
#endif
    }
  }
  if(stochastic)for(const auto *program:{&*random_q,&*random_decision,&*next_target_logits}) {
    EXPECT_EQ(program->live_output_blocks(),0u);EXPECT_EQ(program->active_output_uses(),0u);
  }
  if(next_target_sampler){EXPECT_EQ(next_target_sampler->live_output_blocks(),0u);EXPECT_EQ(next_target_sampler->active_output_uses(),0u);}
  if(short_verifier){EXPECT_EQ(short_verifier->program.live_output_blocks(),0u);EXPECT_EQ(short_verifier->program.active_output_uses(),0u);}
  if(short_decision){EXPECT_EQ(short_decision->live_output_blocks(),0u);EXPECT_EQ(short_decision->active_output_uses(),0u);}
  if(multi_executed) {
    ::testing::Test::RecordProperty("random_max_drafts",int(max_drafts));
    ::testing::Test::RecordProperty("random_actual_drafts",int(multi_actual_drafts));
    ::testing::Test::RecordProperty("random_accepted_prefix",int(multi_accepted));
    ::testing::Test::RecordProperty("repaired_prefixes",int(multi_repaired_prefixes));
    ::testing::Test::RecordProperty("target_head_rows",int(multi_actual_drafts+1));
    ::testing::Test::RecordProperty("actual_episode_draws",int(2*multi_actual_drafts+1));
    ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(declared_budget));
    ::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(peak_backing));
    ::testing::Test::RecordProperty("scope","Natural recursive original MoE NextN proposals and true per-row target MoE states; one M3 original vocabulary head, common p/q prefix decisions, complete0/1/2 target-conditioned repair and atomic pending/RNG; upstream layer fixture remains external, target body is sequential, no full-model quality/TPS/parallel-target claim.");
    return;
  }
  EXPECT_EQ(repairs,accepted_counts[1]);
  if(!stochastic){EXPECT_GT(repairs,0u);EXPECT_GT(accepted_counts[0],0u);EXPECT_GT(accepted_counts[1],0u);}
  if(stochastic) {
    EXPECT_EQ(accepted_counts[0]+accepted_counts[1],limits.single_row()?0u:1u);
    ::testing::Test::RecordProperty("random_actual_proposal",natural_proposal);
    ::testing::Test::RecordProperty("random_accepted",int(accepted_counts[1]));
    ::testing::Test::RecordProperty("random_rejected",int(accepted_counts[0]));
    const bool request_limited=limits.quota!=std::numeric_limits<std::size_t>::max()||
        limits.remaining_context!=std::numeric_limits<std::size_t>::max()||!limits.eos.empty();
    ::testing::Test::RecordProperty("random_committed_draws",limits.single_row()?1:request_limited?3:4);
    ::testing::Test::RecordProperty("bounded_tail_heads",int(bounded_tails));
    ::testing::Test::RecordProperty("whole_state_publications",int(publications));
    ::testing::Test::RecordProperty("cancelled_unpublished_drafts",int(cancellations));
    ::testing::Test::RecordProperty("rejected_stale_publishes",int(stale_rejections));
    ::testing::Test::RecordProperty("accepted_target_conditioned_repairs",int(repairs));
    ::testing::Test::RecordProperty("target_verifier_full_vocabulary",int(model.vocabulary));
    ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(declared_budget));
    ::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(peak_backing));
    ::testing::Test::RecordProperty("stochastic_policy","actual NextN q categorical proposal0 -> actual target row1; real full-V target p0/q/p1 common accept1/final2 -> direct I32 result lease; full original target/NextN resources plus Cursor metadata in same expected-base Head. Cancel/stale/failure advances0, K1 success advances3, actual residual/bonus target-only continuation advances1.");
    ::testing::Test::RecordProperty("scope","One natural finite original-role target/NextN stochastic K1 and actual pending continuation on the existing common Qwen component controller. Previous-layer input/history remain explicit fixtures; no full-model quality, natural acceptance-rate, all identities/backends, or performance claim.");
    return;
  }
  ::testing::Test::RecordProperty("actual_submissions",int(draft.native_submissions()+target.program.native_submissions()+verifier.program.native_submissions()));
  ::testing::Test::RecordProperty("target_submissions",int(target.program.native_submissions()));
  ::testing::Test::RecordProperty("nextn_submissions",int(draft.native_submissions()));
  ::testing::Test::RecordProperty("target_verifier_submissions",int(verifier.program.native_submissions()));
  ::testing::Test::RecordProperty("accepted_target_conditioned_repairs",int(repairs));
  ::testing::Test::RecordProperty("actual_accepted_kv_repair_submissions",int(repair_submissions));
  ::testing::Test::RecordProperty("accepted_zero_drafts",int(accepted_counts[0]));
  ::testing::Test::RecordProperty("accepted_one_draft",int(accepted_counts[1]));
  ::testing::Test::RecordProperty("actual_pending_embedding_feedback_checks",int(feedbacks+repairs));
  ::testing::Test::RecordProperty("whole_state_publications",int(publications));
  ::testing::Test::RecordProperty("cancelled_unpublished_drafts",int(cancellations));
  ::testing::Test::RecordProperty("rejected_stale_publishes",int(stale_rejections));
  ::testing::Test::RecordProperty("actual_target_reseed_checks",int(reseed_checks));
  ::testing::Test::RecordProperty("original_native_reference_submissions",0);
  ::testing::Test::RecordProperty("natural_k1_trials",1);::testing::Test::RecordProperty("controlled_matching_trials",1);::testing::Test::RecordProperty("controlled_rejection_trials",1);
  ::testing::Test::RecordProperty("natural_accepted_drafts",int(natural_accepted));::testing::Test::RecordProperty("natural_proposal",natural_proposal);
  ::testing::Test::RecordProperty("complete_target_draft_resources",int(std::size_t(kResources)+std::size_t(kTargetResources)));
  ::testing::Test::RecordProperty("original_target_roles",int(target.role_count));
  ::testing::Test::RecordProperty("original_target_source_bytes",std::to_string(target.source_bytes));
  ::testing::Test::RecordProperty("first_full_target_projection_values",std::to_string(target.full_projection_values));
  ::testing::Test::RecordProperty("first_full_target_verifier_values",std::to_string(full_verifier_values));
  ::testing::Test::RecordProperty("target_verifier_full_vocabulary",int(model.vocabulary));
  ::testing::Test::RecordProperty("target_verifier_matmuls",int(verifier.matmuls));
  ::testing::Test::RecordProperty("target_verifier_rows",2);
  ::testing::Test::RecordProperty("target_verifier_provider",verifier.provider);
  if(model.moe) {
    ::testing::Test::RecordProperty("target_resident_slot_bytes",std::to_string(target.program.supply->layout().slot_bytes()));
    ::testing::Test::RecordProperty("nextn_resident_slot_bytes",std::to_string(draft.supply->layout().slot_bytes()));
    ::testing::Test::RecordProperty("target_expert_domain",int(target.program.supply->layout().domain()));
    ::testing::Test::RecordProperty("target_resident_capacity",int(target.program.supply->layout().capacity()));
    ::testing::Test::RecordProperty("target_resident_episodes",int(target.program.supply->episodes));
    ::testing::Test::RecordProperty("nextn_resident_episodes",int(draft.supply->episodes));
    ::testing::Test::RecordProperty("target_expert_source_codec",draft.recipe.original_bfloat?"BF16":ggml_type_name(target.parameters24->source->spec(target24_roles::Role::kExpertGate).native));
    ::testing::Test::RecordProperty("nextn_expert_source_codec",draft.recipe.original_bfloat?"BF16":ggml_type_name(draft.recipe.original24->source->spec(full24::Role::kExpertGate).native));
  }
  ::testing::Test::RecordProperty("first_full_nextn_head_values",std::to_string(draft.original_full_head_values));
  ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(declared_budget));::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(peak_backing));
  ::testing::Test::RecordProperty("scope",std::string(target.result_ct?
      "original CompressedTensors asymmetric4 target layer and complete BF16 NextN with explicitly supplied original BF16 vocabulary roles; upstream target input/history fixture,not a full trained trunk; original BF16 scale/product, BF16 A/F32 accumulation/BF16 return; seven three-node packed recipes without whole-W materialization; original I64 shape metadata checked separately":target.result_gptq?
      "original GPTQ-v1 target layer and complete BF16 NextN with explicitly supplied original vocabulary roles; upstream target input/history fixture,not a full trained trunk; original F16 scale product then BF16 W and BF16 A/F32 dot/BF16 result; all seven public recipes contract without whole-W materialization":target.result_nvfp4?
      "original ModelOpt NVFP4 target23 and complete original BF16 NextN retain distinct original roles and one actual tied BF16 vocabulary; layer22 input/history remain explicit fixtures,not a full trained trunk; Nvfp4Linear preserves A QDQ and BF16 return,not a raw-F32 port":draft.recipe.original_bfloat?
      "original BF16 Qwen target layer and complete original BF16 NextN retain their distinct actual source roles; upstream target input/history remain explicit fixtures,not a full trained trunk; F32-fused RoPE with BF16 output is not HF eager intermediate-bit equivalence":model.moe?
      "original Q24 MAIN target91 and complete NextN92 retain distinct original per-role GGUF codecs and sources; layer90 input/history fixture,not92-layer target":
      "original Q27 MAIN target63 and explicitly selected complete NextN64 retain distinct authenticated source/vocabulary contracts; layer62 input/history fixture,not64-layer target")+
      "; draft/target use their explicitly supplied original vocabulary contracts; targettap follows globalnorm and shifts to next-token NextN input; one M2 verifier,natural K1 and separate actualTop1/Top2 controls; accept0 keeps aftercurrent,accept1 repairs recursive draft KV with actualtargettap before12member sameHead publication; no full-model quality,natural acceptance-rate or TPS claim");
}

// Different draft/target vocabularies are a caller contract, not a second
// controller. The MAIN target uses its own Q6_K head/globalnorm even when the
// draft source is the Q3_K sidecar. Only the target head is bound here; no
// unused MAIN NextN banks are uploaded for this pairing.
[[maybe_unused]] void run_packed_q27_target_verifier(const c::ModelArtifactProfileDescription &draft_profile,
    std::shared_ptr<const full::Data> draft_source,std::shared_ptr<const full::Data> target_vocabulary_source,
    std::shared_ptr<const target_roles::Data> target_source,c::Context &context,const c::StorageOwner &owner,bool cuda,bool stochastic=false,hyle::inference::mtp::SamplingLimits limits={}) {
  if(!draft_source||!target_vocabulary_source||!target_source)
    throw std::runtime_error("packed Q27 K1 requires explicit original draft,target and vocabulary sources");
  const auto model=packed_q27_model(draft_profile,*draft_source);
  const auto &target_identity=target_vocabulary_source->identity();
  if(target_identity.repository!=full::kPrimaryIdentity.repository||target_identity.revision!=full::kPrimaryIdentity.revision||
      target_identity.variant!=full::kPrimaryIdentity.variant||target_identity.file!=full::kPrimaryIdentity.file)
    throw std::runtime_error("this pairing requires the explicitly selected MAIN Q27 target vocabulary");
  const auto complete=std::make_shared<const full::Roles>(full::bind(std::move(draft_source),owner));
  const auto prefix=full::input_roles(*complete);
  const auto target=std::make_shared<const target_roles::Bound>(target_roles::bind(std::move(target_source),owner));
  const auto &head_spec=target_vocabulary_source->spec(full::role_name(full::Role::kHead));
  if(head_spec.native!=GGML_TYPE_Q6_K||head_spec.k!=model.hidden||head_spec.n!=model.vocabulary)
    throw std::runtime_error("selected MAIN target verifier needs its original full Q6_K vocabulary head");
  const auto head_bytes=target_vocabulary_source->bytes(full::role_name(full::Role::kHead));
  const auto head=orig::detail::bind(head_spec.name,head_spec.description,owner,head_bytes,&target_vocabulary_source->identity());
  auto vocabulary=target_vocabulary(target_vocabulary_source,head,head_bytes,head_spec.native);
  std::size_t source_bytes=0;for(std::size_t slot=0;slot<complete->source->size();++slot)
    source_bytes+=complete->source->spec(complete->source->name(slot)).bytes;
  ::testing::Test::RecordProperty("original_draft_roles",int(complete->source->size()));
  ::testing::Test::RecordProperty("original_draft_source_bytes",std::to_string(source_bytes));
  ::testing::Test::RecordProperty("draft_artifact",std::string(complete->source->identity().file));
  ::testing::Test::RecordProperty("target_vocabulary_artifact",std::string(target_identity.file));
  ::testing::Test::RecordProperty("draft_head_codec",int(complete->source->spec(full::role_name(full::Role::kHead)).native));
  ::testing::Test::RecordProperty("target_head_codec",int(head_spec.native));
  // S01 owns both norms. Fixed builder639-642 chooses its shared_head_norm
  // when present; its separate output_norm remains authenticated source data.
  if(complete->source->size()==18) {
    const auto standalone=complete->source->norm(full::kOutputNormName);
    if(standalone.size()!=std::size_t(model.hidden))throw std::runtime_error("sidecar independent output_norm extent");
    ::testing::Test::RecordProperty("draft_head_norm_source",std::string(full::role_name(full::Role::kHeadNorm)));
    ::testing::Test::RecordProperty("draft_independent_output_norm_bytes",std::to_string(complete->source->bytes(full::kOutputNormName).size()));
  }
  auto draft=Program::create(build(model,owner,prefix,nullptr,nullptr,{},complete),context,owner,cuda);
  auto plan=prepare_target(target,std::move(vocabulary),context,owner,cuda);
  run_original_target_verifier(std::move(draft),std::move(plan),context,owner,cuda,stochastic,limits);
}

// Same-artifact GGUF target/draft/vocabulary adapter. The Data has exactly
// the29 authenticated original names; no primary/sidecar pin substitution.
[[maybe_unused]] void run_named_q27_target_verifier(const c::ModelArtifactProfileDescription &profile,
    std::shared_ptr<const full::Data> source,c::Context &context,const c::StorageOwner &owner,bool cuda,bool stochastic=false,hyle::inference::mtp::SamplingLimits limits={}) {
  if(!source||source->size()!=29)throw std::runtime_error("named Q27 full verifier requires its29 actual original roles");
  std::size_t source_bytes=0,target_bytes=0;bool has_bfloat=false;
  for(std::size_t slot=0;slot<source->size();++slot) {
    const auto &spec=source->spec(source->name(slot));
    if(spec.bytes>std::numeric_limits<std::size_t>::max()-source_bytes)
      throw std::runtime_error("named Q27 original byte sum overflows");
    source_bytes+=spec.bytes;has_bfloat=has_bfloat||spec.native==GGML_TYPE_BF16;
  }
  for(std::size_t slot=0;slot<target_roles::kRoles;++slot) {
    const auto bytes=source->spec(target_roles::spec(static_cast<target_roles::Role>(slot)).name).bytes;
    if(bytes>std::numeric_limits<std::size_t>::max()-target_bytes)
      throw std::runtime_error("named Q27 target byte sum overflows");
    target_bytes+=bytes;
  }
  const auto model=packed_q27_model(profile,*source);
  const auto complete=std::make_shared<const full::Roles>(full::bind(std::move(source),owner));
  const auto prefix=full::input_roles(*complete);const auto &head=complete->at(full::role_name(full::Role::kHead));
  const auto &spec=complete->source->spec(full::role_name(full::Role::kHead));
  auto vocabulary=target_vocabulary(complete,head,complete->source->bytes(full::role_name(full::Role::kHead)),spec.native);
  auto draft=Program::create(build(model,owner,prefix,nullptr,nullptr,{},complete),context,owner,cuda);
  auto target=prepare_target({},std::move(vocabulary),context,owner,cuda,complete,model);
  if(target.role_count!=target_roles::kRoles||target.source_bytes!=target_bytes)
    throw std::runtime_error("named Q27 actual target original extent differs");
  ::testing::Test::RecordProperty("original_source_roles",29);
  ::testing::Test::RecordProperty("original_source_bytes",std::to_string(source_bytes));
  ::testing::Test::RecordProperty("projection_precision",has_bfloat?
      "F32 state; actual BF16-W uses actual A_RNE_BF16/F32-accumulation/F32-return; packed original AQ unchanged":
      "F32 state; original per-role GGUF codec and native activation quantization retained; no dense weight shadow");
  run_original_target_verifier(std::move(draft),std::move(target),context,owner,cuda,stochastic,limits);
}

// One callable source entry into the bounded prefix runner. Artifact metadata and
// ready roles vary; the state, verifier and accepted-prefix repair stay shared.
#if defined(HYLE_TEST_QWEN_B07_ARTIFACTS_HEADER)
// Same Program/StateHead prefix verifier as B04/Q14 and primary Q24. Only cold
// source/descriptor bindings differ; absent expert IDs go to the owned driver.
void run_named_q24_target_verifier(std::shared_ptr<const full::Data> source,c::Context &context,
    const c::StorageOwner &owner,bool cuda,bool stochastic=false
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ,hyle::inference::mtp::SamplingLimits limits={},std::size_t max_drafts=1) {
  namespace artifact=qwen_gguf_q24_artifacts::b07;
  const auto expert_root=directory("HYLE_QWEN_GGUF_24_EXPERT_DIR");
  if(expert_root.empty())throw std::runtime_error("set HYLE_QWEN_GGUF_24_EXPERT_DIR to authenticated selected Q8_0 expert planes");
  const auto request=directory("HYLE_QWEN_GGUF_24_EXPERT_REQUEST_FILE");
  const bool wait=directory("HYLE_QWEN_GGUF_24_WAIT_FOR_EXPERTS")=="1";
  const auto kv=gguf24::kv_data(source);const auto model=packed_q24_model(artifact::kProfile,*source);
  const auto actual=std::make_shared<const full::Roles>(full::bind(source,owner));
  const auto prefix=gguf24::input_roles(*actual,owner);
  const auto nextn_source=gguf_experts::Source::create(artifact::expert_contract(gguf_experts::Section::kNextn));
  const auto target_source=gguf_experts::Source::create(artifact::expert_contract(gguf_experts::Section::kTarget91));
  const auto complete=packed_moe::nextn_view(actual,nextn_source);const auto target=packed_moe::target_view(actual,target_source);
  const auto supplied=[&](std::shared_ptr<gguf_experts::Source> selected,ExpertSource original) {
    const auto frontier=std::make_shared<GgufFrontierSupply>(GgufFrontierSupply{std::move(selected),expert_root,request,wait});
    auto supply=std::make_shared<ExpertSupply>();supply->source=std::move(original);
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
    if(cuda) {
      if(!device)throw std::runtime_error("B07 CUDA source requires the exact compiled storage resource");
      supply->cuda=std::make_shared<resident24::CudaSupply>(take(resident24::CudaSupply::create(context,owner,*device,supply->source.descriptions),"bounded actual B07 CUDA supply"));
    } else
#else
    if(cuda)throw std::runtime_error("B07 CUDA source entry requires its CUDA provider build");
#endif
    {supply->cpu=std::make_shared<resident24::CpuSupply>(take(resident24::CpuSupply::create(owner,supply->source.descriptions),"bounded actual B07 CPU supply"));}
    supply->before_upload=[frontier](std::span<const std::int32_t> ids){frontier->ensure(ids);};return supply;
  };
  auto nextn_supply=supplied(nextn_source,expert_source(complete->source));nextn_supply->roles=complete;
  auto target_supply=supplied(target_source,expert_source(target->source));
  auto draft=Program::create(build(model,owner,prefix,nullptr,&kv,{},{},complete),context,owner,cuda,false,nextn_supply);
  auto head=target_vocabulary(actual,complete->at(full24::Role::kHead),source->bytes("output.weight"),source->spec("output.weight").native);
  auto plan=prepare_target24(target,std::move(head),target_supply,context,owner,cuda,model);
  ::testing::Test::RecordProperty("original_source_roles",int(gguf24::kStaticRoles));
  ::testing::Test::RecordProperty("original_source_bytes",std::to_string(gguf24::kStaticBytes));
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(artifact::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(artifact::kReceiptSha256));
  ::testing::Test::RecordProperty("projection_precision","F32 state; actual per-role Q8_0/F32 bytes; no complete expert banks or dense weight shadow");
  run_original_target_verifier(std::move(draft),std::move(plan),context,owner,cuda,stochastic,limits,max_drafts);
}
#endif
void run_bfloat_target_verifier(const Model &model,std::shared_ptr<const bfloat::Bound> source,
    const bfloat::DenseNormConventions &rules,std::string_view target_layer,std::string_view target_norm,
    bfloat::VocabularyNames vocabulary,c::Context &context,const c::StorageOwner &owner,bool cuda,
    std::span<const qwen_nvfp4_target_fragment::ProjectionNames> target_projections={},
    std::span<const qwen_gptq_projection_fragment::Names> target_gptq={},
    std::span<const qwen_ct_asymmetric_projection_fragment::Names> target_ct={},bool stochastic=false,hyle::inference::mtp::SamplingLimits limits={}) {
  if(!source||!source->source||!model.profile||model.moe||model.storage!=c::DataType::kBFloat16||
      model.profile->artifact_repository!=source->source->identity().repository||
      model.profile->artifact_revision!=source->source->identity().revision)
    throw std::runtime_error("BF16 dense K1 requires its actual original source/profile; selected-MoE verification is not silently substituted");
  std::size_t source_bytes=0;
  for(std::size_t slot=0;slot<source->source->size();++slot) {
    const auto bytes=take(source->source->description(source->source->role_name(slot)).required_bytes(),"original BF16 source role extent");
    if(bytes>std::numeric_limits<std::size_t>::max()-source_bytes)throw std::runtime_error("original BF16 source byte sum overflows");
    source_bytes+=bytes;
  }
  ::testing::Test::RecordProperty("artifact_repository",std::string(source->source->identity().repository));
  ::testing::Test::RecordProperty("artifact_revision",std::string(source->source->identity().revision));
  ::testing::Test::RecordProperty("artifact_variant",std::string(source->source->identity().variant));
  ::testing::Test::RecordProperty("original_role_count",int(source->source->size()));
  ::testing::Test::RecordProperty("original_role_bytes",std::to_string(source_bytes));
  auto draft=Program::create(build_bfloat(model,source,rules,{},vocabulary),context,owner,cuda);
  auto target_head=target_vocabulary(source,source->at(vocabulary.head),
      source->source->bytes(vocabulary.head),GGML_TYPE_BF16);
  auto target=prepare_bfloat_target(model,std::move(source),rules,target_layer,target_norm,std::move(target_head),context,owner,cuda,target_projections,target_gptq,target_ct);
  run_original_target_verifier(std::move(draft),std::move(target),context,owner,cuda,stochastic,limits);
}

// Actual B05 caller supplies authenticated static roles and separate target /
// NextN selected sources. Both use the same bounded-prefix controller.
[[maybe_unused]] void run_bfloat_moe_target_verifier(const Model &model,std::shared_ptr<const bfloat::Bound> source,
    const bfloat::DenseNormConventions &rules,const bfloat_moe_target::Names &target_names,
    std::shared_ptr<ExpertSupply> nextn_supply,std::shared_ptr<ExpertSupply> target_supply,
    bfloat::VocabularyNames vocabulary,c::Context &context,const c::StorageOwner &owner,bool cuda,
    std::shared_ptr<MixedPlacement> mixed={},bool stochastic=false,hyle::inference::mtp::SamplingLimits limits={},std::size_t max_drafts=1) {
  if(!source||!source->source||!model.profile||!model.moe||model.storage!=c::DataType::kBFloat16||
      !nextn_supply||!target_supply||!nextn_supply->source.owner||!target_supply->source.owner||
      model.profile->artifact_repository!=source->source->identity().repository||
      model.profile->artifact_revision!=source->source->identity().revision)
    throw std::runtime_error("BF16 MoE K1 requires its actual original static and selected-source owners");
  const std::array resident{nextn_supply->layout().resident(0),nextn_supply->layout().resident(1),nextn_supply->layout().resident(2)};
  auto draft=Program::create(build_bfloat(model,source,rules,resident,vocabulary),context,owner,cuda,false,std::move(nextn_supply),mixed);
  auto target_head=target_vocabulary(source,source->at(vocabulary.head),source->source->bytes(vocabulary.head),GGML_TYPE_BF16);
  auto target=prepare_bfloat_moe_target(model,std::move(source),rules,target_names,std::move(target_head),std::move(target_supply),context,owner,cuda,std::move(mixed));
  run_original_target_verifier(std::move(draft),std::move(target),context,owner,cuda,stochastic,limits,max_drafts);
}

TEST(OriginalRoleInput, Nvfp4SubgroupsDecodeIndependentlyOfTheWriter) {
  const nvfp4_test::Fixture fixture(5,128,1,19);
  const auto bytes=std::as_bytes(std::span(fixture.bytes));
  for(std::size_t row=0;row<5;++row) {
    const auto packed=bytes.subspan(row*72,72);
    for(std::size_t group=0;group<4;++group) {
      const auto decoded=original_reference::detail::weight(packed,GGML_TYPE_NVFP4,group);
      for(std::size_t lane=0;lane<32;++lane)
        EXPECT_EQ(float(decoded.scale*decoded.codes[lane]+decoded.minimum*decoded.offsets[lane]),
            fixture.decoded[row*128+group*32+lane]);
    }
  }
}

TEST(OriginalRoleInput, BoundedSha256AuthenticatesBlockAndPaddingBoundaries) {
  struct Case final { std::size_t bytes; const char *sha256; };
  constexpr std::array cases{
      Case{0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      Case{55, "2900465fcb533e05a158fd2b3be0e5e3b03740d83060aa3580e0d98a96bf2384"},
      Case{56, "31454ff48ef36af2f08fd511bdc37d9d5855ac23e992e5ff5445cb6b7674a674"},
      Case{63, "5f6401b96532c36de4e65beec0409b69b1d181864c8009b7a04f43e5d56350d1"},
      Case{64, "94eb5de4943613fd048dc93393ab06877405faa39c11f53e9386083339833e7e"},
      Case{65, "fc518669b6eb4b4dd91827ecacef86689c725bd5bab888fd3b26dbb196eec954"},
      Case{4097, "042a02b55ba342cdd30322331816b6e4bca073f33bdb80149dc74a6f949802f4"},
  };
  for (const auto &entry : cases) {
    std::vector<std::byte> payload(entry.bytes);
    for (std::size_t i = 0; i < payload.size(); ++i)
      payload[i] = static_cast<std::byte>((i * 37u + 11u) & 255u);
    EXPECT_EQ(original_role_input::sha256(payload), entry.sha256) << "bytes=" << entry.bytes;
  }
  constexpr std::string_view abc = "abc";
  EXPECT_EQ(original_role_input::sha256(std::as_bytes(std::span(abc.data(), abc.size()))),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

class QwenMtpStateConsumer:public ::testing::Test {
protected:
  static inline std::unique_ptr<const orig::q27::Data> data27;
  static inline std::unique_ptr<const orig::q24::Data> data24;
  static inline std::shared_ptr<const full::Data> data27_full;
#if defined(HYLE_TEST_QWEN_GGUF_ARTIFACTS_HEADER)
  static inline std::shared_ptr<const full::Data> data27_sidecar;
  static bool prepare_original27_sidecar() {
    if(data27_sidecar)return true;
    const auto path=directory("HYLE_QWEN27_ORIGINAL_SIDECAR_DIR");if(path.empty())return false;
    const auto manifest=qwen_gguf_original_artifacts::s01::manifest();
    data27_sidecar=full::Data::load(path,full::kSidecarIdentity,manifest);return true;
  }
#endif
#if defined(HYLE_TEST_QWEN_B04_ARTIFACTS_HEADER)
  static inline std::shared_ptr<const full::Data> data27_fidelity;
  static bool prepare_original27_fidelity() {
    if(data27_fidelity)return true;
    const auto path=directory("HYLE_QWEN27_FIDELITY_DIR");if(path.empty())return false;
    const auto manifest=qwen_gguf_original_artifacts::b04::manifest();
    data27_fidelity=full::Data::load(path,qwen_gguf_original_artifacts::b04::kIdentity,manifest);
    return true;
  }
#endif
#if defined(HYLE_TEST_QWEN_Q14_ARTIFACTS_HEADER)
  static inline std::shared_ptr<const full::Data> data_q14;
  static bool prepare_original_q14() {
    if(data_q14)return true;
    const auto path=directory("HYLE_QWEN_Q14_ORIGINAL_DIR");if(path.empty())return false;
    const auto manifest=qwen_gguf_original_artifacts::q14::manifest();
    data_q14=full::Data::load(path,qwen_gguf_original_artifacts::q14::kIdentity,manifest);
    return true;
  }
#endif
#if defined(HYLE_TEST_QWEN_B07_ARTIFACTS_HEADER)
  static inline std::shared_ptr<const full::Data> data24_fidelity;
  static bool prepare_original24_fidelity() {
    if(data24_fidelity)return true;
    const auto path=directory("HYLE_QWEN24_FIDELITY_DIR");if(path.empty())return false;
    const auto manifest=qwen_gguf_q24_artifacts::b07::manifest();
    data24_fidelity=full::Data::load(path,gguf24::kIdentity,manifest);return true;
  }
#endif
  static inline std::shared_ptr<const full24::Data> data24_full;
  static inline std::shared_ptr<const target_roles::Data> data27_target;
  static inline std::shared_ptr<const target24_roles::Data> data24_target;
  static inline std::shared_ptr<const bfloat::Data> data27_bfloat;
  static inline std::shared_ptr<const bfloat::Data> data24_bfloat;
#if defined(HYLE_TEST_QWEN_Q03_ARTIFACTS_HEADER)
  static inline std::shared_ptr<const bfloat::Data> data_q03;
  static bool prepare_original_q03() {
    if(data_q03)return true;
    const auto path=directory("HYLE_QWEN_Q03_ORIGINAL_DIR");if(path.empty())return false;
    const auto manifest=q03::specs();
    data_q03=bfloat::Data::load(path,q03::kIdentity,manifest);return true;
  }
#endif
#if defined(HYLE_TEST_QWEN_Q01_ARTIFACTS_HEADER)
  static inline std::shared_ptr<const bfloat::Data> data_q01;
  static bool prepare_original_q01() {
    if(data_q01)return true;
    const auto path=directory("HYLE_QWEN_Q01_ORIGINAL_DIR");if(path.empty())return false;
    const auto manifest=q01::specs();
    data_q01=bfloat::Data::load(path,q01::kIdentity,manifest);return true;
  }
#endif
#if defined(HYLE_TEST_QWEN_Q09_ARTIFACTS_HEADER)
  static inline std::shared_ptr<const bfloat::Data> data_q09;
  static bool prepare_original_q09() {
    if(data_q09)return true;
    const auto path=directory("HYLE_QWEN_Q09_ORIGINAL_DIR");if(path.empty())return false;
    const auto manifest=q09::specs();
    auto source=bfloat::Data::load(path,q09::kIdentity,manifest);
    q09::validate_shapes(*source);data_q09=std::move(source);return true;
  }
#endif
#if defined(HYLE_TEST_QWEN_Q10_ARTIFACTS_HEADER)
  static inline std::shared_ptr<const bfloat::Data> data_q10;
  static bool prepare_original_q10() {
    if(data_q10)return true;
    const auto path=directory("HYLE_QWEN_Q10_ORIGINAL_DIR");if(path.empty())return false;
    const auto manifest=q10::specs();
    data_q10=bfloat::Data::load(path,q10::kIdentity,manifest);return true;
  }
#endif
  static void SetUpTestSuite() {
    const auto path27=directory("HYLE_QWEN27_MTP_INPUT_DIR"),path24=directory("HYLE_QWEN24_MTP_KV_DIR");
    if(!path24.empty())data24=std::make_unique<const orig::q24::Data>(orig::q24::Data::load(path24));
    const auto complete27=directory("HYLE_QWEN27_ORIGINAL_NEXTN_DIR");
    if(!complete27.empty()) {
      if(path27.empty())throw std::runtime_error("MAIN Q27 complete source needs its explicit original prefix directory");
      const auto manifest=full::primary_manifest(std::filesystem::absolute(complete27),std::filesystem::absolute(path27));
      data27_full=full::Data::load({},full::kPrimaryIdentity,manifest);
    }
  }
  static bool prepare_original27_input() {
    // Legacy generated-vocabulary prefix cases alone need the old owned
    // projection vector. Complete sources borrow the authenticated mmap bank.
    if(data27)return true;
    const auto path=directory("HYLE_QWEN27_MTP_INPUT_DIR");if(path.empty())return false;
    data27=std::make_unique<const orig::q27::Data>(orig::q27::Data::load(path));return true;
  }
  static bool prepare_original24_data() {
    // Only the two new complete-original cases need the17GB source. Legacy
    // regressions do not rehash/map it merely because the env remains set.
    if(data24_full)return true;
    const auto path=directory("HYLE_QWEN24_ORIGINAL_NEXTN_DIR");if(path.empty())return false;
    data24_full=std::make_shared<const full24::Data>(full24::Data::load(path));return true;
  }
  static bool prepare_original27_target() {
    if(data27_target)return true;
    const auto path=directory("HYLE_QWEN27_ORIGINAL_TARGET_DIR");if(path.empty())return false;
    data27_target=std::make_shared<const target_roles::Data>(target_roles::Data::load(path));return true;
  }
  static bool prepare_original24_target() {
    if(data24_target)return true;
    const auto path=directory("HYLE_QWEN24_ORIGINAL_TARGET_DIR");if(path.empty())return false;
    data24_target=std::make_shared<const target24_roles::Data>(target24_roles::Data::load(path));return true;
  }
  static bool prepare_original27_bfloat() {
    if(data27_bfloat)return true;
    const auto path=directory("HYLE_QWEN_BFLOAT_27_DIR");if(path.empty())return false;
    const auto manifest=bfloat27::specs();
    data27_bfloat=bfloat::Data::load(path,bfloat27::kIdentity,manifest);return true;
  }
  static bool prepare_original24_bfloat() {
    if(data24_bfloat)return true;
    const auto path=directory("HYLE_QWEN_BFLOAT_24_DIR");if(path.empty())return false;
    const auto manifest=bfloat24::specs();
    data24_bfloat=bfloat::Data::load(path,bfloat24::kIdentity,manifest);return true;
  }
  static void TearDownTestSuite(){
#if defined(HYLE_TEST_QWEN_Q14_ARTIFACTS_HEADER)
    data_q14.reset();
#endif
#if defined(HYLE_TEST_QWEN_GGUF_ARTIFACTS_HEADER)
    data27_sidecar.reset();
#endif
#if defined(HYLE_TEST_QWEN_Q03_ARTIFACTS_HEADER)
    data_q03.reset();
#endif
#if defined(HYLE_TEST_QWEN_Q01_ARTIFACTS_HEADER)
    data_q01.reset();
#endif
#if defined(HYLE_TEST_QWEN_Q10_ARTIFACTS_HEADER)
    data_q10.reset();
#endif
#if defined(HYLE_TEST_QWEN_Q09_ARTIFACTS_HEADER)
    data_q09.reset();
#endif
#if defined(HYLE_TEST_QWEN_B04_ARTIFACTS_HEADER)
    data27_fidelity.reset();
#endif
    data24_bfloat.reset();data27_bfloat.reset();data24_target.reset();data27_target.reset();data24_full.reset();data27_full.reset();data27.reset();data24.reset();}
};
#if defined(HYLE_TEST_QWEN_STATE_CPU)
struct SystemResource final {m::MemoryResourceRef ref() noexcept{return m::system_resource();}};
TEST_F(QwenMtpStateConsumer,CpuQ27WholeNextNPrefixPublishesAndContinues) {
  if(!prepare_original27_input())GTEST_SKIP()<<"Set HYLE_QWEN27_MTP_INPUT_DIR to authenticated MAIN Q27 roles";
  auto context=take(g::cpu::create_context(),"ordinary CPU context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU storage owner");
  const auto roles=orig::q27::bind(*data27,owner);run(q27,roles,data27.get(),nullptr,context,owner,false);
}
TEST_F(QwenMtpStateConsumer,CpuQ27OriginalNextNAndVocabularyPublishesAndContinues) {
  if(!data27_full)GTEST_SKIP()<<"Set HYLE_QWEN27_MTP_INPUT_DIR and HYLE_QWEN27_ORIGINAL_NEXTN_DIR to authenticated MAIN Q27 roles";
  auto context=take(g::cpu::create_context(),"ordinary CPU context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU storage owner");
  const auto complete=std::make_shared<const full::Roles>(full::bind(data27_full,owner));
  const auto roles=full::input_roles(*complete);
  run(q27,roles,nullptr,nullptr,context,owner,false,complete);
}
TEST_F(QwenMtpStateConsumer,CpuQ27OriginalTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!data27_full||!prepare_original27_target())GTEST_SKIP()<<"Set original Q27 NextN directories and HYLE_QWEN27_ORIGINAL_TARGET_DIR to authenticated target63 roles";
  auto context=take(g::cpu::create_context(),"ordinary CPU target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU target verifier owner");
  const auto complete=std::make_shared<const full::Roles>(full::bind(data27_full,owner));
  const auto roles=full::input_roles(*complete);
  const auto target=std::make_shared<const target_roles::Bound>(target_roles::bind(data27_target,owner));
  auto draft=Program::create(build(q27,owner,roles,nullptr,nullptr,{},complete),context,owner,false);
  auto target_head=target_vocabulary(complete,complete->at(full::role_name(full::Role::kHead)),complete->source->bytes(full::role_name(full::Role::kHead)),complete->source->spec(full::role_name(full::Role::kHead)).native);
  auto target_plan=prepare_target(target,std::move(target_head),context,owner,false);
  run_original_target_verifier(std::move(draft),std::move(target_plan),context,owner,false);
}
#if defined(HYLE_TEST_QWEN_GGUF_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CpuQ27SidecarOriginalNextNAndMainTargetFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original27_sidecar()||!data27_full||!prepare_original27_target())
    GTEST_SKIP()<<"Set HYLE_QWEN27_ORIGINAL_SIDECAR_DIR plus original MAIN NextN/prefix/target directories; compile with the generated authenticated S01 artifact header";
  auto context=take(g::cpu::create_context(),"ordinary S01/Main K1 context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary S01/Main storage owner");
  ::testing::Test::RecordProperty("draft_manifest_receipt_sha256",std::string(qwen_gguf_original_artifacts::s01::kReceiptSha256));
  run_packed_q27_target_verifier(qwen_gguf_original_artifacts::s01::kProfile,data27_sidecar,data27_full,data27_target,context,owner,false);
}
#endif
#if defined(HYLE_TEST_QWEN_Q03_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CpuQ03OriginalNvfp4TargetAndBFloatNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q03())GTEST_SKIP()<<"Set HYLE_QWEN_Q03_ORIGINAL_DIR to the complete authenticated49 original Q03 roles";
  auto context=take(g::cpu::create_context(),"ordinary Q03 Cpu context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary Q03 CPU owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data_q03,owner));
  const auto model=bfloat_model(q03::kProfile,q03::kGeometry,q03::kIntermediate,false,q03::kVocabulary,q03::kRopeTheta);
  const auto projections=q03::target_projection_names();
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(q03::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(q03::kReceiptSha256));
  ::testing::Test::RecordProperty("target_projection_contract","ModelOpt Nvfp4Linear original U8/E4/W-global/A-global; actual A QDQ and BF16 return, no raw-F32 port");
  ::testing::Test::RecordProperty("vocabulary_binding","one original tied BF16 bank, same readonly lease for Gather and draft/target fullV heads");
  run_bfloat_target_verifier(model,source,q03::kNormConventions,q03::kTargetLayer,q03::kTargetNorm,
      q03::kVocabularyNames,context,owner,false,projections);
}
#endif
#if defined(HYLE_TEST_QWEN_Q01_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CpuQ01OriginalGptqTargetAndBFloatNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q01())GTEST_SKIP()<<"Set HYLE_QWEN_Q01_ORIGINAL_DIR to the complete authenticated49 original Q01 roles";
  auto context=take(g::cpu::create_context(),"ordinary Q01 Cpu context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary Q01 CPU owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data_q01,owner));
  const auto model=bfloat_model(q01::kProfile,q01::kGeometry,q01::kIntermediate,false,q01::kVocabulary,q01::kRopeTheta);
  const auto projections=q01::target_projection_names();
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(q01::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(q01::kReceiptSha256));
  ::testing::Test::RecordProperty("target_projection_contract","original GPTQ-v1 Q/Z/F16 SF/g_idx; F16 scale product then BF16 W, BF16 A/F32 dot/BF16 return; seven packed contractions required");
  ::testing::Test::RecordProperty("vocabulary_binding","one original tied BF16 bank, same readonly lease for Gather and draft/target fullV heads");
  run_bfloat_target_verifier(model,source,q01::kNormConventions,q01::kTargetLayer,q01::kTargetNorm,
      q01::kVocabularyNames,context,owner,false,{},projections);
}
#endif
#if defined(HYLE_TEST_QWEN_Q09_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CpuQ09OriginalCtAsymmetricTargetAndBFloatNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q09())GTEST_SKIP()<<"Set HYLE_QWEN_Q09_ORIGINAL_DIR to the complete authenticated50 original Q09 roles";
  auto context=take(g::cpu::create_context(),"ordinary Q09 Cpu context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary Q09 CPU owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data_q09,owner));
  const auto model=bfloat_model(q09::kProfile,q09::kGeometry,q09::kIntermediate,false,q09::kVocabulary,q09::kRopeTheta);
  const auto projections=q09::target_projection_names();
  ::testing::Test::RecordProperty("target_layer",63);
  ::testing::Test::RecordProperty("target_group_size",32);
  ::testing::Test::RecordProperty("target_shape_metadata_roles",7);
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(q09::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(q09::kReceiptSha256));
  ::testing::Test::RecordProperty("target_projection_contract","original CT asymmetric codes/Z/BF16 SF; no GPTQ zero+1/F16 store or AWQ-GEMM4 layout; BF16 A/F32 accumulation/BF16 result; seven three-node packed contractions required");
  ::testing::Test::RecordProperty("vocabulary_binding","two original untied BF16 banks; actual embedding for Gather and independent original head for draft/target fullV logits");
  run_bfloat_target_verifier(model,source,q09::kNormConventions,q09::kTargetLayer,q09::kTargetNorm,
      q09::kVocabularyNames,context,owner,false,{}, {},projections);
}
#endif
#if defined(HYLE_TEST_QWEN_Q10_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CpuQ10OriginalGptqTargetAndBFloatNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q10())GTEST_SKIP()<<"Set HYLE_QWEN_Q10_ORIGINAL_DIR to the complete authenticated50 original Q10 roles";
  auto context=take(g::cpu::create_context(),"ordinary Q10 Cpu context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary Q10 CPU owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data_q10,owner));
  const auto model=bfloat_model(q10::kProfile,q10::kGeometry,q10::kIntermediate,false,q10::kVocabulary,q10::kRopeTheta);
  const auto projections=q10::target_projection_names();
  ::testing::Test::RecordProperty("target_layer",63);
  ::testing::Test::RecordProperty("target_group_size",64);
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(q10::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(q10::kReceiptSha256));
  ::testing::Test::RecordProperty("target_projection_contract","original GPTQ-v1 Q/Z/F16 SF/g_idx; F16 scale product then BF16 W, BF16 A/F32 dot/BF16 return; seven packed contractions required");
  ::testing::Test::RecordProperty("vocabulary_binding","two original untied BF16 banks; actual embedding for Gather and independent original head for draft/target fullV logits");
  run_bfloat_target_verifier(model,source,q10::kNormConventions,q10::kTargetLayer,q10::kTargetNorm,
      q10::kVocabularyNames,context,owner,false,{},projections);
}
#endif
#if defined(HYLE_TEST_QWEN_B04_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CpuQ27FidelityOriginalTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original27_fidelity())GTEST_SKIP()<<"Set HYLE_QWEN27_FIDELITY_DIR to the authenticated B04 original29 loose roles";
  auto context=take(g::cpu::create_context(),"ordinary B04 original target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"B04 actual original storage owner");
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(qwen_gguf_original_artifacts::b04::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(qwen_gguf_original_artifacts::b04::kReceiptSha256));
  ::testing::Test::RecordProperty("original_header_sha256",std::string(qwen_gguf_original_artifacts::b04::kHeaderSha256));
  ::testing::Test::RecordProperty("original_artifact",std::string(qwen_gguf_original_artifacts::b04::kIdentity.file));
  run_named_q27_target_verifier(qwen_gguf_original_artifacts::b04::kProfile,data27_fidelity,context,owner,false);
}
#endif
#if defined(HYLE_TEST_QWEN_Q14_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CpuQ14OriginalNvfp4NextNAndPackedTargetFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q14())GTEST_SKIP()<<"Set HYLE_QWEN_Q14_ORIGINAL_DIR to the authenticated Q14 original29 roles";
  auto context=take(g::cpu::create_context(),"ordinary Q14 original target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"Q14 original CPU owner");
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(qwen_gguf_original_artifacts::q14::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(qwen_gguf_original_artifacts::q14::kReceiptSha256));
  ::testing::Test::RecordProperty("original_header_sha256",std::string(qwen_gguf_original_artifacts::q14::kHeaderSha256));
  run_named_q27_target_verifier(qwen_gguf_original_artifacts::q14::kProfile,data_q14,context,owner,false);
}
#endif
TEST_F(QwenMtpStateConsumer,CpuQ27OriginalBFloatTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original27_bfloat())GTEST_SKIP()<<"Set HYLE_QWEN_BFLOAT_27_DIR to authenticated official B02 BF16 NextN,target63 and shared vocabulary roles";
  auto context=take(g::cpu::create_context(),"ordinary CPU BF16 target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU BF16 target verifier owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data27_bfloat,owner));
  const auto model=bfloat_model(d5_qwen38_gdn_fixture::kOfficialProfile,bfloat27::kGeometry,
      bfloat27::kIntermediate,false,bfloat27::kVocabulary,bfloat27::kRopeTheta);
  ::testing::Test::RecordProperty("original_bfloat_manifest_sha256",std::string(bfloat27::kManifestSha256));
  run_bfloat_target_verifier(model,source,bfloat27::kNormConventions,bfloat27::kTargetLayer,bfloat27::kTargetNorm,bfloat27::kVocabularyNames,context,owner,false);
}
#if defined(HYLE_TEST_QWEN_B07_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CpuQ24FidelityOriginalTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original24_fidelity())GTEST_SKIP()<<"Set HYLE_QWEN24_FIDELITY_DIR to the authenticated B07 static33 original loose roles";
  auto context=take(g::cpu::create_context(),"ordinary B07 CPU target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"actual B07 CPU storage owner");
  run_named_q24_target_verifier(data24_fidelity,context,owner,false);
}
#endif
TEST_F(QwenMtpStateConsumer,CpuQ24WholeNextNPrefixPublishesAndContinues) {
  if(!data24)GTEST_SKIP()<<"Set HYLE_QWEN24_MTP_KV_DIR to authenticated MAIN Q24 roles";
  auto context=take(g::cpu::create_context(),"ordinary CPU context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU storage owner");
  const auto roles=orig::q24::bind(*data24,owner);run(q24,roles,nullptr,data24.get(),context,owner,false);
}
TEST_F(QwenMtpStateConsumer,CpuQ24OriginalNextNAndVocabularyPublishesAndContinues) {
  if(!data24||!prepare_original24_data())GTEST_SKIP()<<"Set HYLE_QWEN24_MTP_KV_DIR and HYLE_QWEN24_ORIGINAL_NEXTN_DIR to authenticated MAIN Q24 roles";
  auto context=take(g::cpu::create_context(),"ordinary CPU context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU storage owner");
  const auto roles=orig::q24::bind(*data24,owner);
  const auto complete=packed_moe::adapt(std::make_shared<const full24::Roles>(full24::bind(data24_full,owner)));
  auto supply=std::make_shared<ExpertSupply>();supply->roles=complete;
  supply->source=expert_source(complete->source,std::array{full24::Role::kExpertGate,full24::Role::kExpertUp,full24::Role::kExpertDown});
  supply->cpu=std::make_shared<resident24::CpuSupply>(take(resident24::CpuSupply::create(owner,supply->source.descriptions),"bounded original Q24 CPU supply"));
  run(q24,roles,nullptr,data24.get(),context,owner,false,{},complete,supply);
}
TEST_F(QwenMtpStateConsumer,CpuQ24OriginalTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!data24||!prepare_original24_data()||!prepare_original24_target())GTEST_SKIP()<<"Set original Q24 NextN directories and HYLE_QWEN24_ORIGINAL_TARGET_DIR to authenticated target91 roles";
  auto context=take(g::cpu::create_context(),"ordinary CPU Q24 target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU Q24 target verifier owner");
  const auto roles=orig::q24::bind(*data24,owner);
  const auto complete=packed_moe::adapt(std::make_shared<const full24::Roles>(full24::bind(data24_full,owner)));
  const auto target=packed_moe::adapt(std::make_shared<const target24_roles::Bound>(target24_roles::bind(data24_target,owner)));
  auto nextn_supply=std::make_shared<ExpertSupply>();nextn_supply->roles=complete;
  nextn_supply->source=expert_source(complete->source,std::array{full24::Role::kExpertGate,full24::Role::kExpertUp,full24::Role::kExpertDown});
  nextn_supply->cpu=std::make_shared<resident24::CpuSupply>(take(resident24::CpuSupply::create(owner,nextn_supply->source.descriptions),"bounded original Q24 NextN CPU supply"));
  auto target_supply=std::make_shared<ExpertSupply>();
  target_supply->source=expert_source(target->source,std::array{target24_roles::Role::kExpertGate,target24_roles::Role::kExpertUp,target24_roles::Role::kExpertDown});
  target_supply->cpu=std::make_shared<resident24::CpuSupply>(take(resident24::CpuSupply::create(owner,target_supply->source.descriptions),"bounded original Q24 target CPU supply"));
  auto draft=Program::create(build(q24,owner,roles,nullptr,data24.get(),{},{},complete),context,owner,false,false,nextn_supply);
  auto target_head=target_vocabulary(complete,complete->at(full24::Role::kHead),complete->source->bytes(full24::Role::kHead),complete->source->spec(full24::Role::kHead).native);
  auto target_plan=prepare_target24(target,std::move(target_head),target_supply,context,owner,false);
  run_original_target_verifier(std::move(draft),std::move(target_plan),context,owner,false);
}
TEST_F(QwenMtpStateConsumer,CpuQ24OriginalTargetAndNextNRandomVerifierContinuesAtomically) {
  if(!data24||!prepare_original24_data()||!prepare_original24_target())GTEST_SKIP()<<"Set original Q24 NextN directories and HYLE_QWEN24_ORIGINAL_TARGET_DIR to authenticated target91 roles";
  auto context=take(g::cpu::create_context(),"ordinary CPU Q24 target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU Q24 target verifier owner");
  const auto roles=orig::q24::bind(*data24,owner);
  const auto complete=packed_moe::adapt(std::make_shared<const full24::Roles>(full24::bind(data24_full,owner)));
  const auto target=packed_moe::adapt(std::make_shared<const target24_roles::Bound>(target24_roles::bind(data24_target,owner)));
  auto nextn_supply=std::make_shared<ExpertSupply>();nextn_supply->roles=complete;
  nextn_supply->source=expert_source(complete->source,std::array{full24::Role::kExpertGate,full24::Role::kExpertUp,full24::Role::kExpertDown});
  nextn_supply->cpu=std::make_shared<resident24::CpuSupply>(take(resident24::CpuSupply::create(owner,nextn_supply->source.descriptions),"bounded original Q24 NextN CPU supply"));
  auto target_supply=std::make_shared<ExpertSupply>();
  target_supply->source=expert_source(target->source,std::array{target24_roles::Role::kExpertGate,target24_roles::Role::kExpertUp,target24_roles::Role::kExpertDown});
  target_supply->cpu=std::make_shared<resident24::CpuSupply>(take(resident24::CpuSupply::create(owner,target_supply->source.descriptions),"bounded original Q24 target CPU supply"));
  auto draft=Program::create(build(q24,owner,roles,nullptr,data24.get(),{},{},complete),context,owner,false,false,nextn_supply);
  auto target_head=target_vocabulary(complete,complete->at(full24::Role::kHead),complete->source->bytes(full24::Role::kHead),complete->source->spec(full24::Role::kHead).native);
  auto target_plan=prepare_target24(target,std::move(target_head),target_supply,context,owner,false);
  run_original_target_verifier(std::move(draft),std::move(target_plan),context,owner,false,true);
}
TEST_F(QwenMtpStateConsumer,CpuQ24OriginalBFloatTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original24_bfloat())GTEST_SKIP()<<"Set HYLE_QWEN_BFLOAT_24_DIR to authenticated official B05 BF16 static33 roles";
  const auto expert_root=directory("HYLE_QWEN_BFLOAT_24_EXPERT_DIR");
  if(expert_root.empty())GTEST_SKIP()<<"Set HYLE_QWEN_BFLOAT_24_EXPERT_DIR to authenticated original selected BF16 planes";
  const auto request=directory("HYLE_QWEN_BFLOAT_FRONTIER_FILE");
  const bool wait=directory("HYLE_QWEN_BFLOAT_WAIT_FOR_EXPERTS")=="1";
  if(wait&&request.empty())throw std::runtime_error("explicit BF16 external expert wait requires its frontier file");
  auto context=take(g::cpu::create_context(),"ordinary CPU selected BF16 target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"ordinary CPU selected BF16 target verifier owner");
  const auto parameters=std::make_shared<const bfloat::Bound>(bfloat::bind(data24_bfloat,owner));
  const auto model=bfloat_model(d5_qwen38_24t_fixture::kOfficialProfile,bfloat24::kGeometry,
      bfloat24::kIntermediate,true,bfloat24::kVocabulary,bfloat24::kRopeTheta);
  const auto nextn_source=bfloat_experts::Source::create(bfloat24::expert_contract(bfloat_experts::Section::kNextn));
  const auto target_source=bfloat_experts::Source::create(bfloat24::expert_contract(bfloat_experts::Section::kTarget91));
  const auto nextn_frontier=std::make_shared<BfloatFrontierSupply>(BfloatFrontierSupply{nextn_source,expert_root,request,wait});
  const auto target_frontier=std::make_shared<BfloatFrontierSupply>(BfloatFrontierSupply{target_source,expert_root,request,wait});
  const auto supplied=[&](const std::shared_ptr<BfloatFrontierSupply> &frontier) {
    auto result=std::make_shared<ExpertSupply>();
    result->source=expert_source(std::shared_ptr<const bfloat_experts::Source>(frontier->source));
    result->cpu=std::make_shared<resident24::CpuSupply>(take(resident24::CpuSupply::create(owner,result->source.descriptions),"bounded original selected BF16 CPU supply"));
    result->before_upload=[frontier](std::span<const std::int32_t> ids){frontier->ensure(ids);};return result;
  };
  const auto nextn_supply=supplied(nextn_frontier),target_supply=supplied(target_frontier);
  ::testing::Test::RecordProperty("backend","ordinary CPU Graph");
  ::testing::Test::RecordProperty("artifact_repository",std::string(bfloat24::kIdentity.repository));
  ::testing::Test::RecordProperty("artifact_revision",std::string(bfloat24::kIdentity.revision));
  ::testing::Test::RecordProperty("artifact_variant",std::string(bfloat24::kIdentity.variant));
  ::testing::Test::RecordProperty("original_role_count",int(data24_bfloat->size()));
  ::testing::Test::RecordProperty("original_static_role_count",int(data24_bfloat->size()));
  ::testing::Test::RecordProperty("original_role_bytes",std::to_string(bfloat24::kOriginalBytes));
  ::testing::Test::RecordProperty("original_bfloat_manifest_sha256",std::string(bfloat24::kManifestSha256));
  run_bfloat_moe_target_verifier(model,parameters,bfloat24::kNormConventions,bfloat24::kTargetNames,
      nextn_supply,target_supply,bfloat24::kVocabularyNames,context,owner,false);
  const auto attachments=nextn_frontier->attachments+target_frontier->attachments;
  ::testing::Test::RecordProperty("selected_expert_source_attachments",std::to_string(attachments));
  ::testing::Test::RecordProperty("selected_expert_plane_files",std::to_string(attachments*3));
  ::testing::Test::RecordProperty("explicit_expert_frontier_requests",std::to_string(nextn_frontier->requests+target_frontier->requests));
  ::testing::Test::RecordProperty("selected_expert_source_scope","Only actual selected original BF16 gate/up/down planes with independent per-ID catalog hashes; full512 routing IDs remain intact; no whole-bank SHA, full-model quality or TPS claim");
}

#endif
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
TEST_F(QwenMtpStateConsumer,CudaQ27WholeNextNPrefixPublishesAndContinues) {
  if(!prepare_original27_input())GTEST_SKIP()<<"Set HYLE_QWEN27_MTP_INPUT_DIR to authenticated MAIN Q27 roles";
  auto context=take(g::cuda::create_context(),"ordinary CUDA context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual CUDA memory")),"ordinary CUDA storage owner");
  const auto roles=orig::q27::bind(*data27,owner);run(q27,roles,data27.get(),nullptr,context,owner,true);
}
TEST_F(QwenMtpStateConsumer,CudaQ27OriginalNextNAndVocabularyPublishesAndContinues) {
  if(!data27_full)GTEST_SKIP()<<"Set HYLE_QWEN27_MTP_INPUT_DIR and HYLE_QWEN27_ORIGINAL_NEXTN_DIR to authenticated MAIN Q27 roles";
  auto context=take(g::cuda::create_context(),"ordinary CUDA context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual CUDA memory")),"ordinary CUDA storage owner");
  const auto complete=std::make_shared<const full::Roles>(full::bind(data27_full,owner));
  const auto roles=full::input_roles(*complete);
  run(q27,roles,nullptr,nullptr,context,owner,true,complete);
}
TEST_F(QwenMtpStateConsumer,CudaQ27OriginalTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!data27_full||!prepare_original27_target())GTEST_SKIP()<<"Set original Q27 NextN directories and HYLE_QWEN27_ORIGINAL_TARGET_DIR to authenticated target63 roles";
  auto context=take(g::cuda::create_context(),"ordinary CUDA target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual CUDA target verifier memory")),"ordinary CUDA target verifier owner");
  const auto complete=std::make_shared<const full::Roles>(full::bind(data27_full,owner));
  const auto roles=full::input_roles(*complete);
  const auto target=std::make_shared<const target_roles::Bound>(target_roles::bind(data27_target,owner));
  auto draft=Program::create(build(q27,owner,roles,nullptr,nullptr,{},complete),context,owner,true);
  auto target_head=target_vocabulary(complete,complete->at(full::role_name(full::Role::kHead)),complete->source->bytes(full::role_name(full::Role::kHead)),complete->source->spec(full::role_name(full::Role::kHead)).native);
  auto target_plan=prepare_target(target,std::move(target_head),context,owner,true);
  run_original_target_verifier(std::move(draft),std::move(target_plan),context,owner,true);
}
#if defined(HYLE_TEST_QWEN_GGUF_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CudaQ27SidecarOriginalNextNAndMainTargetFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original27_sidecar()||!data27_full||!prepare_original27_target())
    GTEST_SKIP()<<"Set HYLE_QWEN27_ORIGINAL_SIDECAR_DIR plus original MAIN NextN/prefix/target directories; compile with the generated authenticated S01 artifact header";
  auto context=take(g::cuda::create_context(),"ordinary S01/Main K1 context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual CUDA memory")),"ordinary S01/Main storage owner");
  ::testing::Test::RecordProperty("draft_manifest_receipt_sha256",std::string(qwen_gguf_original_artifacts::s01::kReceiptSha256));
  run_packed_q27_target_verifier(qwen_gguf_original_artifacts::s01::kProfile,data27_sidecar,data27_full,data27_target,context,owner,true);
}
#endif
#if defined(HYLE_TEST_QWEN_Q03_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CudaQ03OriginalNvfp4TargetAndBFloatNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q03())GTEST_SKIP()<<"Set HYLE_QWEN_Q03_ORIGINAL_DIR to the complete authenticated49 original Q03 roles";
  auto context=take(g::cuda::create_context(),"ordinary Q03 Cuda context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual Q03 CUDA memory")),"ordinary Q03 CUDA owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data_q03,owner));
  const auto model=bfloat_model(q03::kProfile,q03::kGeometry,q03::kIntermediate,false,q03::kVocabulary,q03::kRopeTheta);
  const auto projections=q03::target_projection_names();
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(q03::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(q03::kReceiptSha256));
  ::testing::Test::RecordProperty("target_projection_contract","ModelOpt Nvfp4Linear original U8/E4/W-global/A-global; actual A QDQ and BF16 return, no raw-F32 port");
  ::testing::Test::RecordProperty("vocabulary_binding","one original tied BF16 bank, same readonly lease for Gather and draft/target fullV heads");
  run_bfloat_target_verifier(model,source,q03::kNormConventions,q03::kTargetLayer,q03::kTargetNorm,
      q03::kVocabularyNames,context,owner,true,projections);
}
#endif
#if defined(HYLE_TEST_QWEN_Q01_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CudaQ01OriginalGptqTargetAndBFloatNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q01())GTEST_SKIP()<<"Set HYLE_QWEN_Q01_ORIGINAL_DIR to the complete authenticated49 original Q01 roles";
  auto context=take(g::cuda::create_context(),"ordinary Q01 Cuda context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual Q01 CUDA memory")),"ordinary Q01 CUDA owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data_q01,owner));
  const auto model=bfloat_model(q01::kProfile,q01::kGeometry,q01::kIntermediate,false,q01::kVocabulary,q01::kRopeTheta);
  const auto projections=q01::target_projection_names();
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(q01::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(q01::kReceiptSha256));
  ::testing::Test::RecordProperty("target_projection_contract","original GPTQ-v1 Q/Z/F16 SF/g_idx; F16 scale product then BF16 W, BF16 A/F32 dot/BF16 return; seven packed contractions required");
  ::testing::Test::RecordProperty("vocabulary_binding","one original tied BF16 bank, same readonly lease for Gather and draft/target fullV heads");
  run_bfloat_target_verifier(model,source,q01::kNormConventions,q01::kTargetLayer,q01::kTargetNorm,
      q01::kVocabularyNames,context,owner,true,{},projections);
}
#endif
#if defined(HYLE_TEST_QWEN_Q09_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CudaQ09OriginalCtAsymmetricTargetAndBFloatNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q09())GTEST_SKIP()<<"Set HYLE_QWEN_Q09_ORIGINAL_DIR to the complete authenticated50 original Q09 roles";
  auto context=take(g::cuda::create_context(),"ordinary Q09 Cuda context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual Q09 CUDA memory")),"ordinary Q09 CUDA owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data_q09,owner));
  const auto model=bfloat_model(q09::kProfile,q09::kGeometry,q09::kIntermediate,false,q09::kVocabulary,q09::kRopeTheta);
  const auto projections=q09::target_projection_names();
  ::testing::Test::RecordProperty("target_layer",63);
  ::testing::Test::RecordProperty("target_group_size",32);
  ::testing::Test::RecordProperty("target_shape_metadata_roles",7);
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(q09::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(q09::kReceiptSha256));
  ::testing::Test::RecordProperty("target_projection_contract","original CT asymmetric codes/Z/BF16 SF; no GPTQ zero+1/F16 store or AWQ-GEMM4 layout; BF16 A/F32 accumulation/BF16 result; seven three-node packed contractions required");
  ::testing::Test::RecordProperty("vocabulary_binding","two original untied BF16 banks; actual embedding for Gather and independent original head for draft/target fullV logits");
  run_bfloat_target_verifier(model,source,q09::kNormConventions,q09::kTargetLayer,q09::kTargetNorm,
      q09::kVocabularyNames,context,owner,true,{}, {},projections);
}
#endif
#if defined(HYLE_TEST_QWEN_Q10_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CudaQ10OriginalGptqTargetAndBFloatNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q10())GTEST_SKIP()<<"Set HYLE_QWEN_Q10_ORIGINAL_DIR to the complete authenticated50 original Q10 roles";
  auto context=take(g::cuda::create_context(),"ordinary Q10 Cuda context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual Q10 CUDA memory")),"ordinary Q10 CUDA owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data_q10,owner));
  const auto model=bfloat_model(q10::kProfile,q10::kGeometry,q10::kIntermediate,false,q10::kVocabulary,q10::kRopeTheta);
  const auto projections=q10::target_projection_names();
  ::testing::Test::RecordProperty("target_layer",63);
  ::testing::Test::RecordProperty("target_group_size",64);
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(q10::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(q10::kReceiptSha256));
  ::testing::Test::RecordProperty("target_projection_contract","original GPTQ-v1 Q/Z/F16 SF/g_idx; F16 scale product then BF16 W, BF16 A/F32 dot/BF16 return; seven packed contractions required");
  ::testing::Test::RecordProperty("vocabulary_binding","two original untied BF16 banks; actual embedding for Gather and independent original head for draft/target fullV logits");
  run_bfloat_target_verifier(model,source,q10::kNormConventions,q10::kTargetLayer,q10::kTargetNorm,
      q10::kVocabularyNames,context,owner,true,{},projections);
}
#endif
#if defined(HYLE_TEST_QWEN_B04_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CudaQ27FidelityOriginalTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original27_fidelity())GTEST_SKIP()<<"Set HYLE_QWEN27_FIDELITY_DIR to the authenticated B04 original29 loose roles";
  auto context=take(g::cuda::create_context(),"ordinary B04 original target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"B04 actual CUDA storage")),"B04 actual original storage owner");
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(qwen_gguf_original_artifacts::b04::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(qwen_gguf_original_artifacts::b04::kReceiptSha256));
  ::testing::Test::RecordProperty("original_header_sha256",std::string(qwen_gguf_original_artifacts::b04::kHeaderSha256));
  ::testing::Test::RecordProperty("original_artifact",std::string(qwen_gguf_original_artifacts::b04::kIdentity.file));
  run_named_q27_target_verifier(qwen_gguf_original_artifacts::b04::kProfile,data27_fidelity,context,owner,true);
}
#endif
#if defined(HYLE_TEST_QWEN_Q14_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CudaQ14OriginalNvfp4NextNAndPackedTargetFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original_q14())GTEST_SKIP()<<"Set HYLE_QWEN_Q14_ORIGINAL_DIR to the authenticated Q14 original29 roles";
  auto context=take(g::cuda::create_context(),"ordinary Q14 original target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),
      "Q14 original CUDA storage")),"Q14 original CUDA owner");
  ::testing::Test::RecordProperty("original_manifest_sha256",std::string(qwen_gguf_original_artifacts::q14::kManifestSha256));
  ::testing::Test::RecordProperty("original_acquisition_sha256",std::string(qwen_gguf_original_artifacts::q14::kReceiptSha256));
  ::testing::Test::RecordProperty("original_header_sha256",std::string(qwen_gguf_original_artifacts::q14::kHeaderSha256));
  run_named_q27_target_verifier(qwen_gguf_original_artifacts::q14::kProfile,data_q14,context,owner,true);
}
#endif
TEST_F(QwenMtpStateConsumer,CudaQ27OriginalBFloatTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original27_bfloat())GTEST_SKIP()<<"Set HYLE_QWEN_BFLOAT_27_DIR to authenticated official B02 BF16 NextN,target63 and shared vocabulary roles";
  auto context=take(g::cuda::create_context(),"ordinary CUDA BF16 target verifier context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual CUDA BF16 target verifier memory")),"ordinary CUDA BF16 target verifier owner");
  const auto source=std::make_shared<const bfloat::Bound>(bfloat::bind(data27_bfloat,owner));
  const auto model=bfloat_model(d5_qwen38_gdn_fixture::kOfficialProfile,bfloat27::kGeometry,
      bfloat27::kIntermediate,false,bfloat27::kVocabulary,bfloat27::kRopeTheta);
  ::testing::Test::RecordProperty("original_bfloat_manifest_sha256",std::string(bfloat27::kManifestSha256));
  run_bfloat_target_verifier(model,source,bfloat27::kNormConventions,bfloat27::kTargetLayer,bfloat27::kTargetNorm,bfloat27::kVocabularyNames,context,owner,true);
}
TEST_F(QwenMtpStateConsumer,CudaQ24WholeNextNPrefixPublishesAndContinues) {
  if(!data24)GTEST_SKIP()<<"Set HYLE_QWEN24_MTP_KV_DIR to authenticated MAIN Q24 roles";
  auto context=take(g::cuda::create_context(),"ordinary CUDA context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual CUDA memory")),"ordinary CUDA storage owner");
  const auto roles=orig::q24::bind(*data24,owner);run(q24,roles,nullptr,data24.get(),context,owner,true);
}
struct SharedCudaResource final {
  std::shared_ptr<c::cuda::MemoryResource> storage;
  m::MemoryResourceRef ref() noexcept{return storage->ref();}
};
#if defined(HYLE_TEST_QWEN_STATE_CPU)
void run_mixed_q24_bfloat(std::shared_ptr<const bfloat::Data> data24_bfloat,bool stochastic,hyle::inference::mtp::SamplingLimits limits={},std::size_t max_drafts=1) {
  const auto expert_root=directory("HYLE_QWEN_BFLOAT_24_EXPERT_DIR");
  if(expert_root.empty())GTEST_SKIP()<<"Set HYLE_QWEN_BFLOAT_24_EXPERT_DIR to authenticated original selected BF16 planes";
  const auto request=directory("HYLE_QWEN_BFLOAT_FRONTIER_FILE");
  const bool wait=directory("HYLE_QWEN_BFLOAT_WAIT_FOR_EXPERTS")=="1";
  if(wait&&request.empty())throw std::runtime_error("explicit BF16 external expert wait requires its frontier file");
  auto context=take(g::cuda::create_context(),"ordinary mixed CUDA body context");
  const auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"mixed actual CUDA device"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaResource{device}),"mixed actual device owner");
  auto placement=std::make_shared<MixedPlacement>();
  placement->cpu=take(g::cpu::create_context(),"ordinary mixed CPU vocabulary context");
  placement->cpu_queue=take(placement->cpu.create_queue(),"ordinary mixed CPU vocabulary queue");
  placement->host=take(c::StorageOwner::adopt(placement->cpu,SystemResource{}),"mixed original vocabulary host owner");
  placement->device=owner;
  placement->staging=take(c::StorageOwner::adopt(context,
      take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost),"mixed actual pinned staging")),"mixed retained staging owner");
  placement->vocabulary=bfloat24::kVocabularyNames;
  const auto parameters=std::make_shared<const bfloat::Bound>(bind_mixed_bfloat(data24_bfloat,*placement));
  const auto model=bfloat_model(d5_qwen38_24t_fixture::kOfficialProfile,bfloat24::kGeometry,
      bfloat24::kIntermediate,true,bfloat24::kVocabulary,bfloat24::kRopeTheta);
  const auto supplied=[&](bfloat_experts::Section section) {
    const auto source=bfloat_experts::Source::create(bfloat24::expert_contract(section));
    const auto frontier=std::make_shared<BfloatFrontierSupply>(BfloatFrontierSupply{source,expert_root,request,wait});
    auto supply=std::make_shared<ExpertSupply>();
    supply->source=expert_source(std::shared_ptr<const bfloat_experts::Source>(source));
    supply->cuda=std::make_shared<resident24::CudaSupply>(take(resident24::CudaSupply::create(context,owner,*device,supply->source.descriptions),"mixed original BF16 CUDA supply"));
    supply->before_upload=[frontier](std::span<const std::int32_t> ids){frontier->ensure(ids);};
    return std::pair{supply,frontier};
  };
  const auto nextn=supplied(bfloat_experts::Section::kNextn),target=supplied(bfloat_experts::Section::kTarget91);
  ::testing::Test::RecordProperty("backend","ordinary mixed CPU vocabulary + CUDA body Graph");
  ::testing::Test::RecordProperty("artifact_repository",std::string(bfloat24::kIdentity.repository));
  ::testing::Test::RecordProperty("artifact_revision",std::string(bfloat24::kIdentity.revision));
  ::testing::Test::RecordProperty("artifact_variant",std::string(bfloat24::kIdentity.variant));
  ::testing::Test::RecordProperty("original_role_count",int(data24_bfloat->size()));
  ::testing::Test::RecordProperty("original_static_role_count",int(data24_bfloat->size()));
  ::testing::Test::RecordProperty("original_role_bytes",std::to_string(bfloat24::kOriginalBytes));
  ::testing::Test::RecordProperty("original_bfloat_manifest_sha256",std::string(bfloat24::kManifestSha256));
  const auto &embedding=parameters->at(placement->vocabulary.embedding),&head=parameters->at(placement->vocabulary.head);
  ::testing::Test::RecordProperty("mixed_host_vocabulary_bytes",std::to_string(embedding.lease.size()+head.lease.size()));
  ::testing::Test::RecordProperty("mixed_device_static_bytes",std::to_string(bfloat24::kOriginalBytes-embedding.lease.size()-head.lease.size()));
  ::testing::Test::RecordProperty("mixed_expert_slot_bytes",std::to_string(nextn.first->layout().slot_bytes()));
  run_bfloat_moe_target_verifier(model,parameters,bfloat24::kNormConventions,bfloat24::kTargetNames,
      nextn.first,target.first,bfloat24::kVocabularyNames,context,owner,true,placement,stochastic,limits,max_drafts);
  const auto attachments=nextn.second->attachments+target.second->attachments;
  ::testing::Test::RecordProperty("selected_expert_source_attachments",std::to_string(attachments));
  ::testing::Test::RecordProperty("selected_expert_plane_files",std::to_string(attachments*3));
  ::testing::Test::RecordProperty("explicit_expert_frontier_requests",std::to_string(nextn.second->requests+target.second->requests));
  ::testing::Test::RecordProperty("mixed_scope","Only original BF16 vocabulary Gather/fullV head/TopK are CPU-required; body/expert computation and all state/pending are CUDA-required. Ordinary Graph inserts actual crossing copies; no complete-model quality, all-CUDA, TPS or measured-VRAM claim.");
}

TEST_F(QwenMtpStateConsumer,MixedQ24OriginalBFloatTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original24_bfloat())GTEST_SKIP()<<"Set HYLE_QWEN_BFLOAT_24_DIR to authenticated official B05 static33 roles";
  run_mixed_q24_bfloat(data24_bfloat,false);
}

TEST_F(QwenMtpStateConsumer,MixedQ24OriginalBFloatTargetAndNextNRandomVerifierContinuesAtomically) {
  if(!prepare_original24_bfloat())GTEST_SKIP()<<"Set HYLE_QWEN_BFLOAT_24_DIR to authenticated official B05 static33 roles";
  run_mixed_q24_bfloat(data24_bfloat,true);
}
#endif
#if defined(HYLE_TEST_QWEN_B07_ARTIFACTS_HEADER)
TEST_F(QwenMtpStateConsumer,CudaQ24FidelityOriginalTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!prepare_original24_fidelity())GTEST_SKIP()<<"Set HYLE_QWEN24_FIDELITY_DIR to the authenticated B07 static33 original loose roles";
  auto context=take(g::cuda::create_context(),"ordinary B07 CUDA target verifier context");
  const auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual B07 CUDA memory"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaResource{device}),"actual B07 CUDA storage owner");
  run_named_q24_target_verifier(data24_fidelity,context,owner,true,false,device.get());
}
#endif
TEST_F(QwenMtpStateConsumer,CudaQ24OriginalNextNAndVocabularyPublishesAndContinues) {
  if(!data24||!prepare_original24_data())GTEST_SKIP()<<"Set HYLE_QWEN24_MTP_KV_DIR and HYLE_QWEN24_ORIGINAL_NEXTN_DIR to authenticated MAIN Q24 roles";
  auto context=take(g::cuda::create_context(),"ordinary CUDA context");
  const auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual shared CUDA memory"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaResource{device}),"ordinary shared CUDA storage owner");
  const auto roles=orig::q24::bind(*data24,owner);
  const auto complete=packed_moe::adapt(std::make_shared<const full24::Roles>(full24::bind(data24_full,owner)));
  auto supply=std::make_shared<ExpertSupply>();supply->roles=complete;
  supply->source=expert_source(complete->source,std::array{full24::Role::kExpertGate,full24::Role::kExpertUp,full24::Role::kExpertDown});
  supply->cuda=std::make_shared<resident24::CudaSupply>(take(resident24::CudaSupply::create(context,owner,*device,supply->source.descriptions),"bounded original Q24 CUDA supply"));
  run(q24,roles,nullptr,data24.get(),context,owner,true,{},complete,supply);
}
TEST_F(QwenMtpStateConsumer,CudaQ24OriginalTargetAndNextNFullVocabularyVerifierPublishesRepairedPrefix) {
  if(!data24||!prepare_original24_data()||!prepare_original24_target())GTEST_SKIP()<<"Set original Q24 NextN directories and HYLE_QWEN24_ORIGINAL_TARGET_DIR to authenticated target91 roles";
  auto context=take(g::cuda::create_context(),"ordinary CUDA Q24 target verifier context");
  const auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual shared CUDA target memory"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaResource{device}),"ordinary shared CUDA Q24 target owner");
  const auto roles=orig::q24::bind(*data24,owner);
  const auto complete=packed_moe::adapt(std::make_shared<const full24::Roles>(full24::bind(data24_full,owner)));
  const auto target=packed_moe::adapt(std::make_shared<const target24_roles::Bound>(target24_roles::bind(data24_target,owner)));
  auto nextn_supply=std::make_shared<ExpertSupply>();nextn_supply->roles=complete;
  nextn_supply->source=expert_source(complete->source,std::array{full24::Role::kExpertGate,full24::Role::kExpertUp,full24::Role::kExpertDown});
  nextn_supply->cuda=std::make_shared<resident24::CudaSupply>(take(resident24::CudaSupply::create(context,owner,*device,nextn_supply->source.descriptions),"bounded original Q24 NextN CUDA supply"));
  auto target_supply=std::make_shared<ExpertSupply>();
  target_supply->source=expert_source(target->source,std::array{target24_roles::Role::kExpertGate,target24_roles::Role::kExpertUp,target24_roles::Role::kExpertDown});
  target_supply->cuda=std::make_shared<resident24::CudaSupply>(take(resident24::CudaSupply::create(context,owner,*device,target_supply->source.descriptions),"bounded original Q24 target CUDA supply"));
  auto draft=Program::create(build(q24,owner,roles,nullptr,data24.get(),{},{},complete),context,owner,true,false,nextn_supply);
  auto target_head=target_vocabulary(complete,complete->at(full24::Role::kHead),complete->source->bytes(full24::Role::kHead),complete->source->spec(full24::Role::kHead).native);
  auto target_plan=prepare_target24(target,std::move(target_head),target_supply,context,owner,true);
  run_original_target_verifier(std::move(draft),std::move(target_plan),context,owner,true);
}
void run_cuda_original_q24_random(const orig::q24::Data &kv,
    std::shared_ptr<const full24::Data> nextn_data,std::shared_ptr<const target24_roles::Data> target_data,
    std::size_t max_drafts) {
  auto context=take(g::cuda::create_context(),"ordinary CUDA Q24 target verifier context");
  const auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"actual shared CUDA target memory"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaResource{device}),"ordinary shared CUDA Q24 target owner");
  const auto roles=orig::q24::bind(kv,owner);
  const auto complete=packed_moe::adapt(std::make_shared<const full24::Roles>(full24::bind(nextn_data,owner)));
  const auto target=packed_moe::adapt(std::make_shared<const target24_roles::Bound>(target24_roles::bind(target_data,owner)));
  auto nextn_supply=std::make_shared<ExpertSupply>();nextn_supply->roles=complete;
  nextn_supply->source=expert_source(complete->source,std::array{full24::Role::kExpertGate,full24::Role::kExpertUp,full24::Role::kExpertDown});
  nextn_supply->cuda=std::make_shared<resident24::CudaSupply>(take(resident24::CudaSupply::create(context,owner,*device,nextn_supply->source.descriptions),"bounded original Q24 NextN CUDA supply"));
  auto target_supply=std::make_shared<ExpertSupply>();
  target_supply->source=expert_source(target->source,std::array{target24_roles::Role::kExpertGate,target24_roles::Role::kExpertUp,target24_roles::Role::kExpertDown});
  target_supply->cuda=std::make_shared<resident24::CudaSupply>(take(resident24::CudaSupply::create(context,owner,*device,target_supply->source.descriptions),"bounded original Q24 target CUDA supply"));
  auto draft=Program::create(build(q24,owner,roles,nullptr,&kv,{},{},complete),context,owner,true,false,nextn_supply);
  auto target_head=target_vocabulary(complete,complete->at(full24::Role::kHead),complete->source->bytes(full24::Role::kHead),complete->source->spec(full24::Role::kHead).native);
  auto target_plan=prepare_target24(target,std::move(target_head),target_supply,context,owner,true);
  run_original_target_verifier(std::move(draft),std::move(target_plan),context,owner,true,true,{},max_drafts);
}
TEST_F(QwenMtpStateConsumer,CudaQ24OriginalTargetAndNextNRandomVerifierContinuesAtomically) {
  if(!data24||!prepare_original24_data()||!prepare_original24_target())GTEST_SKIP()<<"Set original Q24 NextN and target91 directories";
  run_cuda_original_q24_random(*data24,data24_full,data24_target,1);
}
TEST_F(QwenMtpStateConsumer,CudaQ24OriginalTargetAndNextNK2RandomVerifierContinuesAtomically) {
  if(!data24||!prepare_original24_data()||!prepare_original24_target())GTEST_SKIP()<<"Set original Q24 NextN and target91 directories";
  run_cuda_original_q24_random(*data24,data24_full,data24_target,2);
}

#endif
} // namespace
