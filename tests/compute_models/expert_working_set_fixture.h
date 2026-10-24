#pragma once

#include "ct_packed_grouped_fixture.h"
#include "nvfp4_reference_fixture.h"
#include "hyle/compute/weight_assembly.h"
#include "hyle/compute/weight_import.h"
#include "hyle/graph/weight_binding.h"

#include <cstring>
#include <gtest/gtest.h>

namespace expert_working_set_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace f = ct_packed_graph_test;
namespace nv = nvfp4_reference_test;
using nv::take;
using nv::check;
using nv::dense;
using Type = c::DataType;
enum class Family { kNvfp4, kCtOdd };
struct BankSpec final { std::uint64_t id; c::TensorDesc description; };
inline constexpr std::int64_t local_experts = 4, global_experts = 512, tokens = 2, slots_count = 3;
inline constexpr std::uint64_t weights_id = 10, zeros_id = 20, scales_id = 30,
    weight_global_id = 40, input_global_id = 45, map_id = 50;
inline std::array<std::int32_t,4> selection(unsigned epoch) {
  if (epoch == 0) return {17,511,3,256};
  if (epoch == 1) return {9,127,255,0};
  return {511,17,256,3};
}
inline std::array<std::int32_t,6> global_ids(unsigned epoch) {
  const auto set = selection(epoch);
  return {set[2],set[0],set[2],set[3],set[1],set[0]};
}
template<class T> std::vector<std::byte> raw(std::span<const T> values) {
  const auto bytes = std::as_bytes(values);
  return {bytes.begin(),bytes.end()};
}
template<class T> std::vector<std::byte> raw(const std::vector<T> &values) {
  return raw(std::span<const T>(values));
}

struct Case final {
  Family family;
  std::int64_t n, k;
  c::TensorDesc a_desc, ids_desc, output_desc;
  std::vector<BankSpec> banks;
  std::vector<g::WeightBinding> bindings;
  g::Graph graph;
  g::ValueId a{}, ids{}, output{};
  explicit Case(Family chosen) : family(chosen), n(chosen == Family::kNvfp4 ? 2048 : 7),
      k(chosen == Family::kNvfp4 ? 8192 : 256),
      a_desc(dense(chosen == Family::kNvfp4 ? Type::kBFloat16 : Type::kFloat32,{tokens,1,k})),
      ids_desc(dense(Type::kInt32,{tokens,slots_count})),
      output_desc(dense(chosen == Family::kNvfp4 ? Type::kBFloat16 : Type::kFloat32,
          {tokens,slots_count,n})) {
    if (family == Family::kNvfp4) banks = {
        {weights_id,dense(Type::kUInt8,{local_experts,n,k/2})},
        {scales_id,dense(Type::kFloat8E4M3FN,{local_experts,n,k/16})},
        {weight_global_id,dense(Type::kFloat32,{local_experts})},
        {input_global_id,dense(Type::kFloat32,{local_experts})}};
    else banks = {{weights_id,dense(Type::kInt32,{local_experts*n,k/8})},
        {zeros_id,dense(Type::kInt32,{(local_experts*n+7)/8,k/128})},
        {scales_id,dense(Type::kFloat16,{local_experts*n,k/128})}};
    banks.push_back({map_id,dense(Type::kInt32,{global_experts})});
    auto builder = take(g::GraphBuilder::create(),"working-set builder");
    a = take(builder.input(a_desc),"working-set A");
    ids = take(builder.input(ids_desc),"original global IDs");
    std::vector<g::ValueId> ports;
    for (const auto &bank : banks) {
      const auto value = take(builder.input(bank.description),"aggregate bank input");
      bindings.push_back({value,bank.id}); ports.push_back(value);
    }
    const auto local_ids = take(builder.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},
        ports.back(),ids),"checked global-to-local Gather");
    if (family == Family::kNvfp4) output = take(builder.add(
        c::model_ops::grouped_nvfp4_linear(),a,ports[0],ports[1],ports[2],ports[3],local_ids),
        "original selected NVFP4 consumer");
    else {
      const auto decoded = take(builder.result(take(builder.add_node(
          c::model_ops::artifact_weight_decode(c::model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4,
              128,Type::kBFloat16),std::span(ports).first(3)),"original CT flat decode")),"CT decoded value");
      const auto shape = take(c::TensorShape::create(std::array{local_experts,n,k}),"CT expert shape");
      const auto weight = take(builder.add(take(c::View::reshape(shape),"CT reshape spec"),decoded),"CT reshape");
      output = take(builder.add(c::model_ops::grouped_matmul(),weight,a,local_ids),"original grouped CT consumer");
    }
    check(builder.publish(output),"publish actual grouped output");
    graph = take(std::move(builder).finish(),"finish working-set graph");
  }
  std::size_t logical_bytes() const {
    std::size_t result = 0;
    for (const auto &bank : banks) result += bank.description.required_bytes().value();
    return result;
  }
  std::size_t capacity() const { return (logical_bytes()+4096+255)/256*256; }
  std::vector<std::byte> activation() const {
    if (family == Family::kNvfp4) {
      std::vector<std::uint16_t> values(static_cast<std::size_t>(tokens*k),0);
      values[0] = nv::bf16(6.F); values[static_cast<std::size_t>(k+32)] = nv::bf16(6.F);
      return raw(values);
    }
    std::vector<float> values(static_cast<std::size_t>(tokens*k),0.F);
    values[0] = 1.F; values[static_cast<std::size_t>(k+128)] = 1.F;
    return raw(values);
  }
};

// Source storage contains only four selected original experts, never the512
// source tensors. All source buffers are discarded immediately after upload.
struct Sources final {
  std::vector<m::Buffer> storage;
  std::vector<c::WeightSlice> slices;
  std::vector<std::byte> expected;
  std::vector<std::vector<std::byte>> original_bytes;
  Sources() = default;
  void add(std::uint64_t id,c::TensorDesc desc,std::span<const std::byte> bytes,
           std::size_t destination_offset) {
    auto data = take(m::Buffer::create(m::system_resource(),bytes.size(),64),"selected host fragment");
    check(data.view().write(bytes),"fill selected original source");
    const auto view = take(c::ConstTensorView::create(desc,data.view()),"selected source view");
    storage.push_back(std::move(data)); slices.push_back({id,view,destination_offset});
  }
  static unsigned code(std::int32_t global,std::int64_t n,std::int64_t k,unsigned epoch) {
    constexpr std::array<unsigned,8> codes{1,2,3,4,5,7,10,13};
    return codes[static_cast<std::size_t>((global+n*3+k+epoch*5)%8)];
  }
  static std::uint8_t scale(std::int32_t global,std::int64_t n,std::int64_t group,unsigned epoch) {
    constexpr std::array<std::uint8_t,3> values{0x30,0x38,0x40};
    return values[static_cast<std::size_t>((global+n+group+epoch)%3)];
  }
  static float wg(std::int32_t global,unsigned epoch) { return std::ldexp(1.F,int((unsigned(global)+epoch)%3)-1); }
  explicit Sources(const Case &f,unsigned epoch) {
    const auto selected = selection(epoch);
    const auto routed = global_ids(epoch);
    if (f.family == Family::kNvfp4) {
      std::vector<std::uint16_t> result(static_cast<std::size_t>(tokens*slots_count*f.n));
      for (std::size_t local=0;local<selected.size();++local) {
        const auto global=selected[local];
        std::vector<std::uint8_t> w(static_cast<std::size_t>(f.n*f.k/2));
        for (std::int64_t n=0;n<f.n;++n) for (std::int64_t pair=0;pair<f.k/2;++pair)
          w[static_cast<std::size_t>(n*(f.k/2)+pair)] = static_cast<std::uint8_t>(
              code(global,n,pair*2,epoch)|(code(global,n,pair*2+1,epoch)<<4));
        add(weights_id,dense(Type::kUInt8,{f.n,f.k/2}),raw(w),local*w.size());
        std::vector<std::uint8_t> sf(static_cast<std::size_t>(f.n*f.k/16));
        for (std::int64_t n=0;n<f.n;++n) for (std::int64_t group=0;group<f.k/16;++group)
          sf[static_cast<std::size_t>(n*(f.k/16)+group)] = scale(global,n,group,epoch);
        add(scales_id,dense(Type::kFloat8E4M3FN,{f.n,f.k/16}),raw(sf),local*sf.size());
        const std::array<float,1> global_w{wg(global,epoch)}, global_a{wg(global,epoch+1)};
        add(weight_global_id,dense(Type::kFloat32,{1}),std::as_bytes(std::span(global_w)),local*4);
        add(input_global_id,dense(Type::kFloat32,{1}),std::as_bytes(std::span(global_a)),local*4);
      }
      // One-hot6 is restored exactly by every selected power-of-two IG. Only
      // this real code participates in the dot; all W/SF bytes are still bound.
      for (std::int64_t t=0;t<tokens;++t) for (std::int64_t s=0;s<slots_count;++s)
        for (std::int64_t n=0;n<f.n;++n) {
          const auto global=routed[static_cast<std::size_t>(t*slots_count+s)];
          const auto hot=t==0?0:32;
          const auto weight=nv::e2m1[code(global,n,hot,epoch)]*
              nv::e4m3(scale(global,n,hot/16,epoch))*wg(global,epoch);
          result[static_cast<std::size_t>((t*slots_count+s)*f.n+n)] = nv::bf16(6.F*weight);
        }
      expected=raw(result);
    } else {
      std::vector<ct_packed_test::Fixture> originals;
      std::vector<c::ConstTensorView> zeros;
      for (std::size_t local=0;local<selected.size();++local) {
        originals.emplace_back(true,unsigned(selected[local])+epoch*13,f.n,f.k,128);
        const auto &w=originals.back();
        const auto codes=f::bytes(w.codes),sf=f::floating(Type::kFloat16,w.scales),z=f::bytes(w.zeros);
        add(weights_id,dense(Type::kInt32,{f.n,f.k/8}),codes,local*codes.size());
        add(scales_id,dense(Type::kFloat16,{f.n,f.k/128}),sf,local*sf.size());
        auto backing=take(m::Buffer::create(m::system_resource(),z.size(),64),"original padded expert Z");
        check(backing.view().write(z),"write original expert Z");
        zeros.push_back(take(c::ConstTensorView::create(dense(Type::kInt32,{(f.n+7)/8,f.k/128}),backing.view()),"Z source"));
        storage.push_back(std::move(backing));
      }
      const auto zdesc=dense(Type::kInt32,{(local_experts*f.n+7)/8,f.k/128});
      auto packed=take(m::Buffer::create(m::system_resource(),zdesc.required_bytes().value(),64),"tight selected Z bank");
      check(c::pack_int4_zero_rows(zeros,f.n,take(c::TensorView::create(zdesc,packed.view()),"Z destination")),"explicit oddN zero packing");
      std::vector<std::byte> tight(packed.size()),naive;
      check(packed.view().read(tight),"read packed Z");
      for (const auto &source : zeros) {
        std::vector<std::byte> bytes(source.description().required_bytes().value());
        check(source.storage().read(bytes),"read original padded Z"); naive.insert(naive.end(),bytes.begin(),bytes.end());
      }
      EXPECT_EQ(tight.size(),naive.size()); EXPECT_NE(tight,naive);
      // Check every real selected row using the independent logical zero list.
      for (std::int64_t row=0;row<local_experts*f.n;++row) for (std::int64_t group=0;group<f.k/128;++group) {
        std::uint32_t word;
        std::memcpy(&word,tight.data()+static_cast<std::size_t>(((row/8)*(f.k/128)+group)*4),4);
        const auto actual=(word>>(4*(row%8)))&15u;
        const auto &original=originals[static_cast<std::size_t>(row/f.n)];
        EXPECT_EQ(actual,static_cast<unsigned>(original.logical_zeros[static_cast<std::size_t>((row%f.n)*(f.k/128)+group)]+8));
      }
      slices.push_back({zeros_id,take(c::ConstTensorView::create(zdesc,packed.view()),"tight original zeros"),0});
      storage.push_back(std::move(packed));
      std::vector<float> result(static_cast<std::size_t>(tokens*slots_count*f.n));
      for (std::int64_t t=0;t<tokens;++t) for (std::int64_t s=0;s<slots_count;++s) {
        const auto global=routed[static_cast<std::size_t>(t*slots_count+s)];
        const auto local=static_cast<std::size_t>(std::find(selected.begin(),selected.end(),global)-selected.begin());
        const auto decoded=originals[local].decoded(Type::kFloat16,Type::kBFloat16,Type::kBFloat16);
        for (std::int64_t n=0;n<f.n;++n) {
          float sum=0.F;
          for (std::int64_t k=0;k<f.k;++k) {
            const double activation=k==(t==0?0:128)?1.:0.;
            sum=nv::f32(double(sum)+activation*decoded[static_cast<std::size_t>(n*f.k+k)]);
          }
          result[static_cast<std::size_t>((t*slots_count+s)*f.n+n)]=sum;
        }
      }
      expected=raw(result);
    }
    std::vector<std::int32_t> map(static_cast<std::size_t>(global_experts),-1);
    for (std::size_t local=0;local<selected.size();++local) map[static_cast<std::size_t>(selected[local])]=static_cast<std::int32_t>(local);
    add(map_id,dense(Type::kInt32,{global_experts}),raw(map),0);
    std::reverse(slices.begin(),slices.end()); // Real upload order differs from bank/expert order.
    std::size_t logical=0; for (const auto &slice:slices) logical+=slice.source.description().required_bytes().value();
    EXPECT_EQ(logical,f.logical_bytes());
    for (const auto &buffer:storage) {
      original_bytes.emplace_back(buffer.size());
      check(buffer.view().read(original_bytes.back()),"snapshot original selected source bytes");
    }
  }
  void expect_unchanged() const {
    ASSERT_EQ(storage.size(),original_bytes.size());
    for (std::size_t i=0;i<storage.size();++i) {
      std::vector<std::byte> current(storage[i].size());
      check(storage[i].view().read(current),"verify original selected source bytes");
      EXPECT_EQ(current,original_bytes[i]);
    }
  }
};

struct Episode final {
  g::Invocation invocation;
  std::array<m::Buffer,2> inputs;
  g::GraphOutput output;
  c::ReadLease lease;
  std::vector<std::byte> expected;
};
template<class Api> void expect_output(Api &api,const Case &f,const Episode &episode) {
  const auto tensor=take(episode.output.tensor(),"owned original result tensor");
  std::vector<std::byte> bytes(episode.expected.size());
  check(api.read(tensor.storage(),bytes),"read actual grouped result"); EXPECT_EQ(bytes,episode.expected);
  const auto view=take(episode.lease.tensor(f.output_desc),"independent original result storage lease");
  check(api.read(view.storage(),bytes),"read independent result lease"); EXPECT_EQ(bytes,episode.expected);
}
template<class Api,class Batch> Episode execute(Api &api,const Case &f,const g::Executable &executable,
    const Batch &batch,unsigned epoch,std::vector<std::byte> expected,c::LeasePool &output_pool,
    std::int32_t last_id=-2) {
  Episode result;
  result.invocation=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"actual working-set invocation");
  const auto activation=f.activation(); auto ids=global_ids(epoch); if (last_id!=-2) ids.back()=last_id;
  const auto id_bytes=std::as_bytes(std::span(ids));
  const std::array<std::span<const std::byte>,2> payload{activation,id_bytes};
  const std::array desc{f.a_desc,f.ids_desc}; const std::array ports{f.a,f.ids};
  for (std::size_t i=0;i<result.inputs.size();++i) {
    result.inputs[i]=take(m::Buffer::create(api.resource,payload[i].size(),256),"dynamic A/global IDs");
    check(api.write(result.inputs[i].view(),payload[i]),"write dynamic working-set source");
    check(result.invocation.bind(ports[i],take(c::ConstTensorView::create(desc[i],result.inputs[i].view()),"dynamic input view")),"bind dynamic source");
  }
  check(api.bind(result.invocation,f.bindings,batch),"atomic same-generation bank+map binding");
  if (output_pool.valid()) check(result.invocation.bind(f.output,f.output_desc,
      take(output_pool.allocate(f.output_desc.required_bytes().value()),"actual grouped result writer")),"bind output writer");
  check(api.prepare(result.invocation),"prepare original graph");
  auto accepted=take(result.invocation.submit(api.queues),"submit original graph");
  const auto status=accepted.wait();
  if (last_id!=-2) {
    EXPECT_FALSE(status.ok()); EXPECT_FALSE(result.invocation.result(f.output));
    return result;
  }
  check(status,"actual working-set completion");
  result.output=take(result.invocation.result(f.output),"owned actual result");
  result.lease=take(result.output.storage_lease(),"retain independent output permission");
  result.expected=std::move(expected);
  expect_output(api,f,result);
  return result;
}
template<class Batch> void expect_report(const Batch &batch,std::size_t logical,std::size_t capacity) {
  check(batch.wait(),"wait original weight upload");
  const auto report=take(batch.report(),"actual accepted upload report");
  EXPECT_EQ(report.logical_bytes,logical); EXPECT_EQ(report.staged_bytes,logical);
  EXPECT_GE(report.submitted_bytes,logical); EXPECT_LE(report.submitted_bytes,capacity);
  EXPECT_EQ(report.completed_bytes,report.submitted_bytes); EXPECT_FALSE(report.copied_bytes_unknown);
}
// Backend is a thin adapter: resource/queues fields, create_slots/upload/bind/
// compile/prepare/write/read methods. It preserves the actual native owners.
template<class Api> void run_lifetimes(Api &api,Family family) {
  Case f(family);
  if (family==Family::kNvfp4) {
    EXPECT_EQ(f.n,2048); EXPECT_EQ(f.k,8192);
    EXPECT_EQ(std::uint64_t(global_experts)*std::uint64_t(f.n)*std::uint64_t(f.k)/2,4294967296ULL);
    EXPECT_EQ(f.logical_bytes(),37750816u);
  }
  auto compiled=take(api.compile(f.graph),"compile ordinary working-set graph");
  if (family==Family::kCtOdd) {
    bool packed=false;
    for (std::size_t i=0;i<compiled.report.step_count();++i)
      packed|=take(compiled.report.step(i),"actual CT step").policy==f::kPackedPolicy;
    EXPECT_TRUE(packed) << "Working-set test must select the existing bounded packed CT route";
  }
  auto slots=take(api.create_slots(f.banks,f.capacity()),"two real bounded upload slots");
  const auto ownership=take(compiled.executable.output_ownership(),"actual result ownership");
  c::LeasePool pool;
  if (ownership==g::OutputOwnership::kCaller) {
    const auto bytes=f.output_desc.required_bytes().value();
    pool=take(c::LeasePool::create(api.resource,3*((bytes+255)/256*256),{.max_blocks=3}),"three finite result blocks");
  } else EXPECT_EQ(ownership,g::OutputOwnership::kInvocation);
  std::array<Episode,3> episodes;
  std::array<typename Api::Batch,2> batches;
  for (unsigned epoch=0;epoch<2;++epoch) {
    Sources source(f,epoch); auto expected=source.expected;
    batches[epoch]=take(api.upload(slots,epoch,source.slices),"actual selected-fragment upload");
    source.expect_unchanged();
    source=Sources{}; // Host original payloads are gone before compute.
    episodes[epoch]=execute(api,f,compiled.executable,batches[epoch],epoch,std::move(expected),pool);
    expect_report(batches[epoch],f.logical_bytes(),f.capacity());
  }
  expect_output(api,f,episodes[0]); expect_output(api,f,episodes[1]);
  EXPECT_NE(episodes[0].expected,episodes[1].expected);
  Sources third(f,2); auto expected=third.expected;
  EXPECT_FALSE(api.upload(slots,0,third.slices)); // Actual retained WeightBatch.
  batches[0]={};
  EXPECT_FALSE(api.upload(slots,0,third.slices)); // Invocation/InputSource still owns it.
  third.expect_unchanged();
  episodes[0]=Episode{}; // Release actual result, lease, invocation and dynamic inputs.
  batches[0]=take(api.upload(slots,0,third.slices),"slot0 reused after real owner release");
  third.expect_unchanged();
  third=Sources{};
  episodes[2]=execute(api,f,compiled.executable,batches[0],2,std::move(expected),pool);
  expect_report(batches[0],f.logical_bytes(),f.capacity());
  expect_output(api,f,episodes[1]);
  batches={};
  episodes[1].invocation={}; episodes[2].invocation={}; compiled=g::CompiledGraph{};
  expect_output(api,f,episodes[1]); expect_output(api,f,episodes[2]);
  check(slots.close(),"close original native upload slots");
  ::testing::Test::RecordProperty("actual_selected_global_experts",4);
  ::testing::Test::RecordProperty("global_expert_domain",512);
  ::testing::Test::RecordProperty("successful_upload_generations",3);
  ::testing::Test::RecordProperty("actual_graph_submissions",3);
  ::testing::Test::RecordProperty("uploaded_logical_bytes",std::to_string(f.logical_bytes()*3));
  ::testing::Test::RecordProperty("original_fragment_slices_per_generation",family==Family::kNvfp4?17:10);
}

class ReadFaultResource final : public m::MemoryResource {
 public:
  explicit ReadFaultResource(std::span<std::byte> bytes) : bytes_(bytes) {}
  unsigned reads=0,fail_at=0; std::size_t successful_bytes=0,partial_bytes=0;
  std::string_view name() const noexcept override { return "expert-source-read-fault"; }
  m::MemoryProperties properties() const noexcept override {
    return {m::MemoryProperty::kHostAccessible|m::MemoryProperty::kHostAddressIdentity|
        m::MemoryProperty::kHostDeviceCoherent,1,bytes_.size(),{this,1}};
  }
  hyle::Result<m::ResourceAllocation> allocate(std::size_t,std::size_t) noexcept override {
    return std::unexpected(hyle::Status::Unimplemented("adopted bounded test source"));
  }
  void deallocate(m::ResourceAllocation) noexcept override {}
  hyle::Result<m::ResourceHostMapping> map_host(void *,std::size_t offset,std::size_t size,m::HostAccess) noexcept override {
    if (offset>bytes_.size()||size>bytes_.size()-offset)
      return std::unexpected(hyle::Status::OutOfRange("fault source range"));
    return m::ResourceHostMapping{bytes_.data()+offset,nullptr};
  }
  hyle::Status read(void *,std::size_t offset,std::span<std::byte> destination) noexcept override {
    if (offset>bytes_.size()||destination.size()>bytes_.size()-offset)
      return hyle::Status::OutOfRange("fault source read range");
    ++reads;
    const auto count=reads==fail_at?(std::min)(std::size_t{2},destination.size()):destination.size();
    std::memcpy(destination.data(),bytes_.data()+offset,count);
    if (reads==fail_at) { partial_bytes+=count; return hyle::Status::Unavailable("intentional partial second host read"); }
    successful_bytes+=count; return hyle::Status::Ok();
  }
 private:
  std::span<std::byte> bytes_;
};

template<class Api> void run_failures(Api &api) {
  Case f(Family::kCtOdd);
  auto compiled=take(api.compile(f.graph),"checked remap graph");
  auto slots=take(api.create_slots(f.banks,f.capacity()),"checked resource slots");
  Sources source(f,0); const auto expected=source.expected;
  auto batch=take(api.upload(slots,0,source.slices),"valid immutable selected generation");
  source.expect_unchanged();
  source=Sources{};
  c::LeasePool pool;
  if (take(compiled.executable.output_ownership(),"checked result ownership")==g::OutputOwnership::kCaller) {
    const auto bytes=f.output_desc.required_bytes().value();
    pool=take(c::LeasePool::create(api.resource,(bytes+255)/256*256,{.max_blocks=1}),"checked one-result block");
  }
  for (unsigned attempt=0;attempt<5;++attempt) {
    const auto bad=attempt==1?1:attempt==3?static_cast<std::int32_t>(global_experts):-2;
    SCOPED_TRACE(attempt);
    auto episode=execute(api,f,compiled.executable,batch,0,expected,pool,bad);
  }
  expect_report(batch,f.logical_bytes(),f.capacity());
  check(slots.close(),"close checked-remap upload");

  // Isolate real copy failure/accounting from arithmetic-domain failures.
  alignas(64) std::array<std::byte,8> backing{
      std::byte{1},std::byte{3},std::byte{5},std::byte{7},std::byte{9},std::byte{11},std::byte{13},std::byte{15}};
  const auto original=backing;
  ReadFaultResource spy(backing); int token=0;
  auto adopted=take(m::Buffer::adopt(m::MemoryResourceRef(spy),{&token,backing.size(),64}),"canonical fault source");
  const auto half=dense(Type::kUInt8,{4});
  const std::array banks{BankSpec{99,dense(Type::kUInt8,{8})}};
  auto copies=take(api.create_slots(banks,4096),"original fault-test slots");
  const std::array slices{
      c::WeightSlice{99,take(c::ConstTensorView::create(half,adopted.view()),"logical first half in larger storage"),0},
      c::WeightSlice{99,take(c::ConstTensorView::create(half,adopted.view().subview(4,4).value()),"second source half"),4}};
  {
    auto first=take(api.upload(copies,0,slices),"prime original accepted episode");
    expect_report(first,8,4096);
  }
  spy.reads=0; spy.successful_bytes=0; spy.partial_bytes=0;
  const std::array hole{slices[0]}; EXPECT_FALSE(api.upload(copies,0,hole));
  auto overlap=slices; overlap[1].destination_offset=0; EXPECT_FALSE(api.upload(copies,0,overlap));
  std::array<c::WeightSlice,65> too_many;
  too_many.fill(slices[0]); EXPECT_FALSE(api.upload(copies,0,too_many));
  EXPECT_EQ(spy.reads,0u); EXPECT_EQ(backing,original);
  spy.fail_at=2;
  const auto failed=api.upload(copies,0,slices);
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().message(),"intentional partial second host read");
  const auto report=take(copies.report(0),"failed source report is still observable");
  EXPECT_EQ(spy.successful_bytes,4u); EXPECT_EQ(spy.partial_bytes,2u);
  EXPECT_EQ(report.logical_bytes,8u); EXPECT_EQ(report.staged_bytes,spy.successful_bytes);
  EXPECT_EQ(report.submitted_bytes,0u); EXPECT_EQ(report.completed_bytes,0u); EXPECT_TRUE(report.copied_bytes_unknown);
  EXPECT_EQ(backing,original);
  spy.fail_at=0; spy.reads=0; spy.successful_bytes=0; spy.partial_bytes=0;
  auto recovered=take(api.upload(copies,0,slices),"fresh upload after unaccepted host failure");
  expect_report(recovered,8,4096); EXPECT_EQ(spy.successful_bytes,8u);
  check(copies.close(),"close original fault-test slots");
  ::testing::Test::RecordProperty("successful_grouped_submissions",3);
  ::testing::Test::RecordProperty("expected_failed_grouped_completions",2);
  ::testing::Test::RecordProperty("expected_partial_host_read_failures",1);
  ::testing::Test::RecordProperty("failed_host_read_proven_successful_prefix_bytes",4);
}
} // namespace expert_working_set_test
