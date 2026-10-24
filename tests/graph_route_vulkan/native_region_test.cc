#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "hyle/graph/vulkan.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"
#include "hyle/graph/batch.h"
#include "hyle/compute/vulkan_batch.h"
#include "../compute_provider_native/half_test_support.h"
#if defined(HYLE_TEST_VULKAN_CPU_CONSTANT)
#include "hyle/graph/cpu.h"
#endif
#include "hyle/compute/vulkan_memory.h"
#include "hyle/mem/quota_allocator.h"
#include "../../src/compute_route_vulkan/vulkan_copy_internal.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"
#include "../../src/graph/execution_source_internal.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace vn = v::detail;

const c::TensorDesc description = c::TensorDesc::dense(c::DataType::kFloat32,
    std::array<std::int64_t, 1>{4}).value();

template<class Query=hyle::simd::bfloat16_t,class Cache=Query,class Output=Query>
hyle::Status floating_page_case(bool mla) {
  const auto require = [](bool value, const char *message) {
    return value ? hyle::Status::Ok() : hyle::Status::Internal(message);
  };
  constexpr unsigned rows = 4, heads = 8, pages = 4, page_size = 16;
  const unsigned depth = mla ? 576 : 256, value_depth = mla ? 512 : 256;
  const unsigned kv_heads = mla ? 1 : 2;
  const auto storage = c::CacheStorage::plain(half_test::data_type<Cache>()).value();
  c::BatchSchema schema;
  if (mla) {
    HYLE_ASSIGN_OR_RETURN(schema, c::BatchSchema::create_mla({
        .query_type=half_test::data_type<Query>(), .output_type=half_test::data_type<Query>(), .cache_storage=c::CacheStorage::plain(half_test::data_type<Query>()).value(), .layout=c::MlaCacheLayout::kCkv512Rope64,
        .max_requests=2, .max_query_rows=rows, .max_kv_rows=64, .max_pages=pages,
        .max_page_refs=5, .query_heads=heads, .page_size=page_size}));
  } else {
    HYLE_ASSIGN_OR_RETURN(schema, c::BatchSchema::create({
        .storage=c::BatchStorage::kPaged, .query_type=half_test::data_type<Query>(),
        .output_type=half_test::data_type<Output>(), .key_storage=storage, .value_storage=storage,
        .max_requests=2, .max_query_rows=rows, .max_kv_rows=64, .max_pages=pages,
        .max_page_refs=5, .query_heads=heads, .kv_heads=kv_heads, .qk_head_dim=depth,
        .value_head_dim=value_depth, .page_size=page_size}));
  }
  const std::array<std::int32_t,3> qp{0,1,4}, kp{0,2,5};
  const std::array<std::int32_t,5> ids{3,1,0,2,1};
  const std::array<std::int32_t,2> tails{15,1};
  HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  ::testing::Test::RecordProperty("vulkan_device", context.environment().device_name);
  HYLE_RETURN_IF_ERROR(require(!context.environment().software_device, "paged attention requires actual Vulkan device"));
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context,v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(context,v::MemoryKind::kStaging));
  struct Close { v::Context &context; ~Close() { context.close().IgnoreError(); } } close{context};
  std::vector<Query> query(std::size_t(rows)*heads*depth);
  std::vector<Cache> keys(std::size_t(pages)*page_size*kv_heads*depth);
  std::vector<Cache> values(mla ? 0 : std::size_t(pages)*page_size*kv_heads*value_depth);
  for (std::size_t i=0;i<query.size();++i) query[i]=half_test::encode<Query>(float(.17*std::sin(double(i)*.23)));
  for (std::size_t i=0;i<keys.size();++i) keys[i]=half_test::encode<Cache>(float(.19*std::cos(double(i)*.13)));
  for (std::size_t i=0;i<values.size();++i) values[i]=half_test::encode<Cache>(float(.23*std::sin(double(i)*.07)));
  HYLE_ASSIGN_OR_RETURN(auto qd, c::TensorDesc::dense(half_test::data_type<Query>(),
      std::array<std::int64_t,3>{rows,heads,depth}));
  HYLE_ASSIGN_OR_RETURN(auto kd, c::batch_cache_description(schema,mla?c::BatchCacheRole::kMla:c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto vd, c::batch_cache_description(schema,mla?c::BatchCacheRole::kMla:c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto qt, context.upload(qd,std::as_bytes(std::span(query))));
  HYLE_ASSIGN_OR_RETURN(auto q, device.import_readonly(qt));
  const auto role=mla?c::BatchCacheRole::kMla:c::BatchCacheRole::kKey;
  HYLE_ASSIGN_OR_RETURN(auto key_arena,c::PagedArena::create(schema,role,device.ref(),{.max_backing_bytes=keys.size()*2+64}));
  c::PagedArena value_arena;
  if(!mla){HYLE_ASSIGN_OR_RETURN(value_arena,c::PagedArena::create(schema,c::BatchCacheRole::kValue,
      device.ref(),{.max_backing_bytes=values.size()*2+64}));}
  const auto page_bytes=key_arena.page_bytes();
  HYLE_ASSIGN_OR_RETURN(auto host,c::LeasePool::create(staging.ref(),page_bytes+64));
  const std::array specs{c::CopyResourceSpec{staging.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto copy_program,v::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,v::CopyBatchInvocation::create(copy_program));
  const auto fill=[&](c::PagedArena &arena,std::span<const Cache> payload)->hyle::Result<c::PagedArenaRead>{
    std::array<c::ReadLease,pages> ready;
    for(unsigned page=0;page<pages;++page){
      HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));HYLE_RETURN_IF_ERROR(copy.reset());
      HYLE_ASSIGN_OR_RETURN(auto source_writer,host.allocate(arena.page_bytes(),64));
      HYLE_RETURN_IF_ERROR(source_writer.view()->write(std::as_bytes(payload).subspan(page*arena.page_bytes(),arena.page_bytes())));
      HYLE_ASSIGN_OR_RETURN(auto source,source_writer.publish());
      HYLE_RETURN_IF_ERROR(copy.bind_source({0},source));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(writer.lease())));
      const std::array<c::CopyPhase,1> phases{{{}}};
      const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,arena.page_bytes()}}};
      HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto done,copy.submit());
      HYLE_RETURN_IF_ERROR(done.wait());HYLE_ASSIGN_OR_RETURN(ready[page],done.output({0},{1},0,arena.page_bytes()));
      HYLE_RETURN_IF_ERROR(writer.complete(ready[page],page_size));
    }
    return arena.read(layout,ready);
  };
  HYLE_ASSIGN_OR_RETURN(auto key_frame,fill(key_arena,keys));
  c::PagedArenaRead value_frame;
  if(!mla){HYLE_ASSIGN_OR_RETURN(value_frame,fill(value_arena,values));}
  const float scale=1.F/std::sqrt(float(depth));
  const auto verify=[&](const v::Tensor &tensor)->hyle::Status{
    HYLE_ASSIGN_OR_RETURN(auto expected_description,c::batch_output_description(schema.capacity(),rows));
    HYLE_RETURN_IF_ERROR(require(tensor.description()==expected_description,"paged output description changed"));
    std::vector<Output> actual(std::size_t(rows)*heads*value_depth);
    HYLE_RETURN_IF_ERROR(context.download(tensor,std::as_writable_bytes(std::span(actual))));
    double maximum=0;
    for(unsigned request=0;request<2;++request){
      const int count=qp[request+1]-qp[request],history=request?33:31;
      for(int row=0;row<count;++row)for(unsigned head=0;head<heads;++head){
        const int end=history-count+row+1,begin=std::max(0,end-1-16);
        std::vector<double> scores(std::size_t(end-begin));double high=-INFINITY,sum=0;
        const auto physical=[&](int j){return unsigned(ids[kp[request]+j/16])*16u+unsigned(j%16);};
        for(int j=begin;j<end;++j){double dot=0;
          for(unsigned d=0;d<depth;++d)dot+=double(half_test::decode(query[(std::size_t(qp[request]+row)*heads+head)*depth+d]))*
              half_test::decode(keys[(std::size_t(physical(j))*kv_heads+head/(heads/kv_heads))*depth+d]);
          scores[std::size_t(j-begin)]=dot*scale;high=std::max(high,dot*scale);}
        for(auto &score:scores){score=std::exp(score-high);sum+=score;}
        for(unsigned d=0;d<value_depth;++d){double expected=0;
          for(int j=begin;j<end;++j){const auto index=(std::size_t(physical(j))*kv_heads+head/(heads/kv_heads))*(mla?depth:value_depth)+d;
            expected+=scores[std::size_t(j-begin)]/sum*half_test::decode(mla?keys[index]:values[index]);}
          const auto decoded=half_test::decode(actual[(std::size_t(qp[request]+row)*heads+head)*value_depth+d]);
          const auto error=std::abs(double(decoded)-expected);maximum=std::max(maximum,error);
          constexpr double tolerance=std::is_same_v<Output,hyle::simd::bfloat16_t>?.008:.0025;
          HYLE_RETURN_IF_ERROR(require(std::isfinite(decoded)&&error<tolerance,"paged independent FP64 attention differs"));}
      }
    }
    std::printf("paged %s D%u/%u query=%u cache=%u output=%u FP64 maxAbs=%.9g\n",mla?"compact MLA":"attention",depth,value_depth,
        unsigned(half_test::data_type<Query>()),unsigned(half_test::data_type<Cache>()),unsigned(half_test::data_type<Output>()),maximum);
    return hyle::Status::Ok();
  };
  // The independent Direct compact path has one cache payload, without a
  // materialized V tensor. Graph below consumes that same exact page frame.
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=8192,.scale=scale,.causal=true,.window_left=16}));
  {
    HYLE_ASSIGN_OR_RETURN(auto direct,v::PagedBatchInvocation::create(context,recipe,device,staging,
        {.max_metadata_backing_bytes=4096}));
    HYLE_RETURN_IF_ERROR(direct.prepare_batch(layout));
    if(mla){HYLE_RETURN_IF_ERROR(direct.bind(q,key_frame));}
    else {HYLE_RETURN_IF_ERROR(direct.bind(q,key_frame,value_frame));}
    HYLE_ASSIGN_OR_RETURN(auto done,direct.submit());HYLE_ASSIGN_OR_RETURN(auto result,done.wait());
    HYLE_RETURN_IF_ERROR(verify(result->tensors()[0]));
  }
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto qi,builder.input(qd));HYLE_ASSIGN_OR_RETURN(auto ki,builder.input(kd));
  auto vi=ki; if(!mla){HYLE_ASSIGN_OR_RETURN(vi,builder.input(vd));}
  g::NodeId node;
  if(mla){HYLE_ASSIGN_OR_RETURN(auto operation,g::batch::paged_mla_attention(layout,{.scale=scale,.causal=true,.window_left=16}));
    HYLE_ASSIGN_OR_RETURN(node,builder.add_node(c::Operation::custom(operation),std::array{qi,ki}));}
  else {HYLE_ASSIGN_OR_RETURN(auto operation,g::batch::paged_attention(layout,{.scale=scale,.causal=true,.window_left=16}));
    HYLE_ASSIGN_OR_RETURN(node,builder.add_node(c::Operation::custom(operation),std::array{qi,ki,vi}));}
  HYLE_ASSIGN_OR_RETURN(auto output,builder.result(node));HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto source,g::vulkan::execution_source(context));
  const std::array domains{g::StorageDomain{device.ref()},g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::batch::compile(graph,{.storage_domains=domains,.targets=targets}));
  const std::array layouts{g::batch::NodeLayout{node,layout}};
  std::array<g::Invocation,2> slots;
  for(auto &slot:slots){
    HYLE_ASSIGN_OR_RETURN(slot,compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::batch::prepare(slot,{},layouts));HYLE_RETURN_IF_ERROR(slot.bind(qi,qd,q));
    HYLE_RETURN_IF_ERROR(g::batch::bind(slot,ki,key_frame));
    if(!mla){HYLE_RETURN_IF_ERROR(g::batch::bind(slot,vi,value_frame));}
    HYLE_RETURN_IF_ERROR(slot.prepare_replay({}));
  }
  const auto observe=[&](g::GraphOutput &out)->hyle::Status{
    HYLE_ASSIGN_OR_RETURN(auto lease,out.storage_lease());HYLE_ASSIGN_OR_RETURN(auto desc,out.description());
    HYLE_ASSIGN_OR_RETURN(auto tensor,device.binding(std::move(lease),desc));return verify(tensor);
  };
  HYLE_ASSIGN_OR_RETURN(auto first,slots[0].submit());HYLE_RETURN_IF_ERROR(first.wait());first={};
  HYLE_ASSIGN_OR_RETURN(auto old,slots[0].result(output));HYLE_RETURN_IF_ERROR(observe(old));
  HYLE_RETURN_IF_ERROR(require(!slots[0].submit(),"retained Graph output allowed native slot reuse"));
  HYLE_ASSIGN_OR_RETURN(auto second,slots[1].submit());HYLE_RETURN_IF_ERROR(second.wait());second={};
  {HYLE_ASSIGN_OR_RETURN(auto out,slots[1].result(output));HYLE_RETURN_IF_ERROR(observe(out));}
  HYLE_RETURN_IF_ERROR(observe(old));old={};
  // No prepare or bind: the original recording and output storage replay.
  HYLE_ASSIGN_OR_RETURN(auto replay,slots[0].submit());HYLE_RETURN_IF_ERROR(replay.wait());replay={};
  HYLE_ASSIGN_OR_RETURN(auto out,slots[0].result(output));return observe(out);
}

TEST(VulkanNativeRegion, BfloatPagedAndCompactMlaKeepOriginalSlotsAndOwners) {
  for(bool mla:{false,true}){SCOPED_TRACE(mla?"compact MLA 576/512":"Qwen D256");
    const auto status=floating_page_case(mla);ASSERT_TRUE(status.ok())<<status.message();}
}

TEST(VulkanNativeRegion, IndependentQueryCacheOutputTypesKeepOriginalSlotsAndOwners) {
  using F16=hyle::simd::float16_t;using BF16=hyle::simd::bfloat16_t;
  auto status=floating_page_case<BF16,F16,BF16>(false);ASSERT_TRUE(status.ok())<<status.message();
  status=floating_page_case<F16,BF16,F16>(false);ASSERT_TRUE(status.ok())<<status.message();
  status=floating_page_case<F16,BF16,BF16>(false);ASSERT_TRUE(status.ok())<<status.message();
}

TEST(VulkanNativeRegion, OriginalGraphChargesNativeBackingAndRetainsEscapedOwner) {
  auto context = v::Context::create().value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const auto source = g::vulkan::execution_source(context).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto y = builder.add(c::Unary{c::UnaryKind::kNeg}, x).value();
  ASSERT_TRUE(builder.publish(y).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  const auto compile = [&](c::ResourceBudget budget) {
    return g::compile(graph, {.storage_domains = domains, .targets = targets},
        {.resource_budget = std::move(budget)});
  };
  // This native Neg recipe has separate 16-byte intermediate and output
  // blocks. A logical-output-only quota must reject the actual 32-byte slot.
  const std::array insufficient_bytes{
      c::ResourceBudgetLimit{hyle::mem::system_resource(), SIZE_MAX},
      c::ResourceBudgetLimit{device.ref(), 16}};
  auto denied = c::ResourceBudget::create({.limits = insufficient_bytes,
      .allow_unattributed = true}).value();
  auto compiled_denied = compile(denied);
  ASSERT_TRUE(compiled_denied) << compiled_denied.error().message();
  auto refused = compiled_denied->executable.create_invocation();
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().IsResourceExhausted()) << refused.error().message();
  EXPECT_EQ(denied.usage(device.ref())->live_bytes, 0U);

  const std::array exact_bytes{
      c::ResourceBudgetLimit{hyle::mem::system_resource(), SIZE_MAX},
      c::ResourceBudgetLimit{device.ref(), 32}};
  auto budget = c::ResourceBudget::create({.limits = exact_bytes,
      .allow_unattributed = true}).value();
  v::Tensor escaped;
  {
    auto compiled = compile(budget);
    ASSERT_TRUE(compiled) << compiled.error().message();
    auto slot = compiled->executable.create_invocation();
    ASSERT_TRUE(slot) << slot.error().message();
    EXPECT_EQ(budget.usage(device.ref())->live_bytes, 32U);
    const std::array values{1.0f, -2.0f, 3.0f, -4.0f};
    auto original = context.upload(description, std::as_bytes(std::span(values))).value();
    auto backing = vn::MemoryAccess::export_storage(vn::Access::runtime(context),
        device.ref(), original, hyle::mem::system_allocator()).value();
    auto binding = c::ConstTensorView::create(description, backing.view()).value();
    ASSERT_TRUE(slot->bind(x, binding).ok());
    ASSERT_TRUE(slot->prepare_replay({}).ok());
    auto done = slot->submit();
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    auto output = slot->result(y);
    ASSERT_TRUE(output) << output.error().message();
    auto lease = output->storage_lease();
    ASSERT_TRUE(lease) << lease.error().message();
    auto native = device.binding(std::move(*lease), description);
    ASSERT_TRUE(native) << native.error().message();
    escaped = std::move(*native);
  }
  EXPECT_EQ(budget.usage(device.ref())->live_bytes, 16U);
  std::array<float, 4> actual{};
  ASSERT_TRUE(context.download(escaped, std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<float, 4>{-1.0f, 2.0f, -3.0f, 4.0f}));
  escaped = {};
  EXPECT_EQ(budget.usage(device.ref())->live_bytes, 0U);
}

hyle::Status execute(g::Invocation &slot, g::ValueId input, c::ConstTensorView binding) {
  auto status = slot.bind(input, binding);
  EXPECT_TRUE(status.ok()) << "bind: " << status.message();
  if (!status.ok()) return status;
  status = slot.prepare_replay({});
  EXPECT_TRUE(status.ok()) << "prepare_replay: " << status.message();
  if (!status.ok()) return status;
  auto accepted = slot.submit();
  if (!accepted) {
    ADD_FAILURE() << "submit: " << accepted.error().message();
    return std::move(accepted).error();
  }
  status = accepted->wait();
  EXPECT_TRUE(status.ok()) << "completion: " << status.message();
  return status;
}
void expect(const v::Context &context, c::ConstTensorView actual,
            const std::array<float, 4> &expected) {
  auto native = vn::MemoryAccess::tensor_view(vn::Access::runtime(context), actual,
                                             hyle::mem::system_allocator());
  ASSERT_TRUE(native) << native.error().message();
  std::array<float, 4> host{};
  ASSERT_TRUE(context.download(*native, std::as_writable_bytes(std::span(host))).ok());
  for (std::size_t i = 0; i < host.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(host[i]), std::bit_cast<std::uint32_t>(expected[i])) << i;
}
c::ReadLease input(const v::Context &context, const v::MemoryResource &memory,
                   const std::array<float, 4> &values) {
  auto native = context.upload(description, std::as_bytes(std::span(values))).value();
  return memory.import_readonly(native).value();
}

struct TrackedHost final : hyle::mem::MemoryResource {
  unsigned reads = 0;
  unsigned writes = 0;
  bool fail = false;
  bool fail_write = false;
  std::string_view name() const noexcept override { return "constant test host"; }
  hyle::mem::MemoryProperties properties() const noexcept override {
    return hyle::mem::system_resource().properties();
  }
  hyle::Result<hyle::mem::ResourceAllocation> allocate(std::size_t bytes,
      std::size_t alignment) noexcept override {
    return hyle::mem::system_resource().resource().allocate(bytes, alignment);
  }
  void deallocate(hyle::mem::ResourceAllocation allocation) noexcept override {
    hyle::mem::system_resource().resource().deallocate(allocation);
  }
  hyle::Result<hyle::mem::ResourceHostMapping> map_host(void *handle,
      std::size_t offset, std::size_t bytes, hyle::mem::HostAccess access) noexcept override {
    return hyle::mem::system_resource().resource().map_host(handle, offset, bytes, access);
  }
  hyle::Status unmap_host(void *handle, void *token, std::size_t offset,
      std::size_t bytes, hyle::mem::HostAccess access) noexcept override {
    return hyle::mem::system_resource().resource().unmap_host(handle, token, offset, bytes, access);
  }
  hyle::Status read(void *handle, std::size_t offset, std::span<std::byte> bytes) noexcept override {
    ++reads;
    if (fail) return hyle::Status::Unavailable("injected host constant read failure");
    return hyle::mem::system_resource().resource().read(handle, offset, bytes);
  }
  hyle::Status write(void *handle, std::size_t offset, std::span<const std::byte> bytes) noexcept override {
    ++writes;
    if (fail_write) return hyle::Status::Unimplemented("injected accepted host constant write failure");
    return hyle::mem::system_resource().resource().write(handle, offset, bytes);
  }
};

struct OneFenceFailure final {
  vn::Device &device;
  PFN_vkWaitForFences original;
  unsigned calls = 0;
  static inline OneFenceFailure *active = nullptr;
  explicit OneFenceFailure(const v::Context &context)
      : device(*vn::Access::runtime(context)->device), original(device.fn.WaitForFences) {
    active = this; device.fn.WaitForFences = invoke;
  }
  ~OneFenceFailure() { device.fn.WaitForFences = original; active = nullptr; }
  static VKAPI_ATTR VkResult VKAPI_CALL invoke(VkDevice device, std::uint32_t count,
      const VkFence *fences, VkBool32 all, std::uint64_t timeout) {
    if (++active->calls == 1) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original(device, count, fences, all, timeout);
  }
};

TEST(VulkanNativeRegion, ReplayCopyRecordsOnceAndDetachesRetiredCompletion) {
  auto context = v::Context::create().value();
  RecordProperty("vulkan_device", context.environment().device_name);
  RecordProperty("vulkan_driver", context.environment().driver_info);
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto source = hyle::mem::Buffer::create(staging.ref(), 16, 4).value();
  auto destination = hyle::mem::Buffer::create(device.ref(), 16, 4).value();
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  struct Drain final {
    v::Context &context;
    ~Drain() { context.close().IgnoreError(); }
  } drain{context}; // Drain an accepted Run before its borrowed allocator dies.
  auto recording = vn::prepare_copy_recording_runtime(vn::Access::runtime(context),
      1, metadata.get(), 16, vn::CopyRecordingMode::kReplayable).value();
  const std::array<c::detail::CopyBackendRange, 1> ranges{{{destination.view(), source.view()}}};
  ASSERT_TRUE(vn::bind_copy_recording(*recording, ranges).ok()); // The only cold recording.
  const auto cold_bytes = metadata.used();
  const auto destination_tensor = c::ConstTensorView::create(description, destination.view()).value();
  const std::array payloads{std::array<float, 4>{0.25f, -1.5f, 3.75f, -0.0f},
                            std::array<float, 4>{-2.25f, 0.5f, -7.0f, 1.25f}};
  c::Completion previous;
  for (const auto &payload : payloads) {
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(payload))).ok());
    auto accepted = vn::submit_copy_recording(*recording);
    ASSERT_TRUE(accepted) << accepted.error().message();
    // No observation has retired this original Run or detached its owner.
    // This gate does not depend on how quickly the physical GPU finishes.
    const auto pending = c::detail::CompletionAccess::dependency(*accepted);
    ASSERT_TRUE(pending.pending());
    ASSERT_NE(pending.state, nullptr);
    EXPECT_TRUE(vn::copy_recording_reuse_gate(*recording).IsUnavailable());
    auto retained = vn::submit_copy_recording(*recording);
    ASSERT_FALSE(retained);
    EXPECT_TRUE(retained.error().IsUnavailable()) << retained.error().message();
    if (previous.valid()) {
      // wait() settled the previous Completion inline. Its observation and
      // cancellation must not inspect or cancel this newly accepted episode.
      const auto old = c::detail::CompletionAccess::dependency(previous);
      EXPECT_FALSE(old.pending());
      EXPECT_EQ(old.state, nullptr);
      EXPECT_TRUE(previous.wait().ok());
      auto polled = previous.poll();
      ASSERT_TRUE(polled) << polled.error().message();
      EXPECT_EQ(*polled, c::CompletionPoll::kComplete);
      EXPECT_TRUE(previous.request_cancel().ok());
    }
    ASSERT_TRUE(accepted->wait().ok());
    const auto retired = c::detail::CompletionAccess::dependency(*accepted);
    EXPECT_FALSE(retired.pending());
    EXPECT_EQ(retired.state, nullptr);
    ASSERT_TRUE(vn::copy_recording_reuse_gate(*recording).ok());
    expect(context, destination_tensor, payload);
    previous = std::move(*accepted); // Keep the old terminal object across reuse.
    EXPECT_EQ(metadata.used(), cold_bytes);
    EXPECT_EQ(vn::copy_recording_report(*recording).native_calls, 1U);
  }
  EXPECT_EQ(vn::copy_recording_report(*recording).phase_completions, 2U);
  ASSERT_TRUE(context.close().ok());
  EXPECT_TRUE(vn::copy_recording_reuse_gate(*recording).IsFailedPrecondition());
  auto closed = vn::submit_copy_recording(*recording);
  ASSERT_FALSE(closed);
  EXPECT_TRUE(closed.error().IsFailedPrecondition()) << closed.error().message();
}

TEST(VulkanNativeRegion, UnifiedInvocationKeepsGeneratedViewsSlotsAndOwners) {
  auto context = v::Context::create().value();
  RecordProperty("vulkan_device", context.environment().device_name);
  RecordProperty("vulkan_driver", context.environment().driver_info);
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const auto source = g::vulkan::execution_source(context).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, x).value();
  const auto restored = builder.add(c::Unary{c::UnaryKind::kNeg}, negative).value();
  const auto shape = c::TensorShape::create(std::array<std::int64_t, 2>{2, 2}).value();
  const auto computed_view = builder.add(c::View::reshape(shape).value(), restored).value();
  const auto bound_view = builder.add(c::View::reshape(shape).value(), x).value();
  for (const auto value : {negative, computed_view, bound_view}) ASSERT_TRUE(builder.publish(value).ok());
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {domains, targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.region_count(), 1U);
  EXPECT_EQ(compiled->executable.required_queue_count(), 0U);
  EXPECT_EQ(compiled->executable.workspace_bytes(), 0U);
  auto first = compiled->executable.create_invocation();
  ASSERT_TRUE(first) << first.error().message();
  auto second = compiled->executable.create_invocation();
  ASSERT_TRUE(second) << second.error().message();
  const std::array<float, 4> a{0.25f, -1.5f, 3.75f, -0.0f}, b{-2.25f, 0.5f, -7.0f, 1.25f};
  auto a_input = input(context, memory, a);
  auto b_input = input(context, memory, b);
  auto a_binding = a_input.tensor(description).value();
  auto b_binding = b_input.tensor(description).value();
  ASSERT_TRUE(execute(*first, x, a_binding).ok());
  auto old = first->result(negative).value();
  auto old_computed = first->result(computed_view).value();
  auto old_bound = first->result(bound_view).value();
  expect(context, old.tensor().value(), {-a[0], -a[1], -a[2], -a[3]});
  expect(context, old_computed.tensor().value(), a);
  expect(context, old_bound.tensor().value(), a);
  EXPECT_FALSE(first->bind(x, b_binding).ok());
  EXPECT_FALSE(first->submit());
  ASSERT_TRUE(second->prepare_resources().ok());
  ASSERT_TRUE(second->bind(x, description, b_input).ok());
  ASSERT_TRUE(second->prepare_replay({}).ok());
  {
    auto accepted = second->submit();
    ASSERT_TRUE(accepted) << accepted.error().message();
    ASSERT_TRUE(accepted->wait().ok());
  }
  expect(context, second->output(computed_view).value(), b);
  expect(context, second->output(bound_view).value(), b);
  expect(context, old.tensor().value(), {-a[0], -a[1], -a[2], -a[3]});
  old.reset(); old_computed.reset(); old_bound.reset();
  ASSERT_TRUE(execute(*first, x, b_binding).ok());
  auto retained = first->result(negative).value();
  auto leased = retained.storage_lease();
  ASSERT_TRUE(leased) << leased.error().message();
  expect(context, leased->tensor(description).value(), {-b[0], -b[1], -b[2], -b[3]});
  retained.reset();
  EXPECT_FALSE(first->prepare_replay({}).ok()); // Escaped native lease is not an internal baseline ref.
  leased->reset();
  ASSERT_TRUE(first->prepare_replay({}).ok());
  {
    auto run = first->submit(); ASSERT_TRUE(run) << run.error().message();
    ASSERT_TRUE(run->wait().ok());
  }
  auto final = first->result(negative).value();
  std::weak_ptr<vn::Buffer> weak;
  {
    auto tensor = vn::MemoryAccess::tensor_view(vn::Access::runtime(context), final.tensor().value(),
                                               hyle::mem::system_allocator()).value();
    weak = vn::Access::buffer(tensor);
  }
  *first = {}; *second = {}; compiled->executable = {}; graph = {};
  ASSERT_TRUE(context.close().ok());
  expect(context, final.tensor().value(), {-b[0], -b[1], -b[2], -b[3]});
  EXPECT_FALSE(weak.expired());
  final.reset();
  EXPECT_TRUE(weak.expired()); // No Buffer -> ResultStorage/Invocation back-reference.
}

TEST(VulkanNativeRegion, UnifiedCompileCapturesBorrowedNativeConstant) {
  auto context = v::Context::create().value();
  RecordProperty("vulkan_device", context.environment().device_name);
  RecordProperty("vulkan_driver", context.environment().driver_info);
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const auto source = g::vulkan::execution_source(context).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto weight = builder.constant(description).value();
  const auto sum = builder.add(c::Binary{c::BinaryKind::kAdd}, x, weight).value();
  const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, sum).value();
  ASSERT_TRUE(builder.publish(sum).ok());
  ASSERT_TRUE(builder.publish(negative).ok());
  auto graph = std::move(builder).finish().value();
  auto constant = input(context, memory, {0.5f, -0.25f, 1.25f, 2.0f});
  const std::array constants{g::ConstantBinding{weight, constant.tensor(description).value(),
                                             g::ConstantPolicy::kBorrow}};
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {domains, targets}, {.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.borrowed_constant_count(), 1U);
  EXPECT_EQ(compiled->executable.workspace_bytes(), 0U);
  auto slot = compiled->executable.create_invocation();
  ASSERT_TRUE(slot) << slot.error().message();
  for (const auto values : {std::array<float, 4>{0.25f, 1.5f, -2.75f, 0.0f},
                            std::array<float, 4>{-2.0f, 3.25f, 1.5f, -0.5f}}) {
    auto inputs = input(context, memory, values);
    ASSERT_TRUE(execute(*slot, x, inputs.tensor(description).value()).ok());
    const std::array<float, 4> expected{values[0] + 0.5f, values[1] - 0.25f,
                                      values[2] + 1.25f, values[3] + 2.0f};
    expect(context, slot->output(sum).value(), expected);
    expect(context, slot->output(negative).value(), {-expected[0], -expected[1], -expected[2], -expected[3]});
  }
}

TEST(VulkanNativeRegion, UnifiedCompileMaterializesHostConstantsOnce) {
  auto context = v::Context::create().value();
  RecordProperty("vulkan_device", context.environment().device_name);
  RecordProperty("vulkan_driver", context.environment().driver_info);
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{g::vulkan::execution_source(context).value(), g::StorageDomainId{0}}};
  const auto single = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{1}).value();
  const auto odd = c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 1>{3}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto weight = builder.constant(description).value();
  const auto scalar = builder.constant(single).value();
  const auto raw = builder.constant(odd).value();
  const auto sum = builder.add(c::Binary{c::BinaryKind::kAdd}, x, weight).value();
  const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, sum).value();
  for (const auto value : {sum, negative, scalar, raw}) ASSERT_TRUE(builder.publish(value).ok());
  auto graph = std::move(builder).finish().value();
  const std::array placements{
      g::NodePlacement{graph.producer(sum).value(), g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
      g::NodePlacement{graph.producer(negative).value(), g::ExecutionTargetId{0}, g::PlacementStrength::kRequired}};
  TrackedHost host;
  auto weights = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 16, 4).value();
  auto one = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 4, 4).value();
  auto bytes = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 3, 1).value();
  const std::array<float, 4> w{0.5f, -0.25f, 1.25f, 2.0f};
  const std::array<float, 1> s{3.75f};
  const std::array<std::byte, 3> raw_values{std::byte{1}, std::byte{127}, std::byte{255}};
  ASSERT_TRUE(weights.view().write(std::as_bytes(std::span(w))).ok());
  ASSERT_TRUE(one.view().write(std::as_bytes(std::span(s))).ok());
  ASSERT_TRUE(bytes.view().write(raw_values).ok());
  const std::array constants{
      g::ConstantBinding{weight, c::ConstTensorView::create(description, weights.view()).value(), g::ConstantPolicy::kMaterialize},
      g::ConstantBinding{scalar, c::ConstTensorView::create(single, one.view()).value(), g::ConstantPolicy::kMaterialize},
      g::ConstantBinding{raw, c::ConstTensorView::create(odd, bytes.view()).value(), g::ConstantPolicy::kMaterialize}};
  const std::array limits{c::ResourceBudgetLimit{memory.ref(), 22},
                          c::ResourceBudgetLimit{hyle::mem::system_resource(), 16 * 1024 * 1024}};
  auto selected_budget = c::ResourceBudget::create({.limits = limits, .allow_unattributed = true});
  ASSERT_TRUE(selected_budget) << selected_budget.error().message();
  auto budget = std::move(*selected_budget);
  auto rejected = g::compile(graph, {domains, targets},
      {.constants = constants, .resource_budget = budget, .node_placements = placements});
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted()) << rejected.error().message();
  EXPECT_EQ(host.reads, 0U); // The full canonical batch is admitted before accepting any upload.
  auto compiled = g::compile(graph, {domains, targets},
      {.constants = constants, .node_placements = placements});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(host.reads, 3U);
  // Query the exact batch policy without reading or allocating through it.
  auto d16 = hyle::mem::Buffer::create(memory.ref(), 16, 4).value();
  auto d4 = hyle::mem::Buffer::create(memory.ref(), 4, 4).value();
  auto d3 = hyle::mem::Buffer::create(memory.ref(), 3, 1).value();
  const std::array copies{
      g::detail::ConstantCopy{c::TensorView::create(description, d16.view()).value(), constants[0].tensor},
      g::detail::ConstantCopy{c::TensorView::create(single, d4.view()).value(), constants[1].tensor},
      g::detail::ConstantCopy{c::TensorView::create(odd, d3.view()).value(), constants[2].tensor}};
  EXPECT_EQ(g::detail::ExecutionSourceAccess::constant_copy_scratch_bytes(targets[0].source, copies).value(), 16U);
  EXPECT_EQ(host.reads, 3U);
  EXPECT_EQ(compiled->report.materialized_constant_bytes(), 23U);
  EXPECT_EQ(compiled->report.materialized_constant_count(), 3U);
  ASSERT_TRUE(weights.reset().ok()); ASSERT_TRUE(one.reset().ok()); ASSERT_TRUE(bytes.reset().ok());
  auto slot = compiled->executable.create_invocation().value();
  for (const auto values : {std::array<float, 4>{0.25f, 1.5f, -2.75f, 0.0f},
                            std::array<float, 4>{-2.0f, 3.25f, 1.5f, -0.5f}}) {
    auto inputs = input(context, memory, values);
    ASSERT_TRUE(execute(slot, x, inputs.tensor(description).value()).ok());
    const std::array<float, 4> expected{values[0] + w[0], values[1] + w[1], values[2] + w[2], values[3] + w[3]};
    expect(context, slot.output(sum).value(), expected);
    expect(context, slot.output(negative).value(), {-expected[0], -expected[1], -expected[2], -expected[3]});
    for (const auto value : {scalar, raw}) {
      auto native = vn::MemoryAccess::tensor_view(vn::Access::runtime(context), slot.output(value).value(),
          hyle::mem::system_allocator()).value();
      if (value == scalar) {
        std::array<float, 1> actual{};
        ASSERT_TRUE(context.download(native, std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, s);
      } else {
        std::array<std::byte, 3> actual{};
        ASSERT_TRUE(context.download(native, actual).ok()); EXPECT_EQ(actual, raw_values);
      }
    }
  }
  EXPECT_EQ(host.reads, 3U); // Replay never reads the released source or invokes cold copy again.
}

TEST(VulkanNativeRegion, ConstantCopyPreflightsWholeBatchAndDoesNotRetryReadFailure) {
  auto context = v::Context::create().value();
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto foreign_context = v::Context::create().value();
  auto foreign = v::MemoryResource::create(foreign_context, v::MemoryKind::kDevice).value();
  auto destination = hyle::mem::Buffer::create(memory.ref(), 16, 4).value();
  auto other = hyle::mem::Buffer::create(foreign.ref(), 16, 4).value();
  TrackedHost host;
  auto source = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 16, 4).value();
  const auto input_view = c::ConstTensorView::create(description, source.view()).value();
  const auto target_view = c::TensorView::create(description, destination.view()).value();
  const auto wrong_view = c::TensorView::create(description, other.view()).value();
  const auto native = vn::MemoryAccess::tensor_view(vn::Access::runtime(context), target_view.as_const(),
      hyle::mem::system_allocator()).value();
  const std::array<float, 4> sentinel{7, 8, 9, 10};
  ASSERT_TRUE(context.upload(native, std::as_bytes(std::span(sentinel))).ok());
  const auto &runtime = vn::Access::runtime(context);
  ASSERT_TRUE(vn::MemoryAccess::can_copy_constant(runtime, {target_view, input_view}).value());
  EXPECT_FALSE(vn::MemoryAccess::can_copy_constant(runtime, {wrong_view, input_view}).value());
  EXPECT_EQ(host.reads, 0U);
  const std::array invalid_batch{vn::ConstantCopy{target_view, input_view},
                                vn::ConstantCopy{wrong_view, input_view}};
  EXPECT_TRUE(vn::MemoryAccess::copy_constants(runtime, invalid_batch,
      hyle::mem::system_allocator()).IsUnimplemented());
  EXPECT_EQ(host.reads, 0U);
  expect(context, target_view.as_const(), sentinel);
  auto builder = g::GraphBuilder::create().value();
  const auto constant = builder.constant(description).value();
  const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, constant).value();
  ASSERT_TRUE(builder.publish(negative).ok());
  auto graph = std::move(builder).finish().value();
  const std::array constants{g::ConstantBinding{constant, input_view, g::ConstantPolicy::kMaterialize}};
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{g::vulkan::execution_source(context).value(), g::StorageDomainId{0}}};
  host.fail = true;
  auto failed = g::compile(graph, {domains, targets}, {.constants = constants});
  ASSERT_FALSE(failed); EXPECT_TRUE(failed.error().IsUnavailable()) << failed.error().message();
  EXPECT_EQ(host.reads, 1U); // Actual copy failure is fatal, never retried on another route.
  EXPECT_EQ(runtime->pending, nullptr);
  host.fail = false;
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(sentinel))).ok());
  OneFenceFailure observation(context);
  const std::array accepted_batch{vn::ConstantCopy{target_view, input_view}};
  const auto terminal = vn::MemoryAccess::copy_constants(runtime, accepted_batch,
      hyle::mem::system_allocator());
  EXPECT_TRUE(terminal.IsResourceExhausted()) << terminal.message();
  EXPECT_EQ(observation.calls, 1U); // Original queue drain retires the real accepted prefix.
  EXPECT_EQ(host.reads, 2U); // No copy retry after the real native error.
  EXPECT_EQ(runtime->pending, nullptr);
}
TEST(VulkanNativeRegion, ColdConstantsCopyBothDirectionsAndAlignedDeviceSpans) {
  auto context = v::Context::create().value();
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const auto &runtime = vn::Access::runtime(context);
  const std::array<float, 4> a{0.25f, -1.5f, 3.75f, -0.0f}, b{-2.25f, 0.5f, -7.0f, 1.25f};
  auto ready = input(context, memory, a);
  auto source = ready.tensor(description).value();
  auto uploaded = hyle::mem::Buffer::create(memory.ref(), 16, 4).value();
  auto copied = hyle::mem::Buffer::create(memory.ref(), 16, 4).value();
  TrackedHost host;
  auto incoming = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 16, 4).value();
  auto outgoing = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 16, 4).value();
  ASSERT_TRUE(incoming.view().write(std::as_bytes(std::span(b))).ok());
  host.writes = 0;
  const std::array<vn::ConstantCopy, 3> copies{{
      {c::TensorView::create(description, uploaded.view()).value(),
       c::ConstTensorView::create(description, incoming.view()).value()},
      {c::TensorView::create(description, copied.view()).value(), source},
      {c::TensorView::create(description, outgoing.view()).value(), source}}};
  std::array<g::detail::ConstantCopy, 3> queries;
  const auto execution_source = g::vulkan::execution_source(context).value();
  for (std::size_t i = 0; i < copies.size(); ++i) {
    ASSERT_TRUE(vn::MemoryAccess::can_copy_constant(runtime, copies[i]).value());
    queries[i] = {copies[i].destination, copies[i].source};
  }
  EXPECT_EQ(g::detail::ExecutionSourceAccess::constant_copy_scratch_bytes(execution_source, queries).value(), 16U);
  EXPECT_EQ(g::detail::ExecutionSourceAccess::constant_copy_scratch_bytes(
      execution_source, std::span(queries).subspan(1, 1)).value(), 0U);
  EXPECT_EQ(host.reads, 0U); EXPECT_EQ(host.writes, 0U);
  ASSERT_TRUE(vn::MemoryAccess::copy_constants(runtime, copies, hyle::mem::system_allocator()).ok());
  EXPECT_EQ(host.reads, 1U); EXPECT_EQ(host.writes, 1U);
  expect(context, copies[0].destination.as_const(), b);
  expect(context, copies[1].destination.as_const(), a);
  std::array<float, 4> actual{};
  ASSERT_TRUE(outgoing.view().read(std::as_writable_bytes(std::span(actual))).ok());
  for (std::size_t i = 0; i < a.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(a[i]));

  const auto odd = c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 1>{3}).value();
  const auto clipped = c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 1>{2}).value();
  auto device = hyle::mem::Buffer::create(memory.ref(), 3, 1).value();
  auto bytes = hyle::mem::Buffer::create(hyle::mem::system_resource(), 3, 1).value();
  auto result = hyle::mem::Buffer::create(hyle::mem::system_resource(), 3, 1).value();
  const std::array<std::byte, 3> payload{std::byte{1}, std::byte{127}, std::byte{255}};
  ASSERT_TRUE(bytes.view().write(payload).ok());
  const auto device_view = c::TensorView::create(odd, device.view()).value();
  const std::array<vn::ConstantCopy, 1> upload{{{device_view, c::ConstTensorView::create(odd, bytes.view()).value()}}};
  ASSERT_TRUE(vn::MemoryAccess::copy_constants(runtime, upload, hyle::mem::system_allocator()).ok());
  const auto clipped_device = c::TensorView::create(clipped, device.view().subview(0, 2).value()).value();
  const auto clipped_host = c::TensorView::create(clipped, bytes.view().subview(0, 2).value()).value();
  // roundup(2)==roundup(3) does not authorize reading or overwriting byte 2.
  EXPECT_FALSE(vn::MemoryAccess::can_copy_constant(runtime, {clipped_device, clipped_host.as_const()}).value());
  EXPECT_FALSE(vn::MemoryAccess::can_copy_constant(runtime, {clipped_host, clipped_device.as_const()}).value());
  const auto short_device = c::TensorView::create(clipped, device.view()).value();
  EXPECT_FALSE(vn::MemoryAccess::can_copy_constant(runtime, {short_device, clipped_host.as_const()}).value());
  EXPECT_FALSE(vn::MemoryAccess::can_copy_constant(runtime, {clipped_host, short_device.as_const()}).value());
  const std::array<vn::ConstantCopy, 1> clipped_copy{{{clipped_device, clipped_host.as_const()}}};
  EXPECT_TRUE(vn::MemoryAccess::copy_constants(runtime, clipped_copy,
      hyle::mem::system_allocator()).IsUnimplemented());
  const std::array<vn::ConstantCopy, 1> download{{{
      c::TensorView::create(odd, result.view()).value(), device_view.as_const()}}};
  ASSERT_TRUE(vn::MemoryAccess::copy_constants(runtime, download, hyle::mem::system_allocator()).ok());
  std::array<std::byte, 3> actual_bytes{};
  ASSERT_TRUE(result.view().read(actual_bytes).ok()); EXPECT_EQ(actual_bytes, payload);
  ASSERT_TRUE(bytes.view().read(actual_bytes).ok()); EXPECT_EQ(actual_bytes, payload);
  EXPECT_FALSE(vn::MemoryAccess::can_copy_constant(runtime, {device_view, device_view.as_const()}).value());

  // An odd complete owned carrier retains its private transfer padding. The
  // cold device copy uses one mapped staging span, without widening subviews.
  auto odd_copy = hyle::mem::Buffer::create(memory.ref(), 3, 1).value();
  const auto odd_destination = c::TensorView::create(odd, odd_copy.view()).value();
  const std::array<vn::ConstantCopy, 1> odd_device_copy{{{odd_destination, device_view.as_const()}}};
  ASSERT_TRUE(vn::MemoryAccess::can_copy_constant(runtime, odd_device_copy[0]).value());
  EXPECT_EQ(vn::MemoryAccess::constant_copy_staging_bytes(runtime, odd_device_copy[0]).value(), 4U);
  ASSERT_TRUE(vn::MemoryAccess::copy_constants(runtime, odd_device_copy, hyle::mem::system_allocator()).ok());
  const std::array<vn::ConstantCopy, 1> odd_readback{{{download[0].destination, odd_destination.as_const()}}};
  ASSERT_TRUE(vn::MemoryAccess::copy_constants(runtime, odd_readback, hyle::mem::system_allocator()).ok());
  ASSERT_TRUE(result.view().read(actual_bytes).ok()); EXPECT_EQ(actual_bytes, payload);

  // A larger D2D leg must neither reserve nor allocate its payload as staging
  // for the smaller host leg. Both still execute through their real carriers.
  const std::array<std::byte, 3> changed{std::byte{3}, std::byte{5}, std::byte{7}};
  ASSERT_TRUE(bytes.view().write(changed).ok());
  const std::array<vn::ConstantCopy, 2> mixed{{{copies[0].destination, source}, upload[0]}};
  const std::array<g::detail::ConstantCopy, 2> mixed_queries{{
      {mixed[0].destination, mixed[0].source}, {mixed[1].destination, mixed[1].source}}};
  EXPECT_EQ(g::detail::ExecutionSourceAccess::constant_copy_scratch_bytes(
      execution_source, mixed_queries).value(), 4U);
  EXPECT_EQ(g::detail::ExecutionSourceAccess::constant_copy_scratch_bytes(
      execution_source, std::span(mixed_queries).subspan(1, 1)).value(), 4U);
  ASSERT_TRUE(vn::MemoryAccess::copy_constants(runtime, mixed, hyle::mem::system_allocator()).ok());
  expect(context, mixed[0].destination.as_const(), a); // Was b before this mixed batch.
  ASSERT_TRUE(vn::MemoryAccess::copy_constants(runtime, download, hyle::mem::system_allocator()).ok());
  ASSERT_TRUE(result.view().read(actual_bytes).ok()); EXPECT_EQ(actual_bytes, changed);

  const auto empty = c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 1>{0}).value();
  auto empty_device = hyle::mem::Buffer::create(memory.ref(), 0, 1).value();
  auto empty_host = hyle::mem::Buffer::create(hyle::mem::system_resource(), 0, 1).value();
  const std::array<vn::ConstantCopy, 1> no_work{{{c::TensorView::create(empty, empty_host.view()).value(),
      c::ConstTensorView::create(empty, empty_device.view()).value()}}};
  hyle::mem::QuotaAllocator denied(hyle::mem::system_allocator(), 0);
  ASSERT_TRUE(vn::MemoryAccess::copy_constants(runtime, no_work, denied.get()).ok());
}

TEST(VulkanNativeRegion, ColdReadbackPreflightsAndDoesNotRetryAcceptedFailures) {
  auto context = v::Context::create().value();
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto foreign_context = v::Context::create().value();
  auto foreign_memory = v::MemoryResource::create(foreign_context, v::MemoryKind::kDevice).value();
  auto source = input(context, memory, {1, 2, 3, 4});
  auto foreign_source = input(foreign_context, foreign_memory, {5, 6, 7, 8});
  TrackedHost host;
  auto output = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 16, 4).value();
  const std::array<float, 4> sentinel{7, 8, 9, 10};
  ASSERT_TRUE(output.view().write(std::as_bytes(std::span(sentinel))).ok());
  host.writes = 0;
  const auto destination = c::TensorView::create(description, output.view()).value();
  const std::array<vn::ConstantCopy, 2> invalid{{{destination, source.tensor(description).value()},
      {destination, foreign_source.tensor(description).value()}}};
  const auto &runtime = vn::Access::runtime(context);
  {
    OneFenceFailure no_acceptance(context);
    EXPECT_TRUE(vn::MemoryAccess::copy_constants(runtime, invalid,
        hyle::mem::system_allocator()).IsUnimplemented());
    EXPECT_EQ(no_acceptance.calls, 0U); EXPECT_EQ(host.writes, 0U);
  }
  const std::array<vn::ConstantCopy, 1> valid{{invalid[0]}};
  host.fail_write = true;
  auto failed = vn::MemoryAccess::copy_constants(runtime, valid, hyle::mem::system_allocator());
  EXPECT_TRUE(failed.IsUnimplemented()) << failed.message();
  EXPECT_EQ(failed.message(), "injected accepted host constant write failure");
  EXPECT_EQ(host.writes, 1U); EXPECT_EQ(runtime->pending, nullptr);
  host.fail_write = false;
  {
    OneFenceFailure terminal(context);
    failed = vn::MemoryAccess::copy_constants(runtime, valid, hyle::mem::system_allocator());
    EXPECT_TRUE(failed.IsResourceExhausted()) << failed.message();
    EXPECT_EQ(terminal.calls, 1U); EXPECT_EQ(host.writes, 1U);
    EXPECT_EQ(runtime->pending, nullptr);
  }
  std::array<float, 4> actual{};
  ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, sentinel);
}

#if defined(HYLE_TEST_VULKAN_CPU_CONSTANT)
TEST(VulkanNativeRegion, DeviceConstantMaterializesForCpuAndNativeMath) {
  auto context = v::Context::create().value();
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto cpu = g::cpu::create_context().value();
  auto queue = cpu.create_queue().value();
  TrackedHost host;
  const std::array domains{g::StorageDomain{hyle::mem::MemoryResourceRef(host)},
                          g::StorageDomain{memory.ref()}};
  const std::array targets{
      g::ExecutionTarget{g::ExecutionSource::acquire(cpu).value(), g::StorageDomainId{0}},
      g::ExecutionTarget{g::vulkan::execution_source(context).value(), g::StorageDomainId{1}}};
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto weight = builder.constant(description).value();
  const auto sum = builder.add(c::Binary{c::BinaryKind::kAdd}, x, weight).value();
  const auto cpu_negative = builder.add(c::Unary{c::UnaryKind::kNeg}, weight).value();
  const auto native_negative = builder.add(c::Unary{c::UnaryKind::kNeg}, weight).value();
  for (const auto value : {sum, cpu_negative, native_negative}) ASSERT_TRUE(builder.publish(value).ok());
  auto graph = std::move(builder).finish().value();
  const std::array placements{
      g::NodePlacement{graph.producer(sum).value(), g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
      g::NodePlacement{graph.producer(cpu_negative).value(), g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
      g::NodePlacement{graph.producer(native_negative).value(), g::ExecutionTargetId{1}, g::PlacementStrength::kRequired}};
  const std::array<float, 4> w{0.5f, -0.25f, 1.25f, 2.0f};
  auto ready = input(context, memory, w);
  const std::array constants{g::ConstantBinding{weight, ready.tensor(description).value(), g::ConstantPolicy::kMaterialize}};
  host.fail_write = true;
  auto rejected = g::compile(graph, {domains, targets}, {.constants = constants, .node_placements = placements});
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented()) << rejected.error().message();
  EXPECT_EQ(rejected.error().message(), "injected accepted host constant write failure");
  EXPECT_EQ(host.writes, 1U); // A copy was accepted; unsupported host write cannot select another route.
  EXPECT_EQ(vn::Access::runtime(context)->pending, nullptr);
  host.fail_write = false;
  auto compiled = g::compile(graph, {domains, targets}, {.constants = constants, .node_placements = placements});
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto cold_writes = host.writes;
  EXPECT_EQ(cold_writes, 2U);
  ready.reset(); // Compiled constants own both completed physical copies.
  std::vector<g::QueueBinding> bindings;
  for (std::size_t i = 0; i < compiled->report.lane_count(); ++i) {
    auto lane = compiled->report.lane(i); ASSERT_TRUE(lane) << lane.error().message();
    if (!lane->queue_required) continue;
    ASSERT_EQ(lane->runtime_mode, c::ProviderRuntimeMode::kSharedQueue);
    ASSERT_EQ(lane->target, g::ExecutionTargetId{0});
    bindings.push_back({lane->lane, &queue});
  }
  auto activation = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 16, 64).value();
  const auto writes_before_replay = host.writes;
  std::array<std::array<hyle::mem::Buffer, 2>, 2> cpu_outputs;
  std::array<g::Invocation, 2> slots;
  g::GraphOutput old;
  for (std::size_t phase = 0; phase < slots.size(); ++phase) {
    const std::array<float, 4> values{float(phase) + 0.25f, -1.5f, 3.75f, -0.5f};
    ASSERT_TRUE(activation.view().write(std::as_bytes(std::span(values))).ok());
    slots[phase] = compiled->executable.create_invocation().value();
    ASSERT_TRUE(slots[phase].bind(x, c::ConstTensorView::create(description, activation.view()).value()).ok());
    const std::array cpu_values{sum, cpu_negative};
    for (std::size_t i = 0; i < cpu_values.size(); ++i) {
      cpu_outputs[phase][i] = hyle::mem::Buffer::create(hyle::mem::MemoryResourceRef(host), 16, 64).value();
      ASSERT_TRUE(slots[phase].bind(cpu_values[i],
          c::TensorView::create(description, cpu_outputs[phase][i].view()).value()).ok());
    }
    auto prepared = slots[phase].prepare_replay(bindings); ASSERT_TRUE(prepared.ok()) << prepared.message();
    {
      auto run = slots[phase].submit(bindings); ASSERT_TRUE(run) << run.error().message();
      const auto status = run->wait(); ASSERT_TRUE(status.ok()) << status.message();
    }
    for (const auto value : {sum, cpu_negative}) {
      auto result = slots[phase].result(value); ASSERT_TRUE(result) << result.error().message();
      auto tensor = result->tensor(); ASSERT_TRUE(tensor) << tensor.error().message();
      std::array<float, 4> actual{};
      ASSERT_TRUE(tensor->storage().read(std::as_writable_bytes(std::span(actual))).ok());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(actual[i], value == sum ? values[i] + w[i] : -w[i]);
      if (phase == 0 && value == sum) old = *result;
    }
    expect(context, slots[phase].output(native_negative).value(), {-w[0], -w[1], -w[2], -w[3]});
  }
  std::array<float, 4> retained{};
  ASSERT_TRUE(old.tensor()->storage().read(std::as_writable_bytes(std::span(retained))).ok());
  EXPECT_EQ(retained, (std::array<float, 4>{0.75f, -1.75f, 5.0f, 1.5f}));
  EXPECT_EQ(host.writes, writes_before_replay + 2U); // Only the two explicit activation writes.
}
#endif
} // namespace
