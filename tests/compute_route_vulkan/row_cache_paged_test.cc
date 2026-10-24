#include "../compute_models/row_cache_native_fixture.h"
#if defined(HYLE_ROW_CACHE_PAGED_APPLE)
#include "hyle/compute/apple_row_cache.h"
namespace native_row=hyle::compute::apple;
#else
#include "hyle/compute/vulkan_row_cache.h"
namespace native_row=hyle::compute::vulkan;
#endif
namespace {
namespace c=hyle::compute;namespace n=native_row;namespace f=row_cache_native_test;
struct Fixture final {
  n::Context context=take_context();
#if defined(HYLE_ROW_CACHE_PAGED_APPLE)
  n::MemoryResource device=n::MemoryResource::create(context).value(),staging=n::MemoryResource::create(context).value();
#else
  n::MemoryResource device=n::MemoryResource::create(context,n::MemoryKind::kDevice).value(),staging=n::MemoryResource::create(context,n::MemoryKind::kStaging).value();
#endif
  c::LeasePool host=c::LeasePool::create(staging.ref(),65536).value(),pool=c::LeasePool::create(device.ref(),65536).value();
  n::CopyBatchProgram copy_program=n::CopyBatchProgram::prepare(context,std::array{c::CopyResourceSpec{staging.ref(),65536},c::CopyResourceSpec{device.ref(),65536}}).value();
  n::CopyBatchInvocation copy=n::CopyBatchInvocation::create(copy_program).value();
  static n::Context take_context(){return f::take(n::Context::create(),"native context");}
  ~Fixture(){context.close().IgnoreError();}
  c::WriteLease write(c::WriteLease destination,std::span<const std::byte>bytes){
    auto hw=host.allocate(bytes.size()).value();EXPECT_TRUE(hw.view()->write(bytes).ok());auto source=hw.publish().value();
    EXPECT_TRUE(copy.reset().ok());EXPECT_TRUE(copy.bind_source({0},source).ok());EXPECT_TRUE(copy.bind_mutable({1},std::move(destination)).ok());
    const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,bytes.size()}}};EXPECT_TRUE(copy.prepare_replay(phases,ranges).ok());
    {auto done=f::take(copy.submit(),"native upload");EXPECT_TRUE(done.wait().ok());}
    auto result=f::take(copy.unbind_mutable({1}),"native original writer");EXPECT_TRUE(copy.reset().ok());return result;
  }
  c::ReadLease load(std::span<const std::byte>bytes){const std::array<std::byte,4> zero{};if(bytes.empty())bytes=zero;
    return write(pool.allocate(bytes.size()).value(),bytes).publish().value();}
  std::vector<std::uint16_t> read(const c::ResultLease<n::Results>&result,c::TensorDesc description){
    std::vector<std::uint16_t> values(description.element_count().value());if(values.empty())return values;
#if defined(HYLE_ROW_CACHE_PAGED_APPLE)
    auto lease=device.import_readonly(result->arrays()[0],description).value();EXPECT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(values))).ok());
#else
    EXPECT_TRUE(context.download(result->tensors()[0],std::as_writable_bytes(std::span(values))).ok());
#endif
    return values;
  }
};
void exercise(){Fixture route;
  for(auto encoding:{c::RowCacheEncoding::float8_block64_tail64(),c::RowCacheEncoding::float4_block32()}){
    SCOPED_TRACE(encoding.depth());auto schema=c::RowCacheBatchSchema::create(f::capacity(encoding)).value();
    auto arena=c::PagedArena::create(schema,route.device.ref(),{.max_backing_bytes=10*encoding.row_bytes()+1}).value();const auto data=f::data(encoding);
    std::array<c::ReadLease,5> ready;
    // Page0 is deliberately uninitialized/unowned. Invalid logical IDs may
    // not read it even as the native checked decoder's speculative fallback.
    for(unsigned page:{1u,3u,4u}){auto writer=arena.acquire(page).value();writer.lease()=route.write(std::move(writer.lease()),std::as_bytes(std::span(data.bytes).subspan(page*2*encoding.row_bytes(),2*encoding.row_bytes())));ready[page]=writer.publish(2).value();}
    const auto desc=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{4,3}).value();
    auto slot=f::take(n::PagedRowCacheGatherInvocation::create(route.context,route.device,route.staging,schema,desc),"native row reader");
    for(int phase:{0,1}){auto version=f::layout(schema,phase);const auto selected=f::ids(phase);auto ids=route.load(std::as_bytes(std::span(selected)));
      ASSERT_TRUE(slot.prepare_layout(version).ok());ASSERT_TRUE(slot.prepare_layout(version).ok());
      auto frame=arena.read(version,std::array{ready[3],ready[4],ready[1]}).value();
      EXPECT_TRUE(slot.submit(arena.read(f::layout(schema,phase),std::array{ready[1],ready[3],ready[4]}).value(),ids).error().IsInvalidArgument());
      auto done=f::take(slot.submit(frame,ids),"native row submit");auto result=f::take(done.wait(),"native row result");EXPECT_EQ(route.read(result,slot.output_description()),f::expected(encoding,data,selected,phase));
      const auto retained=slot.prepare_layout(version);EXPECT_FALSE(retained.ok())<<"retained native result allowed metadata prepare";
    }
    for(std::int32_t invalid:{-1,3,8}){auto failed=f::take(n::PagedRowCacheGatherInvocation::create(route.context,route.device,route.staging,schema,desc),"native invalid slot");
      auto version=f::layout(schema,0);ASSERT_TRUE(failed.prepare_layout(version).ok());auto selected=f::ids(0);selected[5]=invalid;auto ids=route.load(std::as_bytes(std::span(selected)));
      auto done=f::take(failed.submit(arena.read(version,std::array{ready[1],ready[3],ready[4]}).value(),ids),"native logical invalid submit");
      auto result=done.wait();ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsOutOfRange());
    }
    const std::array<std::int32_t,4> q{0,0,2,4},kv{};const std::array<std::int32_t,3> tail{};auto empty=schema.version({q,kv,{},tail}).value();
    const auto empty_desc=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{4,0}).value();
    auto no=f::take(n::PagedRowCacheGatherInvocation::create(route.context,route.device,route.staging,schema,empty_desc),"native empty row reader");
    ASSERT_TRUE(no.prepare_layout(empty).ok());auto done=f::take(no.submit(arena.read(empty,{}).value(),route.load({})),"native empty submit");auto result=f::take(done.wait(),"native empty result");EXPECT_TRUE(route.read(result,no.output_description()).empty());
    if(encoding.depth()==512){
      auto life_arena=c::PagedArena::create(schema,route.device.ref(),{.max_backing_bytes=10*encoding.row_bytes()+1}).value();std::array<c::ReadLease,3> life_pages;
      unsigned i=0;for(unsigned page:{1u,3u,4u}){auto writer=life_arena.acquire(page).value();writer.lease()=route.write(std::move(writer.lease()),std::as_bytes(std::span(data.bytes).subspan(page*2*encoding.row_bytes(),2*encoding.row_bytes())));life_pages[i++]=writer.publish(2).value();}
      auto version=f::layout(schema,0);auto frame=life_arena.read(version,life_pages).value();auto selected=f::ids(0);auto ids=route.load(std::as_bytes(std::span(selected)));
      auto owner=f::take(n::PagedRowCacheGatherInvocation::create(route.context,route.device,route.staging,schema,desc),"native lifetime slot");ASSERT_TRUE(owner.prepare_layout(version).ok());
      auto submitted=f::take(owner.submit(frame,ids),"native lifetime submit");owner={};frame={};life_pages={};ids.reset();life_arena={};
      auto retained=f::take(submitted.wait(),"native result after caller owners released");
      const auto output=c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{4,3,512}).value();EXPECT_EQ(route.read(retained,output),f::expected(encoding,data,selected,0));
    }
  }
}
}
#if defined(HYLE_ROW_CACHE_PAGED_APPLE)
TEST(ApplePagedRowCache, SelectedReadyPagesKeepLogicalBoundsAndNativeOwners){exercise();}
#else
TEST(VulkanPagedRowCache, SelectedReadyPagesKeepLogicalBoundsAndNativeOwners){exercise();}
#endif
