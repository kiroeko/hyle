#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/batch_transfer.h>
#include <hyle/compute/paged_arena.h>
#include <hyle/mem/quota_allocator.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c=hyle::compute;
namespace m=hyle::mem;
using hyle::Status;
Status require(bool value,const char* message){return value?Status::Ok():Status::Internal(message);}
double decode(std::uint16_t x){
  const unsigned e=(x>>10)&31U,f=x&1023U;
  return (x&0x8000U?-1.:1.)*(e?std::ldexp(1.+double(f)/1024.,int(e)-15):std::ldexp(double(f),-24));
}
Status run(){
  const c::BatchCapacity cap{.max_requests=2,.max_query_rows=2,.max_kv_rows=128,
      .max_pages=8,.max_page_refs=8,.query_heads=8,.kv_heads=2,.head_dim=256};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  m::QuotaAllocator quota(m::system_allocator());
  HYLE_ASSIGN_OR_RETURN(auto arena,c::PagedArena::create(schema,m::system_resource(),
      {.max_backing_bytes=256*1024,.allocator=quota.get()}));
  constexpr std::size_t elements=16*2*256,row_elements=2*256,row_bytes=row_elements*2;
  std::vector<std::uint16_t> expected(elements,0x3000),actual(elements);
  HYLE_ASSIGN_OR_RETURN(auto first,arena.acquire(0));
  HYLE_RETURN_IF_ERROR(first.lease().view()->write(std::as_bytes(std::span(expected))));
  HYLE_ASSIGN_OR_RETURN(auto ready,first.publish());
  HYLE_ASSIGN_OR_RETURN(auto clean,arena.acquire(1));
  HYLE_RETURN_IF_ERROR(clean.lease().view()->fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto shared,clean.publish());
  const std::array<std::int32_t,3> qp{0,1,2},kp{0,1,2};
  const std::array<std::int32_t,2> ids{0,1},tails{16,16};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto frame,arena.read(layout,std::array{ready,shared}));
  const auto generation=ready.generation();
  const auto origin=ready.view()->offset();
  HYLE_RETURN_IF_ERROR(require(!arena.acquire(ready),"shared frame permitted in-place mutation"));
  HYLE_RETURN_IF_ERROR(require(ready.valid() && ready.generation()==generation,"failed acquire consumed ready page"));
  frame={};
  HYLE_ASSIGN_OR_RETURN(auto uses,c::LeaseUsePool::create());
  HYLE_ASSIGN_OR_RETURN(auto use,uses.begin());
  HYLE_RETURN_IF_ERROR(use.read(ready));
  HYLE_RETURN_IF_ERROR(require(!arena.acquire(ready),"admitted native use permitted page mutation"));
  use={};
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cpu::device()));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(m::system_resource(),row_bytes+512));
  HYLE_ASSIGN_OR_RETURN(auto incoming,pool.allocate(row_bytes));
  HYLE_RETURN_IF_ERROR(incoming.view()->fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto source,incoming.publish());
  const std::array specs{c::CopyResourceSpec{m::system_resource(),row_bytes},
      c::CopyResourceSpec{m::system_resource(),arena.page_bytes()}};
  HYLE_ASSIGN_OR_RETURN(auto program,c::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(program,queue));
  const std::array<c::CopyPhase,1> phases{{{}}};
  for(unsigned episode=0;episode<64;++episode){
    HYLE_ASSIGN_OR_RETURN(incoming,source.try_write());
    std::array<std::uint16_t,row_elements> row{};
    row.fill(std::uint16_t(0x3000+16*episode));
    HYLE_RETURN_IF_ERROR(incoming.view()->write(std::as_bytes(std::span(row))));
    HYLE_ASSIGN_OR_RETURN(source,incoming.publish());
    auto reservation=quota.try_reserve_bytes(quota.remaining());
    HYLE_RETURN_IF_ERROR(require(reservation.has_value(),"quota reservation failed"));
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(ready));
    HYLE_RETURN_IF_ERROR(require(!ready.valid(),"successful acquire did not consume ready page"));
    HYLE_RETURN_IF_ERROR(require(writer.lease().generation()==generation,"page generation changed during in-place update"));
    HYLE_RETURN_IF_ERROR(copy.bind_source({0},source));
    HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(writer.lease())));
    const auto offset=(episode%16)*row_bytes;
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,offset,row_bytes}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    {
      HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());
      HYLE_RETURN_IF_ERROR(require(!copy.unbind_mutable({1}),"retained submission permitted mutable withdrawal"));
      HYLE_RETURN_IF_ERROR(submitted.wait());
      HYLE_RETURN_IF_ERROR(require(submitted.report().logical_bytes==row_bytes,"update copied entire page/history"));
      HYLE_RETURN_IF_ERROR(require(!copy.unbind_mutable({1}),"observed retained submission permitted withdrawal"));
    }
    HYLE_ASSIGN_OR_RETURN(writer.lease(),copy.unbind_mutable({1}));
    HYLE_RETURN_IF_ERROR(require(!copy.unbind_mutable({1}),"mutable binding withdrawn twice"));
    HYLE_ASSIGN_OR_RETURN(ready,writer.publish());
    HYLE_RETURN_IF_ERROR(require(ready.generation()==generation && ready.view()->offset()==origin,"update changed original page identity/range"));
    HYLE_RETURN_IF_ERROR(copy.reset());
    std::copy(row.begin(),row.end(),expected.begin()+offset/2);
    HYLE_RETURN_IF_ERROR(ready.view()->read(std::as_writable_bytes(std::span(actual))));
    HYLE_RETURN_IF_ERROR(require(actual==expected,"dirty row or preserved history differs"));
    double actual_sum=0,expected_sum=0;
    for(auto x:actual)actual_sum+=decode(x);
    for(auto x:expected)expected_sum+=decode(x);
    HYLE_RETURN_IF_ERROR(require(std::isfinite(actual_sum) && actual_sum==expected_sum,"readonly numerical reduction differs"));
  }
  HYLE_ASSIGN_OR_RETURN(frame,arena.read(layout,std::array{ready,shared}));
  HYLE_RETURN_IF_ERROR(require(frame.page_count()==2,"updated page no longer participates in exact frame"));
  HYLE_RETURN_IF_ERROR(copy.close());
  HYLE_RETURN_IF_ERROR(queue.drain());
  std::fprintf(stderr,"public dirty page PASS: 64 episodes, 1024 dirty/16384 page bytes, same generation, unchanged rows, shared/native uses and retained submissions guarded; zero new arena metadata allocations\n");
  return Status::Ok();
}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"dirty page: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
