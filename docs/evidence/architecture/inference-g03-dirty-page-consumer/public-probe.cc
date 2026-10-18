#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/batch_transfer.h>
#include <hyle/compute/paged_arena.h>
#include <array>
#include <cstdio>

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
using hyle::Status;
Status require(bool value, const char* message) {
  return value ? Status::Ok() : Status::Internal(message);
}
Status run() {
  const c::BatchCapacity cap{.max_requests=2,.max_query_rows=2,.max_kv_rows=128,
      .max_pages=8,.max_page_refs=8,.query_heads=8,.kv_heads=2,.head_dim=256};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto arena, c::PagedArena::create(schema, m::system_resource(),
      {.max_backing_bytes=256*1024}));
  const auto page_bytes=arena.page_bytes();
  constexpr std::size_t row_bytes=2*256*2;
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(c::cpu::device()));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(m::system_resource(), page_bytes+512));
  HYLE_ASSIGN_OR_RETURN(auto incoming, pool.allocate(page_bytes));
  HYLE_RETURN_IF_ERROR(incoming.view()->fill(std::byte{0x23}));
  HYLE_ASSIGN_OR_RETURN(auto source, incoming.publish());
  const std::array specs{c::CopyResourceSpec{m::system_resource(),page_bytes},
      c::CopyResourceSpec{m::system_resource(),page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto program, c::CopyBatchProgram::prepare(context, specs));
  HYLE_ASSIGN_OR_RETURN(auto copy, c::CopyBatchInvocation::create(program, queue));
  const std::array<c::CopyPhase,1> phases{{{}}};
  for(unsigned partial=0;partial<2;++partial) {
    HYLE_ASSIGN_OR_RETURN(auto writer, arena.acquire(partial));
    HYLE_RETURN_IF_ERROR(writer.lease().view()->fill(std::byte{0x17}));
    HYLE_RETURN_IF_ERROR(copy.bind_source({0},source));
    HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(writer.lease())));
    const auto offset=partial?3*row_bytes:0;
    const auto bytes=partial?row_bytes:page_bytes;
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,offset,bytes}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    {
      HYLE_ASSIGN_OR_RETURN(auto submitted, copy.submit());
      HYLE_RETURN_IF_ERROR(submitted.wait());
      HYLE_RETURN_IF_ERROR(require(submitted.report().logical_bytes==bytes,"dirty byte report differs"));
      auto full=submitted.output({0},{1},0,page_bytes);
      if(!partial) {
        if(!full)return full.error();
        HYLE_RETURN_IF_ERROR(writer.complete(*full));
        std::fprintf(stderr,"BASELINE full page %zu bytes -> PagedPageWriter.complete PASS\n",bytes);
      } else {
        HYLE_RETURN_IF_ERROR(require(!full,"partial update incorrectly published untouched bytes"));
        HYLE_ASSIGN_OR_RETURN(auto dirty,submitted.output({0},{1},offset,bytes));
        const auto completed=writer.complete(dirty);
        HYLE_RETURN_IF_ERROR(require(!completed.ok(),"partial lease incorrectly accepted as whole page"));
        const auto denied=arena.acquire(partial);
        HYLE_RETURN_IF_ERROR(require(!denied,"consumed original writer was speculatively recovered"));
        std::fprintf(stderr,"GAP exact dirty row %zu/%zu bytes: full output refused: %.*s; page completion refused: %.*s\n",
            bytes,page_bytes,int(full.error().message().size()),full.error().message().data(),
            int(completed.message().size()),completed.message().data());
      }
    }
    HYLE_RETURN_IF_ERROR(copy.reset());
  }
  HYLE_RETURN_IF_ERROR(copy.close());
  HYLE_RETURN_IF_ERROR(queue.drain());
  return Status::Ok();
}
}
int main() {
  const auto status=run();
  if(!status.ok())std::fprintf(stderr,"probe failed: %.*s\n",int(status.message().size()),status.message().data());
  return status.ok()?0:1;
}
