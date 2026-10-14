#pragma once
#include <hyle/mem/allocator.h>
#include <atomic>
#include <cstddef>
#include <limits>

namespace hyle_f5_fault_fixture {
struct Allocator final {
  hyle::mem::AllocatorRef backing=hyle::mem::system_allocator();
  std::size_t attempts=0,fail=(std::numeric_limits<std::size_t>::max)(),live=0;
  hyle::mem::AllocatorRef ref()noexcept{
    return {[](void*p,std::size_t bytes,std::size_t alignment)noexcept->void*{
      auto&s=*static_cast<Allocator*>(p);if(s.attempts++==s.fail)return nullptr;
      auto*result=s.backing.allocate(bytes,alignment);s.live+=result!=nullptr;return result;
    },[](void*p,void*data,std::size_t bytes,std::size_t alignment)noexcept{
      auto&s=*static_cast<Allocator*>(p);s.live-=data!=nullptr;s.backing.deallocate(data,bytes,alignment);
    },[](void*p,void*data,std::size_t old_bytes,std::size_t bytes,std::size_t alignment)noexcept->void*{
      auto&s=*static_cast<Allocator*>(p);if(bytes && s.attempts++==s.fail)return nullptr;
      auto*result=s.backing.reallocate(data,old_bytes,bytes,alignment);
      if(!bytes)s.live-=data!=nullptr;else if(result && !data)++s.live;return result;
    },this};
  }
};
// Detached Graph work may allocate while the caller releases its slot. Fault
// selection remains a cold, serialized operation; counters support both owners.
struct ConcurrentAllocator final {
  hyle::mem::AllocatorRef backing=hyle::mem::system_allocator();
  std::atomic<std::size_t> attempts{0},fail{(std::numeric_limits<std::size_t>::max)()},live{0};
  hyle::mem::AllocatorRef ref()noexcept{
    return {[](void*p,std::size_t bytes,std::size_t alignment)noexcept->void*{
      auto&s=*static_cast<ConcurrentAllocator*>(p);
      if(s.attempts.fetch_add(1)==s.fail.load())return nullptr;
      auto*result=s.backing.allocate(bytes,alignment);
      if(result)s.live.fetch_add(1);
      return result;
    },[](void*p,void*data,std::size_t bytes,std::size_t alignment)noexcept{
      auto&s=*static_cast<ConcurrentAllocator*>(p);
      if(data)s.live.fetch_sub(1);
      s.backing.deallocate(data,bytes,alignment);
    },[](void*p,void*data,std::size_t old_bytes,std::size_t bytes,std::size_t alignment)noexcept->void*{
      auto&s=*static_cast<ConcurrentAllocator*>(p);
      if(bytes && s.attempts.fetch_add(1)==s.fail.load())return nullptr;
      auto*result=s.backing.reallocate(data,old_bytes,bytes,alignment);
      if(!bytes && data)s.live.fetch_sub(1);
      else if(result && !data)s.live.fetch_add(1);
      return result;
    },this};
  }
};
} // namespace hyle_f5_fault_fixture
