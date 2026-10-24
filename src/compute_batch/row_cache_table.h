#pragma once
#include "row_cache_map.h"
#include <cstring>
namespace hyle::compute::detail {
// Checked read-only access to the same owned metadata ABI used by RowCacheMap.
// No page permission is created here; the Graph adapter retains exact leases.
struct RowCacheTable final {
  RowCacheMap mapping{};const std::byte* bytes=nullptr;std::uint32_t requests=0,refs=0;
  static std::int32_t i32(const std::byte*p)noexcept{std::int32_t v;std::memcpy(&v,p,4);return v;}
  std::int32_t query(std::size_t i)const noexcept{return i32(bytes+mapping.query_offset+i*4);}
  std::int32_t kv(std::size_t i)const noexcept{return i32(bytes+mapping.kv_offset+i*4);}
  std::int32_t tail(std::size_t i)const noexcept{return i32(bytes+mapping.tail_offset+i*4);}
  std::uint64_t length(std::size_t r)const noexcept{
    return kv(r+1)==kv(r)?0:std::uint64_t(kv(r+1)-kv(r)-1)*mapping.page_size+std::uint32_t(tail(r));
  }
  std::size_t request(std::size_t row)const noexcept{
    std::size_t lo=0,hi=requests;while(lo<hi){const auto mid=lo+(hi-lo)/2;if(query(mid+1)<=std::int64_t(row))lo=mid+1;else hi=mid;}return lo;
  }
  std::int32_t physical(std::size_t r,std::uint64_t id)const noexcept{
    const auto page=i32(bytes+mapping.page_offset+(std::size_t(kv(r))+std::size_t(id/mapping.page_size))*4);
    return std::int32_t(std::uint64_t(std::uint32_t(page))*mapping.page_size+id%mapping.page_size);
  }
  static Result<RowCacheTable> create(RowCacheMap m,const std::byte*data,std::size_t rows)noexcept{
    RowCacheTable t{m,data,std::uint32_t(i32(data+16)),std::uint32_t(i32(data+28))};
    if(t.requests>m.max_requests() || t.refs>m.max_refs() || std::uint32_t(i32(data+20))!=rows ||
       t.query(0)!=0 || t.query(t.requests)!=std::int64_t(rows) || t.kv(0)!=0 || t.kv(t.requests)!=std::int64_t(t.refs))
      return std::unexpected(Status::OutOfRange("row cache metadata counts differ"));
    for(std::size_t r=0;r<t.requests;++r){
      const auto a=t.kv(r),b=t.kv(r+1),tail=t.tail(r);
      if(t.query(r)<0 || t.query(r+1)<t.query(r) || a<0 || b<a || std::uint32_t(b)>t.refs ||
         (a==b?tail!=0:(tail<=0 || std::uint32_t(tail)>m.page_size)))
        return std::unexpected(Status::OutOfRange("row cache metadata request span differs"));
    }
    for(std::size_t p=0;p<t.refs;++p){const auto id=i32(data+m.page_offset+p*4);
      if(id<0 || std::uint32_t(id)>=m.max_pages)return std::unexpected(Status::OutOfRange("row cache metadata page differs"));}
    return t;
  }
};
}
