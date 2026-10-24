#pragma once
#include "row_cache_fixture.h"
#include "hyle/compute/batch_layout.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <stdexcept>
namespace row_cache_native_test {
namespace c=hyle::compute;
template<class T>T take(hyle::Result<T> result,const char*what){if(!result)throw std::runtime_error(std::string(what)+": "+std::string(result.error().message()));return std::move(*result);}
inline c::RowCacheBatchCapacity capacity(c::RowCacheEncoding e){return {.encoding=e,.max_requests=3,.max_query_rows=4,.max_kv_rows=8,.max_pages=5,.max_page_refs=4,.page_size=2};}
inline c::RowCacheLayoutVersion layout(const c::RowCacheBatchSchema&s,int phase){
  const std::array<std::int32_t,4> q=phase?std::array<std::int32_t,4>{0,0,2,4}:std::array<std::int32_t,4>{0,2,2,4};
  const std::array<std::int32_t,4> kv=phase?std::array<std::int32_t,4>{0,0,2,4}:std::array<std::int32_t,4>{0,2,3,4};
  const std::array<std::int32_t,4> pages{4,1,3,1};const std::array<std::int32_t,3> tail=phase?std::array<std::int32_t,3>{0,1,2}:std::array<std::int32_t,3>{1,2,2};
  return take(s.version({q,kv,pages,tail}),"native row layout");
}
inline row_cache_test::Reference data(c::RowCacheEncoding e){auto first=row_cache_test::reference(e,row_cache_test::input_values(e,0)),next=row_cache_test::reference(e,row_cache_test::input_values(e,1));
  first.bytes.insert(first.bytes.end(),next.bytes.begin(),next.bytes.end());first.decoded.insert(first.decoded.end(),next.decoded.begin(),next.decoded.end());return first;}
inline std::array<std::int32_t,12> ids(int phase){return phase?std::array<std::int32_t,12>{2,0,2,1,2,0,3,0,2,1,3,0}:std::array<std::int32_t,12>{2,0,2,1,2,0,1,0,1,0,1,0};}
inline std::vector<std::uint16_t> expected(c::RowCacheEncoding e,const row_cache_test::Reference&source,std::span<const std::int32_t>selected,int phase){std::vector<std::uint16_t> result;
  for(std::size_t i=0;i<selected.size();++i){const auto id=selected[i];const auto page=i/3<2?(id<2?4:1):(phase?(id<2?3:1):1);const auto row=page*2+id%2;
    const auto first=source.decoded.begin()+row*e.depth();result.insert(result.end(),first,first+e.depth());}return result;}
}
