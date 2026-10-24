#include "apple_row_cache_indexer.h"
#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include "apple_checks.h"
namespace hyle::compute::apple::detail {
namespace mx=mlx::core;
namespace {
constexpr const char *kIndexerKey = R"METAL(
    inline uint hyle_key(uint code,uint scale) {
      const uint mantissas[8]={0,1,2,3,4,6,8,12};
      const uint magnitude=mantissas[code&7], sign=(code&8)<<12;
      if(scale==255)return sign|0x7fc0;
      if(!magnitude)return sign;
      const int power=int(scale)-128, top=31-clz(magnitude), exponent=top+power;
      if(exponent>127)return sign|0x7f80;
      if(exponent>=-126)return sign|(uint(exponent+127)<<7)|((magnitude<<uint(7-top))&127);
      const int shift=power+133;
      uint value=0;
      if(shift>=0)value=magnitude<<uint(shift);
      else if(shift>-32) {
        const uint right=uint(-shift), high=magnitude>>right;
        const uint low=magnitude&((1u<<right)-1), halfway=1u<<(right-1);
        value=high+uint(low>halfway || (low==halfway && (high&1)));
      }
      return sign|value;
    }
  )METAL";
mx::array finish_indexer_scores(const mx::array&dots,const mx::array&weights,
    const mx::array&ranges,const mx::array&columns,int tokens,int width,
    int grid_x,int grid_y,mx::Stream s) {
  static const auto finish=mx::fast::metal_kernel("hyle_row_fp4_indexer_reduce",
      {"dots","weights","ranges","columns"},{"scores"},R"METAL(
    const ulong pair=ulong(thread_position_in_grid.y)*GRID_X+thread_position_in_grid.x;
    const ulong width=columns_shape[0], tokens=weights_shape[0];
    if(pair>=tokens*width)return;
    const ulong t=pair/width, k=pair%width, row=columns[k];
    const int begin=ranges[t*ranges_strides[0]], end=ranges[t*ranges_strides[0]+ranges_strides[1]];
    if(row<ulong(begin) || row>=ulong(end)){scores[pair]=-INFINITY;return;}
    float total=0.f;
    for(uint h=0;h<64;++h) {
      const float dot=as_type<float>(uint(dots[t*dots_strides[0]+h*dots_strides[1]+k*dots_strides[2]])<<16);
      const float positive=isnan(dot)?dot:(dot>0.f?dot:0.f);
      const float weight=as_type<float>(uint(weights[t*weights_strides[0]+h*weights_strides[1]])<<16);
      total=total+hyle_bfloat(positive*weight);
    }
    scores[pair]=hyle_bfloat(total);
  )METAL",R"METAL(
    #pragma clang fp contract(off)
    inline float hyle_bfloat(float x) {
      uint bits=as_type<uint>(x);
      if((bits&0x7fffffff)>0x7f800000)return as_type<float>((bits&0xffff0000)|0x00400000);
      return as_type<float>((bits+0x7fff+((bits>>16)&1))&0xffff0000);
    }
  )METAL",false,false,mx::CompileOptions{});
  return finish({mx::view(dots,mx::uint16,s),weights,ranges,columns},
      {{tokens,width}},{mx::float32},{grid_x,grid_y,1},{128,1,1},
      {{"GRID_X",grid_x}},std::nullopt,false,s)[0];
}
} // namespace
Result<mx::array> row_cache_indexer_scores(std::span<const mx::array> inputs,mx::Stream s) {
  const int tokens=inputs[0].shape(0), rows=inputs[1].shape(0);
  auto begin=mx::slice(inputs[3],{0,0},{tokens,1},s);
  auto end=mx::slice(inputs[3],{0,1},{tokens,2},s);
  auto valid=mx::logical_and(mx::greater_equal(begin,mx::array(0),s),
      mx::logical_and(mx::less_equal(begin,end,s),mx::less_equal(end,mx::array(rows),s),s),s);
  auto all_valid=mx::all(valid,false,s);
  auto status=mx::where(all_valid,mx::array(0),mx::array(kIndexOutOfRange),s);
  if(!tokens || !rows)
    return with_device_status(mx::zeros({tokens,rows},mx::float32,s),std::move(status),s);
  // Decode a bounded union of referenced rows, shared by all query heads.
  // The scalar is a real dependency on complete GPU range validation; even
  // another query's otherwise legal row must not be read after a failure.
  static const auto decoder=mx::fast::metal_kernel("hyle_row_fp4_indexer_keys",
      {"bank","ranges","all_valid","columns"},{"keys"},R"METAL(
    const uint lane=thread_index_in_threadgroup;
    const uint offset=threadgroup_position_in_grid.x;
    const ulong row=columns[offset];
    bool referenced=false;
    if(all_valid)for(int t=0;t<ranges_shape[0];++t) {
      const int begin=ranges[t*ranges_strides[0]], end=ranges[t*ranges_strides[0]+ranges_strides[1]];
      if(row>=ulong(begin) && row<ulong(end)){referenced=true;break;}
    }
    if(!referenced){keys[offset*128+lane]=0;return;}
    const uint packed=bank[row*bank_strides[0]+(lane/2)*bank_strides[1]];
    const uint code=(packed>>((lane&1)*4))&15;
    const uint scale=bank[row*bank_strides[0]+(64+lane/32)*bank_strides[1]];
    keys[offset*128+lane]=hyle_key(code,scale);
  )METAL",kIndexerKey,false,false,mx::CompileOptions{});
  // Keep the original explicit BF16 product and sequential F32 head sum.
  // Only the dot is delegated to MLX's mature BF16/F32-accumulator matmul.
  constexpr int tile_rows=256;
  auto query=inputs[0];
  const bool flatten=tokens<=std::numeric_limits<int>::max()/64;
  if(flatten)query=mx::reshape(query,{tokens*64,128},s);
  const auto weights=mx::view(inputs[2],mx::uint16,s);
  std::vector<mx::array> blocks;
  blocks.reserve((std::size_t(rows)+tile_rows-1)/tile_rows);
  for(int first=0;first<rows;) {
    const int width=std::min(tile_rows,rows-first);
    std::vector<mx::array> sources{mx::arange(first,first+width,mx::int32,s)};
    // Sequence decode itself, not merely matmul. Otherwise MLX may schedule
    // independent keys ahead of consumers and retain a whole decoded bank.
    if(!blocks.empty())sources=mx::depends(sources,{blocks.back()});
    auto bits=decoder({inputs[1],inputs[3],all_valid,sources[0]},{{width,128}},{mx::uint16},
        {width*128,1,1},{128,1,1},{},std::nullopt,false,s)[0];
    auto keys=mx::view(bits,mx::bfloat16,s);
    auto dots=mx::matmul(query,mx::transpose(keys,{1,0},s),s);
    if(flatten)dots=mx::reshape(dots,{tokens,64,width},s);
    const auto count=std::uint64_t(tokens)*std::uint64_t(width);
    const int grid_x=static_cast<int>(std::min<std::uint64_t>((count+127)/128*128,65536));
    const auto grid_y=(count+grid_x-1)/grid_x;
    if(grid_y>std::uint64_t(std::numeric_limits<int>::max()))
      return std::unexpected(Status::Unimplemented("Apple row indexer native grid capacity"));
    blocks.push_back(finish_indexer_scores(dots,weights,inputs[3],sources[0],
        tokens,width,grid_x,static_cast<int>(grid_y),s));
    first+=width;
  }
  auto output=mx::concatenate(blocks,1,s);
  return with_device_status(std::move(output),std::move(status),s);
}

Result<mx::array> row_cache_paged_indexer_scores(
    const compute::detail::RowCachePagedIndexerScores &operation,
    std::span<const mx::array> inputs,mx::Stream s) {
  const auto &m=operation.mapping;
  const int tokens=inputs[0].shape(0), columns=int(operation.score_columns);
  if(m.metadata_bytes>INT32_MAX || !m.max_refs())
    return std::unexpected(Status::Unimplemented("Apple paged indexer metadata exceeds native I32 carrier"));
  constexpr const char *metadata=R"METAL(
    inline uint hyle_word(const device uchar*p,ulong stride,uint at) {
      return uint(p[ulong(at)*stride])|(uint(p[ulong(at+1)*stride])<<8)|
          (uint(p[ulong(at+2)*stride])<<16)|(uint(p[ulong(at+3)*stride])<<24);
    }
    inline uint hyle_request(const device uchar*p,ulong stride,uint row,uint requests,uint q_offset) {
      uint lo=0,hi=requests;
      while(lo<hi){const uint mid=lo+(hi-lo)/2;
        if(int(hyle_word(p,stride,(q_offset+mid+1)*4))<=int(row))lo=mid+1;else hi=mid;}
      return lo;
    }
  )METAL";
  static const auto validate=mx::fast::metal_kernel("hyle_paged_fp4_indexer_validate",
      {"metadata","ranges"},{"groups","valid"},R"METAL(
    const uint lane=thread_index_in_threadgroup;
    const ulong stride=metadata_strides[0];
    const uint requests=hyle_word(metadata,stride,16),refs=hyle_word(metadata,stride,28);
    const uint tokens=ranges_shape[0];
    bool good=requests<=MAX_REQUESTS && refs<=MAX_REFS &&
        hyle_word(metadata,stride,20)==tokens;
    if(good)good=int(hyle_word(metadata,stride,Q_OFFSET*4))==0 &&
        int(hyle_word(metadata,stride,(Q_OFFSET+requests)*4))==int(tokens) &&
        int(hyle_word(metadata,stride,KV_OFFSET*4))==0 &&
        int(hyle_word(metadata,stride,(KV_OFFSET+requests)*4))==int(refs);
    const bool header=good;
    if(header) {
      for(ulong r=lane;r<requests;r+=128) {
        const int q0=int(hyle_word(metadata,stride,(Q_OFFSET+uint(r))*4));
        const int q1=int(hyle_word(metadata,stride,(Q_OFFSET+uint(r)+1)*4));
        const int first=int(hyle_word(metadata,stride,(KV_OFFSET+uint(r))*4));
        const int last=int(hyle_word(metadata,stride,(KV_OFFSET+uint(r)+1)*4));
        const int tail=int(hyle_word(metadata,stride,(TAIL_OFFSET+uint(r))*4));
        const bool span=q0>=0 && q1>=q0 && uint(q1)<=tokens && first>=0 &&
            last>=first && uint(last)<=refs &&
            ((first==last && tail==0)||(last>first && tail>0 && uint(tail)<=PAGE_SIZE));
        const ulong length=span && last>first?ulong(last-first-1)*PAGE_SIZE+uint(tail):0;
        good=good && span && length<=SCORE_COLUMNS;
      }
      for(ulong p=lane;p<refs;p+=128) {
        const int page=int(hyle_word(metadata,stride,(PAGE_OFFSET+uint(p))*4));
        good=good && page>=0 && uint(page)<MAX_PAGES;
      }
    }
    for(ulong t=lane;t<tokens;t+=128) {
      uint r=0;bool query_ok=header;
      if(query_ok){r=hyle_request(metadata,stride,uint(t),requests,Q_OFFSET);query_ok=r<requests;}
      uint group=0;
      if(query_ok) {
        const int q0=int(hyle_word(metadata,stride,(Q_OFFSET+r)*4));
        const int q1=int(hyle_word(metadata,stride,(Q_OFFSET+r+1)*4));
        const int first=int(hyle_word(metadata,stride,(KV_OFFSET+r)*4));
        const int last=int(hyle_word(metadata,stride,(KV_OFFSET+r+1)*4));
        const int tail=int(hyle_word(metadata,stride,(TAIL_OFFSET+r)*4));
        const bool span=first>=0 && last>=first && uint(last)<=refs &&
            ((first==last && tail==0)||(last>first && tail>0 && uint(tail)<=PAGE_SIZE));
        const ulong length=span && last>first?ulong(last-first-1)*PAGE_SIZE+uint(tail):0;
        const int begin=ranges[t*ranges_strides[0]],end=ranges[t*ranges_strides[0]+ranges_strides[1]];
        query_ok=span && q0>=0 && q0<=int(t) && q1>int(t) &&
            begin>=0 && end>=begin && ulong(end)<=length && length<=SCORE_COLUMNS;
        if(query_ok)group=REQUEST_GROUPS?r:uint(q0);
      }
      groups[t]=group;good=good && query_ok;
    }
    threadgroup uint failed[128];failed[lane]=uint(!good);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint offset=64;offset;offset/=2){if(lane<offset)failed[lane]|=failed[lane+offset];
      threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(lane==0)valid[0]=failed[0]==0;
  )METAL",metadata,false,false,mx::CompileOptions{});
  const bool request_groups=m.max_requests()<=std::uint32_t(tokens);
  const int groups=std::min(int(m.max_requests()),tokens);
  auto checked=validate({inputs[4],inputs[3]},{{std::max(tokens,1)},{}},{mx::uint32,mx::bool_},
      {128,1,1},{128,1,1},
      {{"MAX_REQUESTS",int(m.max_requests())},{"MAX_REFS",int(m.max_refs())},
       {"MAX_PAGES",int(m.max_pages)},{"PAGE_SIZE",int(m.page_size)},
       {"Q_OFFSET",int(m.query_offset/4)},{"KV_OFFSET",int(m.kv_offset/4)},
       {"PAGE_OFFSET",int(m.page_offset/4)},{"TAIL_OFFSET",int(m.tail_offset/4)},
       {"SCORE_COLUMNS",columns},{"REQUEST_GROUPS",request_groups}},std::nullopt,false,s);
  auto status=mx::where(checked[1],mx::array(0),mx::array(kIndexOutOfRange),s);
  if(!tokens || !columns)
    return with_device_status(mx::zeros({tokens,columns},mx::float32,s),std::move(status),s);
  // Each active request's tile is decoded once. If Rmax>T, its first query row
  // is the compact group key; empty requests consume no decoded group.
  static const std::string preamble=std::string(metadata)+kIndexerKey;
  static const auto decoder=mx::fast::metal_kernel("hyle_paged_fp4_indexer_keys",
      {"bank","ranges","metadata","all_valid","columns"},{"keys"},R"METAL(
    const uint lane=thread_index_in_threadgroup;
    const uint offset=threadgroup_position_in_grid.x,group=threadgroup_position_in_grid.y;
    const uint width=columns_shape[0];
    const ulong out=(ulong(group)*width+offset)*128+lane;
    if(!all_valid){keys[out]=0;return;}
    const ulong stride=metadata_strides[0];
    const uint requests=hyle_word(metadata,stride,16);
    const uint r=REQUEST_GROUPS?group:hyle_request(metadata,stride,group,requests,Q_OFFSET);
    if(r>=requests){keys[out]=0;return;}
    const uint q0=hyle_word(metadata,stride,(Q_OFFSET+r)*4),q1=hyle_word(metadata,stride,(Q_OFFSET+r+1)*4);
    if((!REQUEST_GROUPS && q0!=group)||q0==q1){keys[out]=0;return;}
    const uint column=columns[offset];bool referenced=false;
    for(ulong t=q0;t<q1;++t){const int begin=ranges[t*ranges_strides[0]],end=ranges[t*ranges_strides[0]+ranges_strides[1]];
      if(column>=uint(begin) && column<uint(end)){referenced=true;break;}}
    if(!referenced){keys[out]=0;return;}
    const uint first=hyle_word(metadata,stride,(KV_OFFSET+r)*4);
    const uint page=hyle_word(metadata,stride,(PAGE_OFFSET+first+column/PAGE_SIZE)*4);
    const ulong row=ulong(page)*PAGE_SIZE+column%PAGE_SIZE;
    const uint packed=bank[row*bank_strides[0]+(lane/2)*bank_strides[1]];
    const uint code=(packed>>((lane&1)*4))&15;
    const uint scale=bank[row*bank_strides[0]+(64+lane/32)*bank_strides[1]];
    keys[out]=hyle_key(code,scale);
  )METAL",preamble,false,false,mx::CompileOptions{});
  constexpr int tile_rows=256;
  const auto weights=mx::view(inputs[2],mx::uint16,s);
  const auto query_indices=mx::arange(tokens,mx::uint32,s);
  std::vector<mx::array> blocks;blocks.reserve((std::size_t(columns)+tile_rows-1)/tile_rows);
  for(int first=0;first<columns;) {
    const int width=std::min(tile_rows,columns-first);
    std::vector<mx::array> local_columns{mx::arange(first,first+width,mx::int32,s)};
    if(!blocks.empty())local_columns=mx::depends(local_columns,{blocks.back()});
    auto bits=decoder({inputs[1],inputs[3],inputs[4],checked[1],local_columns[0]},
        {{groups,width,128}},{mx::uint16},{width*128,groups,1},{128,1,1},
        {{"Q_OFFSET",int(m.query_offset/4)},{"KV_OFFSET",int(m.kv_offset/4)},
         {"PAGE_OFFSET",int(m.page_offset/4)},{"PAGE_SIZE",int(m.page_size)},
         {"REQUEST_GROUPS",request_groups}},std::nullopt,false,s)[0];
    auto keys=mx::transpose(mx::view(bits,mx::bfloat16,s),{0,2,1},s);
    auto dots=mx::gather_mm(inputs[0],keys,query_indices,checked[0],false,s);
    const auto count=std::uint64_t(tokens)*std::uint64_t(width);
    const int grid_x=static_cast<int>(std::min<std::uint64_t>((count+127)/128*128,65536));
    const auto grid_y=(count+grid_x-1)/grid_x;
    if(grid_y>std::uint64_t(std::numeric_limits<int>::max()))
      return std::unexpected(Status::Unimplemented("Apple paged indexer native grid capacity"));
    blocks.push_back(finish_indexer_scores(dots,weights,inputs[3],local_columns[0],
        tokens,width,grid_x,static_cast<int>(grid_y),s));
    first+=width;
  }
  return with_device_status(mx::concatenate(blocks,1,s),std::move(status),s);
}
}
