#include "native_device_lengths.h"
#include "device_status_kernel.h"
#include <flashinfer/attention/scheduler.cuh>
#include <limits>
namespace hyle::compute::flashinfer::native {
namespace {
constexpr unsigned kRequestLimit=32;
__global__ void update_lengths(DeviceLengthsShape d,const std::int32_t*query_lengths,
    const std::int32_t*kv_lengths,const unsigned char*bounds,unsigned char*metadata,
    unsigned char*integer,unsigned*status){
  __shared__ std::int32_t qs[kRequestLimit+1],ks[kRequestLimit+1],os[kRequestLimit+1];
  __shared__ std::int32_t rows[kRequestLimit],pages[kRequestLimit],chunks[kRequestLimit],tails[kRequestLimit];
  __shared__ std::int32_t chunk;
  const auto&s=d.spec;
  const auto*maxq=reinterpret_cast<const std::int32_t*>(bounds+s.query_indptr_offset);
  const auto*maxk=reinterpret_cast<const std::int32_t*>(bounds+s.kv_indptr_offset);
  const auto*max_tail=reinterpret_cast<const std::int32_t*>(bounds+s.tail_offset);
  const auto*source_pages=reinterpret_cast<const std::int32_t*>(bounds+s.page_indices_offset);
  if(threadIdx.x==0){
    bool invalid=*status!=0;chunk=*reinterpret_cast<const std::int32_t*>(integer+d.chunk_rows);
    if(chunk<=0 || chunk%std::int32_t(s.page_size))invalid=true;
    qs[0]=ks[0]=os[0]=0;std::uint64_t tiles=0;
    for(unsigned r=0;r<s.requests;++r){
      auto q=query_lengths[r],kv=kv_lengths[r];
      const auto max_pages=maxk[r+1]-maxk[r];
      const auto max_kv=max_pages?(std::int64_t(max_pages)-1)*s.page_size+max_tail[r]:0;
      if(q<0 || q>maxq[r+1]-maxq[r] || kv<0 || kv>max_kv || q>kv){invalid=true;q=kv=0;}
      rows[r]=q;pages[r]=kv/std::int32_t(s.page_size)+(kv%std::int32_t(s.page_size)!=0);
      tails[r]=kv?(kv-1)%std::int32_t(s.page_size)+1:0;
      // Safe even after a malformed private chunk scalar: no division by zero.
      const auto chunk_pages=chunk>0?chunk/std::int32_t(s.page_size):0;
      chunks[r]=chunk_pages>0?pages[r]/chunk_pages+(pages[r]%chunk_pages!=0):1;
      if(!chunks[r])chunks[r]=1;
      const auto next_q=std::int64_t(qs[r])+q,next_k=std::int64_t(ks[r])+pages[r],
          next_o=std::int64_t(os[r])+std::int64_t(q)*chunks[r];
      if(next_q>s.query_rows || next_k>maxk[s.requests] || next_o>INT32_MAX){invalid=true;qs[r+1]=ks[r+1]=os[r+1]=0;}
      else{qs[r+1]=std::int32_t(next_q);ks[r+1]=std::int32_t(next_k);os[r+1]=std::int32_t(next_o);}
      const auto packed=std::uint64_t(q)*(s.query_heads/s.kv_heads);
      tiles+=((packed+d.cta_query-1)/d.cta_query)*chunks[r];
    }
    if(tiles>d.padded_tiles)invalid=true;
    if(invalid){
      atomicCAS(status,0u,static_cast<unsigned>(cuda::detail::DeviceStatusCode::kInvalidSequenceOffsets));
      for(unsigned r=0;r<s.requests;++r){rows[r]=pages[r]=tails[r]=0;chunks[r]=1;qs[r+1]=ks[r+1]=os[r+1]=0;}
    }
  }
  __syncthreads();
  auto q=reinterpret_cast<std::int32_t*>(metadata+s.query_indptr_offset),
      k=reinterpret_cast<std::int32_t*>(metadata+s.kv_indptr_offset),
      ids=reinterpret_cast<std::int32_t*>(metadata+s.page_indices_offset),
      tail=reinterpret_cast<std::int32_t*>(metadata+s.tail_offset);
  auto req=reinterpret_cast<std::int32_t*>(integer+d.request_tiles),
      qt=reinterpret_cast<std::int32_t*>(integer+d.query_tiles),
      kt=reinterpret_cast<std::int32_t*>(integer+d.kv_tiles),
      output=reinterpret_cast<std::int32_t*>(integer+d.output_indptr),
      merge=reinterpret_cast<std::int32_t*>(integer+d.merge_indptr);
  auto mask=reinterpret_cast<bool*>(integer+d.valid_mask);
  for(unsigned i=threadIdx.x;i<d.padded_tiles;i+=blockDim.x){mask[i]=false;req[i]=qt[i]=kt[i]=0;}
  for(std::size_t i=threadIdx.x;i<=s.query_rows;i+=blockDim.x)merge[i]=0;
  for(unsigned r=0;r<s.requests;++r)
    for(std::size_t i=threadIdx.x;i<std::uint32_t(pages[r]);i+=blockDim.x)ids[std::size_t(ks[r])+i]=source_pages[std::size_t(maxk[r])+i];
  __syncthreads();
  if(threadIdx.x==0){
    unsigned at=0;
    for(unsigned r=0;r<s.requests;++r){
      q[r]=qs[r];k[r]=ks[r];output[r]=os[r];tail[r]=tails[r];
      const auto packed=std::uint64_t(rows[r])*(s.query_heads/s.kv_heads);
      const auto tiles=(packed+d.cta_query-1)/d.cta_query;
      for(unsigned t=0;t<tiles;++t)for(unsigned c=0;c<std::uint32_t(chunks[r]);++c){
        req[at]=std::int32_t(r);qt[at]=std::int32_t(t);kt[at]=std::int32_t(c);mask[at]=true;++at;
      }
      for(std::int32_t row=0;row<rows[r];++row)merge[qs[r]+row]=os[r]+row*chunks[r];
    }
    q[s.requests]=qs[s.requests];k[s.requests]=ks[s.requests];output[s.requests]=os[s.requests];
    merge[qs[s.requests]]=os[s.requests];
    *reinterpret_cast<std::uint32_t*>(integer+d.total_rows)=std::uint32_t(qs[s.requests]);
  }
}
}
cudaError_t qualify_device_lengths(const void*state,BatchSpec s,std::uint32_t chunk,
    const std::int32_t*q,const std::int32_t*k,const std::int32_t*tails,
    std::size_t integer_bytes,DeviceLengthsShape&shape,bool&qualified)noexcept{
  qualified=false;
  if(!state || !q || !k || !tails || !s.bounded_layouts || !s.requests || s.requests>kRequestLimit ||
      !s.paged || s.fp8 || !s.causal || s.window_left!=-1 || !s.page_size || s.page_size>INT32_MAX ||
      !s.kv_heads || !s.query_heads || s.query_heads%s.kv_heads || !chunk || chunk>INT32_MAX || chunk%s.page_size ||
      q[0]!=0 || k[0]!=0 || q[s.requests]!=std::int32_t(s.query_rows))return cudaSuccess;
  const auto&p=*static_cast<const ::flashinfer::PrefillPlanInfo*>(state);
  if(!p.enable_cuda_graph || !p.split_kv || p.cta_tile_q<=0 || p.padded_batch_size<=0 ||
      p.total_num_rows!=s.query_rows)return cudaSuccess;
  const auto fits=[&](std::int64_t offset,std::uint64_t bytes){return offset>=0 &&
      std::uint64_t(offset)<=integer_bytes && bytes<=integer_bytes-std::uint64_t(offset);};
  const auto tile_bytes=std::uint64_t(p.padded_batch_size)*4;
  if(!fits(p.request_indices_offset,tile_bytes) || !fits(p.qo_tile_indices_offset,tile_bytes) ||
      !fits(p.kv_tile_indices_offset,tile_bytes) || !fits(p.o_indptr_offset,std::uint64_t(s.requests+1)*4) ||
      !fits(p.merge_indptr_offset,(std::uint64_t(s.query_rows)+1)*4) || !fits(p.block_valid_mask_offset,p.padded_batch_size) ||
      !fits(p.total_num_rows_offset,4) || !fits(p.kv_chunk_size_ptr_offset,4))return cudaErrorInvalidValue;
  std::uint64_t tiles=0,partials=0;
  for(unsigned r=0;r<s.requests;++r){
    const auto rows=q[r+1]-q[r],pages=k[r+1]-k[r];
    if(rows<0 || pages<0 || tails[r]<0 || tails[r]>std::int32_t(s.page_size))return cudaErrorInvalidValue;
    const auto chunks=pages?std::uint64_t(pages)/(chunk/s.page_size)+(std::uint64_t(pages)%(chunk/s.page_size)!=0):1;
    const auto packed=std::uint64_t(rows)*(s.query_heads/s.kv_heads);
    tiles+=((packed+std::uint64_t(p.cta_tile_q)-1)/std::uint64_t(p.cta_tile_q))*chunks;
    partials+=std::uint64_t(rows)*chunks;
  }
  if(tiles>std::uint64_t(p.padded_batch_size) || partials>INT32_MAX)return cudaSuccess;
  shape={s,std::size_t(p.request_indices_offset),std::size_t(p.qo_tile_indices_offset),
      std::size_t(p.kv_tile_indices_offset),std::size_t(p.o_indptr_offset),std::size_t(p.merge_indptr_offset),
      std::size_t(p.block_valid_mask_offset),std::size_t(p.total_num_rows_offset),std::size_t(p.kv_chunk_size_ptr_offset),
      std::uint32_t(p.padded_batch_size),std::uint32_t(p.cta_tile_q)};
  qualified=true;return cudaSuccess;
}
cudaError_t update_device_lengths(DeviceLengthsShape shape,const std::int32_t*q,const std::int32_t*k,
    const void*bounds,void*metadata,void*integer,unsigned*status,cudaStream_t stream)noexcept{
  if(!q || !k || !bounds || !metadata || !integer || !status || !shape.spec.requests ||
      shape.spec.requests>kRequestLimit || !shape.cta_query)return cudaErrorInvalidValue;
  update_lengths<<<1,256,0,stream>>>(shape,q,k,static_cast<const unsigned char*>(bounds),
      static_cast<unsigned char*>(metadata),static_cast<unsigned char*>(integer),status);
  return cudaPeekAtLastError();
}
} // namespace hyle::compute::flashinfer::native
