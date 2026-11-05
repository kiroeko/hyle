#include "cuda_paged_append.h"

#include <cuda_runtime.h>
#include "device_status_kernel.h"

namespace hyle::compute::cuda::detail {
namespace {

// Each block owns one complete, disjoint destination range. All offsets are
// row coordinates in the common wire format; F16/BF16 payloads remain bits.
__global__ void append_rows(
    void *arena_bytes, const void *input_bytes, const void *metadata_bytes,
    compute::detail::PagedAppendNativeSpec spec) {
  const auto *metadata =
      static_cast<const compute::detail::PagedAppendMetadata *>(metadata_bytes);
  const auto active = metadata->active_count;
  if (active > spec.instruction_capacity || metadata->reserved0 != 0 ||
      metadata->reserved1 != 0 || metadata->reserved2 != 0)
    return;
  const auto *instructions = reinterpret_cast<
      const compute::detail::PagedAppendInstruction *>(metadata + 1);
  auto *arena = static_cast<std::uint16_t *>(arena_bytes);
  const auto *input = static_cast<const std::uint16_t *>(input_bytes);
  const auto row_elements = spec.row_bytes / sizeof(std::uint16_t);
  for (std::uint64_t index = blockIdx.x; index < active; index += gridDim.x) {
    const auto instruction = instructions[index];
    // Common validated the whole host episode before upload. These guards
    // also prevent a malformed device record from reading/writing payload.
    if (instruction.reserved0 != 0 || instruction.reserved1 != 0 ||
        instruction.rows == 0 || instruction.source_kind > 1 ||
        instruction.destination_page >= spec.physical_pages ||
        instruction.destination_row > spec.page_rows ||
        instruction.rows > spec.page_rows - instruction.destination_row)
      continue;
    const std::uint16_t *source = nullptr;
    if (instruction.source_kind == 0) {
      if (instruction.source_row > spec.input_rows ||
          instruction.rows > spec.input_rows - instruction.source_row)
        continue;
      source = input + std::size_t(instruction.source_row) * row_elements;
    } else {
      if (instruction.source_page >= spec.physical_pages ||
          instruction.source_row > spec.page_rows ||
          instruction.rows > spec.page_rows - instruction.source_row)
        continue;
      source = arena +
          (std::size_t(instruction.source_page) * spec.page_rows +
           instruction.source_row) * row_elements;
    }
    auto *destination = arena +
        (std::size_t(instruction.destination_page) * spec.page_rows +
         instruction.destination_row) * row_elements;
    const auto elements = std::size_t(instruction.rows) * row_elements;
    for (std::size_t element = threadIdx.x; element < elements;
         element += blockDim.x)
      destination[element] = source[element];
  }
}

} // namespace

namespace {
}

const void *paged_append_kernel() noexcept {
  return reinterpret_cast<const void *>(append_rows);
}

cudaError_t launch_paged_append(
    const void *kernel, void *arena, const void *rows, const void *metadata,
    compute::detail::PagedAppendNativeSpec spec, cudaStream_t stream) noexcept {
  // The bound depends only on cold query/request capacity, never KV history.
  // A grid-stride instruction loop also keeps very large capacities within
  // CUDA's portable one-dimensional grid limit without another native call.
  const auto blocks = spec.instruction_capacity < 65535U
      ? spec.instruction_capacity : 65535U;
  void *arguments[]{&arena, &rows, &metadata, &spec};
  return cudaLaunchKernel(kernel, dim3(blocks), dim3(256), arguments, 0, stream);
}

namespace {
__global__ void append_device_lengths(void*arena_bytes,const void*input_bytes,const void*metadata_bytes,
    const std::int32_t*query_lengths,const std::int32_t*kv_lengths,
    compute::detail::DeviceLengthsPagedAppendSpec domain,unsigned*status){
  const auto spec=domain.storage;
  const auto*metadata=static_cast<const compute::detail::PagedAppendMetadata*>(metadata_bytes);
  const auto*words=reinterpret_cast<const compute::detail::PagedAppendInstruction*>(metadata+1);
  const auto*directory=reinterpret_cast<const compute::detail::DeviceAppendDirectory*>(words);
  __shared__ std::int32_t offsets[33],previous[32],last[32];
  __shared__ bool valid;
  if(threadIdx.x==0){
    valid=*status==0;offsets[0]=0;
    if(!metadata->active_count || metadata->active_count>spec.instruction_capacity || metadata->reserved0 ||
        metadata->reserved1 || metadata->reserved2 || directory->requests!=domain.requests ||
        directory->requests>32 || directory->pages>metadata->active_count-1 ||
        directory->requests>metadata->active_count-1-directory->pages)valid=false;
    for(unsigned i=0;i<6;++i)if(directory->reserved[i])valid=false;
    if(valid){
      const auto*requests=reinterpret_cast<const compute::detail::DeviceAppendRequest*>(words+1);
      for(unsigned r=0;r<domain.requests;++r){
        const auto b=requests[r];const auto q=query_lengths[r],kv=kv_lengths[r];
        const auto old=std::int64_t(kv)-q,next=std::int64_t(offsets[r])+q;
        for(unsigned i=0;i<4;++i)if(b.reserved[i])valid=false;
        if(q<0 || std::uint32_t(q)>b.query_rows || kv<0 || std::uint32_t(kv)>b.maximum_kv_rows ||
            old<b.minimum_previous_rows || old>b.maximum_previous_rows || next<0 || next>spec.input_rows){
          valid=false;offsets[r+1]=previous[r]=last[r]=0;
        }else{offsets[r+1]=std::int32_t(next);previous[r]=std::int32_t(old);last[r]=kv;}
      }
    }
    if(!valid)atomicCAS(status,0u,static_cast<unsigned>(DeviceStatusCode::kInvalidSequenceOffsets));
  }
  __syncthreads();if(!valid)return;
  const auto*pages=reinterpret_cast<const compute::detail::DeviceAppendPage*>(words+1+domain.requests);
  // prepare_append admits whole two-byte words and original two-byte-aligned
  // carriers. Each page has three disjoint ranges; one block copies a whole
  // range, with no per-element row division or source-selection branch.
  auto*arena=static_cast<std::uint16_t*>(arena_bytes);const auto*input=static_cast<const std::uint16_t*>(input_bytes);
  const auto row_words=spec.row_bytes/sizeof(std::uint16_t);
  const auto phase=blockIdx.x%3u;
  for(std::uint64_t index=blockIdx.x/3u;index<directory->pages;index+=gridDim.x/3u){
    const auto p=pages[index];
    if(p.request>=domain.requests || p.destination_page>=spec.physical_pages || !p.initialized_rows ||
        p.initialized_rows>spec.page_rows || p.source_rows>spec.page_rows || p.reserved[0] || p.reserved[1]){
      if(threadIdx.x==0)atomicCAS(status,0u,static_cast<unsigned>(DeviceStatusCode::kIndexOutOfRange));return;
    }
    const auto begin=std::uint64_t(p.logical_page)*spec.page_rows;
    const auto prior=std::uint64_t(previous[p.request]),next=std::uint64_t(last[p.request]);
    const auto inherited=begin<prior?static_cast<std::uint32_t>(min(std::uint64_t(p.initialized_rows),prior-begin)):0u;
    const auto live=begin<next?static_cast<std::uint32_t>(min(std::uint64_t(p.initialized_rows),next-begin)):0u;
    const auto added=live-inherited;
    const auto source_row=added?std::uint64_t(offsets[p.request])+(begin>prior?begin:prior)-prior:0;
    if(inherited>live || (inherited && (p.source_page>=spec.physical_pages || inherited>p.source_rows)) ||
        (added && (source_row>spec.input_rows || added>spec.input_rows-source_row))){
      if(threadIdx.x==0)atomicCAS(status,0u,static_cast<unsigned>(DeviceStatusCode::kIndexOutOfRange));return;
    }
    const auto first=phase==0?0u:phase==1?inherited:live;
    const auto end=phase==0?inherited:phase==1?live:p.initialized_rows;
    const auto count=std::size_t(end-first)*row_words;
    if(!count)continue;
    const std::uint16_t*source=nullptr;
    if(phase==0)source=arena+std::size_t(p.source_page)*spec.page_rows*row_words;
    else if(phase==1)source=input+std::size_t(source_row)*row_words;
    auto*destination=arena+(std::size_t(p.destination_page)*spec.page_rows+first)*row_words;
    if(source){for(std::size_t i=threadIdx.x;i<count;i+=blockDim.x)destination[i]=source[i];}
    else{for(std::size_t i=threadIdx.x;i<count;i+=blockDim.x)destination[i]=0;}
  }
}
}
const void*device_lengths_paged_append_kernel()noexcept{return reinterpret_cast<const void*>(append_device_lengths);}
cudaError_t launch_device_lengths_paged_append(const void*kernel,void*arena,const void*rows,const void*metadata,
    const void*query,const void*kv,compute::detail::DeviceLengthsPagedAppendSpec spec,unsigned*status,cudaStream_t stream)noexcept{
  const auto pages=spec.storage.instruction_capacity<21845U?spec.storage.instruction_capacity:21845U;
  void*arguments[]{&arena,&rows,&metadata,&query,&kv,&spec,&status};
  return cudaLaunchKernel(kernel,dim3(pages*3u),dim3(256),arguments,0,stream);
}
} // namespace hyle::compute::cuda::detail
