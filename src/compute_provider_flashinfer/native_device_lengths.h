#pragma once
#include "native_batch.h"
namespace hyle::compute::flashinfer::native {
// Private qualified launch facts. Effective lengths and their bounds remain
// device data in the original slot's owned carriers, never capture arguments.
struct DeviceLengthsShape final {
  BatchSpec spec;
  std::size_t request_tiles=0,query_tiles=0,kv_tiles=0,output_indptr=0;
  std::size_t merge_indptr=0,valid_mask=0,total_rows=0,chunk_rows=0;
  std::uint32_t padded_tiles=0,cta_query=0;
};
cudaError_t qualify_device_lengths(const void *,BatchSpec,std::uint32_t,
    const std::int32_t *,const std::int32_t *,const std::int32_t *,
    std::size_t,DeviceLengthsShape &,bool &) noexcept;
cudaError_t update_device_lengths(DeviceLengthsShape,const std::int32_t *,
    const std::int32_t *,const void *,void *,void *,unsigned *,cudaStream_t) noexcept;
} // namespace hyle::compute::flashinfer::native
