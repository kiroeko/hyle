#pragma once

#include "hyle/compute/apple.h"
#include "hyle/compute/model_ops.h"
#include "../compute_batch/row_cache_map.h"

namespace hyle::compute::apple::detail {
Result<mlx::core::array> encode_row_cache(
    RowCacheEncoding encoding, const mlx::core::array &input,
    mlx::core::Stream stream);
Result<mlx::core::array> gather_decode_row_cache(
    const model_ops::RowCacheGatherDecode &operation,
    const mlx::core::array &bank, const mlx::core::array &indices,
    mlx::core::Stream stream);
Result<mlx::core::array> gather_decode_row_cache_padded(RowCacheEncoding,
    const mlx::core::array&bank,const mlx::core::array&indices,mlx::core::Stream);
Result<std::vector<mlx::core::array>> map_row_cache_pages_padded(const compute::detail::RowCachePaddedMap&,
    const mlx::core::array&indices,const mlx::core::array&ranges,const mlx::core::array&metadata,mlx::core::Stream);
Result<mlx::core::array> map_row_cache_pages(const compute::detail::RowCacheMap&,
    const mlx::core::array&indices,const mlx::core::array&metadata,mlx::core::Stream);
} // namespace hyle::compute::apple::detail
