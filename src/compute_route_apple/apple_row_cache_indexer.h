#pragma once
#include "hyle/compute/apple.h"
#include "../compute_batch/row_cache_map.h"
namespace hyle::compute::apple::detail {
Result<mlx::core::array> row_cache_indexer_scores(
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
Result<mlx::core::array> row_cache_paged_indexer_scores(
    const compute::detail::RowCachePagedIndexerScores &,
    std::span<const mlx::core::array>, mlx::core::Stream);
}
