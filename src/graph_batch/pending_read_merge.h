#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include "hyle/mem/buffer.h"

namespace hyle::graph::batch::detail {

// Inputs are distinct, sorted views of the same original backing. For each
// offset retain its longest original prefix, without joining adjacent ranges.
template<class Allocator>
void merge_pending_reads(std::vector<mem::ConstBufferView, Allocator> &reads,
                         std::span<const mem::ConstBufferView> other) {
  if (other.size() > reads.max_size() - reads.size())
    throw std::length_error("paged append pending reader count overflows");
  const auto first_size = reads.size();
  reads.resize(first_size + other.size());
  auto first = first_size, second = other.size(), out = reads.size();
  const auto before = [](auto left, auto right) noexcept {
    return left.offset() < right.offset() ||
        (left.offset() == right.offset() && left.size() > right.size());
  };
  while (first && second) {
    if (before(reads[first - 1], other[second - 1]))
      reads[--out] = other[--second];
    else
      reads[--out] = reads[--first];
  }
  while (second) reads[--out] = other[--second];
  reads.erase(std::unique(reads.begin(), reads.end(), [](auto left, auto right) noexcept {
    return left.offset() == right.offset();
  }), reads.end());
}

} // namespace hyle::graph::batch::detail
