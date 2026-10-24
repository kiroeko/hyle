#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace gather_elements_test {
// Different rows and repeated per-row IDs expose Cartesian/flattened-index
// mistakes. K>E is legal: the selected IDs, not their count, need to fit E.
inline constexpr std::size_t kRows=3,kColumns=5,kSelected=7;
inline std::array<double,kRows*kColumns> data(int seed=0){
  std::array<double,kRows*kColumns> result{};
  for(std::size_t row=0;row<kRows;++row)for(std::size_t col=0;col<kColumns;++col)
    result[row*kColumns+col]=(double(row*17+col*3+seed*7)-23)/8;
  return result;
}
inline std::array<std::int32_t,kRows*kSelected> indices(int seed=0){
  std::array<std::int32_t,kRows*kSelected> result{};
  for(std::size_t row=0;row<kRows;++row)for(std::size_t col=0;col<kSelected;++col)
    result[row*kSelected+col]=std::int32_t((row*2+col*3+std::size_t(seed))%kColumns);
  return result;
}
inline std::array<double,kRows*kSelected> reference(std::span<const double> source,
    std::span<const std::int32_t> ids){
  std::array<double,kRows*kSelected> result{};
  for(std::size_t row=0;row<kRows;++row)for(std::size_t col=0;col<kSelected;++col)
    result[row*kSelected+col]=source[row*kColumns+std::size_t(ids[row*kSelected+col])];
  return result;
}
} // namespace gather_elements_test
