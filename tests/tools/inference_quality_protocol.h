#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace inference_quality {
inline void binary_stdout() {
  static_assert(std::endian::native==std::endian::little && sizeof(float)==4 &&
      std::numeric_limits<float>::is_iec559);
#if defined(_WIN32)
  if (_setmode(_fileno(stdout),_O_BINARY)==-1) throw std::runtime_error("binary stdout failed");
#endif
}
template<class T> void scalar(T value) {
  std::cout.write(reinterpret_cast<const char *>(&value),sizeof(value));
}
inline std::int32_t top1(std::span<const float> values) {
  if (values.empty()) throw std::runtime_error("empty logits");
  std::int32_t best=0;
  for (std::size_t i=0;i<values.size();++i) {
    if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite logits");
    if (values[i]>values[std::size_t(best)]) best=static_cast<std::int32_t>(i);
  }
  return best;
}
// Little-endian 56-byte header followed by one F32 row only for kind=2.
// Kinds:1 ready,2 row,3 reset/retirement acknowledgement,4 clean exit,
// 5 tokenized prompt (length int32 tokens follow the header).
// Flags:bit1 means EOS is known,bit0 means the row's top1 is EOS.
inline void emit(std::uint32_t kind,std::uint32_t vocabulary,
    std::uint64_t case_id=0,std::uint64_t position=0,std::uint64_t length=0,
    std::uint64_t rng=0,std::span<const float> logits={},std::uint32_t flags=0) {
  constexpr char magic[8]={'H','Y','Q','L','0','0','0','1'};
  std::cout.write(magic,8);scalar(kind);scalar(vocabulary);scalar(case_id);
  scalar(position);scalar(length);scalar(rng);
  scalar(logits.empty()?std::int32_t{-1}:top1(logits));scalar(flags);
  if (kind==2) {
    if (logits.size()!=vocabulary) throw std::runtime_error("logit vocabulary differs");
    std::cout.write(reinterpret_cast<const char *>(logits.data()),std::streamsize(logits.size_bytes()));
  }
  std::cout.flush();
  if (!std::cout) throw std::runtime_error("quality stream write failed");
}
} // namespace inference_quality
