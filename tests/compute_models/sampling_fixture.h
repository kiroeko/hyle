#pragma once

// Shared public-only numerical consumer. Each route supplies its original
// Program/Invocation/result owner; no reference calls a production RNG/math.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hyle/compute/sampling_ops.h>

namespace sampling_test {
namespace c = hyle::compute;
using Bytes = std::vector<std::byte>;
using Payloads = std::vector<Bytes>;
using Words = std::array<std::uint32_t, 4>;

inline void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
inline void require(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
template<class T> Bytes pack(std::span<const T> values) {
  const auto bytes = std::as_bytes(values);
  return {bytes.begin(), bytes.end()};
}
template<class T> Bytes pack(const std::vector<T> &values) {
  return pack(std::span<const T>(values));
}
template<class T> std::vector<T> unpack(const Bytes &bytes) {
  check(bytes.size() % sizeof(T) == 0, "output scalar byte size differs");
  std::vector<T> values(bytes.size() / sizeof(T));
  if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}
inline c::TensorDesc dense(c::DataType type,
                           std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, {shape.begin(), shape.size()}));
}
inline std::vector<c::TensorDesc> outputs(const c::Operation &operation,
                                         std::span<const c::TensorDesc> inputs) {
  std::vector<c::TensorDesc> result(c::operation_traits(operation).outputs);
  require(c::infer_output_descriptions(operation, inputs, result));
  return result;
}
struct Output final {
  Payloads bytes;
  // Retains actual native outputs. Exercised after both slots/program close.
  std::function<Payloads()> read_again;
};

// Scalar reference in ordinary arrays, qualified separately by upstream KATs.
inline Words philox(Words counter, std::array<std::uint32_t, 2> key) {
  for (int round = 0; round != 10; ++round) {
    const std::uint64_t p0 = std::uint64_t(counter[0]) * 0xd2511f53;
    const std::uint64_t p1 = std::uint64_t(counter[2]) * 0xcd9e8d57;
    const Words next{std::uint32_t(p1 >> 32) ^ counter[1] ^ key[0],
                     std::uint32_t(p1),
                     std::uint32_t(p0 >> 32) ^ counter[3] ^ key[1],
                     std::uint32_t(p0)};
    counter = next;
    key[0] += 0x9e3779b9;
    key[1] += 0xbb67ae85;
  }
  return counter;
}
inline std::uint32_t word(Words identity, std::uint64_t position) {
  const auto block = position / 4;
  return philox({std::uint32_t(block), std::uint32_t(block >> 32),
                  identity[2], identity[3]}, {identity[0], identity[1]})[position % 4];
}
inline std::vector<Words> identities(std::size_t rows, std::uint64_t draw) {
  std::vector<Words> result(rows);
  for (std::size_t row = 0; row < rows; ++row)
    result[row] = {std::uint32_t(row * 0x9e3779b9ULL),
                   0xfedcba98U ^ std::uint32_t(row * 17),
                   std::uint32_t(draw + row),
                   std::uint32_t((draw + row) >> 32)};
  return result;
}
inline void check_words(const Output &out, std::span<const Words> ids,
                          std::size_t count) {
  check(out.bytes.size() == 1, "random output arity differs");
  const auto actual = unpack<std::uint32_t>(out.bytes[0]);
  check(actual.size() == ids.size() * count, "random output shape differs");
  for (std::size_t row = 0; row < ids.size(); ++row)
    for (std::size_t j = 0; j < count; ++j)
      check(actual[row * count + j] == word(ids[row], j),
            "random output differs from independent Philox addressing");
}
inline double narrow_value(std::uint16_t bits, c::DataType type) {
  if (type == c::DataType::kBFloat16)
    return std::bit_cast<float>(std::uint32_t(bits) << 16);
  const auto exponent = (bits >> 10) & 31, fraction = bits & 1023;
  const auto sign = bits & 0x8000 ? -1. : 1.;
  if (exponent == 31)
    return fraction ? std::numeric_limits<double>::quiet_NaN()
                    : sign * std::numeric_limits<double>::infinity();
  return sign * (exponent ? std::ldexp(1. + fraction / 1024., exponent - 15)
                          : std::ldexp(double(fraction), -24));
}
// Mathematical nearest-even encoder; test sizes using narrow dtypes are small.
inline std::uint16_t narrow_bits(float value, c::DataType type) {
  const bool bf = type == c::DataType::kBFloat16;
  const std::uint16_t infinity = bf ? 0x7f80 : 0x7c00;
  const std::uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  if (std::isnan(value)) return std::uint16_t(infinity | 1);
  const auto magnitude = std::abs(double(value));
  if (std::isinf(magnitude)) return std::uint16_t(sign | infinity);
  std::uint16_t low = 0, high = std::uint16_t(infinity - 1);
  while (low < high) {
    const auto mid = std::uint16_t((unsigned(low) + high + 1) / 2);
    if (narrow_value(mid, type) <= magnitude) low = mid;
    else high = std::uint16_t(mid - 1);
  }
  if (low + 1 < infinity) {
    const auto a = magnitude - narrow_value(low, type);
    const auto b = narrow_value(std::uint16_t(low + 1), type) - magnitude;
    if (b < a || (b == a && (low & 1))) ++low;
  }
  return std::uint16_t(low | sign);
}
inline Bytes stored_logits(std::vector<float> &logits, c::DataType type) {
  if (type == c::DataType::kFloat32) return pack(logits);
  std::vector<std::uint16_t> bits(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    bits[i] = narrow_bits(logits[i], type);
    logits[i] = float(narrow_value(bits[i], type));
  }
  return pack(bits);
}
inline std::vector<std::int32_t> check_categorical(const Output &out,
    std::span<const float> logits, std::size_t rows, std::size_t width,
    bool scored) {
  check(out.bytes.size() == (scored ? 2U : 1U), "categorical result arity differs");
  const auto ids = unpack<std::int32_t>(out.bytes[0]);
  check(ids.size() == rows, "categorical result row count differs");
  std::vector<float> scores;
  if (scored) scores = unpack<float>(out.bytes[1]);
  check(!scored || scores.size() == rows, "selected log probability shape differs");
  for (std::size_t row = 0; row < rows; ++row) {
    check(ids[row] >= 0 && std::size_t(ids[row]) < width, "categorical ID out of range");
    const auto chosen = logits[row * width + std::size_t(ids[row])];
    check(std::isfinite(chosen), "categorical selected masked weight");
    if (!scored) continue;
    const auto start = logits.begin() + std::ptrdiff_t(row * width);
    const auto maximum = *std::max_element(start, start + std::ptrdiff_t(width));
    double total = 0;
    for (std::size_t j = 0; j < width; ++j)
      total += std::exp(double(start[std::ptrdiff_t(j)]) - double(maximum));
    const auto expected = (double(chosen) - double(maximum)) - std::log(total);
    check(std::isfinite(scores[row]) &&
          std::abs(double(scores[row]) - expected) <= 2e-5 + 2e-5 * std::abs(expected),
          "selected log probability differs from independent F64 normalizer");
  }
  return ids;
}

template<class Session> void random_journey() {
  for (const int rows : {0, 1, 17}) for (const unsigned count : {0, 1, 3, 4, 5, 17, 65}) {
    Session session(c::sampling_ops::random_bits(count),
        {dense(c::DataType::kUInt32, {rows, 4})});
    auto ids = identities(std::size_t(rows), UINT64_MAX - 7);
    auto first = take(session.run(Payloads{pack(ids)}, 2));
    check_words(first, ids, count);
    std::reverse(ids.begin(), ids.end());
    check_words(take(session.run(Payloads{pack(ids)}, 1)), ids, count);
    auto repeated = take(session.run(Payloads{pack(ids)}, 0));
    check_words(repeated, ids, count);
    session.close();
    check(first.read_again() == first.bytes, "random native owner changed after close");
  }
  Session session(c::sampling_ops::random_bits(17),
      {dense(c::DataType::kUInt32, {17, 4})});
  Output old;
  for (int step = 0; step < 65; ++step) {
    auto ids = identities(17, 0xffffffffULL + std::uint64_t(step));
    std::rotate(ids.begin(), ids.begin() + step % 17, ids.end());
    auto out = take(session.run(Payloads{pack(ids)}, step == 0 ? 2 : step % 2));
    check_words(out, ids, 17);
    if (step == 0) old = std::move(out);
  }
  session.close();
  check(old.read_again() == old.bytes, "65-step random old native result changed");
  std::puts("PASS random: independent corpus/prefix/high key+draw/reorder/empty/65 steps/two slots/old owner");
}

template<class Session> void categorical_journey() {
  constexpr auto f32 = c::DataType::kFloat32;
  for (const auto type : {f32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    constexpr int rows = 13, width = 17;
    const std::vector<c::TensorDesc> desc{dense(type, {rows, width}),
                                         dense(c::DataType::kUInt32, {rows, 4})};
    Session plain(c::sampling_ops::categorical(), desc);
    Session scored(c::sampling_ops::categorical(true), desc);
    std::vector<float> logits(rows * width);
    for (int r = 0; r < rows; ++r) for (int j = 0; j < width; ++j)
      logits[r * width + j] = j % 5 == 0 ? -INFINITY :
          float((r * 7 + j) % 11 - 5) * .25F + (r % 3 == 0 ? 1024.F : -1024.F);
    const auto payload = stored_logits(logits, type);
    auto keys = identities(rows, 0xfedcba9876543210ULL);
    auto first = take(scored.run({payload, pack(keys)}, 2));
    const auto ids = check_categorical(first, logits, rows, width, true);
    const auto unscored = take(plain.run({payload, pack(keys)}, 1));
    check(unpack<std::int32_t>(unscored.bytes[0]) == ids,
          "requesting log probability changed the draw");
    std::reverse(keys.begin(), keys.end());
    for (int r = 0; r < rows / 2; ++r)
      for (int j = 0; j < width; ++j) std::swap(logits[r * width + j], logits[(rows-1-r)*width+j]);
    const auto reordered = take(scored.run({stored_logits(logits, type), pack(keys)}, 1));
    auto expected = ids; std::reverse(expected.begin(), expected.end());
    check(check_categorical(reordered, logits, rows, width, true) == expected,
          "categorical row permutation changed request draw");
    scored.close(); plain.close();
    check(first.read_again() == first.bytes, "categorical old native result changed after close");
  }
  // Large real vocabulary plus independent live-input/slot sequence. Only
  // selected ID/scalar outputs are downloaded; logits stay on the route.
  for (const int rows : {1, 16}) {
    constexpr int width = 248320;
    Session session(c::sampling_ops::categorical(true),
        {dense(f32, {rows, width}), dense(c::DataType::kUInt32, {rows, 4})});
    Output old;
    std::vector<float> logits(std::size_t(rows) * width, -INFINITY);
    for (int step = 0; step < 65; ++step) {
      std::fill(logits.begin(), logits.end(), -INFINITY);
      for (int r = 0; r < rows; ++r) {
        const auto j = (std::size_t(step) * 7919 + std::size_t(r) * 131) % width;
        logits[std::size_t(r) * width + j] = step % 2 ? 1e20F : -1e20F;
      }
      const auto keys = identities(std::size_t(rows), std::uint64_t(step));
      auto result = take(session.run({pack(logits), pack(keys)}, step == 0 ? 2 : step % 2));
      const auto ids = check_categorical(result, logits, std::size_t(rows), width, true);
      for (int r = 0; r < rows; ++r)
        check(std::size_t(ids[std::size_t(r)]) ==
              (std::size_t(step)*7919 + std::size_t(r)*131) % width,
              "full-vocabulary one-positive-mass selection differs");
      if (step == 0) old = std::move(result);
    }
    session.close();
    check(old.read_again() == old.bytes, "full vocabulary old output changed");
  }
  // Actual checked payload failures, followed by the original successful
  // key/draw again: failure cannot advance hidden generator state.
  Session session(c::sampling_ops::categorical(true),
      {dense(f32, {3, 7}), dense(c::DataType::kUInt32, {3, 4})});
  auto keys = identities(3, 73);
  std::vector<float> valid(21, .25F);
  auto before = take(session.run({pack(valid), pack(keys)}, 2));
  for (float invalid : {INFINITY, std::numeric_limits<float>::quiet_NaN(), -INFINITY}) {
    auto bad = valid;
    if (invalid == -INFINITY) std::fill(bad.begin()+7, bad.begin()+14, -INFINITY);
    else bad[9] = invalid;
    auto failed = session.run({pack(bad), pack(keys)}, 1);
    check(!failed && failed.error().IsInvalidArgument(), "invalid row did not fail checked completion");
    auto after = take(session.run({pack(valid), pack(keys)}, 0));
    check(after.bytes == before.bytes, "failed draw changed later successful result");
    check(before.read_again() == before.bytes, "failed draw corrupted old native output");
  }
  session.close();
  Session empty(c::sampling_ops::categorical(true),
      {dense(f32, {0, 7}), dense(c::DataType::kUInt32, {0, 4})});
  check_categorical(take(empty.run({{}, {}}, 0)), {}, 0, 7, true);
  empty.close();
  std::puts("PASS categorical: narrow storage/F64 logprob/reorder/large offset/real vocabulary/65 steps/two slots/checked failures/old owner");
}

template<class Session> void categorical_quality() {
  constexpr int rows = 4096, width = 7, batches = 64;
  constexpr double count = double(rows) * batches;
  for (const std::array<float, width> law : {
      std::array<float, width>{0, 0, 0, 0, 0, 0, 0},
      std::array<float, width>{-2, -1, 0, 1, 2, -.5F, .5F},
      std::array<float, width>{-INFINITY, -2, -INFINITY, 0, 1, -INFINITY, .5F}}) {
    Session session(c::sampling_ops::categorical(true),
        {dense(c::DataType::kFloat32, {rows, width}),
         dense(c::DataType::kUInt32, {rows, 4})});
    std::vector<float> logits(rows * width);
    for (int r = 0; r < rows; ++r) std::copy(law.begin(), law.end(), logits.begin() + r * width);
    std::array<std::uint64_t, width> buckets{};
    for (int batch = 0; batch < batches; ++batch) {
      const auto keys = identities(rows, std::uint64_t(batch) * rows);
      auto output = take(session.run({pack(logits), pack(keys)}, batch % 2));
      for (const auto id : check_categorical(output, logits, rows, width, true)) ++buckets[std::size_t(id)];
    }
    double sum = 0; for (const auto value : law) sum += std::exp(double(value));
    for (int j = 0; j < width; ++j) {
      const auto p = std::exp(double(law[std::size_t(j)])) / sum;
      if (p == 0) check(buckets[std::size_t(j)] == 0, "masked bucket has samples");
      else {
        check(count * p >= 20, "quality fixture has unqualified rare bucket");
        check(std::abs(double(buckets[std::size_t(j)]) - count * p) <=
              7 * std::sqrt(count * p * (1-p)) + 8,
              "categorical fixed-seed distribution exceeds frozen seven-sigma gate");
      }
    }
    session.close();
  }
  std::puts("PASS quality: 262144 fixed-seed samples per uniform/nonuniform/masked law; every bucket checked");
}
template<class Session> void topk_selection() {
  for (const auto &[rows, width] : {std::pair{5,17},std::pair{1,248320},std::pair{16,248320}}) {
    std::vector<float> values(std::size_t(rows) * std::size_t(width));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = float(int((i * 131 + 37) % 1709) - 854) * .125F;
    if (width == 17) {
      const std::array<std::uint32_t,17> edges{0x7fc01234,0xffc05678,0,0x80000000,
          1,2,0x80000001,0x80000002,0x7f800000,0xff800000,
          0x3f800000,0x3f800000,0xbf800000,0xbf800000,0x00800000,0x007fffff,0x7fc00001};
      for (int j = 0; j < width; ++j) values[std::size_t(j)] = std::bit_cast<float>(edges[std::size_t(j)]);
      for (int j = 0; j < width; ++j) values[std::size_t(width+j)] = std::bit_cast<float>(0x7fc01000U+unsigned(j));
      for (int j = 0; j < width; ++j) values[std::size_t(2*width+j)] = j%2 ? -0.F : 0.F;
      for (int j = 0; j < width; ++j) values[std::size_t(3*width+j)] = std::bit_cast<float>(unsigned(j+1));
      for (int j = 0; j < width; ++j) values[std::size_t(4*width+j)] = std::bit_cast<float>(0x80000001U+unsigned(j));
    }
    for (const bool largest : {false,true}) for (const int count : {1,3}) {
      if (count != 1 && width != 17) continue;
      Session session(c::Operation(c::TopK{.count=std::size_t(count),.largest=largest}),
          {dense(c::DataType::kFloat32,{rows,width})});
      auto result = take(session.run({pack(values)},0));
      check(result.bytes.size()==2,"TopK did not return both ports");
      const auto bits=unpack<std::uint32_t>(result.bytes[0]);
      const auto ids=unpack<std::int32_t>(result.bytes[1]);
      for (int r=0;r<rows;++r) {
        std::vector<std::int32_t> order(std::size_t(width),0);
        for (int j=0;j<width;++j) order[std::size_t(j)]=j;
        std::partial_sort(order.begin(),order.begin()+count,order.end(),[&](int a,int b){
          const auto x=values[std::size_t(r*width+a)],y=values[std::size_t(r*width+b)];
          if (std::isnan(x)!=std::isnan(y)) return !std::isnan(x);
          if (!std::isnan(x) && x!=y) return largest ? x>y : x<y;
          return a<b;
        });
        for (int j=0;j<count;++j) {
          const auto expected=order[std::size_t(j)];
          const auto index=std::size_t(r*count+j);
          check(ids[index]==expected,"TopK NaN-last/first tie/numeric order differs");
          check(bits[index]==std::bit_cast<std::uint32_t>(values[std::size_t(r*width+expected)]),
                "TopK changed selected source storage bits");
        }
      }
      session.close();
      check(result.read_again()==result.bytes,"TopK native result changed after close");
    }
  }
  std::puts("PASS TopK: both outputs/NaN payload/ties/signed zero/subnormals/k>1 holdout/full R1+R16 vocabulary");
}
template<class Session> void run() {
  topk_selection<Session>();
  random_journey<Session>();
  categorical_journey<Session>();
  categorical_quality<Session>();
}
} // namespace sampling_test
