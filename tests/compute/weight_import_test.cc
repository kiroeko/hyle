#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/weight_import.h"
#include "../../src/compute/weight_assembly.h"

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
using hyle::Status;

c::TensorDesc dense(c::DataType type, std::int64_t rows, std::int64_t columns) {
  return c::TensorDesc::dense(type, std::array{rows, columns}).value();
}
m::Buffer buffer(std::size_t bytes) {
  return m::Buffer::create(m::system_resource(), bytes, 64).value();
}
c::ConstTensorView input(const c::TensorDesc &desc, m::ConstBufferView storage) {
  return c::ConstTensorView::create(desc, storage).value();
}
c::TensorView output(const c::TensorDesc &desc, m::BufferView storage) {
  return c::TensorView::create(desc, storage).value();
}
std::vector<std::byte> contents(m::ConstBufferView storage) {
  std::vector<std::byte> bytes(storage.size());
  EXPECT_TRUE(storage.read(bytes).ok());
  return bytes;
}
std::uint32_t word(std::span<const std::byte> bytes, std::size_t offset) {
  std::uint32_t result;
  std::memcpy(&result, bytes.data() + offset, sizeof(result));
  return result;
}
std::uint32_t oracle(unsigned code) {
  if (code == 0) return 0x00400000u;
  if (code == 255) return 0x7f800001u;
  return std::bit_cast<std::uint32_t>(std::ldexp(1.0F, int(code) - 127));
}

TEST(Float8WeightImport, AllExponentBytesPreserveOriginalWeightAndRawNaN) {
  constexpr std::int64_t n = 1, k = 32768;
  const auto wd = dense(c::DataType::kFloat8E4M3FN, n, k);
  const auto sd = dense(c::DataType::kUInt8, 1, 256);
  const auto yd = c::float8_weight_description(k, n, c::Float8ScaleMode::kFloat32Block128).value();
  auto w = buffer(wd.required_bytes().value());
  auto s = buffer(256);
  auto y = buffer(yd.required_bytes().value());
  std::vector<std::byte> codes(std::size_t(k), std::byte{0});
  for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = std::byte(i & 255u);
  std::array<std::byte, 256> scales{};
  for (unsigned i = 0; i < scales.size(); ++i) scales[i] = std::byte(i);
  ASSERT_TRUE(w.view().write(codes).ok());
  ASSERT_TRUE(s.view().write(scales).ok());
  ASSERT_TRUE(c::import_float8_block128_weight(input(wd, w.view()), input(sd, s.view()),
                                               output(yd, y.view())).ok());
  const auto actual = contents(y.view());
  EXPECT_TRUE(std::equal(codes.begin(), codes.end(), actual.begin()));
  for (unsigned code = 0; code < 256; ++code)
    EXPECT_EQ(word(actual, std::size_t(k) + code * 4), oracle(code)) << code;
  EXPECT_EQ(contents(w.view()), codes);
  EXPECT_EQ(contents(s.view()), (std::vector<std::byte>{scales.begin(), scales.end()}));
}

TEST(Float8WeightImport, PartialNAndSourceStridesKeepPaddingAndAxisOrder) {
  constexpr std::int64_t n = 129, k = 256;
  const auto wd = c::TensorDesc::strided_bytes(c::DataType::kFloat8E4M3FN,
      std::array{n, k}, std::array<std::int64_t, 2>{519, 2}).value();
  const auto sd = c::TensorDesc::strided_bytes(c::DataType::kUInt8,
      std::array<std::int64_t, 2>{2, 2}, std::array<std::int64_t, 2>{9, 3}).value();
  const auto yd = c::float8_weight_description(k, n, c::Float8ScaleMode::kFloat32Block128).value();
  auto w = buffer(wd.required_bytes().value() + 10);
  auto s = buffer(sd.required_bytes().value() + 6);
  auto y = buffer(yd.required_bytes().value() + 32);
  std::vector<std::byte> wb(w.size(), std::byte{0xa5});
  std::vector<std::byte> sb(s.size(), std::byte{0xa5});
  for (std::size_t row = 0; row < std::size_t(n); ++row)
    for (std::size_t col = 0; col < std::size_t(k); ++col)
      wb[5 + row * 519 + col * 2] = std::byte((row * 17 + col * 29) & 255u);
  constexpr std::array<unsigned, 4> scales{0, 127, 1, 254};
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t col = 0; col < 2; ++col) sb[3 + row * 9 + col * 3] = std::byte(scales[row * 2 + col]);
  ASSERT_TRUE(w.view().write(wb).ok());
  ASSERT_TRUE(s.view().write(sb).ok());
  ASSERT_TRUE(y.view().fill(std::byte{0x6d}).ok());
  ASSERT_TRUE(c::import_float8_block128_weight(
      input(wd, w.view().subview(5, wd.required_bytes().value()).value()),
      input(sd, s.view().subview(3, sd.required_bytes().value()).value()),
      output(yd, y.view().subview(16, yd.required_bytes().value()).value())).ok());
  const auto actual = contents(y.view());
  for (std::size_t i = 0; i < 16; ++i) EXPECT_EQ(actual[i], std::byte{0x6d});
  for (std::size_t i = actual.size() - 16; i < actual.size(); ++i) EXPECT_EQ(actual[i], std::byte{0x6d});
  for (std::size_t row = 0; row < std::size_t(n); ++row)
    for (std::size_t col = 0; col < std::size_t(k); ++col)
      ASSERT_EQ(actual[16 + row * std::size_t(k) + col], wb[5 + row * 519 + col * 2]);
  for (unsigned i = 0; i < 4; ++i)
    EXPECT_EQ(word(actual, 16 + std::size_t(n * k) + i * 4), oracle(scales[i]));
  EXPECT_EQ(contents(w.view()), wb);
  EXPECT_EQ(contents(s.view()), sb);
}

TEST(Float8WeightImport, StoredFloat32ScalesPreserveRawBitsAndExplicitSingletonView) {
  constexpr std::int64_t n = 129, k = 384;
  const auto wd = dense(c::DataType::kFloat8E4M3FN, n, k);
  const auto sd = c::TensorDesc::strided_bytes(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 3}, std::array<std::int64_t, 2>{32, 8}).value();
  const auto yd = c::float8_weight_description(k, n, c::Float8ScaleMode::kFloat32Block128).value();
  auto w = buffer(std::size_t(n * k));
  auto s = buffer(sd.required_bytes().value() + 16);
  auto y = buffer(yd.required_bytes().value() + 32);
  std::vector<std::byte> wb(w.size());
  for (std::size_t i = 0; i < wb.size(); ++i) wb[i] = std::byte(i & 255u);
  std::vector<std::byte> sb(s.size(), std::byte{0xa5});
  // No numeric operation may quiet the sNaN, lose tiny values or round SF.
  constexpr std::array<std::uint32_t, 6> scales{
      0x3f807fccu, 0x80000000u, 0x00000001u, 0x7f800001u, 0xbf123456u, 0x7f800000u};
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t column = 0; column < 3; ++column)
      std::memcpy(sb.data() + 8 + row * 32 + column * 8,
                  &scales[row * 3 + column], sizeof(std::uint32_t));
  ASSERT_TRUE(w.view().write(wb).ok());
  ASSERT_TRUE(s.view().write(sb).ok());
  ASSERT_TRUE(y.view().fill(std::byte{0x6d}).ok());
  const auto target = output(yd, y.view().subview(16, yd.required_bytes().value()).value());
  ASSERT_TRUE(c::import_float8_block128_weight(input(wd, w.view()),
      input(sd, s.view().subview(8, sd.required_bytes().value()).value()), target).ok());
  const auto actual = contents(y.view());
  EXPECT_TRUE(std::equal(wb.begin(), wb.end(), actual.begin() + 16));
  for (std::size_t i = 0; i < scales.size(); ++i)
    EXPECT_EQ(word(actual, 16 + std::size_t(n * k) + i * 4), scales[i]);
  EXPECT_TRUE(std::all_of(actual.begin(), actual.begin() + 16,
      [](std::byte b) { return b == std::byte{0x6d}; }));
  EXPECT_TRUE(std::all_of(actual.end() - 16, actual.end(),
      [](std::byte b) { return b == std::byte{0x6d}; }));
  EXPECT_EQ(contents(w.view()), wb);
  EXPECT_EQ(contents(s.view()), sb);

  const auto four = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 4>{2, 1, 3, 1}).value();
  auto contiguous = buffer(sizeof(scales));
  ASSERT_TRUE(contiguous.view().write(std::as_bytes(std::span(scales))).ok());
  const auto original = contents(y.view());
  // Four axes are producer metadata, not an implicit import convention.
  EXPECT_FALSE(c::import_float8_block128_weight(input(wd, w.view()),
      input(four, contiguous.view()), target).ok());
  EXPECT_FALSE(c::import_float8_block128_weight(input(wd, w.view()),
      input(dense(c::DataType::kFloat16, 2, 3), contiguous.view()), target).ok());
  EXPECT_EQ(contents(y.view()), original);
  const auto flat = dense(c::DataType::kFloat32, 2, 3);
  ASSERT_TRUE(c::import_float8_block128_weight(input(wd, w.view()),
      input(flat, contiguous.view()), target).ok());
  EXPECT_EQ(contents(y.view()), original);
}

TEST(Float8WeightImport, InvalidGeometryAndAliasesRejectBeforeWriting) {
  const auto wd = dense(c::DataType::kFloat8E4M3FN, 1, 128);
  const auto sd = dense(c::DataType::kUInt8, 1, 1);
  const auto yd = c::float8_weight_description(128, 1, c::Float8ScaleMode::kFloat32Block128).value();
  auto w = buffer(256), s = buffer(16), y = buffer(256);
  ASSERT_TRUE(w.view().fill(std::byte{0x38}).ok());
  ASSERT_TRUE(s.view().fill(std::byte{127}).ok());
  ASSERT_TRUE(y.view().fill(std::byte{0x6d}).ok());
  const auto original = contents(y.view());
  const auto source = input(wd, w.view());
  const auto scales = input(sd, s.view());
  const auto target = output(yd, y.view());
  EXPECT_FALSE(c::import_float8_block128_weight({}, scales, target).ok());
  EXPECT_FALSE(c::import_float8_block128_weight(
      input(dense(c::DataType::kUInt8, 1, 128), w.view()), scales, target).ok());
  EXPECT_FALSE(c::import_float8_block128_weight(
      source, input(dense(c::DataType::kUInt8, 1, 2), s.view()), target).ok());
  EXPECT_FALSE(c::import_float8_block128_weight(
      input(dense(c::DataType::kFloat8E4M3FN, 1, 129), w.view()), scales, target).ok());
  EXPECT_FALSE(c::import_float8_block128_weight(source, scales,
      output(dense(c::DataType::kUInt8, 1, 132), y.view())).ok());
  EXPECT_FALSE(c::TensorView::create(yd, y.view().subview(0, 131).value()));
  EXPECT_EQ(contents(y.view()), original);
  EXPECT_FALSE(c::import_float8_block128_weight(
      input(wd, y.view()), scales, target).ok());
  EXPECT_FALSE(c::import_float8_block128_weight(source,
      input(sd, y.view().subview(128, 1).value()), target).ok());
  EXPECT_EQ(contents(y.view()), original);
}

class ImportHostResource final : public m::MemoryResource {
 public:
  explicit ImportHostResource(std::span<std::byte> bytes) : bytes_(bytes) {}
  bool identity = true, coherent = true, fail_flush = false, fail_close = false;
  unsigned maps = 0, closes = 0, flushes = 0;
  std::string_view name() const noexcept override { return "import-host-test"; }
  m::MemoryProperties properties() const noexcept override {
    auto flags = m::MemoryProperty::kHostAccessible;
    if (identity) flags |= m::MemoryProperty::kHostAddressIdentity;
    if (coherent) flags |= m::MemoryProperty::kHostDeviceCoherent;
    return {flags, 4, bytes_.size(), {this, 1}};
  }
  hyle::Result<m::ResourceAllocation> allocate(std::size_t, std::size_t) noexcept override {
    return std::unexpected(Status::Unimplemented("test uses adopted host storage"));
  }
  void deallocate(m::ResourceAllocation) noexcept override {}
  hyle::Result<m::ResourceHostMapping> map_host(
      void *, std::size_t offset, std::size_t size, m::HostAccess) noexcept override {
    ++maps;
    if (offset > bytes_.size() || size > bytes_.size() - offset)
      return std::unexpected(Status::OutOfRange("test map range"));
    return m::ResourceHostMapping{bytes_.data() + offset, nullptr};
  }
  Status unmap_host(void *, void *, std::size_t, std::size_t, m::HostAccess) noexcept override {
    ++closes;
    return fail_close ? Status::Unavailable("test import unmap failure") : Status::Ok();
  }
  Status flush(void *, std::size_t, std::size_t) noexcept override {
    ++flushes;
    return fail_flush ? Status::Unavailable("test import flush failure") : Status::Ok();
  }
 private:
  std::span<std::byte> bytes_;
};

TEST(Float8WeightImport, CanonicalAddressProofAndMappingFailuresRemainObservable) {
  const auto wd = dense(c::DataType::kFloat8E4M3FN, 1, 128);
  const auto sd = dense(c::DataType::kUInt8, 1, 1);
  const auto yd = c::float8_weight_description(128, 1, c::Float8ScaleMode::kFloat32Block128).value();
  alignas(64) std::array<std::byte, 256> backing{};
  backing.fill(std::byte{0x6d});
  ImportHostResource source_resource(backing), output_resource(backing);
  int source_token = 0, output_token = 0;
  auto adopted_source = m::Buffer::adopt(m::MemoryResourceRef(source_resource),
      {&source_token, backing.size(), 64}).value();
  auto adopted_output = m::Buffer::adopt(m::MemoryResourceRef(output_resource),
      {&output_token, backing.size(), 64}).value();
  auto scales = buffer(1);
  ASSERT_TRUE(scales.view().fill(std::byte{127}).ok());
  const auto original = backing;
  EXPECT_FALSE(c::import_float8_block128_weight(input(wd, adopted_source.view()),
      input(sd, scales.view()), output(yd, adopted_output.view())).ok());
  EXPECT_EQ(backing, original); // Different resource/handles still physically alias.
  EXPECT_EQ(source_resource.closes, 1u);
  EXPECT_EQ(output_resource.closes, 1u);
  output_resource.identity = false;
  const auto prior_maps = output_resource.maps;
  EXPECT_FALSE(c::import_float8_block128_weight(input(wd, adopted_source.view()),
      input(sd, scales.view()), output(yd, adopted_output.view())).ok());
  EXPECT_EQ(output_resource.maps, prior_maps);
  EXPECT_EQ(backing, original);

  auto weights = buffer(128);
  ASSERT_TRUE(weights.view().fill(std::byte{0x38}).ok());
  output_resource.identity = true;
  output_resource.coherent = false;
  output_resource.fail_flush = true;
  auto status = c::import_float8_block128_weight(input(wd, weights.view()),
      input(sd, scales.view()), output(yd, adopted_output.view()));
  EXPECT_EQ(status.message(), "test import flush failure");
  EXPECT_EQ(output_resource.flushes, 1u);
  output_resource.coherent = true;
  output_resource.fail_flush = false;
  output_resource.fail_close = true;
  status = c::import_float8_block128_weight(input(wd, weights.view()),
      input(sd, scales.view()), output(yd, adopted_output.view()));
  EXPECT_EQ(status.message(), "test import unmap failure");
}

TEST(E8M0ScaleImport, PositiveFiniteThresholdsAcrossAllExponentBins) {
  std::vector<float> values;
  std::vector<std::byte> expected;
  // Numerical midpoint comparisons are independent of the product GRS mask.
  for(int exponent=-126;exponent<=126;++exponent) {
    const auto lower=std::ldexp(1.F,exponent);
    const auto middle=lower*1.5F;
    for(const auto value:std::array{lower,lower*1.25F,
        std::nextafter(middle,0.F),middle,std::nextafter(middle,INFINITY)}) {
      values.push_back(value);
      expected.push_back(std::byte(exponent+127+(value>=middle)));
    }
  }
  for(const auto bits:std::array<std::uint32_t,5>{1,0x003fffff,0x00400000,0x00400001,0x007fffff}) {
    values.push_back(std::bit_cast<float>(bits));
    expected.push_back(std::byte(bits>0x00400000u?1:0));
  }
  const auto count=static_cast<std::int64_t>(values.size());
  auto source=buffer(values.size()*4),target=buffer(values.size());
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
  ASSERT_TRUE(c::encode_e8m0_scales(input(dense(c::DataType::kFloat32,1,count),source.view()),
      output(dense(c::DataType::kUInt8,1,count),target.view())).ok());
  EXPECT_EQ(contents(target.view()),expected);
}

TEST(E8M0ScaleImport, StridedGridAndCompleteValidationBeforeWriting) {
  const auto sd=c::TensorDesc::strided_bytes(c::DataType::kFloat32,
      std::array<std::int64_t,3>{2,2,3},std::array<std::int64_t,3>{92,36,8}).value();
  const auto yd=c::TensorDesc::dense(c::DataType::kUInt8,
      std::array<std::int64_t,3>{2,2,3}).value();
  auto source=buffer(sd.required_bytes().value()),target=buffer(28);
  std::vector<std::byte> original(source.size(),std::byte{0xa5});
  for(std::size_t i=0;i<12;++i) {
    const auto value=std::ldexp(1.F,int(i)-5);
    const auto offset=(i/6)*92+((i/3)%2)*36+(i%3)*8;
    std::memcpy(original.data()+offset,&value,4);
  }
  ASSERT_TRUE(source.view().write(original).ok());
  ASSERT_TRUE(target.view().fill(std::byte{0x6d}).ok());
  const auto destination=output(yd,target.view().subview(8,12).value());
  ASSERT_TRUE(c::encode_e8m0_scales(input(sd,source.view()),destination).ok());
  const auto actual=contents(target.view());
  for(std::size_t i=0;i<actual.size();++i)
    EXPECT_EQ(actual[i],i>=8&&i<20?std::byte(122+i-8):std::byte{0x6d});
  EXPECT_EQ(contents(source.view()),original);
  // Put every invalid value at the last logical cell: no valid prefix is written.
  for(const auto bad:std::array<std::uint32_t,8>{0,0x80000000,0xbf800000,0x7f800000,
      0xff800000,0x7fc00001,0x7f400000,0x7f7fffff}) {
    auto bytes=original;std::memcpy(bytes.data()+92+36+16,&bad,4);
    ASSERT_TRUE(source.view().write(bytes).ok());
    ASSERT_TRUE(target.view().fill(std::byte{0x6d}).ok());
    EXPECT_TRUE(c::encode_e8m0_scales(input(sd,source.view()),destination).IsInvalidArgument());
    EXPECT_EQ(contents(target.view()),std::vector<std::byte>(28,std::byte{0x6d}));
  }
  EXPECT_FALSE(c::encode_e8m0_scales({},destination).ok());
  EXPECT_FALSE(c::encode_e8m0_scales(input(sd,source.view()),
      output(dense(c::DataType::kUInt8,1,12),target.view())).ok());
}

TEST(E8M0ScaleImport, CrossResourceAliasAndFlushCloseFailuresRemainObservable) {
  const auto sd=dense(c::DataType::kFloat32,1,2),yd=dense(c::DataType::kUInt8,1,2);
  alignas(64) std::array<std::byte,64> backing{};
  const std::array<float,2> values{1.25F,1.5F};
  std::memcpy(backing.data(),values.data(),8);
  ImportHostResource a(backing),b(backing);int at=0,bt=0;
  auto ab=m::Buffer::adopt(m::MemoryResourceRef(a),{&at,backing.size(),64}).value();
  auto bb=m::Buffer::adopt(m::MemoryResourceRef(b),{&bt,backing.size(),64}).value();
  const auto original=backing;
  EXPECT_FALSE(c::encode_e8m0_scales(input(sd,ab.view()),output(yd,bb.view())).ok());
  EXPECT_EQ(backing,original);
  auto source=buffer(8);ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
  b.identity=false;const auto prior=b.maps;
  EXPECT_FALSE(c::encode_e8m0_scales(input(sd,source.view()),output(yd,bb.view())).ok());
  EXPECT_EQ(b.maps,prior);
  b.identity=true;b.coherent=false;b.fail_flush=true;
  EXPECT_EQ(c::encode_e8m0_scales(input(sd,source.view()),output(yd,bb.view())).message(),"test import flush failure");
  b.coherent=true;b.fail_flush=false;b.fail_close=true;
  EXPECT_EQ(c::encode_e8m0_scales(input(sd,source.view()),output(yd,bb.view())).message(),"test import unmap failure");
  EXPECT_EQ(backing[0],std::byte{127});EXPECT_EQ(backing[1],std::byte{128});
}

TEST(Int4ZeroImport, OddRowsCrossExpertBoundariesWithoutSourcePadding) {
  constexpr std::int64_t rows=7,groups=3;
  const auto sd=dense(c::DataType::kInt32,1,groups);
  const auto yd=dense(c::DataType::kInt32,3,groups);
  std::array<m::Buffer,3> sources;
  std::array<c::ConstTensorView,3> views;
  for(std::size_t expert=0;expert<sources.size();++expert) {
    std::array<std::uint32_t,groups> codes{};
    for(std::size_t group=0;group<std::size_t(groups);++group) {
      codes[group]=0xf0000000u; // Deliberately poison each expert's padding.
      for(unsigned row=0;row<unsigned(rows);++row)
        codes[group]|=std::uint32_t((expert*5+group*3+row)&15u)<<(4*row);
    }
    sources[expert]=buffer(sizeof(codes));
    ASSERT_TRUE(sources[expert].view().write(std::as_bytes(std::span(codes))).ok());
    views[expert]=input(sd,sources[expert].view());
  }
  auto target=buffer(yd.required_bytes().value()+16);
  ASSERT_TRUE(target.view().fill(std::byte{0x6d}).ok());
  ASSERT_TRUE(c::pack_int4_zero_rows(views,rows,
      output(yd,target.view().subview(8,yd.required_bytes().value()).value())).ok());
  const auto result=contents(target.view());
  for(std::size_t i=0;i<8;++i) {
    EXPECT_EQ(result[i],std::byte{0x6d});EXPECT_EQ(result[result.size()-1-i],std::byte{0x6d});
  }
  for(std::size_t row=0;row<24;++row)for(std::size_t group=0;group<std::size_t(groups);++group) {
    const auto code=(word(result,8+((row/8)*std::size_t(groups)+group)*4)>>(4*(row%8)))&15u;
    EXPECT_EQ(code,row<21?((row/7)*5+group*3+(row%7))&15u:0u)<<row<<":"<<group;
  }
}

TEST(Int4ZeroImport, StridedSourceAndWholeWordFastPathMatch) {
  for(const auto rows:std::array<std::int64_t,3>{1,9,16}) {
    const auto packed=rows/8+(rows%8!=0);
    const auto sd=c::TensorDesc::strided_bytes(c::DataType::kInt32,
        std::array{packed,std::int64_t{2}},std::array<std::int64_t,2>{28,12}).value();
    const auto dense_sd=dense(c::DataType::kInt32,packed,2);
    auto source=buffer(sd.required_bytes().value()),dense_source=buffer(dense_sd.required_bytes().value());
    std::vector<std::byte> physical(source.size(),std::byte{0xa5});
    std::vector<std::uint32_t> codes(static_cast<std::size_t>(packed*2));
    for(std::size_t row=0;row<static_cast<std::size_t>(packed);++row)for(std::size_t group=0;group<2;++group) {
      const auto value=0x8f3205cau+std::uint32_t(row*31+group*97);
      std::memcpy(physical.data()+row*28+group*12,&value,4);codes[row*2+group]=value;
    }
    ASSERT_TRUE(source.view().write(physical).ok());
    ASSERT_TRUE(dense_source.view().write(std::as_bytes(std::span(codes))).ok());
    const auto yd=dense(c::DataType::kInt32,(3*rows)/8+(3*rows%8!=0),2);
    auto target=buffer(yd.required_bytes().value()),other=buffer(target.size());
    const std::array inputs{input(sd,source.view()),input(dense_sd,dense_source.view()),input(sd,source.view())};
    const std::array dense_inputs{input(dense_sd,dense_source.view()),input(dense_sd,dense_source.view()),input(dense_sd,dense_source.view())};
    ASSERT_TRUE(c::pack_int4_zero_rows(inputs,rows,output(yd,target.view())).ok());
    ASSERT_TRUE(c::pack_int4_zero_rows(dense_inputs,rows,output(yd,other.view())).ok());
    EXPECT_EQ(contents(target.view()),contents(other.view()));
    EXPECT_EQ(contents(source.view()),physical);
  }
}

TEST(Int4ZeroImport, GeometryCrossResourceAliasesAndMappingFailuresStayObservable) {
  const auto sd=dense(c::DataType::kInt32,1,2),yd=dense(c::DataType::kInt32,2,2);
  alignas(64) std::array<std::byte,64> backing{};backing.fill(std::byte{0x6d});
  ImportHostResource a(backing),b(backing);int at=0,bt=0;
  auto ab=m::Buffer::adopt(m::MemoryResourceRef(a),{&at,backing.size(),64}).value();
  auto bb=m::Buffer::adopt(m::MemoryResourceRef(b),{&bt,backing.size(),64}).value();
  const std::array aliased{input(sd,ab.view()),input(sd,ab.view())};
  const auto original=backing;
  EXPECT_FALSE(c::pack_int4_zero_rows(aliased,7,output(yd,bb.view())).ok());
  EXPECT_EQ(backing,original);
  EXPECT_FALSE(c::pack_int4_zero_rows({},7,output(yd,bb.view())).ok());
  EXPECT_FALSE(c::pack_int4_zero_rows(aliased,0,output(yd,bb.view())).ok());
  EXPECT_FALSE(c::pack_int4_zero_rows(aliased,INT64_MAX,output(yd,bb.view())).ok());
  auto source=buffer(8);ASSERT_TRUE(source.view().fill(std::byte{0x12}).ok());
  const std::array inputs{input(sd,source.view()),input(sd,source.view())};
  b.identity=false;
  EXPECT_FALSE(c::pack_int4_zero_rows(inputs,7,output(yd,bb.view())).ok());
  EXPECT_EQ(backing,original);
  b.identity=true;b.coherent=false;b.fail_flush=true;
  EXPECT_EQ(c::pack_int4_zero_rows(inputs,7,output(yd,bb.view())).message(),"test import flush failure");
  b.coherent=true;b.fail_flush=false;b.fail_close=true;
  EXPECT_EQ(c::pack_int4_zero_rows(inputs,7,output(yd,bb.view())).message(),"test import unmap failure");
}

struct AssemblySpec final { std::uint64_t id; c::TensorDesc description; };
struct AssemblyEntry final { AssemblySpec spec; std::size_t offset = 0, bytes = 0; };

TEST(WeightAssembly, UnorderedSlicesUseCatalogBankOrderAndExactAlignment) {
  auto source = buffer(24);
  std::array<std::byte,24> original{};
  for (std::size_t i = 0; i < original.size(); ++i) original[i] = std::byte(19 + i * 7);
  ASSERT_TRUE(source.view().write(original).ok());
  const std::array catalog{
      AssemblySpec{3,dense(c::DataType::kUInt8,1,7)},
      AssemblySpec{5,dense(c::DataType::kInt32,1,2)}, // Intentionally not selected.
      AssemblySpec{9,dense(c::DataType::kFloat32,1,3)}};
  const std::array slices{
      c::WeightSlice{9,input(dense(c::DataType::kFloat32,1,1),source.view().subview(16,4).value()),8},
      c::WeightSlice{3,input(dense(c::DataType::kUInt8,1,3),source.view().subview(4,3).value()),4},
      c::WeightSlice{9,input(dense(c::DataType::kFloat32,1,2),source.view().subview(8,8).value()),0},
      c::WeightSlice{3,input(dense(c::DataType::kUInt8,1,4),source.view().subview(0,4).value()),0}};
  std::array<AssemblyEntry,3> entries{};
  const auto result = c::detail::prepare_weight_slices(std::span<const AssemblySpec>(catalog),
      std::span<const c::WeightSlice>(slices),std::span<AssemblyEntry>(entries),28,16);
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(result->banks,2u);
  EXPECT_EQ(result->logical_bytes,19u);
  EXPECT_EQ(result->physical_bytes,28u);
  EXPECT_EQ(entries[0].spec.id,3u); EXPECT_EQ(entries[0].offset,0u); EXPECT_EQ(entries[0].bytes,7u);
  EXPECT_EQ(entries[1].spec.id,9u); EXPECT_EQ(entries[1].offset,16u); EXPECT_EQ(entries[1].bytes,12u);
  const std::span<const AssemblyEntry> selected(entries.data(),result->banks);
  EXPECT_EQ(c::detail::weight_slice_offset(selected,9).value(),16u);
  const auto absent = c::detail::weight_slice_offset(selected,5);
  ASSERT_FALSE(absent);
  EXPECT_TRUE(absent.error().IsNotFound());
  EXPECT_EQ(contents(source.view()),(std::vector<std::byte>{original.begin(),original.end()}));
}

TEST(WeightAssembly, InvalidCoverageTypeOffsetAndCapacityDoNotTouchSources) {
  auto source = buffer(32);
  ASSERT_TRUE(source.view().fill(std::byte{0x6d}).ok());
  const auto original = contents(source.view());
  const std::array catalog{AssemblySpec{7,dense(c::DataType::kFloat32,1,4)}};
  const auto half = input(dense(c::DataType::kFloat32,1,2),source.view());
  const auto whole = input(catalog[0].description,source.view());
  std::array<AssemblyEntry,1> entries{};
  const auto assemble = [&](std::span<const c::WeightSlice> slices,std::size_t capacity = 16) {
    return c::detail::prepare_weight_slices(std::span<const AssemblySpec>(catalog),slices,
        std::span<AssemblyEntry>(entries),capacity,16);
  };
  const auto expect_error = [&](std::span<const c::WeightSlice> slices,
                                hyle::StatusCode code,std::size_t capacity = 16) {
    const auto result = assemble(slices,capacity);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(),code);
  };
  const std::array hole{c::WeightSlice{7,half,0}};
  expect_error(hole,hyle::StatusCode::InvalidArgument);
  const std::array overlap{c::WeightSlice{7,half,0},c::WeightSlice{7,half,4}};
  expect_error(overlap,hyle::StatusCode::InvalidArgument);
  const std::array wrong_type{c::WeightSlice{7,
      input(dense(c::DataType::kInt32,1,4),source.view()),0}};
  expect_error(wrong_type,hyle::StatusCode::InvalidArgument);
  const std::array misaligned{c::WeightSlice{7,half,1}};
  expect_error(misaligned,hyle::StatusCode::OutOfRange);
  const std::array oversized_offset{c::WeightSlice{7,half,SIZE_MAX}};
  expect_error(oversized_offset,hyle::StatusCode::OutOfRange);
  const std::array complete{c::WeightSlice{7,whole,0}};
  expect_error(complete,hyle::StatusCode::ResourceExhausted,15);
  EXPECT_EQ(contents(source.view()),original);
}

TEST(WeightAssembly, CanonicalHostAddressFactIsRequiredBeforeSourceMapping) {
  alignas(64) std::array<std::byte,16> backing{};
  backing.fill(std::byte{0x6d});
  const auto original = backing;
  ImportHostResource resource(backing); resource.identity = false;
  int token = 0;
  auto source = m::Buffer::adopt(m::MemoryResourceRef(resource),
      {&token,backing.size(),64}).value();
  const auto description = dense(c::DataType::kUInt8,1,4);
  const std::array catalog{AssemblySpec{11,description}};
  const std::array slices{c::WeightSlice{11,input(description,source.view()),0}};
  std::array<AssemblyEntry,1> entries{};
  const auto prepare = [&] {
    return c::detail::prepare_weight_slices(std::span<const AssemblySpec>(catalog),
        std::span<const c::WeightSlice>(slices),std::span<AssemblyEntry>(entries),4,16);
  };
  const auto rejected = prepare();
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
  EXPECT_EQ(resource.maps,0u);
  EXPECT_EQ(resource.closes,0u);
  resource.identity = true;
  const auto result = prepare();
  ASSERT_TRUE(result);
  EXPECT_EQ(result->logical_bytes,4u);
  EXPECT_EQ(resource.maps,0u); // This common helper validates metadata only.
  EXPECT_EQ(backing,original);
}

TEST(WeightAssembly, EncodedBanksUseExplicitByteSlicesWithoutScalarReinterpretation) {
  auto source = buffer(192);
  std::vector<std::byte> original(source.size());
  for (std::size_t i = 0; i < original.size(); ++i)
    original[i] = std::byte((i * 37 + 11) & 255u);
  ASSERT_TRUE(source.view().write(original).ok());
  const auto fp4 = c::float4_grouped_weight_description(1, 3, 32).value();
  const auto fp8 = c::float8_weight_description(
      128, 1, c::Float8ScaleMode::kFloat32Block128).value();
  ASSERT_EQ(fp4.required_bytes().value(), 60u);
  ASSERT_EQ(fp8.required_bytes().value(), 132u);
  const std::array catalog{
      AssemblySpec{3, dense(c::DataType::kUInt8, 1, 7)},
      AssemblySpec{7, fp4}, AssemblySpec{11, fp8}};
  const auto raw = [&](std::int64_t bytes, std::size_t offset) {
    return input(dense(c::DataType::kUInt8, 1, bytes),
                 source.view().subview(offset, std::size_t(bytes)).value());
  };
  // Split inside packed bytes/scale storage without treating logical F32 as
  // the source width. The helper validates the complete physical bank only;
  // it neither decodes these arbitrary bytes nor inserts codec padding.
  const std::array slices{
      c::WeightSlice{11, raw(69, 96), 63},
      c::WeightSlice{7, raw(43, 32), 17},
      c::WeightSlice{3, raw(7, 0), 0},
      c::WeightSlice{11, raw(63, 80), 0},
      c::WeightSlice{7, raw(17, 8), 0}};
  std::array<AssemblyEntry, 3> entries{};
  const auto result = c::detail::prepare_weight_slices(
      std::span<const AssemblySpec>(catalog),
      std::span<const c::WeightSlice>(slices),
      std::span<AssemblyEntry>(entries), 212, 16);
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(result->banks, 3u);
  EXPECT_EQ(result->logical_bytes, 199u);
  EXPECT_EQ(result->physical_bytes, 212u);
  EXPECT_EQ(entries[0].offset, 0u);
  EXPECT_EQ(entries[1].offset, 16u);
  EXPECT_EQ(entries[2].offset, 80u);
  EXPECT_EQ(entries[1].bytes, 60u);
  EXPECT_EQ(entries[2].bytes, 132u);
  EXPECT_EQ(entries[1].spec.description, fp4);
  EXPECT_EQ(entries[2].spec.description, fp8);
  EXPECT_EQ(contents(source.view()), original);
}

TEST(WeightAssembly, EncodedBanksRejectTypedEncodedAndStridedSources) {
  auto source = buffer(128);
  ASSERT_TRUE(source.view().fill(std::byte{0x6d}).ok());
  const auto original = contents(source.view());
  const auto fp4 = c::float4_grouped_weight_description(1, 3, 32).value();
  const std::array catalog{AssemblySpec{7, fp4}};
  std::array<AssemblyEntry, 1> entries{};
  const auto reject = [&](const c::TensorDesc &description) {
    const std::array slices{c::WeightSlice{7, input(description, source.view()), 0}};
    const auto result = c::detail::prepare_weight_slices(
        std::span<const AssemblySpec>(catalog),
        std::span<const c::WeightSlice>(slices),
        std::span<AssemblyEntry>(entries), 128, 16);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsInvalidArgument());
  };
  reject(dense(c::DataType::kFloat32, 1, 15)); // Same logical dtype is not raw bytes.
  reject(dense(c::DataType::kInt8, 1, 60));
  reject(fp4); // Even the exact destination format is not an explicit U8 slice.
  reject(c::TensorDesc::strided_bytes(c::DataType::kUInt8,
      std::array<std::int64_t, 2>{2, 15},
      std::array<std::int64_t, 2>{30, 2}).value());
  EXPECT_EQ(contents(source.view()), original);
}

TEST(WeightAssembly, EncodedCoverageAndOriginalBlockAlignmentStayExact) {
  auto source = buffer(64);
  ASSERT_TRUE(source.view().fill(std::byte{0xa7}).ok());
  const auto original = contents(source.view());
  const auto format = c::StorageFormat::blocked(
      c::StorageFormatId{0x48594c45u, 0x5741u}, 32, 18, 64).value();
  const auto block = c::TensorDesc::blocked_dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 64}, format).value();
  ASSERT_EQ(block.required_bytes().value(), 36u);
  const std::array catalog{
      AssemblySpec{1, dense(c::DataType::kUInt8, 1, 3)}, AssemblySpec{9, block}};
  std::array<AssemblyEntry, 2> entries{};
  const auto raw = [&](std::int64_t bytes) {
    return input(dense(c::DataType::kUInt8, 1, bytes), source.view());
  };
  const auto prepare = [&](std::span<const c::WeightSlice> slices,
                           std::size_t capacity = 100) {
    return c::detail::prepare_weight_slices(
        std::span<const AssemblySpec>(catalog), slices,
        std::span<AssemblyEntry>(entries), capacity, 16);
  };
  const std::array complete{c::WeightSlice{9, raw(19), 17},
      c::WeightSlice{1, raw(3), 0}, c::WeightSlice{9, raw(17), 0}};
  const auto accepted = prepare(complete);
  ASSERT_TRUE(accepted) << accepted.error().message();
  EXPECT_EQ(accepted->logical_bytes, 39u);
  EXPECT_EQ(accepted->physical_bytes, 100u);
  EXPECT_EQ(entries[1].offset, 64u); // Retain the bank's alignment, not just16.
  EXPECT_EQ(entries[1].spec.description, block);
  const auto reject = [&](std::span<const c::WeightSlice> slices,
                          hyle::StatusCode code, std::size_t capacity = 100) {
    const auto result = prepare(slices, capacity);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), code);
  };
  const std::array hole{c::WeightSlice{9, raw(17), 0}};
  reject(hole, hyle::StatusCode::InvalidArgument);
  const std::array overlap{c::WeightSlice{9, raw(20), 0},
                          c::WeightSlice{9, raw(19), 17}};
  reject(overlap, hyle::StatusCode::InvalidArgument);
  const std::array beyond{c::WeightSlice{9, raw(19), 18}};
  reject(beyond, hyle::StatusCode::OutOfRange);
  const std::array overflow{c::WeightSlice{9, raw(17), SIZE_MAX}};
  reject(overflow, hyle::StatusCode::OutOfRange);
  reject(complete, hyle::StatusCode::ResourceExhausted, 99);
  EXPECT_EQ(contents(source.view()), original);
}
} // namespace
