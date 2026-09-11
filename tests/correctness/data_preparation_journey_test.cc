#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/container/base64.h"
#include "hyle/math/scalar.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/ops.h"

TEST(CorrectnessJourney, DataPreparationOwnsTransformsAndReleasesScratch) {
  constexpr std::string_view encoded = "AQIDBAUGBwg=";
  const auto decoded_size = hyle::container::base64_decoded_size(
      std::span<const char>{encoded.data(), encoded.size()});
  ASSERT_TRUE(decoded_size) << static_cast<int>(decoded_size.status);
  ASSERT_EQ(decoded_size.output_size, 8u);

  const auto scratch_size =
      hyle::math::try_bit_ceil(decoded_size.output_size);
  ASSERT_TRUE(scratch_size.has_value());
  auto created = hyle::mem::Buffer::create(
      hyle::mem::system_resource(), *scratch_size, alignof(std::int8_t));
  ASSERT_TRUE(created.has_value()) << created.error().message();
  hyle::mem::Buffer scratch = std::move(*created);

  auto mapped = scratch.view().map_host(hyle::mem::HostAccess::kReadWrite);
  ASSERT_TRUE(mapped.has_value()) << mapped.error().message();
  auto writable = mapped->writable_bytes();
  ASSERT_TRUE(writable.has_value()) << writable.error().message();
  const auto decoded = hyle::container::base64_decode(
      std::span<const char>{encoded.data(), encoded.size()}, *writable);
  ASSERT_TRUE(decoded) << static_cast<int>(decoded.status);
  ASSERT_EQ(decoded.output_count, decoded_size.output_size);

  const auto* values =
      reinterpret_cast<const std::int8_t*>(writable->data());
  EXPECT_EQ(hyle::simd::dot(values, values, decoded.output_count), 204);
  EXPECT_TRUE(mapped->flush().ok());
  EXPECT_TRUE(mapped->close().ok());

  std::array<std::byte, 8> uncommitted{};
  uncommitted.fill(std::byte{0x7f});
  constexpr std::string_view malformed = "AQIDBAUGBwg!";
  const auto rejected = hyle::container::base64_decode(
      std::span<const char>{malformed.data(), malformed.size()}, uncommitted);
  EXPECT_FALSE(rejected);
  EXPECT_NE(rejected.status, hyle::container::Base64Status::kSuccess);

  EXPECT_TRUE(scratch.reset().ok());
  EXPECT_FALSE(scratch.valid());
}
