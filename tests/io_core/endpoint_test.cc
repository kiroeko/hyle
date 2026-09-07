#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string_view>

#include "hyle/io/endpoint.h"

namespace {

using hyle::io::AddressFamily;
using hyle::io::Endpoint;

std::string_view formatted(const Endpoint& endpoint, bool with_port,
                           std::array<char, 96>& storage) {
  auto result = endpoint.format(storage.data(), storage.size(), with_port);
  EXPECT_TRUE(result.has_value());
  return result.has_value() ? std::string_view(storage.data(), *result)
                            : std::string_view{};
}

TEST(IoEndpoint, ParsesAndFormatsIpv4WithoutNativePadding) {
  auto parsed = Endpoint::parse("127.0.0.1", 2048);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->family(), AddressFamily::kIpv4);
  EXPECT_EQ(parsed->port(), 2048);
  EXPECT_EQ(parsed->scope_id(), 0u);
  EXPECT_TRUE(parsed->is_loopback());
  EXPECT_FALSE(parsed->is_unspecified());

  std::array<char, 96> text{};
  EXPECT_EQ(formatted(*parsed, true, text), "127.0.0.1:2048");
}

TEST(IoEndpoint, ParsesBracketedIpv6AndNumericScope) {
  auto parsed = Endpoint::parse("[fe80::1%7]", 443);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->family(), AddressFamily::kIpv6);
  EXPECT_EQ(parsed->scope_id(), 7u);

  std::array<char, 96> text{};
  EXPECT_EQ(formatted(*parsed, true, text), "[fe80::1%7]:443");
}

TEST(IoEndpoint, AnyAndLoopbackAreTypedValues) {
  Endpoint any = Endpoint::any(AddressFamily::kIpv6, 80);
  Endpoint loopback = Endpoint::loopback(AddressFamily::kIpv6, 80);
  EXPECT_TRUE(any.is_unspecified());
  EXPECT_FALSE(any.is_loopback());
  EXPECT_TRUE(loopback.is_loopback());
  EXPECT_NE(any, loopback);
}

TEST(IoEndpoint, InvalidInputReturnsCanonicalStatus) {
  auto hostname = Endpoint::parse("model.internal", 80);
  ASSERT_FALSE(hostname.has_value());
  EXPECT_TRUE(hostname.error().IsInvalidArgument());

  auto bad_scope = Endpoint::parse("fe80::1%eth0", 80);
  ASSERT_FALSE(bad_scope.has_value());
  EXPECT_TRUE(bad_scope.error().IsInvalidArgument());
}

TEST(IoEndpoint, BufferFillReportsFullSizeAndTerminates) {
  auto parsed = Endpoint::parse("127.0.0.1", 2048);
  ASSERT_TRUE(parsed.has_value());
  std::array<char, 5> small{};
  auto result = parsed->format(small.data(), small.size());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, std::string_view("127.0.0.1:2048").size());
  EXPECT_EQ(small.back(), '\0');
  EXPECT_STREQ(small.data(), "127.");

  auto size_only = parsed->format(nullptr, 0);
  ASSERT_TRUE(size_only.has_value());
  EXPECT_EQ(*size_only, *result);
}

}  // namespace
