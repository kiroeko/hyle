#include "hyle/container/base64.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hyle::container {
namespace {

std::span<const char> chars(std::string_view value) noexcept {
  return {value.data(), value.size()};
}

TEST(Base64, Rfc4648StandardVectors) {
  struct Vector {
    std::string_view raw;
    std::string_view encoded;
  };
  constexpr Vector kVectors[] = {
      {"", ""},       {"f", "Zg=="},     {"fo", "Zm8="},
      {"foo", "Zm9v"}, {"foob", "Zm9vYg=="},
      {"fooba", "Zm9vYmE="}, {"foobar", "Zm9vYmFy"},
  };

  for (const Vector& vector : kVectors) {
    const Base64SizeResult encoded_size =
        base64_encoded_size(vector.raw.size());
    ASSERT_TRUE(encoded_size) << vector.raw;
    ASSERT_EQ(encoded_size.output_size, vector.encoded.size()) << vector.raw;

    std::string encoded(encoded_size.output_size + 1, '?');
    const Base64Result encode = base64_encode(
        chars(vector.raw), std::span<char>{encoded.data(), encoded_size.output_size});
    ASSERT_TRUE(encode) << vector.raw;
    EXPECT_EQ(encode.input_count, vector.raw.size());
    EXPECT_EQ(encode.output_count, vector.encoded.size());
    EXPECT_EQ(std::string_view(encoded.data(), encode.output_count),
              vector.encoded);
    EXPECT_EQ(encoded.back(), '?');

    const Base64SizeResult decoded_size =
        base64_decoded_size(chars(vector.encoded));
    ASSERT_TRUE(decoded_size) << vector.encoded;
    ASSERT_EQ(decoded_size.output_size, vector.raw.size());

    std::string decoded(decoded_size.output_size, '\0');
    const Base64Result decode =
        base64_decode(chars(vector.encoded), std::span<char>{decoded});
    ASSERT_TRUE(decode) << vector.encoded;
    EXPECT_EQ(decode.input_count, vector.encoded.size());
    EXPECT_EQ(decode.output_count, vector.raw.size());
    EXPECT_EQ(std::string_view(decoded.data(), decode.output_count),
              vector.raw);
  }
}

TEST(Base64, UrlAlphabetAndEncodePaddingAreExplicit) {
  const std::array<std::byte, 2> raw = {std::byte{0xfb}, std::byte{0xff}};
  std::array<char, 4> output{};

  Base64EncodeOptions url_padded{
      .alphabet = Base64Alphabet::kUrl,
      .padding = Base64EncodePadding::kInclude,
  };
  Base64Result result = base64_encode(raw, output, url_padded);
  ASSERT_TRUE(result);
  EXPECT_EQ(std::string_view(output.data(), result.output_count), "-_8=");

  Base64EncodeOptions url_unpadded{
      .alphabet = Base64Alphabet::kUrl,
      .padding = Base64EncodePadding::kOmit,
  };
  EXPECT_EQ(base64_encoded_size(raw.size(), url_unpadded).output_size, 3u);
  result = base64_encode(raw, output, url_unpadded);
  ASSERT_TRUE(result);
  EXPECT_EQ(std::string_view(output.data(), result.output_count), "-_8");

  std::array<std::byte, 2> decoded{};
  Base64DecodeOptions url_required{
      .alphabet = Base64Alphabet::kUrl,
      .padding = Base64DecodePadding::kRequired,
      .whitespace = Base64Whitespace::kReject,
  };
  result = base64_decode(chars("-_8="), decoded, url_required);
  ASSERT_TRUE(result);
  EXPECT_EQ(decoded, raw);

  EXPECT_EQ(base64_decode(chars("+/8="), decoded, url_required).status,
            Base64Status::kInvalidCharacter);
}

TEST(Base64, DecodePaddingPoliciesRemainCanonical) {
  std::array<char, 4> output{};
  const Base64DecodeOptions required{};
  const Base64DecodeOptions allowed{
      .alphabet = Base64Alphabet::kStandard,
      .padding = Base64DecodePadding::kAllowed,
      .whitespace = Base64Whitespace::kReject,
  };
  const Base64DecodeOptions forbidden{
      .alphabet = Base64Alphabet::kStandard,
      .padding = Base64DecodePadding::kForbidden,
      .whitespace = Base64Whitespace::kReject,
  };

  EXPECT_TRUE(base64_decode(chars("Zg=="), output, required));
  EXPECT_FALSE(base64_decode(chars("Zg"), output, required));
  EXPECT_TRUE(base64_decode(chars("Zg=="), output, allowed));
  EXPECT_TRUE(base64_decode(chars("Zg"), output, allowed));
  EXPECT_EQ(base64_decode(chars("Zg=="), output, forbidden).status,
            Base64Status::kInvalidPadding);
  EXPECT_TRUE(base64_decode(chars("Zg"), output, forbidden));
  EXPECT_EQ(base64_decode(chars("Zg=A"), output, allowed).status,
            Base64Status::kInvalidCharacter);
  EXPECT_EQ(base64_decode(chars("Zg==="), output, allowed).status,
            Base64Status::kInvalidCharacter);

  EXPECT_EQ(base64_decoded_size(chars("Zg"), required).status,
            Base64Status::kInvalidPadding);
  EXPECT_EQ(base64_decoded_size(chars("Zg=="), forbidden).status,
            Base64Status::kInvalidPadding);
  EXPECT_EQ(base64_decoded_size(chars("Z==="), allowed).status,
            Base64Status::kInvalidPadding);
  EXPECT_EQ(base64_decoded_size(chars("Zg=A"), allowed).status,
            Base64Status::kInvalidPadding);

  // Both padded and unpadded modes reject non-zero unused bits.
  EXPECT_EQ(base64_decode(chars("Zh=="), output, required).status,
            Base64Status::kNonCanonicalBits);
  EXPECT_EQ(base64_decode(chars("Zh"), output, allowed).status,
            Base64Status::kNonCanonicalBits);
  const Base64SizeResult detail = base64_decoded_size(chars("Zh"), allowed);
  EXPECT_EQ(detail.status, Base64Status::kNonCanonicalBits);
  EXPECT_EQ(detail.input_count, 1u);
}

TEST(Base64, WhitespacePolicyIsIndependentFromPadding) {
  constexpr std::string_view kSpaced = " Z\tg=\n=\f";
  std::array<char, 1> output{};

  const Base64Result rejected = base64_decode(chars(kSpaced), output);
  EXPECT_EQ(rejected.status, Base64Status::kWhitespaceRejected);
  EXPECT_EQ(rejected.input_count, 0u);
  EXPECT_EQ(rejected.output_count, 0u);

  const Base64DecodeOptions accepting{
      .alphabet = Base64Alphabet::kStandard,
      .padding = Base64DecodePadding::kRequired,
      .whitespace = Base64Whitespace::kAllowAscii,
  };
  const Base64SizeResult size = base64_decoded_size(chars(kSpaced), accepting);
  ASSERT_TRUE(size);
  EXPECT_EQ(size.output_size, 1u);
  const Base64Result decoded = base64_decode(chars(kSpaced), output, accepting);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded.input_count, kSpaced.size());
  ASSERT_EQ(decoded.output_count, 1u);
  EXPECT_EQ(output[0], 'f');

  EXPECT_EQ(base64_decoded_size(chars("Zg\v=="), accepting).status,
            Base64Status::kInvalidCharacter);
}

TEST(Base64, CheckedSizeReportsStableFailureOffsets) {
  Base64SizeResult result = base64_decoded_size(chars("Zm#v"));
  EXPECT_EQ(result.status, Base64Status::kInvalidCharacter);
  EXPECT_EQ(result.input_count, 2u);

  result = base64_decoded_size(chars("A==="));
  EXPECT_EQ(result.status, Base64Status::kInvalidPadding);
  EXPECT_EQ(result.input_count, 3u);

  result = base64_decoded_size(chars("A"));
  EXPECT_EQ(result.status, Base64Status::kInvalidLength);
  EXPECT_EQ(result.input_count, 1u);

  result = base64_decoded_size(chars("AA"));
  EXPECT_EQ(result.status, Base64Status::kInvalidPadding);
  EXPECT_EQ(result.input_count, 2u);
}

TEST(Base64, SizeHelpersHandleBoundsWithoutAllocation) {
  EXPECT_EQ(base64_max_decoded_size(0), 0u);
  EXPECT_EQ(base64_max_decoded_size(2), 1u);
  EXPECT_EQ(base64_max_decoded_size(3), 2u);
  EXPECT_EQ(base64_max_decoded_size(4), 3u);
  EXPECT_EQ(base64_max_decoded_size(5), 3u);

  const Base64SizeResult overflow =
      base64_encoded_size(std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(overflow.status, Base64Status::kSizeOverflow);

  const Base64EncodeOptions bad_encode{
      .alphabet = static_cast<Base64Alphabet>(0xff),
  };
  EXPECT_EQ(base64_encoded_size(3, bad_encode).status,
            Base64Status::kInvalidArgument);
  const Base64DecodeOptions bad_decode{
      .padding = static_cast<Base64DecodePadding>(0xff),
  };
  EXPECT_EQ(base64_decoded_size(chars(""), bad_decode).status,
            Base64Status::kInvalidArgument);
}

TEST(Base64, SmallOutputAndOverlapAreTypedFailures) {
  std::array<char, 8> encoded;
  encoded.fill('!');
  const Base64Result encode = base64_encode(chars("foobar"),
                                            std::span<char>{encoded}.first(7));
  EXPECT_EQ(encode.status, Base64Status::kOutputTooSmall);
  EXPECT_EQ(encode.output_count, 0u);
  EXPECT_EQ(encoded[0], '!');

  std::array<std::byte, 2> decoded{};
  const Base64Result decode = base64_decode(chars("Zm9v"), decoded);
  EXPECT_EQ(decode.status, Base64Status::kOutputTooSmall);
  EXPECT_LE(decode.output_count, decoded.size());

  std::array<char, 16> shared{};
  std::copy_n("foobar", 6, shared.data());
  EXPECT_EQ(base64_encode(
                std::as_bytes(std::span<const char>{shared.data(), 6}), shared)
                .status,
            Base64Status::kInvalidArgument);
  EXPECT_EQ(base64_decode(
                std::span<const char>{shared.data(), 4},
                std::as_writable_bytes(std::span<char>{shared}))
                .status,
            Base64Status::kInvalidArgument);
}

TEST(Base64, EveryByteRoundTripsThroughBothAlphabets) {
  std::array<std::uint8_t, 256> raw{};
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint8_t>(i);
  }

  for (Base64Alphabet alphabet :
       {Base64Alphabet::kStandard, Base64Alphabet::kUrl}) {
    for (Base64EncodePadding encode_padding :
         {Base64EncodePadding::kInclude, Base64EncodePadding::kOmit}) {
      const Base64EncodeOptions encode_options{alphabet, encode_padding};
      const Base64SizeResult size =
          base64_encoded_size(raw.size(), encode_options);
      ASSERT_TRUE(size);
      std::vector<char> encoded(size.output_size);
      const Base64Result encode = base64_encode(
          std::as_bytes(std::span<const std::uint8_t>{raw}), encoded,
          encode_options);
      ASSERT_TRUE(encode);

      const Base64DecodeOptions decode_options{
          .alphabet = alphabet,
          .padding = encode_padding == Base64EncodePadding::kInclude
                         ? Base64DecodePadding::kRequired
                         : Base64DecodePadding::kForbidden,
          .whitespace = Base64Whitespace::kReject,
      };
      std::array<std::uint8_t, 256> decoded{};
      const Base64Result decode = base64_decode(
          std::span<const char>{encoded},
          std::as_writable_bytes(std::span<std::uint8_t>{decoded}),
          decode_options);
      ASSERT_TRUE(decode);
      EXPECT_EQ(decode.output_count, raw.size());
      EXPECT_EQ(decoded, raw);
    }
  }
}

TEST(Base64, ProviderBlockTailsRoundTripAcrossPolicies) {
  std::array<std::uint8_t, 513> raw{};
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint8_t>((i * 131u + 17u) & 0xffu);
  }

  for (std::size_t length = 0; length <= raw.size(); ++length) {
    const auto input = std::as_bytes(
        std::span<const std::uint8_t>{raw.data(), length});
    for (Base64Alphabet alphabet :
         {Base64Alphabet::kStandard, Base64Alphabet::kUrl}) {
      for (Base64EncodePadding encode_padding :
           {Base64EncodePadding::kInclude, Base64EncodePadding::kOmit}) {
        const Base64EncodeOptions encode_options{alphabet, encode_padding};
        const Base64SizeResult encoded_size =
            base64_encoded_size(length, encode_options);
        ASSERT_TRUE(encoded_size) << "length=" << length;
        std::vector<char> encoded(encoded_size.output_size);
        const Base64Result encoded_result =
            base64_encode(input, encoded, encode_options);
        ASSERT_TRUE(encoded_result) << "length=" << length;

        const Base64DecodeOptions decode_options{
            .alphabet = alphabet,
            .padding = encode_padding == Base64EncodePadding::kInclude
                           ? Base64DecodePadding::kRequired
                           : Base64DecodePadding::kForbidden,
            .whitespace = Base64Whitespace::kReject,
        };
        const Base64SizeResult exact =
            base64_decoded_size(std::span<const char>{encoded},
                                decode_options);
        ASSERT_TRUE(exact) << "length=" << length;
        EXPECT_EQ(exact.output_size, length);
        EXPECT_LE(exact.output_size,
                  base64_max_decoded_size(encoded.size()));

        std::vector<std::uint8_t> decoded(length);
        const Base64Result decoded_result = base64_decode(
            std::span<const char>{encoded},
            std::as_writable_bytes(std::span<std::uint8_t>{decoded}),
            decode_options);
        ASSERT_TRUE(decoded_result) << "length=" << length;
        ASSERT_EQ(decoded_result.output_count, length);
        EXPECT_TRUE(std::equal(decoded.begin(), decoded.end(), raw.begin()));
      }
    }
  }
}

TEST(Base64, ShortAdversarialPolicySpaceAgreesAcrossColdAndFastRoutes) {
  // This is deliberately a state-space check rather than another list of
  // friendly vectors.  It crosses both alphabets, all padding policies and
  // both whitespace policies with alphabet-only, wrong-alphabet, padding,
  // accepted whitespace and garbage symbols.  The cold diagnostic and the
  // fast provider route may report different error categories, but they must
  // never disagree about whether an input is accepted.
  constexpr std::array<char, 12> kSymbols = {
      'A', 'B', 'Z', 'g', '+', '/', '-', '_', '=', ' ', '#', '\v'};
  constexpr std::array<Base64Alphabet, 2> kAlphabets = {
      Base64Alphabet::kStandard, Base64Alphabet::kUrl};
  constexpr std::array<Base64DecodePadding, 3> kPadding = {
      Base64DecodePadding::kRequired, Base64DecodePadding::kAllowed,
      Base64DecodePadding::kForbidden};
  constexpr std::array<Base64Whitespace, 2> kWhitespace = {
      Base64Whitespace::kReject, Base64Whitespace::kAllowAscii};

  std::string input;
  std::array<std::byte, 8> output{};
  std::size_t combinations = 1;
  for (std::size_t length = 0; length <= 5; ++length) {
    input.resize(length);
    for (std::size_t encoded = 0; encoded < combinations; ++encoded) {
      std::size_t digits = encoded;
      for (std::size_t i = 0; i < length; ++i) {
        input[i] = kSymbols[digits % kSymbols.size()];
        digits /= kSymbols.size();
      }

      for (Base64Alphabet alphabet : kAlphabets) {
        for (Base64DecodePadding padding : kPadding) {
          for (Base64Whitespace whitespace : kWhitespace) {
            const Base64DecodeOptions options{alphabet, padding, whitespace};
            const Base64SizeResult cold =
                base64_decoded_size(std::span<const char>{input}, options);
            const Base64Result fast = base64_decode(
                std::span<const char>{input}, output, options);
            if (cold.ok() != fast.ok()) {
              ADD_FAILURE() << "cold/fast acceptance mismatch for input="
                            << testing::PrintToString(input)
                            << " alphabet=" << static_cast<int>(alphabet)
                            << " padding=" << static_cast<int>(padding)
                            << " whitespace=" << static_cast<int>(whitespace)
                            << " cold=" << static_cast<int>(cold.status)
                            << " fast=" << static_cast<int>(fast.status);
              return;
            }
            if (cold.ok() &&
                (fast.input_count != input.size() ||
                 fast.output_count != cold.output_size)) {
              ADD_FAILURE() << "cold/fast progress mismatch for input="
                            << testing::PrintToString(input)
                            << " expected_output=" << cold.output_size
                            << " input_count=" << fast.input_count
                            << " output_count=" << fast.output_count;
              return;
            }
          }
        }
      }
    }
    combinations *= kSymbols.size();
  }
}

}  // namespace
}  // namespace hyle::container
