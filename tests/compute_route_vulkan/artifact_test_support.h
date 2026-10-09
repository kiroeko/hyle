#pragma once

#include "hyle/compute/model_ops.h"
#include "tensor_test_support.h"

namespace vulkan_artifact_test {
namespace t = vulkan_tensor_test;
namespace c = hyle::compute;
namespace v = c::vulkan;
using Format = c::model_ops::WeightArtifactFormat;
constexpr std::array types{c::DataType::kFloat32, c::DataType::kFloat16,
                            c::DataType::kBFloat16};

// Independent scalar interpretation of the source formats. No product recipe,
// shader, quantizer or observed GPU result participates in the reference.
struct Inputs final {
  c::Operation operation;
  std::vector<c::TensorDesc> descriptions;
  std::vector<std::vector<std::byte>> payloads;
  c::TensorDesc output;
  std::vector<double> expected;

  Inputs(Format format, int n, int k, int group, c::DataType scale_type,
         c::DataType output_type, int seed = 0)
      : operation(c::model_ops::artifact_weight_decode(format, group, output_type)),
        expected(static_cast<std::size_t>(n * k)) {
    const bool nf4 = format == Format::kNf4 || format == Format::kNf4DoubleQuant;
    const bool nested = format == Format::kNf4DoubleQuant;
    const bool awq = format == Format::kAwqGemm4;
    const int groups = n * k / group;
    const auto add = [&](c::TensorDesc desc, std::span<const std::uint64_t> bits) {
      descriptions.push_back(desc);
      payloads.push_back(t::payload(desc, bits));
    };
    if (nf4) {
      std::vector<std::uint64_t> codes(n * k / 2);
      for (std::size_t i = 0; i < codes.size(); ++i)
        codes[i] = (((i * 7 + seed * 3) % 16) << 4) | ((i * 11 + 5 + seed) % 16);
      add(t::dense(c::DataType::kUInt8, {n, k / 2}), codes);
      constexpr std::array<float, 16> book{
          -1, -.696192801f, -.525073051f, -.394917488f, -.284441382f,
          -.18477343f, -.0910500363f, 0, .0795802996f, .160930201f,
          .246112302f, .337915242f, .440709829f, .562617004f, .722956836f, 1};
      std::vector<float> scales(groups);
      if (nested) {
        std::vector<std::uint64_t> scale_codes(groups), outer_bits((groups + 255) / 256);
        std::array<std::uint64_t, 256> scale_book{};
        for (std::size_t i = 0; i < scale_book.size(); ++i)
          scale_book[i] = t::encode((static_cast<int>(i) - 128) / 127.f, c::DataType::kFloat32);
        for (std::size_t i = 0; i < outer_bits.size(); ++i)
          outer_bits[i] = t::encode(.317f + .021f * (i + seed), c::DataType::kFloat32);
        const float bias = .513f + .031f * seed;
        for (int i = 0; i < groups; ++i) {
          scale_codes[i] = (i * 37 + seed * 19) % 256;
          const float decoded = static_cast<float>(t::decode(scale_book[scale_codes[i]], c::DataType::kFloat32));
          const float outer = static_cast<float>(t::decode(outer_bits[i / 256], c::DataType::kFloat32));
          const float product = decoded * outer;
          scales[i] = product + bias;
        }
        add(t::dense(c::DataType::kUInt8, {groups}), scale_codes);
        add(t::dense(c::DataType::kFloat32, {static_cast<std::int64_t>(outer_bits.size())}), outer_bits);
        add(t::dense(c::DataType::kFloat32, {256}), scale_book);
        const std::array offset{t::encode(bias, c::DataType::kFloat32)};
        add(t::dense(c::DataType::kFloat32, {}), offset);
      } else {
        std::vector<std::uint64_t> scale_bits(groups);
        for (int i = 0; i < groups; ++i) {
          scales[i] = .0713f * (i % 13 + seed + 1);
          scale_bits[i] = t::encode(scales[i], c::DataType::kFloat32);
        }
        add(t::dense(c::DataType::kFloat32, {groups}), scale_bits);
      }
      std::array<std::uint64_t, 16> book_bits{};
      for (std::size_t i = 0; i < book.size(); ++i)
        book_bits[i] = t::encode(book[i], c::DataType::kFloat32);
      add(t::dense(c::DataType::kFloat32, {16}), book_bits);
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto code = i % 2 ? codes[i / 2] & 15 : codes[i / 2] >> 4;
        expected[i] = t::rounded(book[code] * scales[i / group], output_type);
      }
    } else {
      const auto word_count = static_cast<std::size_t>(n * k / 8);
      std::vector<std::uint64_t> words(word_count), zero_words(awq ? groups / 8 : 0);
      for (std::size_t i = 0; i < words.size(); ++i)
        for (int nibble = 0; nibble < 8; ++nibble)
          words[i] |= ((i * 3 + nibble * 7 + seed * 5) % 16) << (4 * nibble);
      for (std::size_t i = 0; i < zero_words.size(); ++i)
        for (int nibble = 0; nibble < 8; ++nibble)
          zero_words[i] |= ((i * 5 + nibble * 3 + seed * 7) % 16) << (4 * nibble);
      add(t::dense(c::DataType::kInt32, {awq ? k : n, awq ? n / 8 : k / 8}), words);
      if (awq)
        add(t::dense(c::DataType::kInt32, {k / group, n / 8}), zero_words);
      std::vector<std::uint64_t> scale_bits(groups);
      for (int i = 0; i < groups; ++i)
        scale_bits[i] = t::encode(.0137f * (1 + i % 17 + seed), scale_type);
      add(t::dense(scale_type, {awq ? k / group : n, awq ? n : k / group}), scale_bits);
      constexpr std::array order{0, 4, 1, 5, 2, 6, 3, 7};
      for (int row = 0; row < n; ++row)
        for (int column = 0; column < k; ++column) {
          const auto word = words[awq ? column * (n / 8) + row / 8
                                     : row * (k / 8) + column / 8];
          const int shift = 4 * (awq ? order[row % 8] : column % 8);
          const auto zero = awq ? (zero_words[(column / group) * (n / 8) + row / 8] >> shift) & 15 : 8;
          const int difference = static_cast<int>((word >> shift) & 15) - static_cast<int>(zero);
          const auto scale = t::decode(scale_bits[awq ? (column / group) * n + row
                                                     : row * (k / group) + column / group], scale_type);
          const float product = static_cast<float>(difference * scale);
          expected[static_cast<std::size_t>(row * k + column)] =
              t::rounded(t::rounded(product, scale_type), output_type);
        }
    }
    std::array<c::TensorDesc, 1> result;
    EXPECT_TRUE(c::infer_output_descriptions(operation, descriptions, result).ok());
    output = result[0];
  }

  std::vector<v::Tensor> upload(const v::Context &context) const {
    std::vector<v::Tensor> tensors;
    for (std::size_t i = 0; i < descriptions.size(); ++i)
      tensors.push_back(context.upload(descriptions[i], payloads[i]).value());
    return tensors;
  }

  void check(const v::Context &context, const v::Tensor &tensor) const {
    ASSERT_EQ(tensor.description(), output);
    const auto actual = t::values(context, tensor);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (output.data_type() == c::DataType::kFloat32)
        EXPECT_NEAR(actual[i], expected[i], 1e-7 + 1e-6 * std::abs(expected[i])) << i;
      else
        EXPECT_EQ(actual[i], expected[i]) << i;
    }
  }
};

template <class Prepare>
void batch(Prepare prepare) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 128 * 1024 * 1024);
  {
    auto context = vulkan_test::context(&quota).value();
    for (auto format : {Format::kNf4, Format::kNf4DoubleQuant, Format::kAwqGemm4,
                        Format::kCompressedTensorsSymmetric4}) {
      const bool nf4 = format == Format::kNf4 || format == Format::kNf4DoubleQuant;
      for (auto scale : types) {
        if (nf4 && scale != c::DataType::kFloat32) continue;
        for (auto output : types)
          for (int group : {64, 128}) {
            const int n = format == Format::kAwqGemm4 ? 16 : 3, k = 3 * group;
            SCOPED_TRACE(::testing::Message() << static_cast<int>(format) << '/'
                << static_cast<int>(scale) << '/' << static_cast<int>(output) << '/' << group);
            Inputs original(format, n, k, group, scale, output);
            Inputs changed(format, n, k, group, scale, output, 1);
            auto program = prepare(context, original.operation, original.descriptions);
            ASSERT_TRUE(program) << program.error().message();
            auto first = program->create_invocation().value();
            auto second = program->create_invocation().value();
            ASSERT_TRUE(first.bind(original.upload(context)).ok());
            ASSERT_TRUE(second.bind(changed.upload(context)).ok());
            v::Tensor retained;
            {
              auto a = first.submit().value(), b = second.submit().value();
              auto second_result = b.wait(), first_result = a.wait();
              ASSERT_TRUE(first_result) << first_result.error().message();
              ASSERT_TRUE(second_result) << second_result.error().message();
              original.check(context, first_result->get()->tensors()[0]);
              changed.check(context, second_result->get()->tensors()[0]);
              retained = first_result->get()->tensors()[0];
            }
            EXPECT_FALSE(first.submit());
            for (int pass = 0; pass < 2; ++pass) {
              auto done = second.submit().value();
              auto result = done.wait();
              ASSERT_TRUE(result) << result.error().message();
              changed.check(context, result->get()->tensors()[0]);
              original.check(context, retained);
            }
            retained = {};
            ASSERT_TRUE(first.bind(changed.upload(context)).ok());
            auto done = first.submit().value();
            auto result = done.wait();
            ASSERT_TRUE(result) << result.error().message();
            changed.check(context, result->get()->tensors()[0]);
          }
      }
    }
    // Nine scale-code bytes and 258 groups exercise partial words and crossing
    // the original 256-scale outer block without conflating the six ports.
    for (int k : {192, 5504}) {
      Inputs data(Format::kNf4DoubleQuant, 3, k, 64,
                  c::DataType::kFloat32, c::DataType::kBFloat16, 2);
      auto program = prepare(context, data.operation, data.descriptions).value();
      auto slot = program.create_invocation().value();
      ASSERT_TRUE(slot.bind(data.upload(context)).ok());
      auto done = slot.submit().value();
      auto result = done.wait();
      ASSERT_TRUE(result) << result.error().message();
      data.check(context, result->get()->tensors()[0]);
    }
    EXPECT_TRUE(context.close().ok());
  }
  EXPECT_EQ(quota.used(), 0U);
}

template <class Decode, class Projection>
void cold_projection(Decode decode, Projection projection) {
  auto context = vulkan_test::context().value();
  for (auto format : {Format::kNf4, Format::kNf4DoubleQuant, Format::kAwqGemm4,
                      Format::kCompressedTensorsSymmetric4})
    for (auto dtype : types) {
      const int n = format == Format::kAwqGemm4 ? 16 : 3, k = 128;
      const bool nf4 = format == Format::kNf4 || format == Format::kNf4DoubleQuant;
      Inputs data(format, n, k, 64, nf4 ? c::DataType::kFloat32 : dtype, dtype);
      auto prepare = decode(context, data.operation, data.descriptions).value();
      auto slot = prepare.create_invocation().value();
      ASSERT_TRUE(slot.bind(data.upload(context)).ok());
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      auto decoded = result->get()->tensors()[0];
      auto weight = decoded.view(c::View::permute(std::array<std::int32_t, 2>{1, 0}).value()).value();
      const auto activation = t::dense(dtype, {2, k});
      auto product = projection(context, activation, weight);
      ASSERT_TRUE(product) << product.error().message();
      auto invocation = product->create_invocation().value();
      for (int pass = 0; pass < 2; ++pass) {
        std::vector<double> values(2 * k, 0), expected(2 * n);
        values[0] = .25 * (pass + 1); values[1] = -.125;
        values[k - 1] = .5;
        values[k] = -.5; values[k + 1] = .25;
        values[2 * k - 1] = .125 * (pass + 1);
        for (int token = 0; token < 2; ++token)
          for (int row = 0; row < n; ++row) {
            double sum = 0;
            for (int column = 0; column < k; ++column)
              sum += data.expected[row * k + column] * values[token * k + column];
            expected[token * n + row] = t::rounded(sum, dtype);
          }
        ASSERT_TRUE(invocation.bind(t::make(context, activation, values)).ok());
        auto done = invocation.submit().value();
        auto output = done.wait();
        ASSERT_TRUE(output) << output.error().message();
        const auto actual = t::values(context, output->get()->tensors()[0]);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
          if (dtype == c::DataType::kFloat32)
            EXPECT_NEAR(actual[i], expected[i], 1e-7 + 2e-6 * std::abs(expected[i]));
          else
            EXPECT_EQ(actual[i], expected[i]);
        }
        data.check(context, decoded);
      }
    }
}
} // namespace vulkan_artifact_test
