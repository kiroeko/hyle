#pragma once

#include <bit>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace g08 {
inline constexpr int kVocabulary = 248320;

struct Data final {
  int rows;
  int count;
  std::vector<float> input;
  std::vector<float> expected_values;
  std::vector<std::int32_t> expected_indices;

  Data(int r, int k) : rows(r), count(k), input(r * kVocabulary),
                       expected_values(r * k), expected_indices(r * k) {
    if ((r != 1 && r != 16) || (k != 1 && k != 3))
      throw std::runtime_error("unexpected frozen TopK shape");
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < kVocabulary; ++col) {
        auto word = std::uint32_t(col) * 0x9e3779b9u ^
                    std::uint32_t(row + 1) * 0x85ebca6bu;
        word ^= word >> 16; word *= 0x7feb352du;
        word ^= word >> 15; word *= 0x846ca68bu; word ^= word >> 16;
        const auto exponent = 124u + ((word >> 24) % 6u);
        input[row * kVocabulary + col] =
            std::bit_cast<float>((word & 0x807fffffu) | (exponent << 23));
      }
      // Every generated magnitude is below8. These explicit normal maxima
      // give a closed-form oracle, independently of any selection algorithm.
      input[row * kVocabulary + kVocabulary - 3] = 9.0F;
      input[row * kVocabulary + kVocabulary - 2] = 8.5F;
      input[row * kVocabulary + kVocabulary - 1] = 9.0F;
      expected_values[row * count] = 9.0F;
      expected_indices[row * count] = kVocabulary - 3;
      if (count == 3) {
        expected_values[row * count + 1] = 9.0F;
        expected_indices[row * count + 1] = kVocabulary - 1;
        expected_values[row * count + 2] = 8.5F;
        expected_indices[row * count + 2] = kVocabulary - 2;
      }
    }
  }
};

class Arm {
public:
  virtual ~Arm() = default;
  virtual void execute() = 0;
  virtual void verify() = 0;
  virtual void finish() = 0;
};

inline void verify_outputs(const Data &data, std::span<const float> values,
                           std::span<const std::int32_t> indices) {
  if (values.size() != data.expected_values.size() ||
      indices.size() != data.expected_indices.size())
    throw std::runtime_error("TopK output arity/shape differs");
  for (std::size_t i = 0; i < values.size(); ++i)
    if (std::bit_cast<std::uint32_t>(values[i]) !=
            std::bit_cast<std::uint32_t>(data.expected_values[i]) ||
        indices[i] != data.expected_indices[i])
      throw std::runtime_error("TopK source bits/first-index oracle differs");
}

std::unique_ptr<Arm> make_candidate(const Data &);
std::unique_ptr<Arm> make_reference(const Data &);
} // namespace g08
