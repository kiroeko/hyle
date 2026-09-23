#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "hyle/compute/model_ops.h"

namespace model_fragment_test {
namespace c = hyle::compute;

struct Case final {
  const char *name;
  c::Operation operation;
  std::vector<c::TensorDesc> inputs;
  std::vector<std::vector<std::byte>> payloads;
  c::TensorDesc output;
  std::vector<float> expected;
};

inline c::TensorDesc dense(c::DataType type,
                           std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

template <class T>
std::vector<std::byte> payload(std::initializer_list<T> values) {
  std::vector<std::byte> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.begin(), result.size());
  return result;
}

inline std::vector<Case> cases() {
  const auto f32 = c::DataType::kFloat32;
  const auto i32 = c::DataType::kInt32;
  std::vector<Case> result;
  result.push_back({
      "grouped-matmul",
      c::model_ops::grouped_matmul(),
      {dense(f32, {3, 2, 3}), dense(f32, {2, 2, 3}), dense(i32, {2, 2})},
      {payload<float>({1, 0, 0, 0, 1, 0, 0, 0, 1,
                       1, 1, 1, 2, 0, 0, 0, 0.5F, 0}),
       payload<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1}),
       payload<std::int32_t>({0, 2, 1, 0})},
      dense(f32, {2, 2, 2}),
      {1, 2, 8, 2.5F, 9, 24, 1, 1}});
  result.push_back({
      "hyper-connection-combine",
      c::model_ops::hyper_connection_combine(1.0e-6F, 4),
      {dense(f32, {2, 24}), dense(f32, {3}), dense(f32, {24})},
      {payload<float>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
       payload<float>({1, 1, 1}),
       payload<float>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})},
      dense(f32, {2, 4, 4}), std::vector<float>(32, 0.25F)});
  result.push_back({
      "hyper-connection-pre",
      c::model_ops::hyper_connection_pre(),
      {dense(f32, {2, 4, 3}), dense(f32, {2, 4})},
      {payload<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
       payload<float>({1, 0, 0, 0, 0, 0.5F, 0.5F, 0})},
      dense(f32, {2, 3}), {1, 2, 3, 17.5F, 18.5F, 19.5F}});
  result.push_back({
      "hyper-connection-post",
      c::model_ops::hyper_connection_post(),
      {dense(f32, {1, 3}), dense(f32, {1, 4, 3}), dense(f32, {1, 4}),
       dense(f32, {1, 4, 4})},
      {payload<float>({1, 2, 3}),
       payload<float>({10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}),
       payload<float>({1, 2, 3, 4}),
       payload<float>({1, 0, 0, 0, 0, 1, 0, 0,
                       0, 0, 1, 0, 0, 0, 0, 1})},
      dense(f32, {1, 4, 3}),
      {11, 22, 33, 42, 54, 66, 73, 86, 99, 104, 118, 132}});
  return result;
}
} // namespace model_fragment_test
