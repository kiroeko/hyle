#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

int main() {
  at::NoGradGuard guard;
  std::size_t checks = 0;
  auto check = [&](bool passed, const char *message) {
    if (!passed) throw std::runtime_error(message);
    ++checks;
  };
  for (std::int64_t width : {1, 2, 4, 8}) {
    // Deliberately odd byte strides; Byte is a storage carrier, not numeric cast.
    const auto row_stride = 3 * width + 5;
    const auto column_stride = width + 1;
    std::vector<std::uint8_t> left(128), right(128), result(6 * width), gathered(6 * width);
    for (std::size_t i = 0; i < 128; ++i) {
      left[i] = static_cast<std::uint8_t>((i * 71 + 183) & 255);
      right[i] = static_cast<std::uint8_t>((i * 103 + 9) & 255);
    }
    auto a = at::from_blob(left.data(), {3, 2, width}, {row_stride, column_stride, 1}, at::kByte);
    auto b = at::from_blob(right.data(), {3, 2, width}, {row_stride, column_stride, 1}, at::kByte);
    std::array<std::uint8_t, 2> condition{0, 127};
    auto c = at::from_blob(condition.data(), {1, 2, 1}, at::kByte).ne(0);
    auto out = at::from_blob(result.data(), {3, 2, width}, at::kByte);
    at::where_out(out, c, a, b);
    for (std::int64_t row = 0; row < 3; ++row)
      for (std::int64_t column = 0; column < 2; ++column)
        for (std::int64_t byte = 0; byte < width; ++byte)
          check(result[(row * 2 + column) * width + byte] ==
                    (column ? left : right)[row * row_stride + column * column_stride + byte],
                "Byte where changed selected storage bits");
    for (bool wide : {false, true}) {
      std::array<std::int32_t, 3> i32{2, 0, 2};
      std::array<std::int64_t, 3> i64{2, 0, 2};
      auto ids = wide ? at::from_blob(i64.data(), {3}, at::kLong)
                      : at::from_blob(i32.data(), {3}, at::kInt);
      auto g = at::from_blob(gathered.data(), {3, 2, width}, at::kByte);
      at::index_select_out(g, a, 0, ids);
      for (std::int64_t row = 0; row < 3; ++row)
        for (std::int64_t column = 0; column < 2; ++column)
          for (std::int64_t byte = 0; byte < width; ++byte)
            check(gathered[(row * 2 + column) * width + byte] ==
                      left[i64[row] * row_stride + column * column_stride + byte],
                  "Byte gather changed selected storage bits");
    }
    std::array<std::uint8_t, 8> scalar{0x80, 0x00, 0xff, 0x7f, 0x01, 0x31, 0xd7, 0xa5};
    auto s = at::from_blob(scalar.data(), {width}, at::kByte);
    out.copy_(s);
    for (std::size_t i = 0; i < result.size(); ++i)
      check(result[i] == scalar[i % width], "Byte fill changed exact scalar bits");
  }
  // Public Assign: stable native sort selects the final logical update per ID;
  // index_copy sees unique IDs only, avoiding original duplicate-write UB.
  for (std::int64_t width : {1, 2, 4, 8}) {
    std::array<std::int64_t, 5> ids{2, 0, 2, 1, 2};
    std::vector<std::uint8_t> base(3 * 2 * width, 0x99), updates(5 * 2 * width);
    for (std::size_t i = 0; i < updates.size(); ++i) updates[i] = (i * 71 + 37) & 255;
    auto index = at::from_blob(ids.data(), {5}, at::kLong);
    const auto [sorted, order] = at::sort(index, std::optional<bool>(true), 0, false);
    auto last = at::cat({sorted.slice(0, 1).ne(sorted.slice(0, 0, 4)), at::ones({1}, at::kBool)});
    auto winners = order.masked_select(last);
    auto unique = sorted.masked_select(last);
    auto rows = at::from_blob(updates.data(), {5, 2, width}, at::kByte).index_select(0, winners);
    auto output = at::from_blob(base.data(), {3, 2, width}, at::kByte);
    output.index_copy_(0, unique, rows);
    const std::array<std::int64_t, 3> expected{1, 3, 4};
    for (std::int64_t row = 0; row < 3; ++row)
      for (std::int64_t byte = 0; byte < 2 * width; ++byte)
        check(base[row * 2 * width + byte] == updates[expected[row] * 2 * width + byte],
              "stable original Assign did not select last logical update");
  }
  // Public maximum rank 8 needs rank 9 only inside the byte carrier.
  std::array<std::uint8_t, 16> rank9_left{}, rank9_right{}, rank9_out{};
  for (std::size_t i = 0; i < 16; ++i) { rank9_left[i] = i; rank9_right[i] = 255 - i; }
  const std::vector<std::int64_t> shape{1,1,1,1,1,1,2,1,8};
  auto a = at::from_blob(rank9_left.data(), shape, at::kByte);
  auto b = at::from_blob(rank9_right.data(), shape, at::kByte);
  auto out = at::from_blob(rank9_out.data(), shape, at::kByte);
  at::where_out(out, at::ones({}, at::kBool), a, b);
  check(rank9_out == rank9_left, "rank9 Byte where mismatch");
  std::array<std::int64_t, 1> id{1};
  auto selected = at::index_select(a, 6, at::from_blob(id.data(), {1}, at::kLong));
  check(std::memcmp(selected.data_ptr(), rank9_left.data() + 8, 8) == 0,
        "rank9 Byte index_select mismatch");
  // Empty original calls are witnesses; not public lifetime/error qualification.
  auto empty = at::empty({0, 2, 8}, at::kByte);
  auto ids = at::empty({0}, at::kLong);
  check(at::index_select(empty, 0, ids).numel() == 0, "empty original gather failed");
  bool rejected = false;
  try { (void)at::index_select(a, 6, at::tensor({-1}, at::kLong)); }
  catch (const c10::Error &) { rejected = true; }
  check(rejected, "negative original index was not rejected");
  std::array<float, 7> tails{-std::numeric_limits<float>::infinity(), -100, -5,
                            -0.0F, 0.0F, std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()};
  auto x = at::from_blob(tails.data(), {7}, at::kFloat);
  for (const auto &entry : std::vector<std::pair<const char *, at::Tensor>>{
           {"silu", at::silu(x)}, {"gelu-erf", at::gelu(x, "none")},
           {"gelu-tanh", at::gelu(x, "tanh")}, {"sigmoid", at::sigmoid(x)}}) {
    std::cout << entry.first;
    for (std::size_t i = 0; i < tails.size(); ++i) std::cout << " " << entry.second.data_ptr<float>()[i];
    std::cout << "\n";
  }
  std::array<float, 4> zeros{0, -0.0F, 0, -0.0F};
  auto z = at::from_blob(zeros.data(), {4}, at::kFloat);
  std::cout << "amin-zero-sign=" << std::signbit(z.amin().item<float>())
            << " amax-zero-sign=" << std::signbit(z.amax().item<float>()) << "\n";
  std::cout << "checks=" << checks << " storage-original-api-witness=PASS\n";
}
