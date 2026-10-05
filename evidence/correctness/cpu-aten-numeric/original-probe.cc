#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

static std::size_t checks = 0;
static void check(bool valid, const char *message) {
  ++checks;
  if (!valid) { std::cerr << message << '\n'; std::exit(2); }
}
static at::Tensor activation(at::Tensor x, int kind) {
  // Original native kernels in a higher-precision intermediate, then a public
  // F32 activation value. No custom element loop or native kernel.
  auto wide = x.to(at::kDouble);
  at::Tensor y;
  if (kind == 0) y = at::silu(wide);
  else if (kind == 1) y = wide.mul(.5).mul(at::erfc(wide.mul(-std::numbers::sqrt2/2)));
  else {
    auto inner = wide.add(wide.pow(3).mul(.044715)).mul(2*std::sqrt(2/std::numbers::pi));
    y = wide.mul(at::sigmoid(inner));
  }
  return at::where(wide.eq(-std::numeric_limits<double>::infinity()),
                   at::zeros_like(y), y).to(at::kFloat);
}
static double oracle(double x, int kind) {
  if (x == -std::numeric_limits<double>::infinity()) return 0;
  if (kind == 0) { const double e = std::exp(-std::abs(x)); return x * (x < 0 ? e / (1 + e) : 1 / (1 + e)); }
  if (kind == 1) return .5 * x * std::erfc(-x / std::numbers::sqrt2);
  const double t = 2*std::sqrt(2/std::numbers::pi) * (x + .044715*x*x*x);
  const double e = std::exp(-std::abs(t));
  return x * (t < 0 ? e/(1+e) : 1/(1+e));
}
static bool near(float actual, float expected) {
  if (std::isnan(expected)) return std::isnan(actual);
  if (std::isinf(expected)) return actual == expected;
  if (expected == 0) return actual == expected && std::signbit(actual) == std::signbit(expected);
  return std::abs(static_cast<double>(actual)-expected) <=
         2e-6 * std::abs(static_cast<double>(expected)) + 4 * std::numeric_limits<float>::denorm_min();
}
static at::Tensor number(at::Tensor a, at::Tensor b, bool minimum) {
  auto result = minimum ? at::fmin(a,b) : at::fmax(a,b);
  auto negative = minimum ? at::signbit(a).logical_or(at::signbit(b))
                          : at::signbit(a).logical_and(at::signbit(b));
  auto zero = at::where(negative, at::full_like(result, -0.0), at::zeros_like(result));
  return at::where(a.eq(0).logical_and(b.eq(0)), zero, result);
}
static at::Tensor reduce(at::Tensor x, at::IntArrayRef axes, bool keep, int kind) {
  auto y = kind == 0 ? at::sum(x,axes,keep) : kind == 1 ? at::mean(x,axes,keep) :
           kind == 2 ? at::amin(x,axes,keep) : at::amax(x,axes,keep);
  if (kind >= 2) {
    auto negative_zero = at::signbit(x).logical_and(x.eq(0));
    auto negative = kind == 2 ? at::any(negative_zero,axes,keep) :
                               at::all(at::signbit(x).logical_or(x.ne(0)),axes,keep);
    auto zero = at::where(negative, at::full_like(y,-0.0), at::zeros_like(y));
    y = at::where(y.eq(0),zero,y);
  }
  return y;
}
int main() {
  at::NoGradGuard guard;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> base{-inf,-100,-90,-40,-20,-12,-9,-8,-6,-5,-4,-1,-0.0F,0,1,4,5,6,12,20,90,100,inf,nan};
  for (int i = -800; i <= 800; ++i) base.push_back(i / 80.0F);
  auto source = at::from_blob(base.data(), {static_cast<std::int64_t>(base.size())},at::kFloat);
  for (auto dtype : {at::kFloat,at::kHalf,at::kBFloat16}) {
    auto stored = source.to(dtype);
    auto x = stored.to(at::kFloat);
    for (int kind = 0; kind != 3; ++kind) {
      auto y = activation(x,kind);
      auto original = kind == 0 ? at::silu(x) : at::gelu(x, kind == 1 ? "none" : "tanh");
      std::cout << "dtype=" << dtype << " activation=" << kind << " raw-neg-inf=" << original[0].item<float>()
                << " qualified-neg-inf=" << y[0].item<float>() << " tail=" << y[1].item<float>() << '\n';
      auto values = y.to(dtype).to(at::kFloat);
      for (std::size_t i = 0; i < base.size(); ++i) {
        float expected = static_cast<float>(oracle(x[i].item<float>(),kind));
        expected = at::scalar_tensor(expected,at::kFloat).to(dtype).item<float>();
        if (!near(values[i].item<float>(), expected))
          std::cerr << "activation kind=" << kind << " input=" << std::setprecision(12) << x[i].item<float>()
                    << " actual=" << values[i].item<float>() << " expected=" << expected << '\n';
        check(near(values[i].item<float>(), expected),"activation precision/special-value witness failed");
      }
      for (bool packed : {false,true}) {
        auto v = at::full_like(x,1e30).to(dtype).to(at::kFloat);
        auto both = at::cat({x,v});
        auto a = packed ? both.slice(0,0,x.numel()) : x;
        auto b = packed ? both.slice(0,x.numel()) : v;
        auto output = activation(a,kind).mul(b).to(dtype).to(at::kFloat);
        for (std::size_t i = 0; i < base.size(); ++i) {
          const float f = static_cast<float>(oracle(a[i].item<float>(),kind));
          float expected = f * b[i].item<float>();
          expected = at::scalar_tensor(expected,at::kFloat).to(dtype).item<float>();
          check(near(output[i].item<float>(), expected),"gate amplification/single-round witness failed");
        }
      }
    }
    auto sp = at::softplus(x,1,20).to(dtype).to(at::kFloat);
    for (std::size_t i = 0; i < base.size(); ++i) {
      const double v = x[i].item<float>();
      float expected = static_cast<float>((std::max)(v,0.0)+std::log1p(std::exp(-std::abs(v))));
      expected = at::scalar_tensor(expected,at::kFloat).to(dtype).item<float>();
      check(near(sp[i].item<float>(), expected),"Softplus stable-tail witness failed");
    }
  }
  std::array<float,8> left{0,-0.0F,0,-0.0F,nan,1,nan,-inf};
  std::array<float,8> right{-0.0F,0,0,-0.0F,1,nan,nan,inf};
  auto a = at::from_blob(left.data(),{8},at::kFloat), b = at::from_blob(right.data(),{8},at::kFloat);
  for (bool minimum : {true,false}) {
    auto y = number(a,b,minimum);
    for (std::size_t i = 0; i < 8; ++i) {
      float expected = minimum ? std::fmin(left[i],right[i]) : std::fmax(left[i],right[i]);
      if (left[i] == 0 && right[i] == 0)
        expected = (minimum ? std::signbit(left[i]) || std::signbit(right[i]) : std::signbit(left[i]) && std::signbit(right[i])) ? -0.0F : 0;
      check(near(y[i].item<float>(),expected),"number Min/Max NaN/zero witness failed");
    }
  }
  std::array<float,12> data{0,-0.0F,0,0, -0.0F,-0.0F,-0.0F,-0.0F,1,2,nan,3};
  auto rows = at::from_blob(data.data(),{3,4},at::kFloat);
  for (bool keep : {false,true}) for (int kind : {2,3}) {
    auto y = reduce(rows,{1},keep,kind).flatten();
    check(std::signbit(y[0].item<float>()) == (kind == 2),"mixed-zero Reduce sign failed");
    check(std::signbit(y[1].item<float>()),"all-negative-zero Reduce sign failed");
    check(std::isnan(y[2].item<float>()),"Reduce NaN propagation failed");
  }
  // Byte-to-owned native packing keeps arbitrary byte strides exact, including
  // addresses which cannot safely be passed to a typed from_blob.
  for (std::int64_t width : {2,4}) {
    std::vector<std::uint8_t> physical(3*(width+1)), expected(3*width);
    for (int i=0;i<3;++i) for (int j=0;j<width;++j) expected[i*width+j]=physical[i*(width+1)+j]=i*31+j*7;
    auto bytes = at::from_blob(physical.data(),{3,width},{width+1,1},at::kByte);
    auto aligned = at::empty({3},width==4?at::kFloat:at::kHalf);
    auto packed = aligned.flatten().view(at::kByte).view({3,width});
    packed.copy_(bytes);
    check(std::memcmp(aligned.data_ptr(),expected.data(),expected.size())==0,"owned native byte packing failed");
  }
  std::cout << "checks=" << checks << " original-numeric-program-witness=PASS\n";
}
