#include <oneapi/dnnl/dnnl.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

float f8(unsigned code) {
  auto v = code & 127u;
  float x = v < 8 ? std::ldexp(float(v), -9)
      : std::ldexp(1.0f + float(v % 8) / 8, int(v / 8) - 7);
  return code & 128 ? -x : x;
}
unsigned nearest(float x, bool four) {
  const float f4[]{0,.5f,1,1.5f,2,3,4,6};
  unsigned best = 0; double distance = INFINITY;
  for (unsigned i=0; i < (four ? 8u : 127u); ++i) {
    double d = std::abs(double(std::abs(x)) - (four ? f4[i] : f8(i)));
    if (d < distance || (d == distance && (i & 1) == 0)) {best=i; distance=d;}
  }
  return best | (std::signbit(x) ? (four ? 8u : 128u) : 0u);
}
int main() {
  using namespace dnnl;
  try {
    engine eng(engine::kind::cpu, 0); stream str(eng);
    for (bool four : {false, true}) {
      std::vector<float> x;
      const float values[]{0,.5f,1,1.5f,2,3,4,6};
      for (unsigned i=0;i < (four ? 7u : 126u);++i) {
        float a = four ? values[i] : f8(i), b = four ? values[i+1] : f8(i+1);
        for (float v : {a,(a+b)/2,std::nextafter((a+b)/2,a),std::nextafter((a+b)/2,b)}) {
          x.push_back(v); x.push_back(-v);
        }
      }
      memory::dims shape{static_cast<int64_t>(x.size())};
      auto dt = four ? memory::data_type::f4_e2m1 : memory::data_type::f8_e4m3;
      memory::desc src(shape,memory::data_type::f32,memory::format_tag::a);
      memory::desc low(shape,dt,memory::format_tag::a);
      std::vector<uint8_t> codes(low.get_size()); std::vector<float> decoded(x.size());
      memory a(src,eng,x.data()), b(low,eng,codes.data()), c(src,eng,decoded.data());
      reorder::primitive_desc pd(eng,src,eng,low);
      reorder(pd).execute(str,a,b); str.wait();
      reorder(b,c).execute(str,b,c); str.wait();
      unsigned wrong=0, decode_wrong=0;
      for (size_t i=0;i<x.size();++i) {
        unsigned got=four ? (codes[i/2] >> (4*(i%2))) & 15 : codes[i];
        auto expected=nearest(x[i],four);
        if (got!=expected) {
          if (wrong < 4) std::cout << "mismatch input_bits=" << std::bit_cast<uint32_t>(x[i])
            << " got=" << got << " expected=" << expected << '\n';
          ++wrong;
        }
        float expected_value = four ? std::copysign(values[got&7], got&8 ? -1.0f : 1.0f) : f8(got);
        if (std::bit_cast<uint32_t>(decoded[i]) != std::bit_cast<uint32_t>(expected_value)) ++decode_wrong;
      }
      std::cout << (four ? "FP4" : "FP8") << " samples=" << x.size()
        << " encode_mismatches=" << wrong << " decode_mismatches=" << decode_wrong
        << " impl=" << pd.impl_info_str() << '\n';
    }
  } catch (const dnnl::error& e) {std::cerr << "dnnl status=" << e.status << ' ' << e.what() << '\n'; return 2;}
}
