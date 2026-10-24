#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

#include "dspark_original_shared_reference.h"

// Opt-in, failure-only test evidence. These are independent scalar equations,
// not a replacement for the native result or a change to its acceptance bound.
namespace packed_projection_failure_capture {
namespace q8=dspark_original_shared_reference;
inline void save(const std::filesystem::path &path,std::span<const std::byte> bytes) {
  std::ofstream file(path,std::ios::binary);file.exceptions(std::ios::badbit|std::ios::failbit);
  file.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
}
inline void q8_failure(std::string_view role,std::span<const float> input,std::span<const std::byte> row,
    std::size_t output,float actual,double center,double bound)noexcept {
#if defined(_WIN32)
  char *buffer=nullptr;std::size_t length=0;
  if(_dupenv_s(&buffer,&length,"HYLE_PACKED_FAILURE_CAPTURE_DIR"))return;
  const std::unique_ptr<char,decltype(&std::free)> owner(buffer,&std::free);
  const char *root=owner.get();
#else
  const char *root=std::getenv("HYLE_PACKED_FAILURE_CAPTURE_DIR");
#endif
  if(!root||!*root)return;
  try {
    if(input.empty()||input.size()%32||row.size()!=input.size()/32*34)
      throw std::runtime_error("failure capture Q8 row extent");
    const std::filesystem::path parent(root);std::filesystem::create_directories(parent);
    std::filesystem::path directory;
    for(std::size_t i=0;i<1024;++i) {
      const auto candidate=parent/("failure-"+std::to_string(i));
      if(std::filesystem::create_directory(candidate)){directory=candidate;break;}
    }
    if(directory.empty())throw std::runtime_error("failure capture directory exhausted");
    save(directory/"input.f32",std::as_bytes(input));save(directory/"weight-row.q8_0",row);
    save(directory/"actual-y.f32",std::as_bytes(std::span<const float>(&actual,1)));
    std::ofstream meta(directory/"metadata.txt");meta.exceptions(std::ios::badbit|std::ios::failbit);
    meta<<std::setprecision(17)<<"role="<<role<<"\nK="<<input.size()<<"\noutput_row="<<output
        <<"\nactual="<<actual<<"\noriginal_oracle="<<center<<"\noriginal_bound="<<bound
        <<"\ncode_sources=independent CPU scalar/AVX equations; not captured native codes\n";
    std::ofstream differences(directory/"quantization.txt");differences.exceptions(std::ios::badbit|std::ios::failbit);
    differences<<std::setprecision(17);
    q8::detail::Sum scalar_sum,avx_sum;std::vector<std::byte> scalar_bytes,avx_bytes;
    for(std::size_t group=0;group<input.size()/32;++group) {
      const auto values=std::span<const float,32>(input.data()+group*32,32);
      const auto scalar=q8::detail::quantize(values,q8::ActivationCodec::kCpuScalarQ8_0,false);
      const auto avx=q8::detail::quantize(values,q8::ActivationCodec::kCpuAvxQ8_0,false);
      float maximum=0;for(float value:values)maximum=std::max(maximum,std::abs(value));
      const float d=q8::detail::rn_divide(maximum,127.F);
      const float scalar_id=maximum?q8::detail::rn_divide(1.F,d):0.F;
      const float avx_id=maximum?q8::detail::rn_divide(127.F,maximum):0.F;
      differences<<"group="<<group<<" max="<<maximum<<" d="<<d<<" scalar_id="<<scalar_id<<" avx_id="<<avx_id
          <<" d_bits=0x"<<std::hex<<std::bit_cast<std::uint32_t>(d)<<" scalar_id_bits=0x"<<std::bit_cast<std::uint32_t>(scalar_id)
          <<" avx_id_bits=0x"<<std::bit_cast<std::uint32_t>(avx_id)<<std::dec<<'\n';
      int scalar_dot=0,avx_dot=0;
      for(std::size_t lane=0;lane<32;++lane) {
        const auto w=q8::detail::code(row[group*34+2+lane]);
        scalar_dot+=w*scalar.codes[lane];avx_dot+=w*avx.codes[lane];
        if(scalar.codes[lane]!=avx.codes[lane])differences<<"different_code group="<<group<<" lane="<<lane
            <<" input="<<values[lane]<<" bits=0x"<<std::hex<<std::bit_cast<std::uint32_t>(values[lane])<<std::dec
            <<" weight="<<w<<" scalar_q="<<scalar.codes[lane]<<" avx_q="<<avx.codes[lane]
            <<" scalar_scaled="<<q8::detail::rn_multiply(values[lane],scalar_id)
            <<" avx_scaled="<<q8::detail::rn_multiply(values[lane],avx_id)<<'\n';
      }
      const double weight_scale=q8::detail::scale(row,group*34);
      scalar_sum.add(weight_scale*double(scalar.scale)*double(scalar_dot));
      avx_sum.add(weight_scale*double(avx.scale)*double(avx_dot));
      const auto append=[](std::vector<std::byte> &bytes,const q8::detail::ActivationBlock &a) {
        const auto bits=q8::detail::half_bits(a.scale);bytes.push_back(std::byte(bits&255));bytes.push_back(std::byte(bits>>8));
        for(int q:a.codes)bytes.push_back(std::byte(std::uint8_t(std::int8_t(q))));
      };
      append(scalar_bytes,scalar);append(avx_bytes,avx);
    }
    save(directory/"independent-scalar.q8_0",scalar_bytes);save(directory/"independent-avx.q8_0",avx_bytes);
    meta<<"scalar_center="<<scalar_sum.value()<<"\navx_center="<<avx_sum.value()
        <<"\nscalar_error="<<std::abs(double(actual)-scalar_sum.value())<<"\navx_error="<<std::abs(double(actual)-avx_sum.value())<<'\n';
    std::fprintf(stderr,"packed projection failure capture: %s\n",directory.string().c_str());
  } catch(const std::exception &error) {
    std::fprintf(stderr,"packed projection capture failed (original numerical failure retained): %s\n",error.what());
  }
}
} // namespace packed_projection_failure_capture
