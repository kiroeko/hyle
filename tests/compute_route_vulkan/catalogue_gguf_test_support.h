#pragma once

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-vulkan.h>

#include "catalogue_test_support.h"
#include "gguf_test_support.h"
#include "iq1_test_support.h"
#include "../compute_models/cpu_c5_gguf_inputs.h"

namespace vulkan_catalogue_gguf {
namespace c = hyle::compute;
namespace in = cpu_c5_inputs;
class Backend final {
public:
  Backend() : value_(ggml_backend_vk_init(0)) {
    if (!value_ || !ggml_backend_is_vk(value_))
      throw std::runtime_error("catalogue reference requires original Vulkan backend");
  }
  ~Backend() {ggml_backend_free(value_);}
  ggml_backend_t get() const {return value_;}
private:
  ggml_backend_t value_;
};
struct Storage final {
  ggml_context *context=nullptr;ggml_backend_buffer_t buffer=nullptr;
  ~Storage(){ggml_backend_buffer_free(buffer);ggml_free(context);}
};

inline std::array<float,8> native_dots(ggml_type type,std::int64_t k,
    std::span<const std::byte> packed,std::span<const float> input) {
  static Backend backend;
  Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*4+ggml_graph_overhead_custom(8,false),nullptr,true});
  if(!storage.context) throw std::runtime_error("original Vulkan reference metadata OOM");
  auto *w=ggml_new_tensor_2d(storage.context,type,k,8);
  auto *a=ggml_new_tensor_2d(storage.context,GGML_TYPE_F32,k,1);
  auto *out=ggml_mul_mat(storage.context,w,a);
  if(!ggml_prec_set_acc(out,GGML_PREC_F32)||ggml_nbytes(w)!=packed.size())
    throw std::runtime_error("original Vulkan reference precision/bytes differ");
  auto *graph=ggml_new_graph_custom(storage.context,8,false);ggml_build_forward_expand(graph,out);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  if(!storage.buffer) throw std::runtime_error("original Vulkan reference storage OOM");
  ggml_backend_tensor_set(w,packed.data(),0,packed.size());
  ggml_backend_tensor_set(a,input.data(),0,input.size_bytes());
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
    throw std::runtime_error("original Vulkan reference native completion failed");
  std::array<float,8> values{};ggml_backend_tensor_get(out,values.data(),0,sizeof(values));return values;
}

struct Bank final {
  ggml_type type;std::int64_t k;std::size_t stride;
  std::vector<std::byte> packed;std::vector<float> decoded;float l1=0;
  explicit Bank(const in::Case &item):type(item.native),k(item.k),
      stride(type==GGML_TYPE_COUNT ? static_cast<std::size_t>(k/256)*
          (item.quant==c::GgufQuantizedType::kIq1Xs?46:item.quant==c::GgufQuantizedType::kIq1Xxs?42:38)
                                  :ggml_row_size(type,k)),packed(8*stride),decoded(8*k) {
    if(type==GGML_TYPE_COUNT) {
      // Registered original IQ1 bytes/grids; no unavailable vendor enum is
      // passed to a native ggml tensor or inferred as backend capability.
      vulkan_iq1_test::Inputs source(item.quant,static_cast<int>(k),1);
      std::copy_n(source.weight.begin(),packed.size(),packed.begin());
      for(int row=0;row<8;++row) for(int col=0;col<k;++col)
        decoded[row*k+col]=source.decoded(row/source.n,row%source.n,col);
    } else if(item.packed) {
      auto source=d5_gguf_test::values(8,k,37);
      auto result=vulkan_gguf_test::pack(item.quant,source,8,static_cast<int>(k),packed.size());
      packed=std::move(result.bytes);decoded=std::move(result.decoded);
    } else {
      const auto *traits=ggml_get_type_traits(type);std::vector<float> source(k);
      for(int row=0;row<8;++row) {
        for(std::int64_t col=0;col<k;++col) source[col]=static_cast<float>((row*13+col*7)%23-11)/32;
        auto *destination=packed.data()+row*stride;
        if(type==GGML_TYPE_F32) std::memcpy(destination,source.data(),stride);
        else if(type==GGML_TYPE_I32) for(std::int64_t col=0;col<k;++col) {
          const auto value=static_cast<std::int32_t>((row*13+col*7)%256);std::memcpy(destination+col*4,&value,4);
        } else traits->from_float_ref(source.data(),destination,k);
        auto *output=decoded.data()+row*k;
        if(type==GGML_TYPE_F32) std::memcpy(output,destination,stride);
        else if(type==GGML_TYPE_I32) for(std::int64_t col=0;col<k;++col) {
          std::int32_t value;std::memcpy(&value,destination+col*4,4);output[col]=static_cast<float>(value);
        } else traits->to_float(destination,output,k);
      }
    }
    for(int row=0;row<8;++row) {
      double norm=0;for(std::int64_t col=0;col<k;++col) norm+=std::abs(decoded[row*k+col]);
      l1=std::max(l1,static_cast<float>(norm));
    }
  }
  std::vector<std::byte> rows(std::int64_t count,std::int64_t first=0) const {
    std::vector<std::byte> result(count*stride);
    for(std::int64_t row=0;row<count;++row)
      std::memcpy(result.data()+row*stride,packed.data()+((first+row)%8)*stride,stride);
    return result;
  }
  std::array<float,8> dots(std::span<const float> input,bool f32_activation=false) const {
    if(type!=GGML_TYPE_COUNT && ggml_is_quantized(type)) return native_dots(type,k,packed,input);
    std::array<float,8> result{};
    for(int row=0;row<8;++row) {
      double sum=0;
      for(std::int64_t col=0;col<k;++col) {
        auto value=input[col];
        if(!f32_activation && type==GGML_TYPE_BF16) value=ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
        if(!f32_activation && type==GGML_TYPE_F16) value=ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        sum+=static_cast<double>(decoded[row*k+col])*value;
      }
      result[row]=static_cast<float>(sum);
    }
    return result;
  }
};
} // namespace vulkan_catalogue_gguf

// Test-only carrier hook: reuse common exact profiles and fragment equations;
// no CPU/backend capability macro or CPU inference fallback is manufactured.
#define HYLE_C5_GGUF_EXTERNAL_BANK vulkan_catalogue_gguf::Bank
#include "../compute_models/cpu_c5_gguf_fixture.h"
#undef HYLE_C5_GGUF_EXTERNAL_BANK

namespace vulkan_catalogue_gguf {
inline void check(const in::Case &item) {
  namespace f=cpu_c5_gguf_fixture;const auto profile=f::profile(in::aliases[item.alias_begin]);
  SCOPED_TRACE(item.id);
  SCOPED_TRACE(profile.artifact_repository);
  SCOPED_TRACE(profile.variant);
  std::printf("VULKAN_CATALOGUE_GGUF id=%s K=%lld N=%lld originalE=%lld executedE=%lld\n",item.id,
      static_cast<long long>(item.k),static_cast<long long>(item.n),static_cast<long long>(item.e),
      static_cast<long long>(item.kind==in::Kind::kExpert?std::min<std::int64_t>(item.e,8):item.e));
  const Bank bank(item);
  if(item.kind==in::Kind::kExpert || item.kind==in::Kind::kHeadGroup)
    vulkan_catalogue_test::run(f::grouped(item,bank,&profile));
  else if(item.kind==in::Kind::kEmbedding || item.kind==in::Kind::kPosition || item.kind==in::Kind::kInteger)
    vulkan_catalogue_test::run(f::lookup(item,bank,&profile));
  else if(item.kind==in::Kind::kHead) for(std::int64_t first=0;first<item.n;first+=4096) {
    vulkan_catalogue_test::run(f::projection(item,bank,&profile,std::min<std::int64_t>(4096,item.n-first),first));
    ASSERT_FALSE(::testing::Test::HasFailure());
  } else vulkan_catalogue_test::run(f::projection(item,bank,&profile,item.n));
}
inline void identities() {
  for(unsigned index=0;index<std::size(in::profiles);++index) {
    const auto profile=cpu_c5_gguf_fixture::profile(index);SCOPED_TRACE(profile.variant);
    const in::Case *selected=nullptr;
    for(const auto &item:in::cases) {
      if(item.kind!=in::Kind::kProjection) continue;
      const auto aliases=std::span(in::aliases).subspan(item.alias_begin,item.alias_count);
      if(std::ranges::find(aliases,index)==aliases.end()) continue;
      if(!selected || item.k*item.n<selected->k*selected->n) selected=&item;
    }
    ASSERT_NE(selected,nullptr);const Bank bank(*selected);
    vulkan_catalogue_test::run(cpu_c5_gguf_fixture::projection(*selected,bank,&profile,selected->n));
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
} // namespace vulkan_catalogue_gguf
