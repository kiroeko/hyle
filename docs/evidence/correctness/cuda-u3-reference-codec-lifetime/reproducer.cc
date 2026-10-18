#include <cstdio>
#include <span>
#include "tests/compute_models/cuda_sidecar_quant_reference.h"
#include "tests/compute_models/cpu_c5_gguf_fixture.h"

std::vector<float> fresh_original(ggml_type type, std::span<const std::byte> packed,
                                 std::span<const float> input, std::int64_t k) {
  cuda_sidecar_quant_reference::Backend backend;
  cuda_sidecar_quant_reference::Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*4+ggml_graph_overhead_custom(8,false),nullptr,true});
  auto *weight=ggml_new_tensor_2d(storage.context,type,k,8);
  auto *activation=ggml_new_tensor_2d(storage.context,GGML_TYPE_F32,k,1);
  auto *output=ggml_mul_mat(storage.context,weight,activation);
  ggml_prec_set_acc(output,GGML_PREC_F32);
  auto *graph=ggml_new_graph_custom(storage.context,8,false);
  ggml_build_forward_expand(graph,output);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  ggml_backend_tensor_set(weight,packed.data(),0,packed.size());
  ggml_backend_tensor_set(activation,input.data(),0,input.size_bytes());
  if (ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
    throw std::runtime_error("forensic original program failed");
  std::vector<float> result(8);
  ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
  return result;
}

std::vector<float> same_address_type_transition(std::span<const std::byte> packed,
                                              std::span<const float> input, std::int64_t k) {
  cuda_sidecar_quant_reference::Backend backend;
  cuda_sidecar_quant_reference::Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*4+ggml_graph_overhead_custom(8,false),nullptr,true});
  auto *weight=ggml_new_tensor_2d(storage.context,GGML_TYPE_IQ3_S,k,8);
  auto *activation=ggml_new_tensor_2d(storage.context,GGML_TYPE_F32,k,1);
  auto *output=ggml_mul_mat(storage.context,weight,activation);
  ggml_prec_set_acc(output,GGML_PREC_F32);
  auto *graph=ggml_new_graph_custom(storage.context,8,false);
  ggml_build_forward_expand(graph,output);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  ggml_backend_tensor_set(weight,packed.data(),0,packed.size());
  ggml_backend_tensor_set(activation,input.data(),0,input.size_bytes());
  for(int warmup=0;warmup!=2;++warmup)
    if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
      throw std::runtime_error("forensic warmup failed");
  // Deterministic equivalent of a newly allocated logical graph reusing the
  // old metadata/data addresses. Equal byte strides, different source codec.
  weight->type=GGML_TYPE_Q3_K;
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
    throw std::runtime_error("forensic transition failed");
  std::vector<float> result(8);
  ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
  return result;
}

int main() {
  using namespace cpu_c5_gguf_fixture;
  const auto item = std::find_if(std::begin(in::cases), std::end(in::cases),
      [](const auto &value) { return std::string_view(value.id) == "g3560f36d25c826bd"; });
  if (item == std::end(in::cases)) return 2;
  Bank bank(*item);
  std::printf("{\"k\":%lld,\"n\":%lld,\"rows\":[", (long long)item->k, (long long)item->n);
  bool first = true;
  for (int pass = 0; pass != 2; ++pass) {
    const auto input = activation(3, item->k, 3 + pass*16);
    for (int slot = 0; slot != 3; ++slot) {
      const auto values = std::span(input).subspan(slot*item->k, item->k);
      const auto original = bank.dots(values, true);
      if (!first) std::printf(","); first = false;
      std::printf("{\"pass\":%d,\"slot\":%d,\"original_cuda\":[", pass, slot);
      for (int row = 0; row != prototypes; ++row)
        std::printf("%s%.9g", row ? "," : "", original[row]);
      std::printf("],\"independent_decoded_f32\":[");
      for (int row = 0; row != prototypes; ++row) {
        double dot = 0;
        for (std::int64_t col = 0; col != item->k; ++col)
          dot += double(bank.decoded[row*item->k+col])*values[col];
        std::printf("%s%.9g", row ? "," : "", dot);
      }
      std::printf("]}");
    }
  }
  const auto input=activation(3,item->k,19);
  const auto selected=std::span(input).subspan(item->k,item->k);
  std::printf("],\"same_bytes_as_iq3_s\":[");
  const auto wrong=fresh_original(GGML_TYPE_IQ3_S,bank.packed,selected,item->k);
  for (int row=0;row!=prototypes;++row)
    std::printf("%s%.9g",row?",":"",wrong[row]);
  const auto transition=same_address_type_transition(bank.packed,selected,item->k);
  std::printf("],\"same_address_iq3_s_to_q3_k\":[");
  for(int row=0;row!=prototypes;++row)
    std::printf("%s%.9g",row?",":"",transition[row]);
  std::printf("]}\n");
}
