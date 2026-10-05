#include <array>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
namespace c=hyle::compute;
template<class T>T take(hyle::Result<T> r){if(!r){std::fprintf(stderr,"invalid diagnostic: %.*s\n",int(r.error().message().size()),r.error().message().data());std::exit(2);}return std::move(*r);}
c::TensorDesc d(c::DataType type,std::initializer_list<std::int64_t> shape){return take(c::TensorDesc::dense(type,{shape.begin(),shape.size()}));}
void desc(const c::TensorDesc &x){
 std::printf("dtype=%d;shape=",int(x.data_type()));for(auto n:x.dimensions())std::printf("%lld,",(long long)n);
 std::printf(";strides=");for(auto n:x.byte_strides())std::printf("%lld,",(long long)n);std::printf(";plain=%d;dense=%d",x.is_plain(),x.is_dense());
}
struct Audit {
 std::array<c::Context,3> contexts;
 const char *names[3]{"ggml","oneDNN","Native"};
 std::size_t rows=0,invalid=0;
 Audit():contexts{take(c::Context::create(c::cpu::device(),c::ggml_cpu::provider())),take(c::Context::create(c::cpu::device(),c::onednn::provider())),take(c::Context::create(c::cpu::device(),c::native::provider()))}{}
 template<class Op> void test(const char *name,Op operation,std::initializer_list<c::TensorDesc> inputs){
  const c::Operation op(operation);
  const auto descriptions=std::span(inputs.begin(),inputs.size());
  const auto inferred=c::infer_single_output(op,descriptions);
  if(!inferred){++invalid;std::printf("%s\tSCHEMA_REJECT\t%.*s\n",name,int(inferred.error().message().size()),inferred.error().message().data());return;}
  for(std::size_t i=0;i<contexts.size();++i){
   auto prepared=c::prepare(contexts[i],op,descriptions,std::span(&*inferred,1),hyle::mem::system_resource());
   ++rows;std::printf("%s\t%s\t%s\t",name,names[i],prepared ? "ADMITTED" : prepared.error().IsUnimplemented() ? "UNIMPLEMENTED" : "OTHER_ERROR");
   for(auto &input:inputs){desc(input);std::printf("|");}std::printf("->");desc(*inferred);
   if(!prepared)std::printf("\t%.*s",int(prepared.error().message().size()),prepared.error().message().data());std::puts("");
  }
 }
};
int main(){
 Audit audit;
 const char *unary[]{"Abs","Neg","Exp","Log","Sqrt","Rsqrt","Tanh","Sigmoid","Relu","Gelu","GeluTanh","Silu","Softplus"};
 const char *binary[]{"Add","Subtract","Multiply","Divide","Minimum","Maximum"};
 const char *reduce[]{"ReduceSum","ReduceMean","ReduceMin","ReduceMax"};
 const char *gate[]{"GateSiluPacked","GateGeluPacked","GateGeluTanhPacked"};
 for(auto type:{c::DataType::kFloat32,c::DataType::kFloat16,c::DataType::kBFloat16}){
  const auto a=d(type,{2,8});
  for(int kind=0;kind<13;++kind)audit.test(unary[kind],c::Unary{c::UnaryKind(kind)},{a});
  for(int kind=0;kind<6;++kind)audit.test(binary[kind],c::Binary{c::BinaryKind(kind)},{a,d(type,{8})});
  for(int kind=0;kind<4;++kind)audit.test(reduce[kind],c::Reduce{.kind=c::ReductionKind(kind),.accumulation_type=c::DataType::kFloat32},{a});
  for(int kind=0;kind<3;++kind)audit.test(gate[kind],c::GatedActivation{.activation=c::GatedActivationKind(kind)},{d(type,{2,16})});
  audit.test("SelectBoolBroadcast",c::Select{},{d(c::DataType::kBool,{2,1}),a,d(type,{8})});
  audit.test("RmsNorm",c::RmsNorm{},{a,d(c::DataType::kFloat32,{8})});
  audit.test("LayerNorm",c::LayerNorm{},{a,d(c::DataType::kFloat32,{8}),d(c::DataType::kFloat32,{8})});
  audit.test("SoftmaxRank2",c::Softmax{},{a});
  audit.test("SoftmaxRank1",c::Softmax{},{d(type,{8})});
  audit.test("SoftmaxRank4Scale",c::Softmax{.scale=.5F},{d(type,{2,2,3,8})});
  audit.test("SoftmaxAxis0",c::Softmax{.axis=0},{a});
  audit.test("LogSoftmax",c::Softmax{.kind=c::SoftmaxKind::kLogSoftmax},{a});
  audit.test("SoftmaxBoolMask",c::Softmax{.mask=c::MaskKind::kBoolean},{a,d(c::DataType::kBool,{2,8})});
  audit.test("SoftmaxAdditiveMask",c::Softmax{.mask=c::MaskKind::kAdditive},{a,a});
  const auto q=d(type,{1,2,2,4}),kv=d(type,{1,3,2,4});
  audit.test("AttentionCausal",c::ScaledDotProductAttention{.causal=true},{q,kv,kv});
  audit.test("AttentionNonCausal",c::ScaledDotProductAttention{},{q,kv,kv});
  audit.test("AttentionBoolMask",c::ScaledDotProductAttention{.mask=c::MaskKind::kBoolean},{q,kv,kv,d(c::DataType::kBool,{1,2,2,3})});
  audit.test("AttentionHeadFirst",c::ScaledDotProductAttention{.layout=c::AttentionLayout::kBatchHeadSequenceDepth},{d(type,{1,2,2,4}),d(type,{1,2,3,4}),d(type,{1,2,3,4})});
  for(auto index:{c::DataType::kInt32,c::DataType::kInt64}){
   audit.test("Gather",c::Gather{},{d(type,{4,8}),d(index,{3})});
   audit.test("ScatterAdd",c::Scatter{.kind=c::ScatterKind::kAdd},{d(type,{4,8}),d(index,{3}),d(type,{3,8})});
   audit.test("ScatterAssign",c::Scatter{},{d(type,{4,8}),d(index,{3}),d(type,{3,8})});
   audit.test("RopeRank4Axis1",c::RoPE{.rotary_dimensions=8,.sequence_axis=1},{d(type,{1,3,2,8}),d(index,{3}),d(c::DataType::kFloat32,{4})});
   audit.test("RopeRank3Axis0",c::RoPE{.rotary_dimensions=8,.sequence_axis=0},{d(type,{3,2,8}),d(index,{3}),d(c::DataType::kFloat32,{4})});
  }
  audit.test("Concat",c::Concat{},{a,d(type,{1,8})});
  audit.test("Copy",c::Copy{},{a});
  audit.test("CastF32",c::Cast{.output_type=c::DataType::kFloat32},{a});
  audit.test("MatMulRowMajor",c::MatMul{},{a,d(type,{8,5})});
  audit.test("MatMulTransposed",c::MatMul{},{a,take(d(type,{5,8}).transposed())});
  audit.test("MatMulBatchTransposed",c::MatMul{},{d(type,{2,3,8}),take(d(type,{5,8}).transposed())});
  audit.test("MatMulBatchRowMajor",c::MatMul{},{d(type,{2,3,8}),d(type,{8,5})});
 }
 for(auto type:{c::DataType::kBool,c::DataType::kFloat64,c::DataType::kInt32,c::DataType::kInt64}){
  audit.test("WideCopy",c::Copy{},{d(type,{2,8})});
  audit.test("WideGather",c::Gather{},{d(type,{4,8}),d(c::DataType::kInt32,{3})});
  audit.test("WideConcat",c::Concat{},{d(type,{2,8}),d(type,{1,8})});
 }
 for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat8E4M3FN,c::DataType::kFloat8E5M2})audit.test("CastNarrow",c::Cast{.output_type=type},{d(c::DataType::kFloat32,{2,8})});
 audit.test("FillF32",c::Fill{.output_shape=take(c::TensorShape::create(std::array<std::int64_t,2>{2,8})),.output_type=c::DataType::kFloat32,.value=c::ScalarValue(2.F)},{});
 std::printf("SUMMARY provider_rows=%zu invalid_schema_cases=%zu\n",audit.rows,audit.invalid);
 return audit.invalid ? 3 : 0;
}
