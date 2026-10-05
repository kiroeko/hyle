#define main previous_probe_main
#include "probe.cc"
#undef main
#include <cstring>

float stored(ggml_tensor *value,std::size_t i){
 if(value->type==GGML_TYPE_F32)return static_cast<float *>(value->data)[i];
 const auto bits=static_cast<std::uint16_t *>(value->data)[i];
 if(value->type==GGML_TYPE_BF16){std::uint32_t f=std::uint32_t(bits)<<16;float x;std::memcpy(&x,&f,4);return x;}
 const int exponent=(bits>>10)&31,fraction=bits&1023;
 const float x=exponent==0 ? std::ldexp(float(fraction),-24) : std::ldexp(float(1024+fraction),exponent-25);
 return bits&0x8000 ? -x : x;
}
int main(int argc,char **argv){
 SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
 if(argc!=2)return 1;
 auto module=LoadLibraryA(argv[1]);auto entry=reinterpret_cast<const CpuProgram *(*)() noexcept>(module ? GetProcAddress(module,"hyle_ggml_cpu_program_v1") : nullptr);
 if(!entry)return 2;program=entry();if(program->abi!=1 || program->tensor_bytes!=sizeof(ggml_tensor) || program->plan_bytes!=sizeof(ggml_cplan))return 3;program->initialize();
 for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16})for(bool masked:{false,true}){
  auto *ctx=ggml_init({8*1024*1024,nullptr,false});auto *a=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,7,3,2,2);
  for(int i=0;i<84;++i)static_cast<float *>(a->data)[i]=float(i%13-6)/8;
  auto *source=type==GGML_TYPE_F32 ? a : ggml_cast(ctx,a,type);
  source=type==GGML_TYPE_F32 ? source : ggml_cast(ctx,source,GGML_TYPE_F32);
  auto *mask=masked ? ggml_new_tensor_2d(ctx,GGML_TYPE_F32,7,3) : nullptr;
  if(mask)for(int i=0;i<21;++i)static_cast<float *>(mask->data)[i]=i%7==0 ? -2.F : 0.F;
  auto *y=ggml_soft_max_ext(ctx,source,mask,.5F,0);if(type!=GGML_TYPE_F32)y=ggml_cast(ctx,y,type);execute(ctx,y);
  double maximum=0;
  for(int row=0;row<12;++row){double sum=0;for(int col=0;col<7;++col)sum+=std::exp(double(static_cast<float *>(a->data)[row*7+col])*.5+(masked && col==0 ? -2 : 0));
   for(int col=0;col<7;++col){const double expected=std::exp(double(static_cast<float *>(a->data)[row*7+col])*.5+(masked && col==0 ? -2 : 0))/sum;maximum=std::max(maximum,std::abs(expected-stored(y,row*7+col)));}}
  const auto tolerance=type==GGML_TYPE_F32 ? 2.e-6 : type==GGML_TYPE_F16 ? 2.e-4 : 2.e-3;
  std::printf("rank4 scaled softmax type=%s additive_mask=%d max_abs=%g %s\n",ggml_type_name(type),masked,maximum,maximum<=tolerance ? "PASS" : "FAIL");if(maximum>tolerance)return 4;ggml_free(ctx);
 }
 {
  auto *ctx=ggml_init({8*1024*1024,nullptr,false});auto *q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,4,2,2,1);auto *k=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,4,3,2,1);auto *v=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,4,3,2,1);
  for(int i=0;i<16;++i)static_cast<float *>(q->data)[i]=float(i%9-4)/8;
  for(int i=0;i<24;++i){static_cast<float *>(k->data)[i]=float(i%11-5)/8;static_cast<float *>(v->data)[i]=float(i%7-3)/4;}
  auto *scores=ggml_mul_mat(ctx,k,q);ggml_prec_set_acc(scores,GGML_PREC_F32);auto *probabilities=ggml_soft_max_ext(ctx,scores,nullptr,.5F,0);
  auto *weights=ggml_cont(ctx,ggml_transpose(ctx,v));auto *y=ggml_mul_mat(ctx,weights,probabilities);ggml_prec_set_acc(y,GGML_PREC_F32);execute(ctx,y);double maximum=0;
  for(int head=0;head<2;++head)for(int row=0;row<2;++row){std::array<double,3> score{};double sum=0;
   for(int key=0;key<3;++key){for(int d=0;d<4;++d)score[key]+=double(static_cast<float *>(q->data)[(head*2+row)*4+d])*static_cast<float *>(k->data)[(head*3+key)*4+d];score[key]=std::exp(score[key]*.5);sum+=score[key];}
   for(int d=0;d<4;++d){double expected=0;for(int key=0;key<3;++key)expected+=score[key]/sum*static_cast<float *>(v->data)[(head*3+key)*4+d];maximum=std::max(maximum,std::abs(expected-stored(y,(head*2+row)*4+d)));}}
  std::printf("noncausal original whole attention region max_abs=%g %s\n",maximum,maximum<=2.e-6 ? "PASS" : "FAIL");if(maximum>2.e-6)return 5;ggml_free(ctx);
 }
 for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}){
  auto *ctx=ggml_init({4*1024*1024,nullptr,false});auto *a=ggml_new_tensor_2d(ctx,type==GGML_TYPE_BF16 ? GGML_TYPE_F32 : type,7,3);auto *y=ggml_fill(ctx,a,2.F);if(type==GGML_TYPE_BF16)y=ggml_cast(ctx,y,type);execute(ctx,y);
  for(int i=0;i<21;++i)if(stored(y,i)!=2.F)return 6;
  std::printf("original Fill type=%s (bf16 through F32 cast) PASS\n",ggml_type_name(type));ggml_free(ctx);
 }
 {
  auto *ctx=ggml_init({4*1024*1024,nullptr,false});auto *base=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,4),*updates=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,2),*indices=ggml_new_tensor_1d(ctx,GGML_TYPE_I64,2);
  std::fill_n(static_cast<float *>(base->data),16,0.F);for(int i=0;i<8;++i)static_cast<float *>(updates->data)[i]=float(i/4+1);
  static_cast<std::int64_t *>(indices->data)[0]=3;static_cast<std::int64_t *>(indices->data)[1]=1;
  auto *y=ggml_set_rows(ctx,base,updates,indices);execute(ctx,y);for(int row=0;row<4;++row)for(int col=0;col<4;++col)if(stored(y,row*4+col)!=(row==3 ? 1.F : row==1 ? 2.F : 0.F))return 7;
  std::puts("original i64 row assign unique indices PASS; duplicate destinations are upstream undefined, NOT public last-wins qualification");ggml_free(ctx);
 }
 FreeLibrary(module);
}
