#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#define NOMINMAX
#include <windows.h>
#include "cpu_program_abi.h"

using hyle::compute::ggml_runtime::CpuProgram;
const CpuProgram *program;
void execute(ggml_context *ctx, ggml_tensor *output) {
  auto *graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, output);
  auto params = ggml_threadpool_params_default(3); params.poll = 0;
  int failure = 0;
  auto *pool = program->create_pool(&params, &failure);
  if (!pool || failure) std::exit(2);
  auto plan = program->plan(graph, 3, pool);
  std::vector<unsigned char> work(plan.work_size); plan.work_data = work.data();
  if (program->compute(graph, &plan) != GGML_STATUS_SUCCESS) std::exit(3);
  program->free_pool(pool);
}
double activation(int kind, double x) {
  if (kind == 0) return std::isnan(x) ? x : x > 0 ? x : 0;
  if (std::isinf(x) && x < 0) return 0;
  if (kind == 1) { const double z = std::exp(-std::abs(x)); return x * (x >= 0 ? 1/(1+z) : z/(1+z)); }
  if (kind == 2) return .5*x*(1+std::erf(x*.7071067811865475244));
  return .5*x*(1+std::tanh(.7978845608028653559*x*(1+.044715*x*x)));
}
int main(int argc, char **argv) {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  if (argc != 2) return 1;
  auto module = LoadLibraryA(argv[1]);
  auto entry = reinterpret_cast<const CpuProgram *(*)() noexcept>(module ? GetProcAddress(module, "hyle_ggml_cpu_program_v1") : nullptr);
  if (!entry) return 4;
  program = entry();
  if (program->abi != 1 || program->tensor_bytes != sizeof(ggml_tensor) || program->plan_bytes != sizeof(ggml_cplan)) return 5;
  program->initialize();
  const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 25> cases{-inf, nan, inf, -0.f, 0.f, -1.e30f, -10000.f, -100.f, -90.f, -80.f, -20.f, -8.f, -5.f, -3.f, -1.f, -.001f, .001f, 1.f, 3.f, 5.f, 8.f, 20.f, 100.f, 10000.f, 1.e30f};
  const char *names[]{"relu", "silu", "gelu-erf", "gelu-tanh"};
  for (int kind = 0; kind != 4; ++kind) for (bool gated : {false, true}) {
    if (kind == 0 && gated) continue;
    auto *ctx = ggml_init({4*1024*1024, nullptr, false});
    auto *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cases.size(), 3);
    auto *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cases.size(), 3);
    for (std::size_t i = 0; i < cases.size()*3; ++i) {
      static_cast<float *>(a->data)[i] = cases[i%cases.size()];
      static_cast<float *>(b->data)[i] = i/cases.size() == 0 ? 1.f : i/cases.size() == 1 ? -2.f : inf;
    }
    ggml_tensor *y = gated ? (kind == 1 ? ggml_swiglu_split(ctx,a,b) : kind == 2 ? ggml_geglu_erf_split(ctx,a,b) : ggml_geglu_split(ctx,a,b))
                           : (kind == 0 ? ggml_relu(ctx,a) : kind == 1 ? ggml_silu(ctx,a) : kind == 2 ? ggml_gelu_erf(ctx,a) : ggml_gelu(ctx,a));
    execute(ctx, y);
    int special_mismatches = 0; double max_abs = 0, max_rel = 0;
    std::printf("%s%s:\n", gated ? "gate-" : "", names[kind]);
    for (std::size_t i=0;i<cases.size()*3;++i) {
      const double expected = activation(kind, cases[i%cases.size()]) * (gated ? static_cast<float *>(b->data)[i] : 1.f);
      const double actual = static_cast<float *>(y->data)[i];
      if (std::isnan(expected) != std::isnan(actual) || std::isinf(expected) != std::isinf(actual) || (std::isinf(expected) && expected != actual)) ++special_mismatches;
      if (std::isfinite(expected) && std::isfinite(actual)) {
        max_abs = std::max(max_abs, std::abs(expected-actual));
        if (expected != 0) max_rel = std::max(max_rel, std::abs((expected-actual)/expected));
      }
      if (i<cases.size()) std::printf("  x=%+.9g actual=%+.9g expected=%+.9g\n",cases[i],actual,expected);
    }
    std::printf("summary special_mismatches=%d max_abs=%g max_rel=%g\n",special_mismatches,max_abs,max_rel);
    ggml_free(ctx);
  }
  for (bool shared : {true,false}) {
    auto *ctx = ggml_init({4*1024*1024,nullptr,false});
    auto *a = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,7,3,2,2);
    auto *b = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,7,5,shared ? 1 : 2,shared ? 1 : 2);
    for(int i=0;i<7*3*4;++i) static_cast<float *>(a->data)[i]=float(i%13-6)/8.f;
    for(int i=0;i<ggml_nelements(b);++i) static_cast<float *>(b->data)[i]=float(i%11-5)/16.f;
    auto *y = ggml_mul_mat(ctx,b,a); ggml_prec_set_acc(y,GGML_PREC_F32); execute(ctx,y);
    double maximum=0;
    for(int batch=0;batch<4;++batch) for(int row=0;row<3;++row) for(int col=0;col<5;++col) {
      double expected=0;
      for(int k=0;k<7;++k) expected+=static_cast<float *>(a->data)[(batch*3+row)*7+k]*static_cast<float *>(b->data)[((shared ? 0 : batch)*5+col)*7+k];
      maximum=std::max(maximum,std::abs(expected-static_cast<float *>(y->data)[(batch*3+row)*5+col]));
    }
    std::printf("F32 rank4 MatMul shared=%d max_abs=%g %s\n",shared,maximum,maximum==0 ? "PASS" : "FAIL");
    if(maximum!=0) return 6;
    ggml_free(ctx);
  }
  FreeLibrary(module);
}
