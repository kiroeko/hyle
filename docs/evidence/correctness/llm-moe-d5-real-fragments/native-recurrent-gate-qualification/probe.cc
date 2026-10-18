#define NOMINMAX
#include <windows.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

constexpr int D = 128;
struct Metric {
  const char* name;
  double max_abs{}, max_ratio{};
  uint64_t failures{}, observations{};
  void add(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) std::abort();
    for (size_t i = 0; i < actual.size(); ++i) {
      const double error = std::abs(double(actual[i]) - expected[i]);
      const double ratio = error / (2e-6 + 4e-5 * std::abs(double(expected[i])));
      ++observations;
      if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || !(ratio <= 1)) ++failures;
      max_abs = std::max(max_abs, error); max_ratio = std::max(max_ratio, ratio);
    }
  }
};
void upload(ggml_tensor* t, const std::vector<float>& v) {
  if (ggml_nbytes(t) != v.size() * sizeof(float)) std::abort();
  ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}
std::vector<float> read(ggml_tensor* t) {
  std::vector<float> v(size_t(ggml_nelements(t)));
  ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float)); return v;
}
template<class T> std::vector<T> load(const std::string& path, size_t count) {
  std::ifstream in(path, std::ios::binary); std::vector<T> result(count);
  if (!in.read(reinterpret_cast<char*>(result.data()), std::streamsize(count * sizeof(T)))) {
    std::fprintf(stderr, "Cannot read %s\n", path.c_str()); std::exit(2);
  }
  return result;
}
double sigmoid(double x) { return x >= 0 ? 1 / (1 + std::exp(-x)) : std::exp(x) / (1 + std::exp(x)); }

bool run(ggml_backend_t backend, int heads, bool official, bool local, const std::string& fixture) {
  const int keys = official ? heads : 16;
  ggml_init_params params{8 * 1024 * 1024, nullptr, true};
  auto* ctx = ggml_init(params); if (!ctx) std::abort();
  auto tensor = [&](int x, int y) { return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, x, y, 1, 1); };
  auto *q = tensor(D, keys), *k = tensor(D, keys), *v = tensor(D, heads);
  auto *a = tensor(1, heads), *b = tensor(1, heads), *dt = tensor(1, heads), *alpha = tensor(1, heads);
  auto* state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, heads, 1);
  auto *z = tensor(D, heads), *scale = tensor(D, 1);
  auto* g = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, a, dt)), alpha);
  auto* beta = ggml_sigmoid(ctx, b);
  auto* packed = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
  auto* core = ggml_reshape_4d(ctx, ggml_view_1d(ctx, packed, D * heads, 0), D, heads, 1, 1);
  auto* state_out = ggml_view_1d(ctx, packed, D * D * heads, sizeof(float) * D * heads);
  auto* normalized = ggml_mul(ctx, ggml_rms_norm(ctx, core, 1e-6f), scale);
  auto* output = ggml_swiglu_split(ctx, z, normalized);
  auto* graph = ggml_new_graph_custom(ctx, 128, false);
  ggml_build_forward_expand(graph, output);
  ggml_build_forward_expand(graph, state_out);
  auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); if (!buffer) std::abort();
  std::vector<float> av(heads), bv(heads), dtv(heads), alphav(heads);
  std::vector<float> qv(D * keys), kv(D * keys), vv(D * heads), zv(D * heads), sv(D);
  std::vector<float> ref_state(D * D * heads), device_state(ref_state.size());
  for (size_t i = 0; i < ref_state.size(); ++i) ref_state[i] = float(int((i * 7 + 5) % 29) - 14) / 256.f;
  device_state = ref_state;
  for (int i = 0; i < D; ++i) sv[i] = 0.5f + float(i % 7) / 16.f;
  if (local) {
    auto logs = load<float>(fixture + "/tensors/model_language_model_layers_0_linear_attn_A_log.bin", 16);
    auto biases = load<uint16_t>(fixture + "/tensors/model_language_model_layers_0_linear_attn_dt_bias.bin", 16);
    for (int h = 0; h < heads; ++h) {
      alphav[h] = -std::exp(logs[h]);
      uint32_t bits = uint32_t(biases[h]) << 16; std::memcpy(&dtv[h], &bits, 4);
    }
  } else {
    const std::array<float, 6> strengths{1.f/256, 1.f/16, .5f, 1.f, 4.f, 16.f};
    for (int h = 0; h < heads; ++h) { alphav[h] = -strengths[h % 6]; dtv[h] = float(h % 9 - 4) / 16.f; }
  }
  upload(dt, dtv); upload(alpha, alphav); upload(scale, sv);
  std::array<Metric, 5> metrics{{{"g"}, {"beta"}, {"core"}, {"state"}, {"output"}}};
  const std::array<float, 12> tails{-80,-40,-20,-15,-8,-4,-1,0,1,4,20,40};
  for (int step = 0; step < 64; ++step) {
    for (int h = 0; h < keys; ++h) {
      const int original_head = official ? h / (heads / 16) : h;
      double qsum = 1e-6, ksum = 1e-6;
      for (int d = 0; d < D; ++d) {
        const int i = h * D + d;
        qv[i] = float((original_head * 13 + d * 7 + step * 3) % 37 - 18) / 128.f;
        kv[i] = float((original_head * 5 + d * 11 + step * 17) % 41 - 20) / 128.f;
        qsum += double(qv[i]) * qv[i]; ksum += double(kv[i]) * kv[i];
      }
      for (int d = 0; d < D; ++d) { qv[h*D+d] = float(qv[h*D+d]/std::sqrt(qsum)); kv[h*D+d] = float(kv[h*D+d]/std::sqrt(ksum)); }
    }
    for (int h = 0; h < heads; ++h) {
      const float x = (step % 4 == 3) ? float((h * 7 + step) % 23 - 11) / 4.f : tails[(h + step) % tails.size()];
      av[h] = x - dtv[h]; bv[h] = float((h * 11 + step * 7) % 81 - 40);
      for (int d = 0; d < D; ++d) {
        vv[h*D+d] = float((h * 7 + d * 3 + step * 13) % 43 - 21) / 128.f;
        zv[h*D+d] = float((h * 5 + d * 11 + step * 3) % 29 - 14) / 16.f;
      }
    }
    upload(q,qv); upload(k,kv); upload(v,vv); upload(a,av); upload(b,bv); upload(z,zv); upload(state,device_state);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();
    std::vector<float> rg(heads), rb(heads), rc(D*heads), ro(D*heads);
    for (int h = 0; h < heads; ++h) {
      // Match only the declared F32 storage addition; the transcendental
      // and recurrence reference are independent stable F64 calculations.
      const double x = float(av[h] + dtv[h]);
      rg[h] = float(double(alphav[h]) * (std::max(x,0.0) + std::log1p(std::exp(-std::abs(x)))));
      rb[h] = float(sigmoid(bv[h]));
      const double decay = std::exp(double(rg[h]));
      const int kh = h % keys;
      for (int col = 0; col < D; ++col) {
        const size_t row = size_t(h * D + col) * D;
        double projected = 0;
        for (int d = 0; d < D; ++d) projected += decay * ref_state[row+d] * kv[kh*D+d];
        const double delta = double(rb[h]) * (double(vv[h*D+col]) - projected);
        double score = 0;
        for (int d = 0; d < D; ++d) {
          const double updated = decay * ref_state[row+d] + delta * kv[kh*D+d];
          score += updated * qv[kh*D+d]; ref_state[row+d] = float(updated);
        }
        rc[h*D+col] = float(score/std::sqrt(double(D)));
      }
      double sum = 0;
      for (int d = 0; d < D; ++d) sum += double(rc[h*D+d]) * rc[h*D+d];
      const double inv = 1 / std::sqrt(sum / D + double(1e-6f));
      for (int d = 0; d < D; ++d) {
        const float norm = float(float(rc[h*D+d] * inv) * sv[d]);
        ro[h*D+d] = float(double(zv[h*D+d]) * sigmoid(zv[h*D+d]) * norm);
      }
    }
    auto actual_g = read(g), actual_b = read(beta), actual_core = read(core), actual_out = read(output);
    device_state = read(state_out); // Actual state, never replaced by the oracle.
    metrics[0].add(actual_g,rg); metrics[1].add(actual_b,rb); metrics[2].add(actual_core,rc);
    metrics[3].add(device_state,ref_state); metrics[4].add(actual_out,ro);
  }
  bool passed = true;
  for (const auto& m : metrics) {
    std::printf("heads=%d layout=%s parameters=%s metric=%s observations=%llu failures=%llu max_abs=%.9g max_gate_ratio=%.9g\n", heads, official?"official":"gguf",local?"qwen35-local":"synthetic",m.name,(unsigned long long)m.observations,(unsigned long long)m.failures,m.max_abs,m.max_ratio);
    passed = passed && m.failures == 0;
  }
  ggml_backend_buffer_free(buffer); ggml_free(ctx); return passed;
}
int main(int argc, char** argv) {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if (argc != 2) return 2;
  auto backend = ggml_backend_cuda_init(0); if (!backend) return 3;
  bool passed = run(backend,16,true,true,argv[1]);
  passed = run(backend,16,false,false,argv[1]) && passed;
  for (int heads : {48,128}) for (bool official : {false,true}) passed = run(backend,heads,official,false,argv[1]) && passed;
  ggml_backend_free(backend); std::printf("verdict=%s\n",passed?"PASS":"FAIL"); return passed?0:1;
}
