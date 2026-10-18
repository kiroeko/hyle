#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

void check(cudnnStatus_t s, const char* what) {
  if (s != CUDNN_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s: %s (%d)\n", what, cudnnGetErrorString(s), int(s));
    std::exit(2);
  }
}
void cuda_check(cudaError_t s, const char* what) {
  if (s != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(s)); std::exit(3); }
}
struct Descriptors {
  std::vector<cudnnBackendDescriptor_t> values;
  ~Descriptors() { for (auto i = values.rbegin(); i != values.rend(); ++i) cudnnBackendDestroyDescriptor(*i); }
  cudnnBackendDescriptor_t make(cudnnBackendDescriptorType_t type) {
    cudnnBackendDescriptor_t d{}; check(cudnnBackendCreateDescriptor(type, &d), "create descriptor"); values.push_back(d); return d;
  }
};
template<class T> void set(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t a, cudnnBackendAttributeType_t t, const T& v) {
  check(cudnnBackendSetAttribute(d, a, t, 1, &v), "set attribute");
}
void finish(cudnnBackendDescriptor_t d) { check(cudnnBackendFinalize(d), "finalize descriptor"); }

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  cuda_check(cudaSetDevice(0), "set device");
  cudaDeviceProp props{}; cuda_check(cudaGetDeviceProperties(&props, 0), "device properties");
  std::printf("cuDNN %zu, device %s, SM %d.%d\n", cudnnGetVersion(), props.name, props.major, props.minor);
  cudnnHandle_t handle{}; check(cudnnCreate(&handle), "create handle");
  int result = 4;
  {
    Descriptors ds;
    auto tensor = [&](int64_t uid) {
      auto t = ds.make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
      set(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
      set(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, int64_t{16});
      set(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, uid);
      // Pointwise-only contiguous reinterpretation; no copy or arithmetic.
      const int64_t dims[]{1, 128, 1, 1}, strides[]{128, 1, 128, 128};
      check(cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims), "dims");
      check(cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides), "strides");
      finish(t); return t;
    };
    auto x = tensor(1), y = tensor(2);
    auto pw = ds.make(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    set(pw, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, CUDNN_POINTWISE_SOFTPLUS_FWD);
    set(pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
    set(pw, CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, CUDNN_TYPE_DOUBLE, 1.0);
    set(pw, CUDNN_ATTR_POINTWISE_NAN_PROPAGATION, CUDNN_TYPE_NAN_PROPOGATION, CUDNN_PROPAGATE_NAN);
    finish(pw);
    auto op = ds.make(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    set(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, pw);
    set(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, x);
    set(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, y);
    set(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1.0f);
    finish(op);
    auto graph = ds.make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, handle);
    set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, op);
    finish(graph);
    for (auto mode : {CUDNN_HEUR_MODE_A}) {
      auto heur = ds.make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
      set(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, graph);
      set(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, mode);
      auto status = cudnnBackendFinalize(heur);
      if (status != CUDNN_STATUS_SUCCESS) { std::printf("heur %d: %s\n", int(mode), cudnnGetErrorString(status)); continue; }
      int64_t count{};
      check(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr), "heur count");
      std::printf("heur %d configs %lld\n", int(mode), static_cast<long long>(count));
      std::vector<cudnnBackendDescriptor_t> configs;
      for (int64_t i = 0; i < count; ++i) configs.push_back(ds.make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
      if (!count) continue;
      check(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &count, configs.data()), "heur results");
      std::printf("heur actual results %lld\n", static_cast<long long>(count));
      configs.resize(size_t(count));
      for (auto config : configs) {
        char detail[4096]{};
        auto engine = ds.make(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        int64_t got{}, engine_id{};
        check(cudnnBackendGetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &got, &engine), "config engine");
        check(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &got, &engine_id), "engine index");
        std::printf("engine=%lld\n", static_cast<long long>(engine_id));
        auto plan = ds.make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        set(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, handle);
        set(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, config);
        status = cudnnBackendFinalize(plan);
        if (status != CUDNN_STATUS_SUCCESS) { cudnnGetLastErrorString(detail, sizeof(detail)); std::printf("plan: %s %s\n", cudnnGetErrorString(status), detail); continue; }
        int64_t workspace_bytes{}, actual{};
        check(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &actual, &workspace_bytes), "workspace size");
        std::printf("plan ready workspace %lld\n", static_cast<long long>(workspace_bytes));
        void *dx{}, *dy{}, *workspace{};
        cuda_check(cudaMalloc(&dx, 128 * sizeof(float)), "alloc x");
        cuda_check(cudaMalloc(&dy, 128 * sizeof(float)), "alloc y");
        if (workspace_bytes) cuda_check(cudaMalloc(&workspace, size_t(workspace_bytes)), "workspace alloc");
        auto pack = ds.make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        const int64_t ids[]{1, 2}; void* pointers[]{dx, dy};
        check(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 2, ids), "pack ids");
        check(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 2, pointers), "pack data");
        set(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, workspace); finish(pack);
        const float inf = std::numeric_limits<float>::infinity();
        const std::vector<float> edges{-inf, -100, -80, -40, -20, -15, -5, -1, -0.0f, 0, 1, 5, 15, 20, 40, 80, 100, inf, std::numeric_limits<float>::quiet_NaN()};
        int failures = 0;
        for (int replay = 0; replay < 2; ++replay) {
          std::vector<float> in(128), out(128);
          for (size_t i = 0; i < in.size(); ++i) in[i] = i < edges.size() ? edges[(i + replay * 7) % edges.size()] : float(int(i) - 64) / float(8 + replay);
          cuda_check(cudaMemcpy(dx, in.data(), 128 * sizeof(float), cudaMemcpyHostToDevice), "upload");
          check(cudnnBackendExecute(handle, plan, pack), "execute");
          cuda_check(cudaMemcpy(out.data(), dy, 128 * sizeof(float), cudaMemcpyDeviceToHost), "download");
          for (size_t i = 0; i < in.size(); ++i) {
            const double a = in[i];
            const double expected = std::isnan(a) ? a : std::max(a, 0.0) + std::log1p(std::exp(-std::abs(a)));
            const float rounded = float(expected);
            bool ok = std::isnan(rounded) ? std::isnan(out[i]) : std::isinf(rounded) ? out[i] == rounded : std::abs(double(out[i]) - rounded) <= std::max(2e-44, std::abs(double(rounded)) * 4e-6);
            if (!ok) ++failures;
            if (i < edges.size() || !ok) std::printf("replay=%d x=%.9g got=%.9g expected=%.9g %s\n", replay, double(in[i]), double(out[i]), double(rounded), ok ? "PASS" : "FAIL");
          }
        }
        std::printf("failures=%d / 256\n", failures);
        cuda_check(cudaFree(dx), "free x"); cuda_check(cudaFree(dy), "free y");
        if (workspace) cuda_check(cudaFree(workspace), "free workspace");
        if (!failures) result = 0;
        else if (result != 0) result = 1;
      }
    }
  }
  check(cudnnDestroy(handle), "destroy handle");
  return result;
}
