#pragma once

// Quantized multirow up/gate projection and SiLU, adapted from ggml CUDA's
// MMQ tiles and llama.cpp b12063's fused gate/up layout (MIT).
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/unary.cuh"
#include <climits>
#include <cstdint>

template <typename T>
static __device__ __forceinline__ const T *hyle_cuda_mmq_glu_weight_row(
    const char *up, const char *gate, int block, int row, int stride) {
  constexpr int rows = 8;
  const char *source = (row / rows) % 2 ? gate : up;
  return reinterpret_cast<const T *>(source) + block +
         ((row / (2 * rows)) * rows + row % rows) * stride;
}

#include "cuda_mmq_glu_load.cuh"

template <ggml_type type, int J>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, false),
                  ggml_cuda_mmq_get_occupancy(type, J, false))
static __global__ void hyle_cuda_mmq_glu_kernel(
    const char *up, const char *gate, const int *y, float *dst,
    int m, int n, int k) {
#if defined(TURING_MMA_AVAILABLE)
  constexpr bool fallback = false;
  constexpr int warp = ggml_cuda_get_physical_warp_size();
  constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int output_tile = I / 2;
  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int blocks_per_iteration = ggml_cuda_mmq_get_K_vram(type, J, fallback) / qk;
  constexpr int y_block_size = sizeof(block_q8_1_mmq) / sizeof(int);
  const int jt = blockIdx.x, it = blockIdx.y;
  extern __shared__ int shared[];
  int *ids = shared;
  int *tile_y = shared + J;
  int *tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, nwarps * warp);
  for (int j = threadIdx.y * warp + threadIdx.x; j < J; j += nwarps * warp)
    ids[j] = j;
  __syncthreads();

  float sum[J * I / (nwarps * warp)] = {0.f};
  const int stride = k / qk, offset = it * output_tile * stride;
  const int *src_y = y + jt * J * y_block_size;
  constexpr ggml_cuda_mmq_vec_dot_t dot = ggml_cuda_mmq_get_vec_dot<type, J, fallback>();
  for (int kb = 0; kb < k / qk; kb += blocks_per_iteration) {
    if constexpr (type == GGML_TYPE_Q4_K)
      hyle_cuda_mmq_glu_load_q4_K<type, J, fallback>(
          up + offset * sizeof(block_q4_K), gate + offset * sizeof(block_q4_K),
          tile_x, kb, 0, stride);
    else
      hyle_cuda_mmq_glu_load_q8_0<type, J, fallback>(
          up + offset * sizeof(block_q8_0), gate + offset * sizeof(block_q8_0),
          tile_x, kb, 0, stride);
    const int *y0 = src_y + m * (kb * qk / QK8_1_MMQ) * y_block_size;
    for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp) {
      const int l = l0 + threadIdx.y * warp + threadIdx.x;
      tile_y[l] = y0[l];
    }
    __syncthreads();
    dot(tile_x, tile_y, sum, 0);
    __syncthreads();
    const int *y1 = src_y + m * ((kb * qk / QK8_1_MMQ) * y_block_size + y_block_size);
    for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp) {
      const int l = l0 + threadIdx.y * warp + threadIdx.x;
      tile_y[l] = y1[l];
    }
    __syncthreads();
    dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
  }

  using tile_C = tile<16, 8, int>;
  constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int ntx = rows_per_warp / tile_C::I;
  const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);
  float *target = dst + jt * J * n + it * output_tile;
  for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J)
    for (int z = 0; z < ntx; ++z)
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        if (j >= m - jt * J) continue;
        const int i = (i0 + z * tile_C::I) / 2 + tile_C::get_i(l);
        const int index = (j0 / tile_C::J + z) * tile_C::ne + l;
        const float value = sum[index], g = sum[index + tile_C::ne / 2];
        target[ids[j] * n + i] = value * ggml_cuda_op_silu_single(g);
      }
#else
  NO_DEVICE_CODE;
#endif
}

// Called by both graph admission and the native matcher; it describes kernel
// layout and hardware requirements rather than model or benchmark policy.
static bool hyle_cuda_mmq_glu_geometry(int device, ggml_type type,
                                        std::uint64_t k, std::uint64_t n,
                                        std::uint64_t m) {
  if ((type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q8_0) ||
      k == 0 || k % 512 || n == 0 || n % 64 || m < 128 || m % 128 ||
      k > INT_MAX || n > INT_MAX || m > INT_MAX || n / 64 > 65535 ||
      k * m > INT_MAX || n * m > INT_MAX || k * n > INT_MAX)
    return false;
  const auto &info = ggml_cuda_info();
  if (device < 0 || device >= info.device_count) return false;
  const auto &gpu = info.devices[device];
  if (!GGML_CUDA_CC_IS_NVIDIA(gpu.cc) || gpu.cc < GGML_CUDA_CC_TURING) return false;
  const auto cfg = ggml_cuda_mmq_get_config(type, 128, false, gpu.cc);
  return cfg.type != GGML_TYPE_COUNT && cfg.I == 128 && cfg.nthreads > 0 &&
         cfg.nthreads % 32 == 0 && mmq_get_nbytes_shared(cfg, gpu.cc) <= gpu.smpbo;
}

static bool hyle_cuda_mmq_glu_supported(const ggml_tensor *up,
    const ggml_tensor *gate, const ggml_tensor *glu, bool require_bound) {
  if (!up || !gate || !glu || up->op != GGML_OP_MUL_MAT ||
      gate->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU ||
      !up->src[0] || !up->src[1] || !gate->src[0] || up->src[1] != gate->src[1])
    return false;
  const ggml_tensor *a = up->src[1], *u = up->src[0], *g = gate->src[0];
  if (g->type != u->type || a->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
      ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
      !ggml_are_same_shape(u, g) || !ggml_are_same_stride(u, g) ||
      !ggml_are_same_shape(up, gate) || !ggml_are_same_shape(up, glu) ||
      !ggml_is_contiguous(u) || !ggml_is_contiguous(g) ||
      !ggml_is_contiguous(a) || !ggml_is_contiguous(glu) ||
      a->ne[2] != 1 || a->ne[3] != 1 || u->ne[2] != 1 || u->ne[3] != 1 ||
      a->ne[0] <= 0 || a->ne[1] <= 0 || u->ne[1] <= 0 || u->ne[0] != a->ne[0] ||
      (require_bound && (!u->data || !g->data || !a->data || !glu->data)))
    return false;
  return hyle_cuda_mmq_glu_geometry(ggml_cuda_get_device(), u->type,
                                    a->ne[0], u->ne[1], a->ne[1]);
}

static void hyle_cuda_mmq_glu_run(ggml_backend_cuda_context &ctx,
    const ggml_tensor *up, const ggml_tensor *gate, const ggml_tensor *glu) {
  const ggml_tensor *a = up->src[1], *u = up->src[0];
  constexpr int J = 128;
  const int k = int(a->ne[0]), m = int(a->ne[1]), n = int(u->ne[1]);
  const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  const auto cfg = ggml_cuda_mmq_get_config(u->type, J, false, cc);
  const int shared = mmq_get_nbytes_shared(cfg, cc);
  ggml_cuda_pool_alloc<char> tmp(ctx.pool());
  auto *q = tmp.alloc(size_t(m) * (k / 128) * sizeof(block_q8_1_mmq));
  auto stream = ctx.stream();
  quantize_mmq_q8_1_cuda(static_cast<const float *>(a->data), nullptr, q, u->type,
      k, k, int64_t(k) * m, int64_t(k) * m, k, m, 1, 1, stream);
  const dim3 block(32, cfg.nthreads / 32), grid(m / J, n / (cfg.I / 2));
  if (u->type == GGML_TYPE_Q4_K) {
    CUDA_SET_SHARED_MEMORY_LIMIT((hyle_cuda_mmq_glu_kernel<GGML_TYPE_Q4_K, J>), shared);
    hyle_cuda_mmq_glu_kernel<GGML_TYPE_Q4_K, J><<<grid, block, shared, stream>>>(
        (const char *)u->data, (const char *)gate->data, (const int *)q,
        (float *)glu->data, m, n, k);
  } else {
    CUDA_SET_SHARED_MEMORY_LIMIT((hyle_cuda_mmq_glu_kernel<GGML_TYPE_Q8_0, J>), shared);
    hyle_cuda_mmq_glu_kernel<GGML_TYPE_Q8_0, J><<<grid, block, shared, stream>>>(
        (const char *)u->data, (const char *)gate->data, (const int *)q,
        (float *)glu->data, m, n, k);
  }
}
