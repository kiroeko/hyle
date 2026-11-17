#pragma once

#include "ggml-backend.h"

#include <cstddef>

// Private cold access in the same TU as the locked native BLAS backend.
// Neither the provider nor a model interprets the backend's private context.
extern "C" bool ggml_hyle_blas_supports(const ggml_tensor *) noexcept;
extern "C" bool ggml_hyle_blas_reserve(ggml_backend_t, std::size_t) noexcept;
extern "C" ggml_status ggml_hyle_blas_compute(ggml_backend_t,
                                             ggml_cgraph *) noexcept;
