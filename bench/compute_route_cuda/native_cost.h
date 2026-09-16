#pragma once

// Included inside the private fixture namespace, after DataSet. This is a raw
// native cost diagnostic, not a SAME_CONTRACT peer or another product runtime.
class NativeHarness final {
public:
  NativeHarness(DataSet &data, bool) {
    primary_ =
        take(cuda::primary_context(data.context), "borrow native context");
    cuda::detail::ScopedPrimaryContext scope(primary_);
    require(scope.status(), "enter native preparation context");
    const auto execution = take(cuda::execution_context(data.context),
                                "borrow native execution context");
    check(cudaExecutionCtxStreamCreate(&stream_, execution,
                                       cudaStreamNonBlocking, 0));
    check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    int processors = 0, shared = 0;
    check(
        cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, 0));
    check(cudaDeviceGetAttribute(&shared,
                                 cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    check(compute::marlin::native::prepare(
        static_cast<int>(data.rows), static_cast<int>(2 * kHidden),
        static_cast<int>(kInner), processors, shared, &first_));
    check(compute::cuda_native::native::sglang_prepare(
        true, true, static_cast<unsigned>(data.rows * kHidden),
        static_cast<unsigned>(kHidden), &middle_));
    weights_ =
        take(cuda::pointer<std::byte>(data.quantized.weights().storage()),
             "borrow native Q4");
    scales_ = take(cuda::pointer<std::byte>(data.quantized.scales().storage()),
                   "borrow native scales");
    down_ = take(cuda::pointer<__half>(data.down.view.as_const().storage()),
                 "borrow native down");
    configure_down(data.rows);
    for (std::size_t i = 0; i < kSlots; ++i) {
      auto &slot = slots_[i];
      slot.input.create(data.resource.ref(), data.rows, kInner);
      slot.input.write(data.inputs[i]);
      slot.projected.create(data.resource.ref(), data.rows, 2 * kHidden);
      slot.activated.create(data.resource.ref(), data.rows, kHidden);
      slot.output.create(data.resource.ref(), data.rows, kOutput);
      slot.first_workspace = take(
          mem::Buffer::create(data.resource.ref(), first_.workspace_bytes, 256),
          "allocate native Marlin workspace");
      slot.down_workspace = take(
          mem::Buffer::create(data.resource.ref(), down_workspace_bytes_, 256),
          "allocate native cuBLAS workspace");
      slot.x = take(cuda::pointer<__half>(slot.input.view.storage()),
                    "native input pointer");
      slot.gate_up = take(cuda::pointer<__half>(slot.projected.view.storage()),
                          "native projection pointer");
      slot.activation =
          take(cuda::pointer<__half>(slot.activated.view.storage()),
               "native activation pointer");
      slot.y = take(cuda::pointer<__half>(slot.output.view.storage()),
                    "native output pointer");
      slot.scratch = take(cuda::pointer<std::byte>(slot.first_workspace.view()),
                          "native Marlin scratch");
      if (down_workspace_bytes_ != 0)
        slot.down_scratch =
            take(cuda::pointer<std::byte>(slot.down_workspace.view()),
                 "native cuBLAS scratch");
      // Same cold lock initialization as the shipping Marlin invocation. The
      // successful device kernel restores locks before the slot can be reused.
      check(cudaMemsetAsync(
          static_cast<std::byte *>(slot.scratch) + first_.locks_offset, 0,
          first_.workspace_bytes - first_.locks_offset, stream_));
      blas(cublasLtCreate(&slot.handle));
    }
    check(cudaStreamSynchronize(stream_));
#if HYLE_CUDA_GRAPH_BENCH
    for (auto &slot : slots_) {
      check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      enqueue(slot);
      cudaGraph_t graph = nullptr;
      check(cudaStreamEndCapture(stream_, &graph));
      check(cudaGraphInstantiateWithFlags(&slot.executable, graph, 0));
      check(cudaGraphDestroy(graph));
    }
#endif
    std::fprintf(stderr,
                 "# NATIVE_COST cold_workspace_bytes_per_slot=%zu "
                 "completion=one-native-event contract=NATURAL_ROUTE\n",
                 first_.workspace_bytes + down_workspace_bytes_);
  }

  ~NativeHarness() {
    cuda::detail::ScopedPrimaryContext scope(primary_);
    require(scope.status(), "enter native teardown context");
    for (auto &slot : slots_) {
#if HYLE_CUDA_GRAPH_BENCH
      check(cudaGraphExecDestroy(slot.executable));
#endif
      blas(cublasLtDestroy(slot.handle));
    }
    for (auto layout : layouts_)
      blas(cublasLtMatrixLayoutDestroy(layout));
    blas(cublasLtMatmulDescDestroy(operation_));
    check(cudaEventDestroy(done_));
    check(cudaStreamDestroy(stream_));
  }

  void run_slot(std::size_t index) {
    cuda::detail::ScopedPrimaryContext scope(primary_);
    require(scope.status(), "enter native call context");
#if HYLE_CUDA_GRAPH_BENCH
    check(cudaGraphLaunch(slots_[index].executable, stream_));
#else
    enqueue(slots_[index]);
#endif
    check(cudaEventRecord(done_, stream_));
    check(cudaEventSynchronize(done_));
  }
  void run() { run_slot((next_slot_++) % kSlots); }
  std::vector<float> output(std::size_t slot) {
    return slots_[slot].output.read();
  }

private:
  struct Slot final {
    Tensor input, projected, activated, output;
    mem::Buffer first_workspace, down_workspace;
    __half *x = nullptr, *gate_up = nullptr, *activation = nullptr,
           *y = nullptr;
    void *scratch = nullptr, *down_scratch = nullptr;
    cublasLtHandle_t handle = nullptr;
#if HYLE_CUDA_GRAPH_BENCH
    cudaGraphExec_t executable = nullptr;
#endif
  };

  static void check(cudaError_t status) {
    require(cuda::status(status), "native CUDA call");
  }
  static void blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "native cuBLAS status=%d\n",
                   static_cast<int>(status));
      fail("native cuBLAS call");
    }
  }

  void configure_down(std::size_t rows) {
    // Exact shipping FP16 row-major descriptor/heuristic policy; no TF32 or
    // output-type split-K reduction. The selected algorithm remains immutable.
    blas(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const std::array dimensions{std::array<std::size_t, 2>{rows, kHidden},
                                std::array<std::size_t, 2>{kHidden, kOutput},
                                std::array<std::size_t, 2>{rows, kOutput}};
    const auto order = CUBLASLT_ORDER_ROW;
    for (std::size_t i = 0; i < layouts_.size(); ++i) {
      blas(cublasLtMatrixLayoutCreate(
          &layouts_[i], CUDA_R_16F, dimensions[i][0], dimensions[i][1],
          static_cast<std::int64_t>(dimensions[i][1])));
      blas(cublasLtMatrixLayoutSetAttribute(
          layouts_[i], CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    }
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    blas(cublasLtCreate(&handle));
    blas(cublasLtMatmulPreferenceCreate(&preference));
    constexpr std::size_t maximum = 32 * 1024 * 1024;
    constexpr std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    constexpr std::uint32_t alignment = 16;
    blas(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maximum,
        sizeof(maximum)));
    blas(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
        sizeof(reduction)));
    for (const auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
      blas(cublasLtMatmulPreferenceSetAttribute(preference, attribute,
                                                &alignment, sizeof(alignment)));
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    blas(cublasLtMatmulAlgoGetHeuristic(handle, operation_, layouts_[0],
                                        layouts_[1], layouts_[2], layouts_[2],
                                        preference, 1, &result, &count));
    if (count != 1)
      fail("native cuBLAS heuristic availability");
    blas(result.state);
    algorithm_ = result.algo;
    down_workspace_bytes_ = result.workspaceSize;
    blas(cublasLtMatmulPreferenceDestroy(preference));
    blas(cublasLtDestroy(handle));
  }

  void enqueue(Slot &slot) {
    check(compute::marlin::native::launch(first_, slot.x, weights_, scales_,
                                          slot.gate_up, slot.scratch, stream_));
    check(compute::cuda_native::native::sglang_launch(
        middle_, slot.gate_up, nullptr, slot.activation, stream_));
    constexpr float alpha = 1, beta = 0;
    blas(cublasLtMatmul(slot.handle, operation_, &alpha, slot.activation,
                        layouts_[0], down_, layouts_[1], &beta, slot.y,
                        layouts_[2], slot.y, layouts_[2], &algorithm_,
                        slot.down_scratch, down_workspace_bytes_, stream_));
  }

  CUcontext primary_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t done_ = nullptr;
  compute::marlin::native::Program first_{};
  compute::cuda_native::native::Program middle_{};
  const void *weights_ = nullptr, *scales_ = nullptr, *down_ = nullptr;
  cublasLtMatmulDesc_t operation_ = nullptr;
  std::array<cublasLtMatrixLayout_t, 3> layouts_{};
  cublasLtMatmulAlgo_t algorithm_{};
  std::size_t down_workspace_bytes_ = 0;
  std::array<Slot, kSlots> slots_;
  std::size_t next_slot_ = 0;
};
