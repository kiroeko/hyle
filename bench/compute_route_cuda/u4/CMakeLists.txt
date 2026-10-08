cmake_minimum_required(VERSION 3.25)
project(hyle_cuda_provider_choices LANGUAGES CXX)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core compute)
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
find_package(Threads REQUIRED)
add_library(u4_nanobench STATIC "${_root}/bench/nanobench_impl.cc")
target_include_directories(u4_nanobench SYSTEM PUBLIC "${_root}/third_party/nanobench")
target_link_libraries(u4_nanobench PUBLIC Threads::Threads)
add_executable(hyle_cuda_provider_choices_bench provider_choices.cc)
target_compile_features(hyle_cuda_provider_choices_bench PRIVATE cxx_std_23)
target_compile_definitions(hyle_cuda_provider_choices_bench PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(hyle_cuda_provider_choices_bench PRIVATE hyle::compute
  hyle::_factor_compute_provider_cuda hyle::_factor_compute_provider_ggml_cuda u4_nanobench)
hyle_stage_runtime_dependencies(hyle_cuda_provider_choices_bench)
add_executable(hyle_cuda_provider_admission admission.cc)
target_compile_features(hyle_cuda_provider_admission PRIVATE cxx_std_23)
target_link_libraries(hyle_cuda_provider_admission PRIVATE hyle::compute
  hyle::_factor_compute_provider_cuda hyle::_factor_compute_provider_ggml_cuda
  hyle::_factor_compute_provider_torch_cuda)
hyle_stage_runtime_dependencies(hyle_cuda_provider_admission)
add_executable(hyle_cuda_matrix_bench matrix_bench.cc)
target_compile_features(hyle_cuda_matrix_bench PRIVATE cxx_std_23)
target_compile_definitions(hyle_cuda_matrix_bench PRIVATE _CRT_SECURE_NO_WARNINGS
  "HYLE_CUDA_MATRIX_PAIRS=\"${CMAKE_CURRENT_LIST_DIR}/matrix-pairs.tsv\"")
target_link_libraries(hyle_cuda_matrix_bench PRIVATE hyle::compute u4_nanobench
  hyle::_factor_compute_provider_cuda hyle::_factor_compute_provider_ggml_cuda
  hyle::_factor_compute_provider_torch_cuda)
hyle_stage_runtime_dependencies(hyle_cuda_matrix_bench)
add_executable(hyle_perf_control_bench "${_root}/bench/integration/perf_control_bench.cc")
target_compile_features(hyle_perf_control_bench PRIVATE cxx_std_23)
target_compile_definitions(hyle_perf_control_bench PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(hyle_perf_control_bench PRIVATE hyle::core u4_nanobench)
