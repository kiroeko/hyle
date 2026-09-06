# container: sequential storage, associative structures and compiled hash/
# bitset algorithms. Intrusive, concurrent, queue and text surfaces live in
# separate internal factors; the consumer-facing hyle::container facade
# aggregates all of them. hash.cc privately includes the vendored
# xxhash + wyhash single-headers (best-of-breed byte hasher: XXH3 short-mid, wyhash
# long); those include dirs are PRIVATE so neither leaks to consumers.
# DynamicBitSet owns its algebra/count dispatcher. The kernel family is
# generated from one direct-Highway implementation header. Container owns the
# pointer/count semantics and one cached whole-buffer call, while Highway owns
# register lowering. No hyle::simd register DSL or simd product target enters
# this component. The proven Apple-NEON CNT+UDOT bit-count body remains a
# Container-owned specialty kernel.
#
# ARCH SPLIT (Apple-Silicon 真机纪律): wide variant TUs gated by HYLE_TARGET_PROCESSOR --
# x86 ships scalar+sse4+avx2+avx512, aarch64 ships scalar+neon. Feeding -mavx2 to arm64 (or the
# x86 intrinsic headers) errors, so non-host-arch TUs are never compiled. FlatHashMap's NEON
# group lives in the private flat-hash probe header (pulled by
# flat_hash_map.cc), so the baseline NEON route compiles automatically on arm64.
include(HyleDispatch)

set(_container_arch_sources "")
set(_container_highway_targets scalar sse4 avx2 neon)
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(HYLE_CONTAINER_X64 TRUE)
  list(APPEND _container_arch_sources hash_xxh3_avx2.cc)
  if(NOT (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    list(APPEND _container_highway_targets avx512)
    set(HYLE_CONTAINER_HAS_AVX512 TRUE)
  endif()
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  set(HYLE_CONTAINER_ARM64 TRUE)
  list(APPEND _container_arch_sources bitset_ops_neon.cc)
endif()
hyle_add_module(container
  SOURCES hash.cc pod_vector.cc flat_hash_map.cc dynamic_bitset.cc
          ${_container_arch_sources})
target_include_directories(hyle_container PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash"
  "${PROJECT_SOURCE_DIR}/third_party/wyhash"
  "${PROJECT_SOURCE_DIR}")
if(HYLE_CONTAINER_HAS_AVX512)
  target_compile_definitions(hyle_container PRIVATE
    HYLE_CONTAINER_HAS_AVX512=1)
endif()

hyle_add_dispatched_kernel(HIGHWAY OWN_TARGET_SET
  TARGET hyle_container
  NAME container_bitset
  IMPL "src/container/bitset_ops_impl.h"
  TARGETS ${_container_highway_targets})

# Per-variant ISA flags: each architecture-specific Container TU gets only its
# target ISA; the dispatcher stays at baseline. Wide TUs exist only for the
# matching build architecture.
if(HYLE_CONTAINER_X64)
  hyle_set_isa(hash_xxh3_avx2.cc AVX2)
elseif(HYLE_CONTAINER_ARM64)
  hyle_set_isa(bitset_ops_neon.cc NEON)
endif()
# Keep IPO/LTO off for the module so a future global -flto can't inline a variant TU's
# wider codegen into the baseline dispatcher (same isolation rule as simd/, conventions §10).
set_target_properties(hyle_container PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
if(NOT WIN32)
  # Generated ISA variant entry points and private vendor helpers are an
  # implementation detail of the Container shared object. Public C++ symbols
  # opt back in through HYLE_CONTAINER_API.
  set_target_properties(hyle_container PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
endif()

hyle_register_package_dependency(
  NAME container_hash_vendors
  OWNER container
  LOCKED_DEPENDENCIES xxhash wyhash
  RUNTIME_FILE_SPECS
    "${PROJECT_SOURCE_DIR}/third_party/xxhash/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|xxhash-LICENSE"
    "${PROJECT_SOURCE_DIR}/third_party/wyhash/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|wyhash-LICENSE")
