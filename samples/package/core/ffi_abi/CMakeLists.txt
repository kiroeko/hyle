cmake_minimum_required(VERSION 3.25)
project(hyle_package_core_ffi_abi LANGUAGES C)

find_package(Hyle 0.1 REQUIRED COMPONENTS core)

# C is used here as the executable ABI schema language. HYLE does not expose a
# separate source-level C product API; managed bindings call the same symbols.
add_executable(hyle_package_core_ffi_abi main.c)
target_link_libraries(hyle_package_core_ffi_abi PRIVATE hyle::core)
hyle_stage_runtime_dependencies(hyle_package_core_ffi_abi)

enable_testing()
add_test(NAME hyle_package_core_ffi_abi COMMAND hyle_package_core_ffi_abi)
