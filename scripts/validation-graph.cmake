# Metadata projection only: no project(), compiler probe, configure or build.
cmake_minimum_required(VERSION 3.28)
set(HYLE_COMPONENTS all)
include("${HYLE_VALIDATION_MANIFEST}")
file(WRITE "${HYLE_VALIDATION_GRAPH}" "")
foreach(_factor IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_factor}" _upper)
  file(APPEND "${HYLE_VALIDATION_GRAPH}"
    "factor|${_factor}|${_HYLE_QUEUED_${_upper}}|${_HYLE_PUBLIC_DEPS_${_upper}}|${_HYLE_PRIVATE_DEPS_${_upper}}|${_HYLE_PRIVATE_BUILD_DEPS_${_upper}}|${_HYLE_SPLIT_PUBLIC_HEADERS_${_upper}}\n")
endforeach()
foreach(_product IN LISTS HYLE_KNOWN_PRODUCTS)
  string(TOUPPER "${_product}" _upper)
  file(APPEND "${HYLE_VALIDATION_GRAPH}"
    "product|${_product}|${_HYLE_PRODUCT_FACTORS_${_upper}}\n")
endforeach()
