#pragma once

#include "hyle/c/mem.h"

namespace hyle::io::c_detail {

void clear_error() noexcept;
void set_error(const char* message) noexcept;
bool normalize_allocator(hyle_allocator* allocator) noexcept;

}  // namespace hyle::io::c_detail
