#pragma once

#include <cstdint>

// str_search.h (internal) -- declares the per-ISA str_count/str_rfind_char variant symbols
// the string_ref.cc dispatcher takes the address of. Each is DEFINED in its own -march TU
// (str_search_{scalar,sse4,avx2,avx512,neon}.cc) from the shared body in str_search_impl.h.
// Names carry an ISA suffix so the variants coexist in one binary (one backend per TU).
//
// NOT a public header -- lives under src/container/, never installed. Public callers use
// string_ref.h's count()/rfind() which forward through detail::str_count / str_rfind_char.

namespace hyle::container::detail {

std::uint32_t str_count_scalar(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_rfind_char_scalar(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_ascii_lower_scalar(const char*, std::uint32_t, char*,
                                     std::uint32_t) noexcept;
std::uint32_t str_ascii_upper_scalar(const char*, std::uint32_t, char*,
                                     std::uint32_t) noexcept;
#if defined(HYLE_ARCH_X64)
std::uint32_t str_count_sse4(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_rfind_char_sse4(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_ascii_lower_sse4(const char*, std::uint32_t, char*,
                                   std::uint32_t) noexcept;
std::uint32_t str_ascii_upper_sse4(const char*, std::uint32_t, char*,
                                   std::uint32_t) noexcept;
std::uint32_t str_count_avx2(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_rfind_char_avx2(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_ascii_lower_avx2(const char*, std::uint32_t, char*,
                                   std::uint32_t) noexcept;
std::uint32_t str_ascii_upper_avx2(const char*, std::uint32_t, char*,
                                   std::uint32_t) noexcept;
std::uint32_t str_count_avx512(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_rfind_char_avx512(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_ascii_lower_avx512(const char*, std::uint32_t, char*,
                                     std::uint32_t) noexcept;
std::uint32_t str_ascii_upper_avx512(const char*, std::uint32_t, char*,
                                     std::uint32_t) noexcept;
#elif defined(HYLE_ARCH_ARM64)
std::uint32_t str_count_neon(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_rfind_char_neon(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_ascii_lower_neon(const char*, std::uint32_t, char*,
                                   std::uint32_t) noexcept;
std::uint32_t str_ascii_upper_neon(const char*, std::uint32_t, char*,
                                   std::uint32_t) noexcept;
#endif

}  // namespace hyle::container::detail
