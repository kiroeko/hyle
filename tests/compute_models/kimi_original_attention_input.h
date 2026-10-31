#pragma once

#include "kimi_original_attention_roles.h"
#include <cstdlib>

namespace kimi_original_attention_input {
namespace roles=kimi_original_attention_roles;
inline std::string environment() {
#if defined(_MSC_VER)
  char *text=nullptr;std::size_t bytes=0;
  if(_dupenv_s(&text,&bytes,"HYLE_KIMI_ORIGINAL_ATTENTION_DIR"))
    throw std::runtime_error("original Kimi fixture environment lookup failed");
  const std::unique_ptr<char,decltype(&std::free)> release(text,&std::free);
  return text?text:"";
#else
  const auto text=std::getenv("HYLE_KIMI_ORIGINAL_ATTENTION_DIR");return text?text:"";
#endif
}
using Input=roles::Input;
inline Input load(const std::filesystem::path &directory) {return roles::load_input(directory);}
} // namespace kimi_original_attention_input
