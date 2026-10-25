#pragma once

#include "kimi_original_attention_roles.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

// Exact finite test input, generated OFFLINE from this batch's complete.json.
// Contains only authenticated identities/hashes, no coefficients or downloader.
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
struct Input final {
  std::filesystem::path directory;
  std::array<std::string,4> union_sha;
  std::array<std::string,23> role_sha;
  roles::Pins pins() const noexcept {
    roles::Pins result;
    for(std::size_t i=0;i<union_sha.size();++i)result.unions[i]=union_sha[i];
    for(std::size_t i=0;i<role_sha.size();++i)result.roles[i]=role_sha[i];
    return result;
  }
};
inline Input load(const std::filesystem::path &directory) {
  std::ifstream stream(directory/"original-role-pins.txt",std::ios::binary);
  if(!stream)throw std::runtime_error("original Kimi completed acquisition pins are absent");
  std::string text;char ch=0;
  while(stream.get(ch)) {
    if(text.size()==16384)throw std::runtime_error("original Kimi pin input exceeds its finite extent");
    text.push_back(ch);
  }
  if(!stream.eof())throw std::runtime_error("original Kimi pin input read failed");
  std::istringstream input(text);std::string schema,repository,revision;
  if(!(input>>schema>>repository>>revision)||schema!="KIMI_ORIGINAL_ATTENTION_V1"||
      repository!=kimi_original_attention_values::repository||revision!=kimi_original_attention_values::revision)
    throw std::runtime_error("original Kimi pin input source identity differs");
  Input result{.directory=directory};std::string name;
  for(std::size_t i=0;i<roles::part_names.size();++i)
    if(!(input>>name>>result.union_sha[i])||name!=roles::part_names[i]||!roles::named::pinned(result.union_sha[i],64))
      throw std::runtime_error("original Kimi union pin order or SHA differs");
  for(std::size_t i=0;i<roles::specs.size();++i)
    if(!(input>>name>>result.role_sha[i])||name!=roles::specs[i].name||!roles::named::pinned(result.role_sha[i],64))
      throw std::runtime_error("original Kimi role pin order or SHA differs");
  if(input>>name)throw std::runtime_error("original Kimi pin input has an unexpected trailing field");
  return result;
}
} // namespace kimi_original_attention_input
