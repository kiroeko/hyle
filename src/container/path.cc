#include "hyle/container/path.h"

#include <cstring>

#include "hyle/macros.h"

namespace hyle::container {
namespace {

// snprintf-style sink: writes into dst[cap] (always leaving room for the NUL),
// while `need` counts every byte that SHOULD be written so the caller learns the
// untruncated length. cap==0 writes nothing.
struct BufWriter {
  char* dst;
  std::size_t cap;
  std::size_t w = 0;
  std::size_t need = 0;

  void put(char c) noexcept {
    if (w + 1 < cap) dst[w++] = c;
    ++need;
  }
  void put(const char* p, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) put(p[i]);
  }
  std::size_t finish() noexcept {
    if (cap > 0) dst[w] = '\0';
    return need;
  }
};

}  // namespace

std::uint32_t normalize(char* path) noexcept {
  const bool is_abs = is_separator(path[0]);
  std::uint32_t w = 0;     // write cursor (output is path[0, w))
  std::uint32_t base = 0;  // floor: a ".." may not pop a component before this
  std::uint32_t r = 0;     // read cursor

  if (is_abs) {
    path[w++] = '/';
    while (is_separator(path[r])) ++r;  // collapse the leading separator run
    base = w;                           // = 1; never pop the root
  }

  while (path[r] != '\0') {
    const std::uint32_t s = r;
    while (path[r] != '\0' && !is_separator(path[r])) ++r;  // component [s, r)
    const std::uint32_t len = r - s;
    while (is_separator(path[r])) ++r;  // consume trailing separator run

    if (len == 1 && path[s] == '.') continue;  // drop "."
    if (len == 2 && path[s] == '.' && path[s + 1] == '.') {
      if (w > base) {  // pop the previous component
        std::uint32_t p = w;
        if (p > base && path[p - 1] == '/') --p;  // step over a left-over sep
        while (p > base && path[p - 1] != '/') --p;
        w = p;
        continue;
      }
      if (is_abs) continue;  // "/.." stays at the root: drop the ".."
      // relative path at the floor: the ".." is irreducible -> keep it (below).
    }

    // Append the component, inserting a single '/' unless we are right after the
    // absolute root or at the very start.
    if (w > 0 && path[w - 1] != '/') path[w++] = '/';
    std::memmove(path + w, path + s, len);  // w <= s always; memmove for overlap
    w += len;
    if (len == 2 && path[s] == '.' && path[s + 1] == '.') {
      base = w;  // lift the floor: a leading ".." can never be popped later
    }
  }

  if (w > 1 && path[w - 1] == '/') --w;  // strip a trailing sep (never the root)
  if (w == 0) path[w++] = '.';           // relative path collapsed to nothing
  path[w] = '\0';
  return w;
}

string_ref join(string_ref a, string_ref b, mem::Allocator alloc) {
  const std::uint32_t n = a.size + 1 + b.size;
  char* s = static_cast<char*>(alloc.allocate(n + 1));
  if (s == nullptr) return string_ref{};
  if (a.size) std::memcpy(s, a.data, a.size);
  s[a.size] = '/';
  if (b.size) std::memcpy(s + a.size + 1, b.data, b.size);
  s[n] = '\0';
  return string_ref{s, n, true};
}

std::size_t join_array(char* dst, std::size_t cap, const string_ref* parts,
                       std::size_t n) {
  BufWriter out{dst, cap};
  // A leading separator is honored only on the first part (an absolute / UNC
  // root); a trailing separator only on the last part. Interior separators are
  // collapsed to one '/' between tokens.
  const bool lead =
      n > 0 && parts[0].size > 0 && is_separator(parts[0].data[0]);
  const bool trail = n > 0 && parts[n - 1].size > 0 &&
                     is_separator(parts[n - 1].data[parts[n - 1].size - 1]);

  if (lead) out.put('/');
  bool first_tok = true;
  for (std::size_t i = 0; i < n; ++i) {
    const string_ref& p = parts[i];
    std::uint32_t j = 0;
    while (j < p.size) {
      while (j < p.size && is_separator(p.data[j])) ++j;
      const std::uint32_t ts = j;
      while (j < p.size && !is_separator(p.data[j])) ++j;
      if (j > ts) {
        if (!first_tok) out.put('/');
        out.put(p.data + ts, j - ts);
        first_tok = false;
      }
    }
  }
  if (trail && !first_tok) out.put('/');  // not for an all-separator input
  return out.finish();
}

CArray<string_ref> split_all(string_ref path, mem::Allocator alloc) {
  CArray<string_ref> out(alloc);
  string_ref rest = path;
  while (rest.size > 0) out.push_back(pop_first(&rest));
  return out;
}

string_ref replace_extension(string_ref path, string_ref ext,
                             mem::Allocator alloc) {
  return concat(strip_extension(path), ext, alloc);
}

}  // namespace hyle::container
