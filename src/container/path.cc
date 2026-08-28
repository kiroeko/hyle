#include "hyle/container/path.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "hyle/core/compiler.h"

namespace hyle::container {
namespace {

// snprintf-style sink: writes into dst[cap] (always leaving room for the NUL),
// while `need` counts every byte that SHOULD be written so the caller learns the
// untruncated length. cap==0 writes nothing.
struct BufWriter {
  explicit BufWriter(std::span<char> output) noexcept
      : dst(output.data()), cap(output.size()) {}

  char* dst;
  std::size_t cap;
  std::size_t w = 0;
  std::size_t need = 0;

  void put(char c) noexcept {
    if (w + 1 < cap) dst[w++] = c;
    if (need != static_cast<std::size_t>(-1)) ++need;
  }
  void put(const char* p, std::size_t n) noexcept {
    const std::size_t available = cap > w + 1 ? cap - w - 1 : 0;
    const std::size_t copied = std::min(available, n);
    if (copied != 0) {
      std::memmove(dst + w, p, copied);
      w += copied;
    }
    if (n > (std::numeric_limits<std::size_t>::max)() - need)
      need = (std::numeric_limits<std::size_t>::max)();
    else
      need += n;
  }
  std::size_t finish() noexcept {
    if (cap > 0) dst[w] = '\0';
    return need;
  }
};

std::uint32_t normalize_in_place(char* path,
                                 std::uint32_t input_size) noexcept {
  const bool is_abs = input_size != 0 && is_separator(path[0]);
  std::uint32_t w = 0;     // write cursor (output is path[0, w))
  std::uint32_t base = 0;  // floor: a ".." may not pop a component before this
  std::uint32_t r = 0;     // read cursor

  if (is_abs) {
    path[w++] = '/';
    while (r < input_size && is_separator(path[r]))
      ++r;  // collapse the leading separator run
    base = w;                           // = 1; never pop the root
  }

  while (r < input_size) {
    const std::uint32_t s = r;
    while (r < input_size && !is_separator(path[r]))
      ++r;  // component [s, r)
    const std::uint32_t len = r - s;
    while (r < input_size && is_separator(path[r]))
      ++r;  // consume trailing separator run

    const bool parent =
        len == 2 && path[s] == '.' && path[s + 1] == '.';
    if (len == 1 && path[s] == '.') continue;  // drop "."
    if (parent) {
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
    if (parent) {
      base = w;  // lift the floor: a leading ".." can never be popped later
    }
  }

  if (w > 1 && path[w - 1] == '/') --w;  // strip a trailing sep (never the root)
  if (w == 0) path[w++] = '.';           // relative path collapsed to nothing
  path[w] = '\0';
  return w;
}

}  // namespace

std::size_t join_path(std::span<const StringView> parts,
                      std::span<char> output) noexcept {
  BufWriter out(output);
  // A leading separator is honored only on the first part (an absolute / UNC
  // root); a trailing separator only on the last part. Interior separators are
  // collapsed to one '/' between tokens.
  const bool lead =
      !parts.empty() && parts.front().size > 0 &&
      is_separator(parts.front().data[0]);
  const bool trail = !parts.empty() && parts.back().size > 0 &&
                     is_separator(parts.back().data[parts.back().size - 1]);

  if (lead) out.put('/');
  bool first_tok = true;
  for (const StringView p : parts) {
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

PodVector<string_ref> split_all(string_ref path, mem::Allocator alloc) {
  PodVector<string_ref> out(alloc);
  string_ref rest = path;
  while (rest.size > 0) out.push_back(pop_first(&rest));
  return out;
}

std::size_t replace_extension(StringView path, StringView ext,
                              std::span<char> output) noexcept {
  BufWriter out(output);
  const StringView stem = strip_extension(path);
  out.put(stem.data, stem.size);
  out.put(ext.data, ext.size);
  return out.finish();
}

std::optional<StringView> normalize_path(StringView path,
                                         std::span<char> output) noexcept {
  const std::size_t input_size = path.size;
  if (input_size == (std::numeric_limits<std::size_t>::max)())
    return std::nullopt;
  const std::size_t required = std::max<std::size_t>(input_size, 1) + 1;
  if (output.size() < required) return std::nullopt;
  if (input_size != 0)
    std::memmove(output.data(), path.data, input_size);
  output[input_size] = '\0';
  const std::uint32_t normalized_size =
      normalize_in_place(output.data(), path.size);
  return StringView{output.data(), normalized_size, true};
}

}  // namespace hyle::container
