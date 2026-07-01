// path lexical-op bench (re-audit gap-fill #51) vs std::filesystem::path. path.h
// is HYLE's pure-string path layer (no syscalls): extension/base/directory/
// normalize/split walk components over a string_ref view, ZERO allocation.
//
// THREE-WAY compare per lexical op, to keep the win HONEST:
//   1. std::filesystem::path -- the std drop-in SOTA, but it CONSTRUCTS a path
//      object (heap-allocates) each call, so it also pays a structural cost.
//   2. std::string_view hand-rolled -- SAME zero-alloc view + SAME back-to-front
//      scan as HYLE, only the API differs. This is the pure ALGORITHM baseline.
//   3. HYLE path::* -- the string_ref view walk.
// HYLE ~= (2) means the scan algorithm is already optimal (no more to squeeze);
// (2)&(3) both crushing (1) is the "view-only vs must-allocate" STRUCTURAL gap,
// not cleverer scanning. NOTE: this measures LEXICAL ops only (no filesystem
// touch); normalize/join_array have no zero-alloc std::string_view peer (they
// PRODUCE a path), so those two stay two-way vs std::filesystem.

#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>

#include "hyle/container/path.h"
#include "hyle/container/string_ref.h"

using hyle::container::base;
using hyle::container::directory;
using hyle::container::extension;
using hyle::container::join_array;
using hyle::container::normalize;
using hyle::container::string_ref;

namespace {
namespace nb = ankerl::nanobench;
// A realistic deep-ish path with an extension (the common parse target).
constexpr const char* kPath = "/usr/local/share/hyle/include/hyle/container/string_ref.h";

// ---- std::string_view hand-rolled lexical parse (the ALGORITHM baseline) -----
// std::filesystem::path is the drop-in SOTA, but it CONSTRUCTS a path object
// (heap-allocates + copies the string) on every op -- so racing it conflates two
// wins: (a) HYLE's zero-alloc view (structural) and (b) HYLE's scan algorithm.
// These string_view versions isolate (b): SAME zero-alloc view input, SAME
// back-to-front scan as path.h, only the API differs. If HYLE ties these, the
// algorithm is already optimal; if BOTH crush std::filesystem, that gap is the
// "must-allocate vs view-only" structural difference, not cleverer scanning.
inline bool sv_sep(char c) noexcept { return c == '/' || c == '\\'; }

std::string_view sv_extension(std::string_view p) noexcept {
  std::size_t ext = std::string_view::npos;
  for (std::size_t i = p.size(); i-- > 0;) {
    char c = p[i];
    if (sv_sep(c)) break;
    if (c == '.') {
      if (ext == std::string_view::npos) ext = i;
    } else if (ext != std::string_view::npos) {
      return p.substr(ext);
    }
  }
  return {};
}

std::string_view sv_base(std::string_view p) noexcept {
  if (p.empty()) return ".";
  bool found = false;
  std::size_t end = p.size();
  for (std::size_t i = p.size(); i-- > 0;) {
    if (sv_sep(p[i])) {
      if (found) return p.substr(i + 1, end - (i + 1));
      end = i;
    } else {
      found = true;
    }
  }
  return found ? p.substr(0, end) : std::string_view{"/"};
}

std::string_view sv_directory(std::string_view p) noexcept {
  if (p.empty()) return ".";
  bool found = false;
  for (std::size_t i = p.size(); i-- > 0;) {
    if (sv_sep(p[i])) {
      if (found) {
        if (i == 0) return "/";
        std::size_t j = i;
        while (j < p.size() && sv_sep(p[j])) --j;
        return (j < p.size()) ? p.substr(0, j + 1) : std::string_view{"/"};
      }
    } else {
      found = true;
    }
  }
  return found ? std::string_view{"."} : std::string_view{"/"};
}
}  // namespace

int bench_path() {
  const string_ref sp{kPath};
  const std::string std_in{kPath};
  const std::string_view sv{kPath};  // algorithm-baseline input (zero-alloc)

  // ---- extension(): ".h" -- three-way (fs alloc / sv algo / HYLE view) --------
  // std::filesystem::path::extension constructs a path (stores/copies the string)
  // then scans; sv_extension + HYLE scan a view in place. Opaque so none folds.
  {
    nb::Bench b;
    b.title("path extension()").relative(true).minEpochIterations(3000000);
    b.run("std::filesystem::path::extension", [&] {
      std::filesystem::path p{std_in};
      nb::doNotOptimizeAway(p.extension().native().size());
    });
    b.run("std::string_view hand-rolled", [&] {
      nb::doNotOptimizeAway(sv_extension(sv).size());
    });
    b.run("HYLE path::extension", [&] {
      nb::doNotOptimizeAway(extension(sp).size);
    });
  }

  // ---- base()/filename(): last component (three-way) -------------------------
  {
    nb::Bench b;
    b.title("path base()/filename()").relative(true).minEpochIterations(3000000);
    b.run("std::filesystem::path::filename", [&] {
      std::filesystem::path p{std_in};
      nb::doNotOptimizeAway(p.filename().native().size());
    });
    b.run("std::string_view hand-rolled", [&] {
      nb::doNotOptimizeAway(sv_base(sv).size());
    });
    b.run("HYLE path::base", [&] {
      nb::doNotOptimizeAway(base(sp).size);
    });
  }

  // ---- directory()/parent_path(): before the last component (three-way) ------
  {
    nb::Bench b;
    b.title("path directory()/parent_path()").relative(true).minEpochIterations(3000000);
    b.run("std::filesystem::path::parent_path", [&] {
      std::filesystem::path p{std_in};
      nb::doNotOptimizeAway(p.parent_path().native().size());
    });
    b.run("std::string_view hand-rolled", [&] {
      nb::doNotOptimizeAway(sv_directory(sv).size());
    });
    b.run("HYLE path::directory", [&] {
      nb::doNotOptimizeAway(directory(sp).size);
    });
  }

  // ---- normalize(): in-place lexical folding ('.' / '..' / separator runs) ----
  // The heaviest pure-string path op: a component-walk that resolves "." / ".." and
  // collapses separators IN PLACE (zero allocation). Rival std::filesystem::path::
  // lexically_normal() does the same lexically but constructs a path object (heap).
  // normalize MUTATES its buffer, so each iteration works on a FRESH copy of a
  // messy path (redundant seps + . + .. to actually exercise the folding).
  {
    constexpr const char* kMessy =
        "/usr//local/./share/../share/hyle/./include/../include/container/";
    char buf[128];
    const std::string std_messy{kMessy};
    nb::Bench b;
    b.title("path normalize() lexical fold").relative(true).minEpochIterations(200000);
    b.run("std::filesystem::path::lexically_normal", [&] {
      std::filesystem::path p{std_messy};
      nb::doNotOptimizeAway(p.lexically_normal().native().size());
    });
    b.run("HYLE path::normalize (in-place)", [&] {
      std::memcpy(buf, kMessy, std::strlen(kMessy) + 1);  // fresh mutable copy
      nb::doNotOptimizeAway(normalize(buf));
    });
  }

  // ---- join_array(): assemble N parts with single '/' separators (no alloc) ---
  // Buffer-fill token walk that collapses interior separator runs + honors lead/
  // trail seps, snprintf-style. The no-alloc producing op of path's allocating
  // class (parallel to normalize). Rival: std::filesystem::path operator/=
  // accumulated over the same parts (which builds a path object -> allocates).
  {
    const string_ref parts[] = {string_ref{"usr"}, string_ref{"local"},
                                string_ref{"share"}, string_ref{"hyle"},
                                string_ref{"include"}, string_ref{"container"}};
    constexpr std::size_t kParts = sizeof(parts) / sizeof(parts[0]);
    char dst[128];
    const char* std_parts[] = {"usr", "local", "share", "hyle", "include", "container"};
    nb::Bench b;
    b.title("path join_array 6 parts (no alloc)").relative(true).minEpochIterations(200000);
    b.run("std::filesystem::path operator/= x6", [&] {
      std::filesystem::path p;
      for (const char* part : std_parts) p /= part;
      nb::doNotOptimizeAway(p.native().size());
    });
    b.run("HYLE path::join_array (buffer-fill)", [&] {
      nb::doNotOptimizeAway(join_array(dst, sizeof(dst), parts, kParts));
    });
  }

  return 0;
}
