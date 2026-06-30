// path lexical-op bench (re-audit gap-fill #51) vs std::filesystem::path. path.h
// is HYLE's pure-string path layer (no syscalls): extension/base/directory/
// normalize/split walk components over a string_ref view, ZERO allocation. The
// drop-in SOTA is std::filesystem::path's lexical ops -- but those construct a
// path object (allocates) and operate on a stored string, so HYLE's view-only
// walk should win. Win or lose, printed; std::filesystem is the rival (always on,
// it's std). NOTE: this measures LEXICAL ops only (no filesystem touch).

#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "hyle/container/path.h"
#include "hyle/container/string_ref.h"

using hyle::container::base;
using hyle::container::directory;
using hyle::container::extension;
using hyle::container::string_ref;

namespace {
namespace nb = ankerl::nanobench;
// A realistic deep-ish path with an extension (the common parse target).
constexpr const char* kPath = "/usr/local/share/hyle/include/hyle/container/string_ref.h";
}  // namespace

int bench_path() {
  const string_ref sp{kPath};
  const std::string std_in{kPath};

  // ---- extension(): ".h" -- HYLE single forward scan vs fs::path::extension ---
  // std::filesystem::path::extension constructs a path (stores/copies the string)
  // then scans; HYLE scans the view in place. Opaque the input so neither folds.
  {
    nb::Bench b;
    b.title("path extension()").relative(true).minEpochIterations(1000000);
    b.run("std::filesystem::path::extension", [&] {
      std::filesystem::path p{std_in};
      nb::doNotOptimizeAway(p.extension().native().size());
    });
    b.run("HYLE path::extension", [&] {
      nb::doNotOptimizeAway(extension(sp).size);
    });
  }

  // ---- base()/filename(): last component ------------------------------------
  {
    nb::Bench b;
    b.title("path base()/filename()").relative(true).minEpochIterations(1000000);
    b.run("std::filesystem::path::filename", [&] {
      std::filesystem::path p{std_in};
      nb::doNotOptimizeAway(p.filename().native().size());
    });
    b.run("HYLE path::base", [&] {
      nb::doNotOptimizeAway(base(sp).size);
    });
  }

  // ---- directory()/parent_path(): everything before the last component -------
  {
    nb::Bench b;
    b.title("path directory()/parent_path()").relative(true).minEpochIterations(1000000);
    b.run("std::filesystem::path::parent_path", [&] {
      std::filesystem::path p{std_in};
      nb::doNotOptimizeAway(p.parent_path().native().size());
    });
    b.run("HYLE path::directory", [&] {
      nb::doNotOptimizeAway(directory(sp).size);
    });
  }

  return 0;
}
