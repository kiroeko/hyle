// Standard-library-only policy test. No model, native API, or device is loaded.
#include "../../engine/inference/version_admission.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <string_view>

namespace {
bool track_allocations = false;
std::size_t allocation_count = 0;
}

void *operator new(std::size_t bytes) {
  if (track_allocations) ++allocation_count;
  if (auto *result = std::malloc(bytes ? bytes : 1)) return result;
  throw std::bad_alloc{};
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

namespace {
namespace i = hyle::inference;
struct Version {};
struct Session { const Version *canonical; };
struct Request { const Session *session; };
using Holder = i::VersionHolder<Session, Version>;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
const auto session_holder = [](const Session *session) {
  return Holder{session, session ? session->canonical : nullptr};
};
const auto no_directory = [](const Version *) { return false; };

template<class Query>
i::VersionAdmission without_allocation(Query &&query) {
  const auto before = allocation_count;
  struct Tracking final {
    Tracking() { track_allocations = true; }
    ~Tracking() { track_allocations = false; }
  };
  i::VersionAdmission result;
  { Tracking tracking; result = query(); }
  require(allocation_count == before, "version admission allocated host storage");
  return result;
}

void shared_aliases_are_not_double_released() {
  Version version;
  Session a{&version}, b{&version}, idle{&version};
  const std::array selected{session_holder(&a), session_holder(&b)};
  const std::array owners{session_holder(&a), session_holder(&b)};
  const auto fits = without_allocation([&] { return i::version_admission({1, 2}, selected, owners, no_directory); });
  require(fits.fits && fits.retiring == 1, "two selected aliases must replace one old version with two new heads");
  const auto full = without_allocation([&] { return i::version_admission({1, 1}, selected, owners, no_directory); });
  require(!full.fits && full.retiring == 1, "alias deduplication incorrectly admitted two heads into N1");
  const std::array with_idle{session_holder(&a), session_holder(&b), session_holder(&idle)};
  const auto held = without_allocation([&] { return i::version_admission({1, 2}, selected, with_idle, no_directory); });
  require(!held.fits && !held.retiring, "an unselected alias was treated as a retiring base");
}

void one_alias_removal_does_not_free_a_version() {
  Version version;
  Session a{&version}, b{&version};
  const std::array selected{session_holder(&a)};
  const std::array owners{session_holder(&a), session_holder(&b)};
  unsigned directory_aliases = 2;
  const auto retained = [&](const Version *candidate) { return candidate == &version && directory_aliases != 0; };
  auto query = [&] { return i::version_admission({1, 1}, selected, owners, retained); };
  require(!without_allocation(query).fits, "directory aliases did not retain the original version");
  --directory_aliases;
  require(!without_allocation(query).fits, "removing one directory alias incorrectly freed the version");
  --directory_aliases;
  require(!without_allocation(query).fits, "a remaining idle session alias was ignored at N1");
  const std::array sole_owner{session_holder(&a)};
  require(without_allocation([&] { return i::version_admission({1, 1}, selected, sole_owner, retained); }).fits,
      "the old sole-owned version could not be replaced");
}

void new_heads_and_capacity_are_actual_current_facts() {
  const std::array fresh{Holder{}};
  const std::array<Holder, 0> none{};
  require(!without_allocation([&] { return i::version_admission({2, 2}, fresh, none, no_directory); }).fits,
      "a fresh request exceeded saturated N");
  require(without_allocation([&] { return i::version_admission({1, 2}, fresh, none, no_directory); }).fits,
      "rereading the lower live count did not admit a fresh head");
  Version version;
  Session a{&version};
  const std::array existing_and_fresh{session_holder(&a), Holder{}};
  const std::array owners{session_holder(&a)};
  const auto replaced = without_allocation([&] {
    return i::version_admission({1, 2}, existing_and_fresh, owners, no_directory);
  });
  require(replaced.fits && replaced.retiring == 1, "mixed replacement/new admission lost its actual head demand");
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  require(!without_allocation([&] { return i::version_admission({maximum, maximum}, fresh, none, no_directory); }).fits,
      "saturated size_t admission wrapped an addition");
  require(without_allocation([&] { return i::version_admission({maximum - 1, maximum}, fresh, none, no_directory); }).fits,
      "valid size_t boundary was rejected");
}

void existing_spans_need_only_inline_projections() {
  Version one, two;
  Session a{&one}, b{&two};
  const std::array<const Session *, 2> selected{&a, &b};
  const std::array<Request, 3> requests{Request{&a}, Request{}, Request{&b}};
  auto chosen = selected | std::views::transform(session_holder);
  auto resident = requests | std::views::transform([](const Request &request) { return session_holder(request.session); });
  const auto fits = without_allocation([&] { return i::version_admission({2, 2}, chosen, resident, no_directory); });
  require(fits.fits && fits.retiring == 2, "span/view projections changed distinct-version replacement facts");
  unsigned scanned = 0;
  const auto counted = [&](const Version *) { ++scanned; return false; };
  const auto spare = without_allocation([&] { return i::version_admission({2, 4}, chosen, resident, counted); });
  require(spare.fits && !spare.retiring && !scanned, "spare capacity unexpectedly scanned retainers");
}
} // namespace

int main() {
  try {
    shared_aliases_are_not_double_released();
    one_alias_removal_does_not_free_a_version();
    new_heads_and_capacity_are_actual_current_facts();
    existing_spans_need_only_inline_projections();
    std::puts("PASS canonical aliases, external retainers, eviction facts, fresh/saturated heads, inline views, no allocations");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
