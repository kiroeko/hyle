#pragma once

#include <concepts>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"

namespace hyle::util {

namespace detail {

struct UniqueResourceAccess;

template <class Stored, class Source>
constexpr decltype(auto) move_or_copy_for_transfer(Source&& source) noexcept {
  if constexpr (std::is_nothrow_constructible_v<Stored, Source&&>) {
    return std::forward<Source>(source);
  } else {
    return static_cast<const std::remove_reference_t<Source>&>(source);
  }
}

template <class Stored, class Source>
concept TransferConstructible = requires(Source&& source) {
  Stored(move_or_copy_for_transfer<Stored>(std::forward<Source>(source)));
};

template <class Stored, class Source>
concept NothrowTransferConstructible =
    TransferConstructible<Stored, Source> &&
    noexcept(Stored(move_or_copy_for_transfer<Stored>(
        std::declval<Source&&>())));

template <class Resource, class Invalid>
concept NothrowSentinelComparable =
    requires(const std::remove_reference_t<Resource>& resource,
             const Invalid& invalid) {
      { resource == invalid } -> std::convertible_to<bool>;
      requires noexcept(static_cast<bool>(resource == invalid));
    };

}  // namespace detail

// Exclusive ownership for a value-like native token that has no domain RAII
// type. Prefer a typed Platform/Mem/IO owner whenever one exists: this fallback
// deliberately carries no error domain or resource-specific operations.
template <class Resource, class Deleter>
class [[nodiscard("discarding a UniqueResource immediately releases it")]]
    UniqueResource final {
  static_assert(std::is_object_v<Resource> && !std::is_array_v<Resource> &&
                !std::is_const_v<Resource> && !std::is_volatile_v<Resource>);
  static_assert(std::is_object_v<Deleter>);
  static_assert(std::is_nothrow_move_constructible_v<Deleter> ||
                    std::is_nothrow_copy_constructible_v<Deleter>,
                "the deleter must move or copy without throwing so ownership "
                "transfer cannot lose cleanup responsibility");
  static_assert(std::is_nothrow_move_constructible_v<Resource> ||
                    std::copy_constructible<Resource>,
                "the resource must move without throwing or remain copyable");
  static_assert(std::invocable<Deleter&, Resource&>);

 public:
  using resource_type = Resource;
  using deleter_type = Deleter;

  template <class R, class D>
    requires std::same_as<std::remove_cvref_t<R>, Resource> &&
             detail::NothrowTransferConstructible<Deleter, D> &&
             detail::TransferConstructible<Resource, R>
  explicit UniqueResource(R&& resource, D&& deleter) noexcept(
      kNothrowDeleterTransfer<D> && kNothrowResourceTransfer<R>)
      : UniqueResource(CheckedConstructionTag{}, true,
                       std::forward<R>(resource), std::forward<D>(deleter)) {}

  UniqueResource(UniqueResource&& other) noexcept(
      kNothrowResourceTransfer<Resource>)
      : deleter_(move_or_copy_deleter_for_owner(other)), allocated_(false) {
    construct_transferred_resource(other);
  }

  UniqueResource& operator=(UniqueResource&& other) noexcept(
      std::is_nothrow_invocable_v<Deleter&, Resource&>)
    requires std::is_nothrow_move_assignable_v<Resource> &&
             std::is_nothrow_move_assignable_v<Deleter>
  {
    if (this == &other) return *this;
    reset();
    deleter_ = std::move(other.deleter_);
    storage_.resource = std::move(other.storage_.resource);
    allocated_ = std::exchange(other.allocated_, false);
    return *this;
  }

  ~UniqueResource() noexcept {
    reset();
    std::destroy_at(std::addressof(storage_.resource));
  }

  UniqueResource(const UniqueResource&) = delete;
  UniqueResource& operator=(const UniqueResource&) = delete;

  [[nodiscard]] bool allocated() const noexcept { return allocated_; }
  explicit operator bool() const noexcept { return allocated(); }

  [[nodiscard]] const Resource& get() const noexcept {
    return storage_.resource;
  }
  [[nodiscard]] Deleter& get_deleter() noexcept { return deleter_; }
  [[nodiscard]] const Deleter& get_deleter() const noexcept { return deleter_; }

  // Disarm without reclaiming. Copy the value-like token from get() first if
  // ownership needs to be handed to another API.
  void release() noexcept { allocated_ = false; }

  void reset() noexcept {
    if (!std::exchange(allocated_, false)) return;
    std::invoke(deleter_, storage_.resource);
  }

  template <class R>
    requires std::is_nothrow_assignable_v<Resource&, R&&>
  void reset(R&& resource) noexcept(
      std::is_nothrow_invocable_v<Deleter&, Resource&>) {
    reset();
    storage_.resource = std::forward<R>(resource);
    allocated_ = true;
  }

 private:
  friend struct detail::UniqueResourceAccess;

  struct CheckedConstructionTag final {};

  union ResourceStorage {
    constexpr ResourceStorage() noexcept {}
    ~ResourceStorage() noexcept {}

    Resource resource;
  };

  template <class R>
  static constexpr bool kNothrowResourceTransfer =
      noexcept(Resource(detail::move_or_copy_for_transfer<Resource>(
          std::declval<R&&>())));

  template <class D>
  static constexpr bool kNothrowDeleterTransfer =
      noexcept(Deleter(detail::move_or_copy_for_transfer<Deleter>(
          std::declval<D&&>())));

  static constexpr decltype(auto) move_or_copy_deleter_for_owner(
      UniqueResource& other) noexcept {
    // A potentially throwing resource copy must leave the source owner usable
    // whenever a nonthrowing deleter copy is available. Otherwise transfer the
    // move-only deleter first and reclaim below if the resource copy fails.
    if constexpr (!kNothrowResourceTransfer<Resource> &&
                  std::is_nothrow_copy_constructible_v<Deleter>) {
      return static_cast<const Deleter&>(other.deleter_);
    } else {
      return detail::move_or_copy_for_transfer<Deleter>(
          std::move(other.deleter_));
    }
  }

  void construct_transferred_resource(UniqueResource& other) noexcept(
      kNothrowResourceTransfer<Resource>) {
    if constexpr (kNothrowResourceTransfer<Resource>) {
      std::construct_at(
          std::addressof(storage_.resource),
          detail::move_or_copy_for_transfer<Resource>(
              std::move(other.storage_.resource)));
    } else {
#if HYLE_HAS_EXCEPTIONS
      try {
        std::construct_at(
            std::addressof(storage_.resource),
            detail::move_or_copy_for_transfer<Resource>(
                std::move(other.storage_.resource)));
      } catch (...) {
        if constexpr (!std::is_nothrow_copy_constructible_v<Deleter>) {
          // The only safe deleter transfer was a move. The source can no
          // longer reclaim with its moved-from deleter, so the constructed
          // destination deleter consumes the still-valid source token.
          if (std::exchange(other.allocated_, false)) {
            std::invoke(deleter_, other.storage_.resource);
          }
        }
        throw;
      }
#else
      std::construct_at(
          std::addressof(storage_.resource),
          detail::move_or_copy_for_transfer<Resource>(
              std::move(other.storage_.resource)));
#endif
    }
    allocated_ = std::exchange(other.allocated_, false);
  }

  template <class R>
    requires kNothrowResourceTransfer<R>
  static void construct_acquired_resource(ResourceStorage& storage,
                                          Deleter&,
                                          R&& resource,
                                          bool) noexcept {
    std::construct_at(
        std::addressof(storage.resource),
        detail::move_or_copy_for_transfer<Resource>(std::forward<R>(resource)));
  }

  template <class R>
    requires (!kNothrowResourceTransfer<R>)
  static void construct_acquired_resource(ResourceStorage& storage,
                                          [[maybe_unused]] Deleter& deleter,
                                          R&& resource,
                                          [[maybe_unused]] bool allocated) {
#if HYLE_HAS_EXCEPTIONS
    try {
      std::construct_at(
          std::addressof(storage.resource),
          detail::move_or_copy_for_transfer<Resource>(
              std::forward<R>(resource)));
    } catch (...) {
      // Construction is the ownership handoff point. If copying storage fails,
      // reclaim the still-valid source token instead of leaking an acquisition
      // written directly in the constructor expression.
      if (allocated) std::invoke(deleter, resource);
      throw;
    }
#else
    std::construct_at(
        std::addressof(storage.resource),
        detail::move_or_copy_for_transfer<Resource>(std::forward<R>(resource)));
#endif
  }

  template <class R, class D>
    requires std::same_as<std::remove_cvref_t<R>, Resource> &&
             detail::NothrowTransferConstructible<Deleter, D> &&
             detail::TransferConstructible<Resource, R>
  explicit UniqueResource(CheckedConstructionTag, bool allocated,
                          R&& resource, D&& deleter) noexcept(
      kNothrowDeleterTransfer<D> && kNothrowResourceTransfer<R>)
      : deleter_(detail::move_or_copy_for_transfer<Deleter>(
            std::forward<D>(deleter))),
        allocated_(allocated) {
    construct_acquired_resource(storage_, deleter_, std::forward<R>(resource),
                                allocated);
  }

  HYLE_NO_UNIQUE_ADDRESS Deleter deleter_;
  ResourceStorage storage_;
  bool allocated_ = true;
};

namespace detail {

struct UniqueResourceAccess final {
 private:
  template <class Resource, class Deleter>
  static consteval bool is_nothrow_make_checked() {
    using Owner = UniqueResource<std::decay_t<Resource>,
                                 std::decay_t<Deleter>>;
    return noexcept(Owner(typename Owner::CheckedConstructionTag{}, false,
                          std::declval<Resource&&>(),
                          std::declval<Deleter&&>()));
  }

 public:
  template <class Resource, class Deleter>
  static auto make_checked(bool allocated, Resource&& resource,
                           Deleter&& deleter) noexcept(
      is_nothrow_make_checked<Resource, Deleter>()) {
    using Owner = UniqueResource<std::decay_t<Resource>,
                                 std::decay_t<Deleter>>;
    return Owner(typename Owner::CheckedConstructionTag{}, allocated,
                 std::forward<Resource>(resource),
                 std::forward<Deleter>(deleter));
  }
};

}  // namespace detail

template <class Resource, class Deleter>
UniqueResource(Resource&&, Deleter&&)
    -> UniqueResource<std::decay_t<Resource>, std::decay_t<Deleter>>;

// Acquisition APIs commonly report failure with a sentinel. Direct
// construction always means "owned"; this factory is the explicit checked
// route that starts inactive when resource == invalid.
template <class Resource, class Invalid, class Deleter>
  requires detail::NothrowSentinelComparable<Resource, Invalid>
[[nodiscard]] auto make_unique_resource_checked(Resource&& resource,
                                                const Invalid& invalid,
                                                Deleter&& deleter) noexcept(
    noexcept(detail::UniqueResourceAccess::make_checked(
        false, std::declval<Resource&&>(), std::declval<Deleter&&>()))) {
  const bool allocated = !static_cast<bool>(resource == invalid);
  return detail::UniqueResourceAccess::make_checked(
      allocated, std::forward<Resource>(resource),
      std::forward<Deleter>(deleter));
}

}  // namespace hyle::util
