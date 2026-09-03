#include "hyle/mem/heap.h"

#include <mimalloc.h>

#include <utility>

#include "hyle/mem/system_inline.h"
#include "hyle/obs/trace.h"

namespace hyle::mem {
namespace {

mi_heap_t* as_native(void* heap) noexcept {
  return static_cast<mi_heap_t*>(heap);
}

void* heap_allocate(mi_heap_t* heap, std::size_t size,
                    std::size_t alignment) noexcept {
  alignment = detail::system_backend_checked_alignment(alignment);
  if (alignment <= detail::kMimallocV3NaturalAlignment) {
    return mi_heap_malloc(
        heap, size < detail::kMimallocV3NaturalAlignment
                  ? detail::kMimallocV3NaturalAlignment
                  : size);
  }
  return mi_heap_malloc_aligned(heap, size, alignment);
}

void* heap_reallocate(mi_heap_t* heap, void* pointer, std::size_t new_size,
                      std::size_t alignment) noexcept {
  alignment = detail::system_backend_checked_alignment(alignment);
  if (alignment <= detail::kMimallocV3NaturalAlignment) {
    return mi_heap_realloc(
        heap, pointer,
        new_size < detail::kMimallocV3NaturalAlignment
            ? detail::kMimallocV3NaturalAlignment
            : new_size);
  }
  return mi_heap_realloc_aligned(heap, pointer, new_size, alignment);
}

void heap_deallocate(void* pointer, std::size_t size,
                     std::size_t alignment) noexcept {
  HYLE_ASSUME(alignment != 0 && (alignment & (alignment - 1)) == 0);
  if (alignment <= detail::kMimallocV3NaturalAlignment) {
    mi_free_size(pointer, detail::system_backend_natural_size(size));
  } else {
    mi_free_size_aligned(pointer, size, alignment);
  }
}

}  // namespace

Result<Heap> Heap::create() noexcept {
  mi_heap_t* heap = mi_heap_new();
  if (heap == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("could not create isolated memory heap"));
  }
  return Heap(heap);
}

Heap::Heap(Heap&& other) noexcept
    : native_heap_(std::exchange(other.native_heap_, nullptr)) {}

Heap& Heap::operator=(Heap&& other) noexcept {
  if (this != &other) {
    release();
    native_heap_ = std::exchange(other.native_heap_, nullptr);
  }
  return *this;
}

Heap::~Heap() { release(); }

AllocatorRef Heap::allocator() noexcept {
  if (!valid()) return {};
  return AllocatorRef(&Heap::allocate_erased, &Heap::deallocate_erased,
                   &Heap::reallocate_erased, this);
}

Status Heap::clear() noexcept {
  if (!valid()) return Status::FailedPrecondition("heap is invalid");
  mi_heap_t* replacement = mi_heap_new();
  if (replacement == nullptr) {
    return Status::ResourceExhausted(
        "could not replace isolated memory heap");
  }
  mi_heap_t* previous = as_native(native_heap_);
  native_heap_ = replacement;
  mi_heap_destroy(previous);
  return Status::Ok();
}

void Heap::collect(bool force) noexcept {
  if (valid()) mi_heap_collect(as_native(native_heap_), force);
}

void Heap::release() noexcept {
  if (!valid()) return;
  mi_heap_destroy(as_native(native_heap_));
  native_heap_ = nullptr;
}

void* Heap::allocate_erased(void* state, std::size_t size,
                            std::size_t alignment) noexcept {
  auto* self = static_cast<Heap*>(state);
  if (size == 0 || !self->valid()) return nullptr;
  void* pointer = heap_allocate(as_native(self->native_heap_), size, alignment);
  if (pointer != nullptr) {
    HYLE_TRACE_ALLOC_NAMED(pointer, size, "hyle.heap");
  }
  return pointer;
}

void Heap::deallocate_erased(void*, void* pointer, std::size_t size,
                             std::size_t alignment) noexcept {
#if defined(HYLE_TRACE)
  if (pointer != nullptr) HYLE_TRACE_FREE_NAMED(pointer, "hyle.heap");
#endif
  heap_deallocate(pointer, size, alignment);
}

void* Heap::reallocate_erased(void* state, void* pointer, std::size_t old_size,
                              std::size_t new_size,
                              std::size_t alignment) noexcept {
  auto* self = static_cast<Heap*>(state);
  if (!self->valid()) return nullptr;
  if (new_size == 0) {
    deallocate_erased(state, pointer, old_size, alignment);
    return nullptr;
  }
  void* result = heap_reallocate(as_native(self->native_heap_), pointer,
                                 new_size, alignment);
  if (result != nullptr) {
    if (pointer != nullptr) HYLE_TRACE_FREE_NAMED(pointer, "hyle.heap");
    HYLE_TRACE_ALLOC_NAMED(result, new_size, "hyle.heap");
  }
  return result;
}

}  // namespace hyle::mem
