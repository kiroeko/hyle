#include "hyle/sched/cancel.h"

#include <cstddef>
#include <mutex>
#include <new>
#include <utility>

namespace hyle::sched {
namespace detail {

struct CancelTree {
  mem::AllocatorRef allocator;
  std::recursive_mutex mutex;
  std::size_t node_count = 0;
  unsigned cancel_depth = 0;
  CancelNode* pending_delete = nullptr;
};

struct CancelNode {
  using CallbackList =
      container::SafeIntrusiveHList<CancelCallbackRecord,
                                    &CancelCallbackRecord::hook>;

  CancelTree* tree = nullptr;
  CancelNode* parent = nullptr;
  CancelNode* first_child = nullptr;
  CancelNode* next_sibling = nullptr;
  CancelNode* previous_sibling = nullptr;
  CancelNode* next_pending = nullptr;
  CallbackList callbacks;
  std::size_t handle_count = 1;
  std::atomic<bool> cancelled{false};
  bool pending_delete = false;
};

namespace {

void cancel_subtree_locked(CancelNode* root) noexcept;

void fire_callbacks_locked(CancelNode* node) noexcept {
  while (!node->callbacks.empty()) {
    CancelCallbackRecord* callback = node->callbacks.pop_front();
    callback->node = nullptr;
    callback->fn(callback->context);
  }
}

void cancel_subtree_locked(CancelNode* root) noexcept {
  if (root == nullptr ||
      root->cancelled.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  fire_callbacks_locked(root);

  // Re-entrant token drops are deferred until the outermost cancel returns,
  // keeping this allocation-free topology walk stable while callbacks run.
  CancelNode* current = root->first_child;
  while (current != nullptr && current != root) {
    const bool was_cancelled =
        current->cancelled.exchange(true, std::memory_order_acq_rel);
    if (!was_cancelled) fire_callbacks_locked(current);
    if (!was_cancelled && current->first_child != nullptr) {
      current = current->first_child;
    } else if (current->next_sibling != nullptr) {
      current = current->next_sibling;
    } else {
      while (current != nullptr && current != root &&
             current->next_sibling == nullptr) {
        current = current->parent;
      }
      if (current == nullptr || current == root) break;
      current = current->next_sibling;
    }
  }
}

void unlink_node_locked(CancelNode* node) noexcept {
  CancelNode* parent = node->parent;
  CancelNode* child = node->first_child;
  if (child != nullptr) {
    CancelNode* tail = child;
    for (;;) {
      tail->parent = parent;
      if (tail->next_sibling == nullptr) break;
      tail = tail->next_sibling;
    }
    if (parent != nullptr) {
      tail->next_sibling = parent->first_child;
      if (parent->first_child != nullptr) {
        parent->first_child->previous_sibling = tail;
      }
      parent->first_child = child;
      child->previous_sibling = nullptr;
      if (parent->cancelled.load(std::memory_order_acquire)) {
        for (CancelNode* current = child;;) {
          CancelNode* next = current == tail ? nullptr : current->next_sibling;
          cancel_subtree_locked(current);
          if (next == nullptr) break;
          current = next;
        }
      }
    } else {
      child->previous_sibling = nullptr;
    }
  }

  if (parent != nullptr) {
    if (node->previous_sibling != nullptr) {
      node->previous_sibling->next_sibling = node->next_sibling;
    } else {
      HYLE_FATAL_ASSERT(parent->first_child == node);
      parent->first_child = node->next_sibling;
    }
    if (node->next_sibling != nullptr) {
      node->next_sibling->previous_sibling = node->previous_sibling;
    }
  }
  node->parent = nullptr;
  node->first_child = nullptr;
  node->next_sibling = nullptr;
  node->previous_sibling = nullptr;
}

void enqueue_delete_locked(CancelNode* node) noexcept {
  if (node->pending_delete) return;
  HYLE_FATAL_ASSERT(node->handle_count == 0);
  HYLE_FATAL_ASSERT(node->callbacks.empty() &&
                    "CancelToken must outlive its CancelCallback objects");
  node->pending_delete = true;
  node->next_pending = node->tree->pending_delete;
  node->tree->pending_delete = node;
}

void drain_deletes_locked(CancelTree* tree) noexcept {
  HYLE_FATAL_ASSERT(tree->cancel_depth == 0);
  ++tree->cancel_depth;
  while (tree->pending_delete != nullptr) {
    CancelNode* node = tree->pending_delete;
    tree->pending_delete = node->next_pending;
    node->next_pending = nullptr;
    unlink_node_locked(node);
    mem::AllocatorRef allocator = tree->allocator;
    node->~CancelNode();
    allocator.deallocate(node, sizeof(CancelNode), alignof(CancelNode));
    HYLE_FATAL_ASSERT(tree->node_count != 0);
    --tree->node_count;
  }
  --tree->cancel_depth;
}

}  // namespace
}  // namespace detail

Result<CancelToken> CancelToken::try_create_root(
    mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid()) {
    return std::unexpected(
        Status::InvalidArgument("cancellation tree requires an allocator"));
  }

  void* tree_storage =
      allocator.allocate(sizeof(detail::CancelTree), alignof(detail::CancelTree));
  if (tree_storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("cancellation tree allocation failed"));
  }
  detail::CancelTree* tree = nullptr;
  try {
    tree = new (tree_storage) detail::CancelTree{};
  } catch (...) {
    allocator.deallocate(tree_storage, sizeof(detail::CancelTree),
                         alignof(detail::CancelTree));
    return std::unexpected(
        Status::Unavailable("cancellation tree lock initialization failed"));
  }
  tree->allocator = allocator;

  void* node_storage =
      allocator.allocate(sizeof(detail::CancelNode), alignof(detail::CancelNode));
  if (node_storage == nullptr) {
    tree->~CancelTree();
    allocator.deallocate(tree, sizeof(detail::CancelTree),
                         alignof(detail::CancelTree));
    return std::unexpected(
        Status::ResourceExhausted("cancellation root allocation failed"));
  }
  auto* node = new (node_storage) detail::CancelNode{};
  node->tree = tree;
  tree->node_count = 1;
  return CancelToken(node);
}

CancelToken CancelToken::root(mem::AllocatorRef allocator) noexcept {
  auto result = try_create_root(allocator);
  HYLE_FATAL_ASSERT(result.has_value());
  return *std::move(result);
}

Result<CancelToken> CancelToken::try_derive() const noexcept {
  if (node_ == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("cannot derive from an invalid cancel token"));
  }
  detail::CancelTree* tree = node_->tree;
  std::lock_guard<std::recursive_mutex> lock(tree->mutex);
  void* storage = tree->allocator.allocate(sizeof(detail::CancelNode),
                                           alignof(detail::CancelNode));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("cancellation child allocation failed"));
  }
  auto* child = new (storage) detail::CancelNode{};
  child->tree = tree;
  child->parent = node_;
  child->next_sibling = node_->first_child;
  if (node_->first_child != nullptr) {
    node_->first_child->previous_sibling = child;
  }
  node_->first_child = child;
  if (node_->cancelled.load(std::memory_order_acquire)) {
    child->cancelled.store(true, std::memory_order_release);
  }
  ++tree->node_count;
  return CancelToken(child);
}

CancelToken CancelToken::derive() const noexcept {
  auto result = try_derive();
  HYLE_FATAL_ASSERT(result.has_value());
  return *std::move(result);
}

CancelToken::CancelToken(const CancelToken& other) noexcept : node_(other.node_) {
  retain_();
}

CancelToken& CancelToken::operator=(const CancelToken& other) noexcept {
  if (this == &other) return *this;
  detail::CancelNode* next = other.node_;
  if (next != nullptr) {
    std::lock_guard<std::recursive_mutex> lock(next->tree->mutex);
    ++next->handle_count;
  }
  reset_();
  node_ = next;
  return *this;
}

CancelToken::CancelToken(CancelToken&& other) noexcept
    : node_(std::exchange(other.node_, nullptr)) {}

CancelToken& CancelToken::operator=(CancelToken&& other) noexcept {
  if (this != &other) {
    reset_();
    node_ = std::exchange(other.node_, nullptr);
  }
  return *this;
}

CancelToken::~CancelToken() { reset_(); }

void CancelToken::retain_() noexcept {
  if (node_ == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(node_->tree->mutex);
  ++node_->handle_count;
}

void CancelToken::reset_() noexcept {
  if (node_ == nullptr) return;
  detail::CancelNode* node = std::exchange(node_, nullptr);
  detail::CancelTree* tree = node->tree;
  bool destroy_tree = false;
  {
    std::lock_guard<std::recursive_mutex> lock(tree->mutex);
    HYLE_FATAL_ASSERT(node->handle_count != 0);
    if (--node->handle_count == 0) {
      detail::enqueue_delete_locked(node);
      if (tree->cancel_depth == 0) detail::drain_deletes_locked(tree);
    }
    destroy_tree = tree->node_count == 0;
  }
  if (destroy_tree) {
    mem::AllocatorRef allocator = tree->allocator;
    tree->~CancelTree();
    allocator.deallocate(tree, sizeof(detail::CancelTree),
                         alignof(detail::CancelTree));
  }
}

void CancelToken::cancel() const noexcept {
  if (node_ == nullptr) return;
  detail::CancelTree* tree = node_->tree;
  std::lock_guard<std::recursive_mutex> lock(tree->mutex);
  ++tree->cancel_depth;
  detail::cancel_subtree_locked(node_);
  HYLE_FATAL_ASSERT(tree->cancel_depth != 0);
  --tree->cancel_depth;
  if (tree->cancel_depth == 0 && tree->pending_delete != nullptr) {
    detail::drain_deletes_locked(tree);
  }
}

bool CancelToken::is_cancelled() const noexcept {
  return node_ != nullptr &&
         node_->cancelled.load(std::memory_order_acquire);
}

std::atomic<bool>* CancelToken::flag_() const noexcept {
  return node_ != nullptr ? &node_->cancelled : nullptr;
}

CancelCallback::CancelCallback(const CancelToken& token, Fn fn,
                               void* context) noexcept {
  HYLE_FATAL_ASSERT(fn != nullptr);
  detail::CancelNode* node = token.node_;
  if (node == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(node->tree->mutex);
  if (node->cancelled.load(std::memory_order_acquire)) {
    fn(context);
    return;
  }
  record_.node = node;
  record_.tree = node->tree;
  record_.fn = fn;
  record_.context = context;
  node->callbacks.push_front(&record_);
}

CancelCallback::~CancelCallback() { unregister(); }

void CancelCallback::unregister() noexcept {
  detail::CancelTree* tree = record_.tree;
  if (tree == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(tree->mutex);
  detail::CancelNode* node = record_.node;
  if (node == nullptr) {
    record_.tree = nullptr;
    return;
  }
  HYLE_FATAL_ASSERT(record_.hook.linked());
  node->callbacks.remove(&record_);
  record_.node = nullptr;
  record_.tree = nullptr;
}

}  // namespace hyle::sched
