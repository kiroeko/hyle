/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include "codegen_probe.h"

#include <cstdio>
#include <initializer_list>
#include <new>

namespace {

using List = HyleContainerIntrusiveProbeList;
using Node = HyleContainerIntrusiveProbeNode;

bool expect(bool condition, const char *message) noexcept {
  if (!condition)
    std::fprintf(stderr, "container_intrusive codegen probe: %s\n", message);
  return condition;
}

template <class ListType>
bool expect_sequence(
    ListType &list,
    std::initializer_list<typename ListType::value_type *> expected) noexcept {
  auto position = list.begin();
  for (Node *node : expected) {
    if (position == list.end() || &*position != node)
      return false;
    ++position;
  }
  return position == list.end();
}

bool exercise_push_pop_remove_and_iterators() noexcept {
  List list;
  Node a{1}, b{2}, c{3};
  hyle_container_intrusive_probe_push_back(&list, &a);
  hyle_container_intrusive_probe_push_front(&list, &b);
  hyle_container_intrusive_probe_push_back(&list, &c);
  if (!expect(hyle_container_intrusive_probe_node_of(&list) == &b,
              "node_of did not recover the first node") ||
      !expect(hyle_container_intrusive_probe_iterator_next(&list) == &a,
              "iterator increment did not reach the second node") ||
      !expect(hyle_container_intrusive_probe_pop_front(&list) == &b,
              "pop_front returned the wrong node") ||
      !expect(hyle_container_intrusive_probe_pop_back(&list) == &c,
              "pop_back returned the wrong node"))
    return false;
  hyle_container_intrusive_probe_remove(&list, &a);
  return expect(list.empty() && !a.hook.linked() && !b.hook.linked() &&
                    !c.hook.linked(),
                "push/pop/remove left a linked node");
}

bool exercise_auto_linked_node_destruction() noexcept {
  alignas(Node) unsigned char storage[sizeof(Node)]{};
  auto *node = ::new (static_cast<void *>(storage)) Node{7};
  List list;
  list.push_back(node);
  hyle_container_intrusive_probe_destroy_linked_node(node);
  // The node lifetime has ended, so only inspect the surviving list. Its
  // sentinel must have been repaired without requiring list.remove().
  return expect(list.empty(),
                "linked auto node destruction did not repair the list");
}

bool exercise_append_and_splice_all() noexcept {
  {
    List destination;
    List source;
    Node a{1}, b{2}, c{3};
    destination.push_back(&a);
    source.push_back(&b);
    source.push_back(&c);
    hyle_container_intrusive_probe_append(&destination, &source);
    if (!expect(source.empty() &&
                    expect_sequence(destination, {&a, &b, &c}),
                "append did not transfer the complete source"))
      return false;
  }
  {
    List destination;
    List source;
    Node a{1}, b{2}, c{3}, d{4};
    destination.push_back(&a);
    destination.push_back(&d);
    source.push_back(&b);
    source.push_back(&c);
    hyle_container_intrusive_probe_splice_all(&destination, &source, &d);
    if (!expect(source.empty() &&
                    expect_sequence(destination, {&a, &b, &c, &d}),
                "whole-list splice produced the wrong order"))
      return false;
  }
  return true;
}

bool exercise_single_and_range_splice() noexcept {
  {
    List destination;
    List source;
    Node a{1}, b{2}, c{3}, d{4};
    destination.push_back(&a);
    destination.push_back(&c);
    source.push_back(&b);
    source.push_back(&d);
    Node *moved = hyle_container_intrusive_probe_splice_single(
        &destination, &source, &c, &b);
    if (!expect(moved == &b && expect_sequence(destination, {&a, &b, &c}) &&
                    expect_sequence(source, {&d}),
                "single-node splice produced the wrong order"))
      return false;
  }
  {
    List destination;
    List source;
    Node a{1}, b{2}, c{3}, d{4}, e{5}, f{6};
    destination.push_back(&a);
    destination.push_back(&f);
    source.push_back(&b);
    source.push_back(&c);
    source.push_back(&d);
    source.push_back(&e);
    Node *moved = hyle_container_intrusive_probe_splice_range(
        &destination, &source, &f, &b, &e);
    if (!expect(moved == &b &&
                    expect_sequence(destination, {&a, &b, &c, &d, &f}) &&
                    expect_sequence(source, {&e}),
                "range splice produced the wrong order"))
      return false;
  }
  return true;
}

bool exercise_swap_and_owner_assume() noexcept {
  List left;
  List right;
  Node a{1}, b{2}, c{3};
  left.push_back(&a);
  left.push_back(&b);
  right.push_back(&c);
  hyle_container_intrusive_probe_swap(&left, &right);
  return expect(expect_sequence(left, {&c}) &&
                    expect_sequence(right, {&a, &b}),
                "swap produced the wrong order") &&
         expect(hyle_container_intrusive_probe_owner_assume(&left, &right) ==
                        hyle_container_intrusive_probe_owner_control(&left,
                                                                    &right),
                    "valid distinct-owner assumption changed the result");
}

bool exercise_safe_link_hot_operations() noexcept {
  using SafeList = HyleContainerSafeIntrusiveProbeList;
  using SafeNode = HyleContainerSafeIntrusiveProbeNode;
  SafeList list;
  SafeNode a{1}, b{2};
  hyle_container_safe_intrusive_probe_push_back(&list, &a);
  hyle_container_safe_intrusive_probe_push_back(&list, &b);
  if (!expect(hyle_container_safe_intrusive_probe_node_of(&list) == &a,
              "safe node_of did not recover the first node") ||
      !expect(hyle_container_safe_intrusive_probe_pop_front(&list) == &a,
              "safe pop_front returned the wrong node") ||
      !expect(!a.hook.linked() && b.hook.linked(),
              "safe pop_front did not reset exactly one hook"))
    return false;
  hyle_container_safe_intrusive_probe_remove(&list, &b);
  if (!expect(list.empty() && !b.hook.linked(),
              "safe remove did not reset the hook"))
    return false;

  alignas(hyle::container::SafeIntrusiveListHook) unsigned char storage
      [sizeof(hyle::container::SafeIntrusiveListHook)]{};
  auto *unlinked_hook =
      ::new (static_cast<void *>(storage))
          hyle::container::SafeIntrusiveListHook;
  hyle_container_safe_intrusive_probe_destroy_unlinked_hook(unlinked_hook);
  return true;
}

} // namespace

int main() {
  const std::size_t hook_size = hyle_container_intrusive_probe_hook_size();
  const std::size_t list_size = hyle_container_intrusive_probe_list_size();
  const std::size_t iterator_size =
      hyle_container_intrusive_probe_iterator_size();
  const std::size_t safe_hook_size =
      hyle_container_safe_intrusive_probe_hook_size();
  const std::size_t safe_list_size =
      hyle_container_safe_intrusive_probe_list_size();
  const std::size_t safe_iterator_size =
      hyle_container_safe_intrusive_probe_iterator_size();
  if (!expect(hook_size == 16 && list_size == 16 && iterator_size == 8 &&
                  safe_hook_size == 16 && safe_list_size == 16 &&
                  safe_iterator_size == 8 && sizeof(void *) == 8,
              "64-bit auto/safe layout is not hook/list/iterator = 16/16/8") ||
      !exercise_push_pop_remove_and_iterators() ||
      !exercise_auto_linked_node_destruction() ||
      !exercise_append_and_splice_all() ||
      !exercise_single_and_range_splice() ||
      !exercise_swap_and_owner_assume() ||
      !exercise_safe_link_hot_operations())
    return 1;

  std::printf("container_intrusive_codegen_layout "
              "hook=%zu list=%zu iterator=%zu safe_hook=%zu safe_list=%zu "
              "safe_iterator=%zu pointer=%zu\n",
              hook_size, list_size, iterator_size, safe_hook_size,
              safe_list_size, safe_iterator_size, sizeof(void *));
  return 0;
}
