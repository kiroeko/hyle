/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include "codegen_probe.h"

#include "hyle/core/compiler.h"

extern "C" {

HYLE_NOINLINE std::size_t
hyle_container_intrusive_probe_hook_size() noexcept {
  return sizeof(hyle::container::IntrusiveListHook);
}

HYLE_NOINLINE std::size_t
hyle_container_intrusive_probe_list_size() noexcept {
  return sizeof(HyleContainerIntrusiveProbeList);
}

HYLE_NOINLINE std::size_t
hyle_container_intrusive_probe_iterator_size() noexcept {
  return sizeof(HyleContainerIntrusiveProbeList::iterator);
}

HYLE_NOINLINE std::size_t
hyle_container_safe_intrusive_probe_hook_size() noexcept {
  return sizeof(hyle::container::SafeIntrusiveListHook);
}

HYLE_NOINLINE std::size_t
hyle_container_safe_intrusive_probe_list_size() noexcept {
  return sizeof(HyleContainerSafeIntrusiveProbeList);
}

HYLE_NOINLINE std::size_t
hyle_container_safe_intrusive_probe_iterator_size() noexcept {
  return sizeof(HyleContainerSafeIntrusiveProbeList::iterator);
}

HYLE_NOINLINE void hyle_container_intrusive_probe_push_front(
    HyleContainerIntrusiveProbeList *list,
    HyleContainerIntrusiveProbeNode *node) noexcept {
  list->push_front(node);
}

HYLE_NOINLINE void hyle_container_intrusive_probe_push_back(
    HyleContainerIntrusiveProbeList *list,
    HyleContainerIntrusiveProbeNode *node) noexcept {
  list->push_back(node);
}

HYLE_NOINLINE HyleContainerIntrusiveProbeNode *
hyle_container_intrusive_probe_pop_front(
    HyleContainerIntrusiveProbeList *list) noexcept {
  return list->pop_front();
}

HYLE_NOINLINE HyleContainerIntrusiveProbeNode *
hyle_container_intrusive_probe_pop_back(
    HyleContainerIntrusiveProbeList *list) noexcept {
  return list->pop_back();
}

HYLE_NOINLINE void hyle_container_intrusive_probe_remove(
    HyleContainerIntrusiveProbeList *list,
    HyleContainerIntrusiveProbeNode *node) noexcept {
  list->remove(node);
}

HYLE_NOINLINE void hyle_container_intrusive_probe_destroy_linked_node(
    HyleContainerIntrusiveProbeNode *node) noexcept {
  node->~HyleContainerIntrusiveProbeNode();
}

HYLE_NOINLINE void hyle_container_safe_intrusive_probe_push_back(
    HyleContainerSafeIntrusiveProbeList *list,
    HyleContainerSafeIntrusiveProbeNode *node) noexcept {
  list->push_back(node);
}

HYLE_NOINLINE HyleContainerSafeIntrusiveProbeNode *
hyle_container_safe_intrusive_probe_pop_front(
    HyleContainerSafeIntrusiveProbeList *list) noexcept {
  return list->pop_front();
}

HYLE_NOINLINE void hyle_container_safe_intrusive_probe_remove(
    HyleContainerSafeIntrusiveProbeList *list,
    HyleContainerSafeIntrusiveProbeNode *node) noexcept {
  list->remove(node);
}

HYLE_NOINLINE HyleContainerSafeIntrusiveProbeNode *
hyle_container_safe_intrusive_probe_node_of(
    HyleContainerSafeIntrusiveProbeList *list) noexcept {
  return &*list->begin();
}

HYLE_NOINLINE void hyle_container_safe_intrusive_probe_destroy_unlinked_hook(
    hyle::container::SafeIntrusiveListHook *hook) noexcept {
  hook->~SafeIntrusiveListHook();
}

HYLE_NOINLINE void hyle_container_intrusive_probe_append(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source) noexcept {
  destination->append(*source);
}

HYLE_NOINLINE void hyle_container_intrusive_probe_splice_all(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source,
    HyleContainerIntrusiveProbeNode *before) noexcept {
  destination->splice(destination->iterator_to(before), *source);
}

HYLE_NOINLINE HyleContainerIntrusiveProbeNode *
hyle_container_intrusive_probe_splice_single(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source,
    HyleContainerIntrusiveProbeNode *before,
    HyleContainerIntrusiveProbeNode *item) noexcept {
  const auto position = destination->iterator_to(before);
  const auto source_item = source->iterator_to(item);
  return &*destination->splice(position, *source, source_item);
}

HYLE_NOINLINE HyleContainerIntrusiveProbeNode *
hyle_container_intrusive_probe_splice_range(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source,
    HyleContainerIntrusiveProbeNode *before,
    HyleContainerIntrusiveProbeNode *first,
    HyleContainerIntrusiveProbeNode *last) noexcept {
  const auto position = destination->iterator_to(before);
  const auto range_first = source->iterator_to(first);
  const auto after_range = source->iterator_to(last);
  return &*destination->splice(position, *source, range_first, after_range);
}

HYLE_NOINLINE void hyle_container_intrusive_probe_swap(
    HyleContainerIntrusiveProbeList *left,
    HyleContainerIntrusiveProbeList *right) noexcept {
  left->swap(*right);
}

HYLE_NOINLINE HyleContainerIntrusiveProbeNode *
hyle_container_intrusive_probe_node_of(
    HyleContainerIntrusiveProbeList *list) noexcept {
  return &*list->begin();
}

HYLE_NOINLINE HyleContainerIntrusiveProbeNode *
hyle_container_intrusive_probe_iterator_next(
    HyleContainerIntrusiveProbeList *list) noexcept {
  auto position = list->begin();
  ++position;
  return &*position;
}

HYLE_NOINLINE int hyle_container_intrusive_probe_owner_assume(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source) noexcept {
  HYLE_ASSUME(destination != source);
  // The same-signature control below returns the same constant. Any difference
  // in Release is therefore work emitted solely for the owner precondition.
  return 1;
}

HYLE_NOINLINE int hyle_container_intrusive_probe_owner_control(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source) noexcept {
  (void)destination;
  (void)source;
  return 1;
}

} // extern "C"
