/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/container/intrusive_list.h"

struct HyleContainerIntrusiveProbeNode {
  std::uint64_t value = 0;
  hyle::container::IntrusiveListHook hook;

  explicit HyleContainerIntrusiveProbeNode(std::uint64_t input) noexcept
      : value(input) {}
};

using HyleContainerIntrusiveProbeList =
    hyle::container::IntrusiveList<HyleContainerIntrusiveProbeNode,
                                   &HyleContainerIntrusiveProbeNode::hook>;

struct HyleContainerSafeIntrusiveProbeNode {
  std::uint64_t value = 0;
  hyle::container::SafeIntrusiveListHook hook;

  explicit HyleContainerSafeIntrusiveProbeNode(std::uint64_t input) noexcept
      : value(input) {}
};

using HyleContainerSafeIntrusiveProbeList =
    hyle::container::SafeIntrusiveList<
        HyleContainerSafeIntrusiveProbeNode,
        &HyleContainerSafeIntrusiveProbeNode::hook>;

// This gate describes the supported 64-bit product ABI, not a portable lower
// bound.  Failing at compile time is preferable to silently changing the hook,
// sentinel, or iterator footprint on a new target.
static_assert(sizeof(void *) == 8);
static_assert(std::is_standard_layout_v<HyleContainerIntrusiveProbeNode>);
static_assert(offsetof(HyleContainerIntrusiveProbeNode, hook) == 8);
static_assert(sizeof(hyle::container::IntrusiveListHook) == 16);
static_assert(sizeof(hyle::container::SafeIntrusiveListHook) == 16);
static_assert(sizeof(HyleContainerIntrusiveProbeList) == 16);
static_assert(sizeof(HyleContainerSafeIntrusiveProbeList) == 16);
static_assert(sizeof(HyleContainerIntrusiveProbeList::iterator) == 8);
static_assert(sizeof(HyleContainerSafeIntrusiveProbeList::iterator) == 8);
static_assert(std::is_standard_layout_v<HyleContainerSafeIntrusiveProbeNode>);
static_assert(offsetof(HyleContainerSafeIntrusiveProbeNode, hook) == 8);

extern "C" {

std::size_t hyle_container_intrusive_probe_hook_size() noexcept;
std::size_t hyle_container_intrusive_probe_list_size() noexcept;
std::size_t hyle_container_intrusive_probe_iterator_size() noexcept;
std::size_t hyle_container_safe_intrusive_probe_hook_size() noexcept;
std::size_t hyle_container_safe_intrusive_probe_list_size() noexcept;
std::size_t hyle_container_safe_intrusive_probe_iterator_size() noexcept;

void hyle_container_intrusive_probe_push_front(
    HyleContainerIntrusiveProbeList *list,
    HyleContainerIntrusiveProbeNode *node) noexcept;
void hyle_container_intrusive_probe_push_back(
    HyleContainerIntrusiveProbeList *list,
    HyleContainerIntrusiveProbeNode *node) noexcept;
HyleContainerIntrusiveProbeNode *hyle_container_intrusive_probe_pop_front(
    HyleContainerIntrusiveProbeList *list) noexcept;
HyleContainerIntrusiveProbeNode *hyle_container_intrusive_probe_pop_back(
    HyleContainerIntrusiveProbeList *list) noexcept;
void hyle_container_intrusive_probe_remove(
    HyleContainerIntrusiveProbeList *list,
    HyleContainerIntrusiveProbeNode *node) noexcept;
void hyle_container_intrusive_probe_destroy_linked_node(
    HyleContainerIntrusiveProbeNode *node) noexcept;

void hyle_container_safe_intrusive_probe_push_back(
    HyleContainerSafeIntrusiveProbeList *list,
    HyleContainerSafeIntrusiveProbeNode *node) noexcept;
HyleContainerSafeIntrusiveProbeNode *
hyle_container_safe_intrusive_probe_pop_front(
    HyleContainerSafeIntrusiveProbeList *list) noexcept;
void hyle_container_safe_intrusive_probe_remove(
    HyleContainerSafeIntrusiveProbeList *list,
    HyleContainerSafeIntrusiveProbeNode *node) noexcept;
HyleContainerSafeIntrusiveProbeNode *
hyle_container_safe_intrusive_probe_node_of(
    HyleContainerSafeIntrusiveProbeList *list) noexcept;
void hyle_container_safe_intrusive_probe_destroy_unlinked_hook(
    hyle::container::SafeIntrusiveListHook *hook) noexcept;
void hyle_container_intrusive_probe_append(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source) noexcept;
void hyle_container_intrusive_probe_splice_all(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source,
    HyleContainerIntrusiveProbeNode *before) noexcept;
HyleContainerIntrusiveProbeNode *hyle_container_intrusive_probe_splice_single(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source,
    HyleContainerIntrusiveProbeNode *before,
    HyleContainerIntrusiveProbeNode *item) noexcept;
HyleContainerIntrusiveProbeNode *hyle_container_intrusive_probe_splice_range(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source,
    HyleContainerIntrusiveProbeNode *before,
    HyleContainerIntrusiveProbeNode *first,
    HyleContainerIntrusiveProbeNode *last) noexcept;
void hyle_container_intrusive_probe_swap(
    HyleContainerIntrusiveProbeList *left,
    HyleContainerIntrusiveProbeList *right) noexcept;

HyleContainerIntrusiveProbeNode *hyle_container_intrusive_probe_node_of(
    HyleContainerIntrusiveProbeList *list) noexcept;
HyleContainerIntrusiveProbeNode *hyle_container_intrusive_probe_iterator_next(
    HyleContainerIntrusiveProbeList *list) noexcept;

int hyle_container_intrusive_probe_owner_assume(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source) noexcept;
int hyle_container_intrusive_probe_owner_control(
    HyleContainerIntrusiveProbeList *destination,
    HyleContainerIntrusiveProbeList *source) noexcept;

} // extern "C"
