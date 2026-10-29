#pragma once

// Model adapter implementation, included after the Model/Session definitions.
// Transfer scheduling and native retirement belong to inference::StateTransfer.
namespace qwen35 {
namespace session_transfer {

inline void idle(Model &model, Session *session = nullptr) {
  if (model.active_step || model.carrier_borrowed ||
      (session && (session->pending || session->stopping || session->failed || !session->head ||
          session->head->model != &model || session->head->math != model.math_identity())))
    throw std::runtime_error("complete state transfer requires an idle live committed model boundary");
}

inline void check_frame(const c::PagedArenaFrame &frame, const std::vector<hyle::inference::PagedPageRef> &pages,
    std::size_t length) {
  if (!frame.valid() || frame.page_count() != pages.size())
    throw std::runtime_error("host image frame has an incomplete page inventory");
  const auto layout = frame.layout(); const auto host = layout.host();
  if (layout.facts().requests != 1 || layout.facts().kv_rows != length ||
      host.page_indices.size() != pages.size())
    throw std::runtime_error("host image frame differs from its committed logical boundary");
  for (std::size_t i = 0; i < pages.size(); ++i)
    if (host.page_indices[i] != static_cast<std::int32_t>(pages[i].page()))
      throw std::runtime_error("host image frame page order differs from its logical inventory");
}

struct ExportScope final {
  // Image owns only independent host storage. The temporary copy alone owns
  // this source checkpoint; publishing the image never pins device versions.
  cache::SnapshotRef source;
  std::shared_ptr<cache::HostImage> image;
};

struct RestoreScope final {
  // All original mutable page reservations and policy references survive an
  // abandoned transfer until its original Completion actually retires.
  std::shared_ptr<const cache::HostImage> image;
  std::shared_ptr<ColdResources> resources;
  std::unique_ptr<cache::State> state;
  std::vector<c::PagedPageWriter> pages;
  std::vector<c::ReadLease> readers;
  ~RestoreScope() {
    // StateTransfer drops its original bindings before this lifetime owner.
    // Native acceptance retains us until its F1 uses actually retire. A
    // consumed reservation must be reclaimed explicitly on a failed import;
    // an empty PagedPageWriter alone deliberately does not guess it free.
    readers.clear();
    for (auto &page : pages) if (page.valid() && !page.lease().valid())
      page.discard().IgnoreError(); // A poisoned device stays unavailable.
  }
};

inline c::BatchLayoutVersion layout(Model &model, const std::vector<hyle::inference::PagedPageRef> &pages,
    std::size_t length) {
  std::vector<std::int32_t> ids; ids.reserve(pages.size());
  for (const auto &page : pages) ids.push_back(static_cast<std::int32_t>(page.page()));
  const std::array<std::int32_t, 2> query{0, 0}, kv{0, static_cast<std::int32_t>(ids.size())};
  const std::array last{static_cast<std::int32_t>(length ? (length - 1) % cache::Plan::page_rows + 1 : 0)};
  return take(model.attention_schema.version({query, kv, ids, last}));
}

} // namespace session_transfer

inline cache::HostImageSchema Model::host_image_schema() const {
  cache::HostImageSchema result;
  result.geometry = geometry; result.capacity = capacity;
  result.hidden_row_bytes = geometry.hidden_bytes();
  result.draft_cache = cache_plan.draft_cache; result.frontier_hidden = retains_sampling_frontier();
  for (std::size_t slot = 0; slot < target_kv_layers; ++slot) {
    const auto stored = target_attention_schema(slot).capacity();
    const auto bytes = cache_storage_row_bytes(geometry, {stored.key_storage, stored.value_storage});
    result.attention.push_back({attention_layer_ids[slot], {bytes[0], bytes[1]}});
  }
  for (std::size_t layer = 0; layer < state_schema.size(); ++layer) if (geometry.is_gdn(layer))
    result.recurrent.push_back({layer, take(state_schema[layer].history.required_bytes()),
        take(state_schema[layer].recurrent.required_bytes())});
  const auto stored = attention_schema.capacity();
  const auto bytes = cache_storage_row_bytes(geometry, {stored.key_storage, stored.value_storage});
  result.draft_row_bytes = {bytes[0], bytes[1]};
  return result;
}

inline cache::HostImageIdentity Model::host_image_identity(std::size_t index) const {
  cache::HostImageIdentity result;
  result.model = this; result.math = math_identity(); result.format = &state_schema;
  result.sampling_index = index; result.sampling_configuration = &sampling_configuration(index);
  result.model_lifetime = resources_;
  return result;
}

inline std::shared_ptr<const cache::HostImage> Model::export_session(Session &session,
    const std::shared_ptr<inference::HostStateStorage> &storage) {
  session_transfer::idle(*this, &session);
  if (!storage) throw std::runtime_error("host image storage is empty");
  check(storage->validate_backend(execution));
  validate_mtp_snapshot(*session.head);
  auto scope = std::make_shared<session_transfer::ExportScope>();
  scope->source = session.checkpoint();
  auto &image = *(scope->image = std::make_shared<cache::HostImage>());
  const auto schema = host_image_schema();
  check(cache::validate_host_image_source(schema, *scope->source));
  image.identity = host_image_identity(session.sampling_configuration().index);
  image.boundary = take(cache::host_image_boundary(*scope->source));
  image.layout = take(cache::make_host_image_layout(schema, image.boundary));
  image.total_bytes = take(cache::host_image_total_bytes(image.layout));
  image.host_owner = storage; image.admission = take(storage->reserve_image());
  check(image.validate_metadata(schema, image.identity));
  const auto count = image.layout.size();
  if (count > UINT32_MAX) throw std::runtime_error("host image payload count exceeds native copy capacity");
  std::vector<inference::StateTransferCopy> copies(count);
  std::vector<c::ReadLease> ready(count);
  if (count) {
    std::size_t bytes = 0;
    for (const auto &part : image.layout) {
      bytes = cache::add(bytes, cache::state_pool_padding(bytes));
      bytes = cache::add(bytes, part.bytes);
    }
    image.pool = take(c::LeasePool::create(storage->resource(), cache::add(bytes, 1),
        {.max_blocks = static_cast<std::uint32_t>(count), .backing_alignment = cache::state_pool_alignment,
         .resource_budget = storage->budget()}));
    for (std::size_t i = 0; i < count; ++i) {
      copies[i].destination = take(image.pool.allocate(image.layout[i].bytes, cache::state_pool_alignment));
      copies[i].bytes = image.layout[i].bytes;
    }
  }
  const auto &head = *scope->source->state_identity();
  std::vector<c::ReadLease> bank(head.paged.pages.size());
  std::size_t cursor = 0;
  for (std::size_t slot = 0; slot < target_kv_layers; ++slot)
    for (const auto *frame : {&head.paged.frames[2 * slot], &head.paged.frames[2 * slot + 1]}) {
      session_transfer::check_frame(*frame, head.paged.pages, head.length);
      check(take(frame->view()).page_prefixes(bank));
      for (const auto &lease : bank) copies[cursor++].source = lease;
    }
  for (const auto &part : schema.recurrent) {
    copies[cursor++].source = head.state[part.layer].history;
    copies[cursor++].source = head.state[part.layer].recurrent;
  }
  if (head.carry_hidden.valid()) copies[cursor++].source = head.carry_hidden;
  if (head.prior_hidden.valid()) copies[cursor++].source = head.prior_hidden;
  if (head.draft) {
    bank.resize(head.draft->paged.pages.size());
    for (const auto *frame : {&head.draft->paged.frames[0], &head.draft->paged.frames[1]}) {
      session_transfer::check_frame(*frame, head.draft->paged.pages, head.draft->frontier);
      check(take(frame->view()).page_prefixes(bank));
      for (const auto &lease : bank) copies[cursor++].source = lease;
    }
  }
  if (cursor != count) throw std::runtime_error("complete state image enumeration count is inconsistent");
  for (std::size_t i = 0; i < count; ++i)
    if (copies[i].source.size() != image.layout[i].bytes)
      throw std::runtime_error("complete state image payload does not match its exact initialized byte extent");
  auto transfer = take(inference::StateTransfer::prepare(execution, copies, scope, storage->budget()));
  check(transfer.submit()); check(transfer.wait());
  for (std::size_t i = 0; i < count; ++i) ready[i] = take(take(transfer.take_writer(i)).publish());
  check(image.publish_payload(std::move(ready))); check(image.validate(schema, image.identity));
  return scope->image;
}

inline std::shared_ptr<Session> Model::restore_session(std::shared_ptr<const cache::HostImage> image) {
  session_transfer::idle(*this);
  if (!image) throw std::runtime_error("complete host image is empty");
  if (!state_storage_prepared)
    throw std::runtime_error("host image restore requires its original prepared model state storage");
  const auto schema = host_image_schema();
  check(image->validate(schema, host_image_identity(image->identity.sampling_index)));
  if (snapshot_versions.live >= cache_plan.resident_sessions)
    throw std::runtime_error("complete host restore needs a free committed-version slot within N");
  // Full validation precedes any fresh version, page reservation or native use.
  auto session = std::shared_ptr<Session>(new Session(*this, resources_,
      sampling_configuration(image->identity.sampling_index)));
  if (!session->tails.try_resize(draft_bank)) throw std::bad_alloc{};
  auto scope = std::make_shared<session_transfer::RestoreScope>();
  scope->image = std::move(image); scope->resources = resources_;
  scope->state = reserve_state(session->head);
  auto &state = *scope->state;
  const auto &boundary = scope->image->boundary;
  state.length = boundary.length; state.next_id = boundary.evaluated_next_id;
  state.sampling = boundary.evaluated_sampling;
  const auto &parts = scope->image->layout; const auto count = parts.size();
  scope->pages.resize(count); scope->readers.resize(count);
  const auto page_count = state.length / cache::Plan::page_rows + (state.length % cache::Plan::page_rows != 0);
  state.paged.pages.reserve(page_count);
  for (std::size_t page = 0; page < page_count; ++page) {
    auto reservation = take(page_inventory.acquire());
    state.paged.pages.push_back(std::move(reservation.reference));
    for (std::size_t bank = 0; bank < reservation.writers.size(); ++bank)
      scope->pages[bank * page_count + page] = std::move(reservation.writers[bank]);
  }
  std::size_t draft_begin = count, draft_pages = 0;
  if (boundary.draft) {
    state.draft.emplace(); auto &draft = *state.draft;
    if (!draft.paged.frames.try_resize(2)) throw std::bad_alloc{};
    draft.frontier = boundary.draft->frontier; draft.lag = boundary.draft->lag;
    draft.catchup_token = boundary.draft->catchup_token;
    draft_pages = draft.frontier / cache::Plan::page_rows + (draft.frontier % cache::Plan::page_rows != 0);
    draft_begin = count - draft_pages * 2; draft.paged.pages.reserve(draft_pages);
    for (std::size_t page = 0; page < draft_pages; ++page) {
      auto reservation = take(draft_inventory.acquire());
      draft.paged.pages.push_back(std::move(reservation.reference));
      scope->pages[draft_begin + page] = std::move(reservation.writers[0]);
      scope->pages[draft_begin + draft_pages + page] = std::move(reservation.writers[1]);
    }
  }
  std::vector<inference::StateTransferCopy> copies(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto &part = parts[i]; auto &copy = copies[i];
    copy.source = scope->image->payload(i); copy.bytes = part.bytes;
    if (scope->pages[i].valid()) copy.destination = std::move(scope->pages[i].lease());
    else {
      switch (part.role) {
      case cache::HostPayloadRole::kConvHistory:
        copy.destination = take(state_pools[part.layer].acquire_history(part.bytes)); break;
      case cache::HostPayloadRole::kGdnState:
        copy.destination = take(state_pools[part.layer].acquire_recurrent(part.bytes)); break;
      case cache::HostPayloadRole::kCarryHidden:
      case cache::HostPayloadRole::kPriorHidden:
        copy.destination = take(hidden_pool.allocate(part.bytes, cache::state_pool_alignment)); break;
      default: throw std::runtime_error("host image page role has no original page writer");
      }
    }
  }
  auto transfer = take(inference::StateTransfer::prepare(execution, copies, scope, allocation_budget));
  check(transfer.submit()); check(transfer.wait());
  for (std::size_t i = 0; i < count; ++i) {
    auto writer = take(transfer.take_writer(i)); const auto &part = parts[i];
    if (!scope->pages[i].valid()) { scope->readers[i] = take(writer.publish()); continue; }
    auto &page = scope->pages[i]; page.lease() = std::move(writer);
    if (part.initialized_rows == cache::Plan::page_rows)
      scope->readers[i] = take(page.publish(part.initialized_rows));
    else {
      auto published = take(page.publish_appendable(part.initialized_rows));
      scope->readers[i] = std::move(published.prefix);
      const auto bank = part.role == cache::HostPayloadRole::kTargetKey ? 0u : 1u;
      if (part.role == cache::HostPayloadRole::kDraftKey || part.role == cache::HostPayloadRole::kDraftValue)
        session->draft_tails[part.role == cache::HostPayloadRole::kDraftKey ? 0 : 1] = std::move(published.appender);
      else session->tails[geometry.attention_slot(part.layer) * 2 + bank] = std::move(published.appender);
    }
  }
  const auto target_layout = session_transfer::layout(*this, state.paged.pages, state.length);
  for (std::size_t slot = 0; slot < target_kv_layers; ++slot) {
    state.paged.frames[2 * slot] = working_frame(take(page_inventory.banks[2 * slot].read(target_layout,
        std::span(scope->readers).subspan(slot * 2 * page_count, page_count))));
    state.paged.frames[2 * slot + 1] = working_frame(take(page_inventory.banks[2 * slot + 1].read(target_layout,
        std::span(scope->readers).subspan((slot * 2 + 1) * page_count, page_count))));
  }
  for (std::size_t i = 0; i < count; ++i) switch (parts[i].role) {
    case cache::HostPayloadRole::kConvHistory: state.state[parts[i].layer].history = scope->readers[i]; break;
    case cache::HostPayloadRole::kGdnState: state.state[parts[i].layer].recurrent = scope->readers[i]; break;
    case cache::HostPayloadRole::kCarryHidden: state.carry_hidden = scope->readers[i]; break;
    case cache::HostPayloadRole::kPriorHidden: state.prior_hidden = scope->readers[i]; break;
    default: break;
  }
  if (state.draft) {
    const auto draft_layout = session_transfer::layout(*this, state.draft->paged.pages, state.draft->frontier);
    state.draft->paged.frames[0] = working_frame(take(draft_inventory.banks[0].read(draft_layout,
        std::span(scope->readers).subspan(draft_begin, draft_pages))));
    state.draft->paged.frames[1] = working_frame(take(draft_inventory.banks[1].read(draft_layout,
        std::span(scope->readers).subspan(draft_begin + draft_pages, draft_pages))));
  }
  validate_mtp_snapshot(state);
  // Registration and every fallible complete-boundary check precede visibility
  // to the caller; the host image remains valid on every failed import.
  std::erase_if(session_links, [](const auto &weak) { return weak.expired(); });
  session_links.push_back(session);
  session->head.commit(std::move(scope->state));
  if (boundary.sampling_wrapper)
    session->head.adopt(sampling_fork(session->checkpoint(), boundary.sampling, boundary.next_id));
  session->tail_epoch = session->head.epoch(); session->draft_tail_epoch = session->head.epoch();
  return session;
}

} // namespace qwen35
