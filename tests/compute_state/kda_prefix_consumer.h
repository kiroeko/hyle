// Included after public_state_consumer's adapters. This starts at the projected
// QKV/gate boundary of GLM Flash KDA, not its weight projections or full model.
template <class Route>
Status exercise_kda_prefixes(Route &route, const char *name) {
  constexpr std::int64_t heads = 64, depth = 128, tokens = 3, kernel = 4;
  constexpr std::size_t width = heads * depth, channels = 3 * width;
  constexpr std::size_t state_elements = heads * depth * depth;
  HYLE_ASSIGN_OR_RETURN(auto state_desc, dense({1, heads, depth, depth}));
  HYLE_ASSIGN_OR_RETURN(auto history_desc, dense({1, kernel - 1, channels}));
  HYLE_ASSIGN_OR_RETURN(auto samples_desc,
                        dense({1, kernel + tokens, channels}));
  HYLE_ASSIGN_OR_RETURN(auto weights_desc, dense({channels, kernel}));
  HYLE_ASSIGN_OR_RETURN(auto norm_desc, dense({depth}));
  HYLE_ASSIGN_OR_RETURN(auto gates_desc,
                        dense({1, tokens + 1, heads, depth}));
  HYLE_ASSIGN_OR_RETURN(auto beta_desc, dense({1, tokens + 1, heads, 1}));
  std::vector<float> initial(state_elements), samples((kernel + tokens) * channels);
  std::vector<float> weights(channels * kernel), scales(depth);
  std::vector<float> gates((tokens + 1) * width), betas((tokens + 1) * heads);
  for (std::size_t i = 0; i < initial.size(); ++i)
    initial[i] = float(int(i % 23) - 11) / 8192.F;
  for (std::size_t i = 0; i < samples.size(); ++i)
    samples[i] = float(int((i * 7 + i / channels) % 37) - 18) / 64.F;
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = float(int(i % 11) - 5) / 64.F;
  std::fill(scales.begin(), scales.end(), 1.F / std::sqrt(float(depth)));
  // Actual KDA post-projection gates: -5*sigmoid(exp(A_log)*(raw+dt)),
  // with a per-key-channel gate and scalar beta per value head.
  for (std::size_t t = 0; t <= tokens; ++t)
    for (std::size_t h = 0; h < heads; ++h) {
      const double alpha = std::exp(double(int(h % 7) - 3) / 8.);
      const double raw_beta = double(int((h + 3 * t) % 13) - 6) / 8.;
      betas[t * heads + h] = float(1. / (1. + std::exp(-raw_beta)));
      for (std::size_t d = 0; d < depth; ++d) {
        const double raw = double(int((t * 17 + h * 3 + d) % 19) - 9) / 8.;
        const double dt = double(int(d % 11) - 5) / 32.;
        gates[t * width + h * depth + d] =
            float(-5. / (1. + std::exp(-alpha * (raw + dt))));
      }
    }
  HYLE_ASSIGN_OR_RETURN(auto s0, route.load(state_desc, initial));
  HYLE_ASSIGN_OR_RETURN(auto projected, route.load(samples_desc, samples));
  HYLE_ASSIGN_OR_RETURN(auto filter, route.load(weights_desc, weights));
  HYLE_ASSIGN_OR_RETURN(auto normalization, route.load(norm_desc, scales));
  HYLE_ASSIGN_OR_RETURN(auto decay, route.load(gates_desc, gates));
  HYLE_ASSIGN_OR_RETURN(auto beta, route.load(beta_desc, betas));
  HYLE_ASSIGN_OR_RETURN(auto history,
      projected.subview(0, (kernel - 1) * channels * sizeof(float)));

  const auto check = [&](const c::ReadLease &actual,
                         const c::ReadLease &expected,
                         const c::TensorDesc &description) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto a, route.read(actual, description));
    HYLE_ASSIGN_OR_RETURN(auto e, route.read(expected, description));
    HYLE_RETURN_IF_ERROR(require(a.size() == e.size(), "KDA extent differs"));
    for (std::size_t i = 0; i < a.size(); ++i)
      if (!std::isfinite(a[i]) || !std::isfinite(e[i]) ||
          std::abs(a[i] - e[i]) > 3.e-6F)
        return Status::Internal("KDA block/prefix sequential reference differs");
    return Status::Ok();
  };
  const auto chain = [&](const c::ReadLease &window, std::size_t count,
                         std::size_t begin, const c::ReadLease &state,
                         bool graph) -> Result<std::vector<c::ReadLease>> {
    const auto rows_count = static_cast<std::int64_t>(count);
    HYLE_ASSIGN_OR_RETURN(auto wd, dense({1, kernel - 1 + rows_count, channels}));
    HYLE_ASSIGN_OR_RETURN(auto cd, dense({1, rows_count, channels}));
    Work convolution{c::model_ops::causal_conv1d(), {wd, weights_desc}, {cd}};
    HYLE_ASSIGN_OR_RETURN(auto convolved,
                          route.math(convolution, std::array{window, filter}, graph));
    Work silu{c::Operation(c::Unary{c::UnaryKind::kSilu}), {cd}, {cd}};
    HYLE_ASSIGN_OR_RETURN(auto activated, route.math(silu, convolved, graph));
    HYLE_ASSIGN_OR_RETURN(auto rows, dense({rows_count * 3 * heads, depth}));
    HYLE_ASSIGN_OR_RETURN(auto selected_rows, dense({rows_count * heads, depth}));
    HYLE_ASSIGN_OR_RETURN(auto qkv_desc, dense({1, rows_count, heads, depth}));
    HYLE_ASSIGN_OR_RETURN(auto ids_desc, c::TensorDesc::dense(
        c::DataType::kInt32, std::array{std::int64_t(count * heads)}));
    std::array<c::ReadLease, 6> inputs;
    for (std::size_t port = 0; port < 3; ++port) {
      std::vector<std::int32_t> ids(count * heads);
      for (std::size_t t = 0; t < count; ++t)
        for (std::size_t h = 0; h < heads; ++h)
          ids[t * heads + h] = std::int32_t((t * 3 + port) * heads + h);
      HYLE_ASSIGN_OR_RETURN(auto selected_ids,
                            route.load_bytes(ids_desc, std::as_bytes(std::span(ids))));
      Work gather{c::Operation(c::Gather{}), {rows, ids_desc}, {selected_rows}};
      HYLE_ASSIGN_OR_RETURN(auto selected,
          route.math(gather, std::array{activated[0], selected_ids}, graph));
      inputs[port] = selected[0];
      if (port < 2) {
        Work norm{c::Operation(c::RmsNorm{.epsilon = 1.e-6F / float(depth)}),
                  {qkv_desc, norm_desc}, {qkv_desc}};
        HYLE_ASSIGN_OR_RETURN(auto normalized,
            route.math(norm, std::array{inputs[port], normalization}, graph));
        inputs[port] = normalized[0];
      }
    }
    HYLE_ASSIGN_OR_RETURN(inputs[3], decay.subview(
        begin * width * sizeof(float), count * width * sizeof(float)));
    HYLE_ASSIGN_OR_RETURN(inputs[4], beta.subview(
        begin * heads * sizeof(float), count * heads * sizeof(float)));
    inputs[5] = state;
    HYLE_ASSIGN_OR_RETURN(auto gd, dense({1, rows_count, heads, depth}));
    HYLE_ASSIGN_OR_RETURN(auto bd, dense({1, rows_count, heads, 1}));
    auto output_state = state_desc;
    if (count > 1) {
      HYLE_ASSIGN_OR_RETURN(output_state,
                            dense({1, rows_count, heads, depth, depth}));
    }
    Work recurrence{c::Operation(c::GatedDeltaNet{
                        .snapshot_count = static_cast<std::uint32_t>(count)}),
                    {qkv_desc, qkv_desc, qkv_desc, gd, bd, state_desc},
                    {qkv_desc, output_state}};
    return route.math(recurrence, inputs, graph);
  };

  const std::array specs{c::StateEntrySpec{.description = state_desc},
      c::StateEntrySpec{.description = history_desc},
      c::StateEntrySpec{.kind = c::StateEntryKind::kHostMetadata,
                        .metadata_bytes = sizeof(std::uint64_t)}};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::StateSchema::create(specs));
  // One shared 128 MiB payload pool is charged by root capacity, plus the
  // separately owned host cursor snapshots.
  HYLE_ASSIGN_OR_RETURN(auto store, c::StateStore::create(
      schema, {.max_backing_bytes = 256u * 1024u * 1024u}));
  HYLE_ASSIGN_OR_RETURN(auto sid, schema.entry(0));
  HYLE_ASSIGN_OR_RETURN(auto hid, schema.entry(1));
  HYLE_ASSIGN_OR_RETURN(auto cid, schema.entry(2));
  const std::uint64_t zero = 0;
  HYLE_ASSIGN_OR_RETURN(auto base, store.import(std::array{
      c::StateMember{sid, s0}, c::StateMember{hid, history},
      c::StateMember{cid, {}, std::as_bytes(std::span(&zero, 1))}}));
  HYLE_ASSIGN_OR_RETURN(auto token_desc, dense({1, 1, heads, depth}));
  HYLE_ASSIGN_OR_RETURN(auto one_sample, dense({1, 1, channels}));
  HYLE_ASSIGN_OR_RETURN(auto full_window, dense({1, kernel, channels}));
  Work append{c::Operation(c::Concat{.axis = 1}),
              {history_desc, one_sample}, {full_window}};
  for (bool graph : {false, true}) {
    std::vector<c::StateVersion> sequential{base};
    std::vector<c::ReadLease> reference_scores;
    for (std::size_t t = 0; t <= tokens; ++t) {
      HYLE_ASSIGN_OR_RETURN(auto prior_state, sequential.back().member(sid));
      HYLE_ASSIGN_OR_RETURN(auto prior_history, sequential.back().member(hid));
      HYLE_ASSIGN_OR_RETURN(auto sample, projected.subview(
          (kernel - 1 + t) * channels * sizeof(float), channels * sizeof(float)));
      HYLE_ASSIGN_OR_RETURN(auto joined,
          route.math(append, std::array{prior_history, sample}, graph));
      HYLE_ASSIGN_OR_RETURN(auto output, chain(joined[0], 1, t, prior_state, graph));
      reference_scores.push_back(output[0]);
      HYLE_ASSIGN_OR_RETURN(auto next_history, joined[0].subview(
          channels * sizeof(float), (kernel - 1) * channels * sizeof(float)));
      HYLE_ASSIGN_OR_RETURN(auto draft, store.begin(sequential.back()));
      HYLE_RETURN_IF_ERROR(draft.replace(sid, output[1]));
      HYLE_RETURN_IF_ERROR(draft.replace(hid, next_history));
      const std::uint64_t cursor = t + 1;
      HYLE_RETURN_IF_ERROR(draft.replace_metadata(cid, std::as_bytes(std::span(&cursor, 1))));
      HYLE_ASSIGN_OR_RETURN(auto next, draft.finalize());
      sequential.push_back(std::move(next));
    }
    HYLE_ASSIGN_OR_RETURN(auto verify_window, projected.subview(
        0, (kernel - 1 + tokens) * channels * sizeof(float)));
    HYLE_ASSIGN_OR_RETURN(auto verified, chain(verify_window, tokens, 0, s0, graph));
    for (std::size_t accepted = 0; accepted <= tokens; ++accepted) {
      HYLE_ASSIGN_OR_RETURN(auto head, c::StateHead::create(store, base));
      if (accepted) {
        HYLE_ASSIGN_OR_RETURN(auto selected, verified[1].subview(
            (accepted - 1) * state_elements * sizeof(float),
            state_elements * sizeof(float)));
        HYLE_ASSIGN_OR_RETURN(auto selected_history, projected.subview(
            accepted * channels * sizeof(float),
            (kernel - 1) * channels * sizeof(float)));
        HYLE_ASSIGN_OR_RETURN(auto draft, store.begin(base));
        HYLE_RETURN_IF_ERROR(draft.replace(sid, selected));
        HYLE_RETURN_IF_ERROR(draft.replace(hid, selected_history));
        const std::uint64_t cursor = accepted;
        HYLE_RETURN_IF_ERROR(draft.replace_metadata(cid, std::as_bytes(std::span(&cursor, 1))));
        HYLE_ASSIGN_OR_RETURN(auto candidate, draft.finalize());
        HYLE_RETURN_IF_ERROR(head.publish(base.identity(), candidate));
        HYLE_ASSIGN_OR_RETURN(auto score, verified[0].subview(
            (accepted - 1) * width * sizeof(float), width * sizeof(float)));
        HYLE_RETURN_IF_ERROR(check(score, reference_scores[accepted - 1], token_desc));
      }
      HYLE_ASSIGN_OR_RETURN(auto current, head.current());
      HYLE_ASSIGN_OR_RETURN(auto selected_state, current.member(sid));
      HYLE_ASSIGN_OR_RETURN(auto selected_history, current.member(hid));
      HYLE_ASSIGN_OR_RETURN(auto expected_state, sequential[accepted].member(sid));
      HYLE_ASSIGN_OR_RETURN(auto expected_history, sequential[accepted].member(hid));
      HYLE_RETURN_IF_ERROR(check(selected_state, expected_state, state_desc));
      HYLE_RETURN_IF_ERROR(check(selected_history, expected_history, history_desc));
      HYLE_ASSIGN_OR_RETURN(auto cursor_member, current.member(cid));
      std::uint64_t cursor = ~std::uint64_t{0};
      HYLE_RETURN_IF_ERROR(cursor_member.view()->read(
          std::as_writable_bytes(std::span(&cursor, 1))));
      HYLE_RETURN_IF_ERROR(require(cursor == accepted, "KDA cursor/state publication differs"));
      HYLE_ASSIGN_OR_RETURN(auto sample, projected.subview(
          (kernel - 1 + accepted) * channels * sizeof(float), channels * sizeof(float)));
      HYLE_ASSIGN_OR_RETURN(auto joined,
          route.math(append, std::array{selected_history, sample}, graph));
      HYLE_ASSIGN_OR_RETURN(auto continued,
                            chain(joined[0], 1, accepted, selected_state, graph));
      HYLE_ASSIGN_OR_RETURN(auto expected_next, sequential[accepted + 1].member(sid));
      HYLE_RETURN_IF_ERROR(check(continued[1], expected_next, state_desc));
      HYLE_RETURN_IF_ERROR(check(continued[0], reference_scores[accepted], token_desc));
      HYLE_ASSIGN_OR_RETURN(auto continued_history, joined[0].subview(
          channels * sizeof(float), (kernel - 1) * channels * sizeof(float)));
      HYLE_ASSIGN_OR_RETURN(auto expected_next_history,
                            sequential[accepted + 1].member(hid));
      HYLE_RETURN_IF_ERROR(check(continued_history, expected_next_history, history_desc));
      HYLE_ASSIGN_OR_RETURN(auto next_draft, store.begin(current));
      HYLE_RETURN_IF_ERROR(next_draft.replace(sid, continued[1]));
      HYLE_RETURN_IF_ERROR(next_draft.replace(hid, continued_history));
      const std::uint64_t next_cursor = accepted + 1;
      HYLE_RETURN_IF_ERROR(next_draft.replace_metadata(
          cid, std::as_bytes(std::span(&next_cursor, 1))));
      HYLE_ASSIGN_OR_RETURN(auto continued_version, next_draft.finalize());
      HYLE_RETURN_IF_ERROR(head.publish(current.identity(), continued_version));
      // A failed/cancelled speculative draft never replaces the shared head.
      // This checks public transaction failure, not an injected native failure.
      HYLE_ASSIGN_OR_RETURN(auto cancelled, store.begin(continued_version));
      HYLE_RETURN_IF_ERROR(cancelled.replace(sid, s0));
      HYLE_RETURN_IF_ERROR(cancelled.cancel());
      const auto rejected = cancelled.finalize();
      HYLE_RETURN_IF_ERROR(require(!rejected && rejected.error().IsCancelled(),
                                  "cancelled KDA draft finalized"));
      HYLE_ASSIGN_OR_RETURN(auto still_current, head.current());
      HYLE_RETURN_IF_ERROR(require(still_current.identity() == continued_version.identity(),
                                  "cancelled KDA draft changed the head"));
      std::printf("KDA %s %s H64/D128 accepted=%zu continuation PASS\n",
                  name, graph ? "Graph" : "Direct", accepted);
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto old_state, base.member(sid));
  HYLE_ASSIGN_OR_RETURN(auto unchanged, route.read(old_state, state_desc));
  HYLE_RETURN_IF_ERROR(require(unchanged == initial, "KDA mutated its old state"));
  HYLE_ASSIGN_OR_RETURN(auto old_history, base.member(hid));
  HYLE_ASSIGN_OR_RETURN(auto old_samples, route.read(old_history, history_desc));
  HYLE_RETURN_IF_ERROR(require(std::equal(old_samples.begin(), old_samples.end(),
                                       samples.begin()), "KDA mutated its old conv history"));
  return store.drain();
}
