#include "execution_report.h"

#include <iomanip>
#include <ostream>
#include <sstream>

#include "../inference/execution_observation.h"

namespace hyle::engine::reporting {

void write_step_observation(std::ostream &output, const inference::StepObservation &step) {
  // Keep the original synchronous endpoint, but emit one formatted record.
  if (!output.good()) { output.write("", 0); output.width(0); return; }
  std::ostringstream record;
  record.imbue(output.getloc());
  record.flags(output.flags() & ~std::ios_base::unitbuf);
  record.precision(output.precision());
  record.width(output.width());
  record.fill(output.fill());
  record.exceptions(output.exceptions());
  const auto emit = [&] {
    output.width(record.width());
    const auto text = record.view();
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.setstate(record.rdstate());
  };
  try {
    if (step.prefix_hit)
      record << "prefix_hit name=" << step.name << " reused_tokens=" << step.reused_tokens
                << " graph_calls=" << step.graph_executed << " visible=" << step.visible << '\n';
    if (step.graph_executed) record << "prepared T=" << (step.graph_input_tokens ? step.graph_input_tokens : step.input_tokens) << " visible=" << step.visible
              << " replay=" << (step.replay_route == graph::ReplayRoute::kCaptured ? "captured"
                  : step.replay_route == graph::ReplayRoute::kHybrid ? "hybrid" : "commands")
              << " cold_batch_seconds=" << step.cold_batch_seconds
              << " capture_recorded=" << step.capture_recorded
              << " capture_reused=" << step.capture_reused << '\n';
    record << step.name << " tokens=" << step.input_tokens << " visible=" << step.visible
              << " seconds=" << step.seconds << " submit_seconds=" << step.submit_seconds
              << " step_seconds=" << step.step_seconds
              << " progress=" << (step.scheduled_progress ? "scheduler" : "direct")
              << " next=" << step.next << '\n';
  } catch (...) {
    // Preserve a formatting exception after publishing its original prefix.
    try { emit(); } catch (...) {}
    throw;
  }
  emit();
}
void write_preparation_observation(std::ostream &output, const inference::StepObservation &step,
                                   const inference::PreparationObservation &preparation) {
  const auto precision = output.precision();
  output << std::setprecision(17)
            << "preparation name=" << step.name << " tokens=" << step.input_tokens
            << " visible=" << step.visible << " slot=" << preparation.slot
            << " phase_slot_prior_uses=" << preparation.phase_slot_prior_uses
            << " native_plan_first_use=" << (preparation.phase_slot_prior_uses == 0)
            << " layout_seconds=" << preparation.layout_seconds
            << " batch_prepare_seconds=" << preparation.batch_prepare_seconds
            << " replay_prepare_seconds=" << preparation.replay_prepare_seconds
            << " report_snapshot_seconds=" << preparation.report_snapshot_seconds
            << " snapshot_after_host_endpoint=1\n";
  for (std::size_t i = 0; i < preparation.reports.size(); ++i) {
    const auto &report = preparation.reports[i];
    const auto &upload = report.metadata_upload;
    output << "batch_plan name=" << step.name << " attention_index=" << i
              << " node=" << preparation.nodes[i].ordinal() << " slot=" << preparation.slot
              << " layout_generation=" << report.layout_generation
              << " metadata_staging_bytes=" << report.metadata_staging_bytes
              << " float_workspace_bytes=" << report.float_workspace_bytes
              << " int_workspace_bytes=" << report.int_workspace_bytes
              << " cta_tile_q=" << report.cta_tile_q
              << " padded_batch_size=" << report.padded_batch_size
              << " split_kv=" << report.split_kv
              << " native_metadata_unknown=" << report.native_metadata_unknown
              << " capture_compatible=" << report.capture_compatible
              << " bf16_partials=" << report.bf16_partials
              << " reused_native_plan=" << report.reused_native_plan
              << " host_waits=" << report.host_waits
              << " upload_resources=" << upload.resources << " upload_copies=" << upload.copies
              << " upload_active_copies=" << upload.active_copies << " upload_phases=" << upload.phases
              << " upload_no_op_copies=" << upload.no_op_copies
              << " upload_logical_bytes=" << upload.logical_bytes
              << " upload_physical_bytes=" << upload.physical_bytes
              << " upload_staging_bytes=" << upload.staging_bytes
              << " upload_metadata_bytes=" << upload.metadata_bytes
              << " upload_native_calls=" << upload.native_calls
              << " upload_native_waits=" << upload.native_waits
              << " upload_phase_completions=" << upload.phase_completions
              << " upload_host_waits=" << upload.host_waits
              << " upload_native_metadata_unknown=" << upload.native_metadata_unknown << '\n';
  }
  output.precision(precision);
}

} // namespace hyle::engine::reporting
