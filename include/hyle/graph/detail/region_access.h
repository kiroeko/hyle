#pragma once
#include <memory>
#include "hyle/graph/executable.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/detail/operation_access.h"
namespace hyle::graph { struct ProviderRegionPreparedVTable; struct ProviderRegionPrepared; }
namespace hyle::compute::detail { class AppendLease; }
namespace hyle::graph::detail {
struct PlanState;
struct InvocationState;
class InputSource;
struct OutputBinding;
class PreparedRegion;
class RegionInvocation;
// Cold cleanup of the actual selected slot, queried through its existing
// artifact extension table. Does not change the prepared-vtable ABI.
inline constexpr compute::ProviderExtensionId kRegionBindingReleaseExtensionId{
    0x48594c4552494d50ULL, 0x52454c4541534501ULL, 1};
struct RegionBindingReleaseVTable final {
  // Pure validation: no generation changes, new work or owner withdrawal.
  Status (*validate)(void *prepared, void *invocation) noexcept = nullptr;
  // Called only after every affected slot validated and recordings whose
  // carriers may change were destroyed. Retired cleanup must not fail, allocate, submit
  // work or invalidate independently escaped output owners. The next ordinary
  // prepare_bindings reinstalls all inputs of this original slot.
  void (*release)(void *prepared, void *invocation) noexcept = nullptr;
  // This exact carrier uses only borrowed TensorView metadata, never input
  // lease owners, native references or input-derived generated output handles.
  // Release leaves those physical argument identities unchanged. Actual
  // backend capture identity and all fresh permissions are still revalidated.
  bool borrows_tensor_views_only = false;
};
// Positive proof from an exact selected slot, separate from borrowed input
// metadata. Ordinary TensorView rebinding preserves all recorded private
// carriers. Any cold configuration/carrier change must invalidate commands
// before replacing them. This grants neither admission nor page permission.
inline constexpr compute::ProviderExtensionId kRegionCaptureBindingsExtensionId{
    0x48594c4543415054ULL, 0x42494e44494e4701ULL, 1};
struct RegionCaptureBindingsVTable final {
  bool (*preserves_private_arguments)(void *prepared,
                                      void *invocation) noexcept = nullptr;
};
// The actual ready control of an exact selected slot (for example a private
// layout mirror). Pure cold query only: no replacement, work or admission.
// Retaining the returned guard prevents that original control from being
// rewritten until every admitted reader has really retired.
inline constexpr compute::ProviderExtensionId kRegionReadyControlExtensionId{
    0x48594c4552435452ULL, 0x4f4c524541445901ULL, 1};
struct RegionReadyControl final {
  compute::ConstTensorView tensor;
  compute::ReadLease read;
};
struct RegionReadyControlVTable final {
  Result<RegionReadyControl> (*inspect)(const ProviderRegionPrepared &,
      void *invocation, std::uint32_t ordinal) noexcept = nullptr;
};
// Optional cold interface of the exact selected prepared artifact. It inspects the real
// compiled region, including private imports/lowering, rather than Graph math.
inline constexpr compute::ProviderExtensionId kRegionProjectedReadExtensionId{
    0x48594c4550524f4aULL, 0x524541445245474eULL, 3};
struct RegionNodeAccess final {
  compute::detail::OperationPreparedAccess access{};
  // Only a real pending owner/importer can opt in. Ready indexed reads do not
  // by themselves establish source ordering or retain pending page controls.
  bool allows_pending = false;
};
struct RegionProjectedReadVTable final {
  // Exact original slot, borrowed only through this cold inspection. An empty
  // footprint must be true of its actual configured computation, not merely
  // supported by the schema or possible in another layout.
  Status (*inspect)(const ProviderRegionPrepared &, void *invocation,
                    std::uint32_t input_port,
                    compute::detail::OperationPreparedAccess *) noexcept = nullptr;
  // The original logical node may share its selected region with other access
  // nodes. projected_input indexes that region's Graph boundary, not a native
  // Program input or the logical node's input list. kNoInput qualifies an
  // access-chain node that does not itself receive the projected capacity.
  // A null invocation requests immutable artifact facts only; actual empty
  // footprint claims must be checked against the supplied original slot.
  Status (*inspect_node)(const ProviderRegionPrepared &, void *invocation,
                        std::uint32_t original_node_ordinal,
                        compute::ProviderExtensionId capability,
                        RegionNodeAccess *) noexcept = nullptr;
  // Cold idle only, under the original Source admission gate. Configuration
  // borrows this call; any needed facts belong to the original selected slot.
  // false means no configuration is required, true means it was applied.
  Result<bool> (*configure_node)(const ProviderRegionPrepared &, void *invocation,
                                std::uint32_t original_node_ordinal,
                                compute::ProviderExtensionId capability,
                                const void *configuration) noexcept = nullptr;
};
struct ReadProjection final {
  ValueId capacity;
  std::span<const ValueId> aliases;
  std::span<const NodeId> readers;
};
struct RegionAccessView final {
  std::uint32_t step=0;
  std::uint64_t graph_generation=0;
  std::span<const std::uint32_t> node_ordinals;
  const compute::detail::ProviderExtensionRef*owner=nullptr;
  void*prepared=nullptr;
  void*invocation=nullptr;
  ExecutionLaneId lane{};
  // Only visit_bound supplies these original prepared port arrays. Borrowed
  // for this cold callback; no tensor execution or ready permission is implied.
  std::span<const compute::ConstTensorView> inputs{};
  std::span<const compute::TensorView> outputs{};
  // Object-local implementation identity for trusted cold configuration.
  // A source capability/provider family alone never authorizes casting a
  // prepared artifact or configuring a sibling implementation's slot.
  const ProviderRegionPreparedVTable *prepared_vtable = nullptr;
};
using RegionAccessFn=Status(*)(void*,const RegionAccessView&) noexcept;
// Borrowed cold proof from the exact selected prepared implementation. The
// typed module qualifies its mathematical reader and actual physical port;
// core independently verifies every real consumer against this artifact.
// Neither a provider family nor a source capability alone grants page access.
struct ProjectedReader final {
  std::uint32_t step = 0;
  const ProviderRegionPreparedVTable *implementation = nullptr;
  const void *prepared = nullptr;
  std::uint32_t input_port = 0;
  // Trusted cold fact of this exact selected reader, not permission to run:
  // replacing/releasing its page guards alone does not change native command
  // arguments or release private carriers. Actual topology/workspace changes
  // must still invalidate through the original before-command-change gate.
  bool permission_changes_preserve_commands = false;
};
// An exact original external-write node in its selected implementation. A
// region may also contain pure nodes, but no second effect/resource access.
// Typed adapters retain its original writers in that Invocation's real slot;
// one region can have only one nominated binding and retired withdrawal.
struct ProjectedWriter final {
  NodeId node;
  std::uint32_t step=0;
  const ProviderRegionPreparedVTable *implementation=nullptr;
  const void *prepared=nullptr;
};
// Pure query of the exact selected writer slot, after actual retirement. True
// only preserves its recorded native arguments/carriers when page permissions
// are withdrawn. It grants no write permission, readiness or future admission.
// Any subsequent native argument/resource change still invalidates commands
// before that change; device metadata content alone may change in cold prepare.
inline constexpr compute::ProviderExtensionId kRegionProjectedWriteExtensionId{
    0x48594c4552494d50ULL, 0x5752495445520101ULL, 1};
struct RegionProjectedWriteVTable final {
  bool (*preserves_commands)(const ProviderRegionPrepared &,
                             void *invocation) noexcept = nullptr;
};
// Explicit qualification of an original native external-write artifact. This
// is separate from release/capture preservation and never grants page rights.
// A null reader endpoint qualifies writer-only use; otherwise the actual
// native writer must support that same ordered reader Runtime. No work runs.
inline constexpr compute::ProviderExtensionId kRegionOrderedWriteExtensionId{
    0x48594c454f524457ULL, 0x5249544545464601ULL, 1};
struct RegionOrderedWriteVTable final {
  Status (*qualify)(const ProviderRegionPrepared &, void *invocation,
      const compute::detail::CompletionEndpoint *reader) noexcept = nullptr;
};
// A typed module validates the logical access chain. Core independently checks
// these exact selected implementations and their physical prepared input port.
struct ResourceProjection final {
  ValueId capacity, metadata;
  NodeId reader;
  std::span<const NodeId> access_nodes;
  compute::ProviderExtensionId capability;
};
// Actual source views of an original bound Store bundle. The capacity is an
// address carrier, never whole-bank read permission; indices have their real
// bounded immutable ready guard retained by that same bundle.
struct StoreProjectionInputs final {
  compute::ConstTensorView capacity{}, indices{};
};
struct OrderedGraphSubmission final {
  // Optional shared-Queue admission, before any consumer command. A delegated
  // native implementation admits its own exact pending reads and real source
  // ordering in its original guarded submit, without inventing a Queue.
  Status (*admit)(void*,ExecutionLaneId,compute::detail::QueueSequence&,
                  compute::LeaseUseBatch&) noexcept = nullptr;
  // Infallible after cold capacity/lifetime qualification. Maps the actual
  // accepted suffix in the same original public observation slot.
  compute::Completion (*map)(void*,compute::Completion&&) noexcept = nullptr;
  void* owner = nullptr;
  // Only these exact inputs defer their missing ready permission to admit.
  // Every other binding still goes through the ordinary lease checks.
  bool (*pending_input)(void*,ValueId) noexcept = nullptr;
  // Exact hidden read already registered by the original selected effect.
  // Without both this fact and actual ordered admission, submit must reject.
  bool (*pending_effect)(void*,std::uint32_t,mem::ConstBufferView) noexcept = nullptr;
  bool valid() const noexcept {
    const bool mapped = map && pending_input && owner;
    return mapped || (!admit && !map && !pending_input && !pending_effect && !owner);
  }
};
// One optional typed resource owner, destroyed after borrowing native slots.
// This carries layouts/frames/source controls, never another execution slot.
struct ResourcePayloadVTable final {
  // Reject retained source controls before mutation, then revoke preparation
  // before changing native resources, including changes that may partly fail.
  // Invalidate selected commands before releasing their native configuration
  // owners. This grants no new work admission.
  Status (*reset)(void *, Invocation &) noexcept = nullptr;
  void (*destroy)(void *) noexcept = nullptr;
};
// Explicit cold/idle opt-in only. Views borrow the callback scope, never a
// different execution slot. Ordinary compile/submit do not call this visitor.
struct RegionAccess final {
  HYLE_GRAPH_API static bool borrows_tensor_views_only(const PreparedRegion &) noexcept;
  HYLE_GRAPH_API static bool preserves_capture_bindings(
      const RegionInvocation &) noexcept;
  HYLE_GRAPH_API static Status withdraw_bindings(Invocation &,
      std::span<const ValueId>) noexcept;
  // Private cold bridge for the exact selected implementation of an original
  // logical node, whether singleton or region. It never exposes provider state
  // or changes public graph topology.
  HYLE_GRAPH_API static Result<bool> configure_node(
      Invocation &, NodeId,
      compute::ProviderExtensionId, const void *) noexcept;
  // Cold/idle only. Call before mutating a selected slot's native resources,
  // including callbacks that may fail after a partial configuration change.
  // Discards only the covering recording and revokes prepared replay.
  HYLE_GRAPH_API static Status invalidate_commands(
      Invocation &, std::uint32_t selected_step) noexcept;
  // Revoke this episode's readiness without discarding qualified physical
  // recordings. Still call invalidate_commands BEFORE replacing any native
  // command resource or argument. Submit requires complete preparation again.
  HYLE_GRAPH_API static Status revoke_preparation(Invocation &) noexcept;
  // Original idle/result/source-control gates; expires cold configuration and
  // projected permissions while retaining ordinary caller bindings/sources.
  HYLE_GRAPH_API static Status reset_configuration(Invocation &) noexcept;
  // Cold adapter check against the retained target Context, independent of
  // tensor bindings. Replay preparation still owns the complete topology.
  HYLE_GRAPH_API static Status validate_queue(const Executable&,ExecutionLaneId,
      const compute::Queue&) noexcept;
  // Retains the original lane target; it creates no Queue or transfer runtime
  // and grants no work admission after Source close.
  HYLE_GRAPH_API static Result<ExecutionSource> execution_source(
      const Executable &, ExecutionLaneId) noexcept;
  // Actual planned external-input carrier, without choosing an implementation
  // or creating transfer state. Published inputs are not configuration ports.
  HYLE_GRAPH_API static Result<mem::MemoryResourceRef> input_resource(
      const Executable &, ValueId) noexcept;
  // Pure query through the immutable Plan's exact selected region and actual
  // slot, under the caller's ordinary Invocation synchronization. It returns
  // real ready authority; no Graph input, copy or execution is manufactured.
  HYLE_GRAPH_API static Result<RegionReadyControl> ready_control(
      const InvocationState *, NodeId, std::uint32_t ordinal) noexcept;
  // Borrowed inspection does not invalidate recordings. An adapter changing
  // native configuration must first invalidate_commands for the selected step.
  HYLE_GRAPH_API static Status visit(const Executable&,RegionAccessFn,void*) noexcept;
  HYLE_GRAPH_API static Status visit(Invocation&,RegionAccessFn,void*) noexcept;
  // Same gate, indexed through the original Plan's selected logical node.
  HYLE_GRAPH_API static Status visit(Invocation&,NodeId,RegionAccessFn,void*) noexcept;
  // Observation only after actual retirement, also for failed/closed recovery.
  // Does not reopen Source admission or authorize configuration/execution.
  HYLE_GRAPH_API static Status visit_retired(Invocation&,RegionAccessFn,void*) noexcept;
  HYLE_GRAPH_API static Status visit_retired(Invocation&,NodeId,RegionAccessFn,void*) noexcept;
  // Requires complete current bindings and the same original idle/result gate.
  // Uses the invocation's existing preallocated port arrays, without copying.
  HYLE_GRAPH_API static Status visit_bound(Invocation&,RegionAccessFn,void*) noexcept;
  HYLE_GRAPH_API static Result<void *> resource_payload(
      Invocation &, const ResourcePayloadVTable *) noexcept;
  // Takes payload ownership only after successful cold admission.
  HYLE_GRAPH_API static Status attach_resource_payload(
      Invocation &, const ResourcePayloadVTable *, void *) noexcept;
  // Cold opt-in capacity for extra records beyond the current binding count;
  // ordinary borrowed invocations need no lease table. Repeated reservations
  // still account for guards installed since the earlier reservation.
  HYLE_GRAPH_API static Status reserve_reads(Invocation&,std::uint32_t,
      ResourceOptions) noexcept;
  // One cold metadata-transfer Queue per actual retained Context/lane. This
  // never replaces the execution Queue or grants tensor permission. Shared
  // owners keep its public wrapper stable through borrowed transfer slots;
  // reset retains this cache. Native-only lanes have no Compute Queue.
  HYLE_GRAPH_API static Result<std::shared_ptr<compute::Queue>> metadata_queue(
      Invocation &, ExecutionLaneId, ResourceOptions) noexcept;
  // Owns immutable input metadata of the selected cold configuration. Only
  // this idle/result-gated path may replace it until the original reset.
  // Rejection preserves the previous binding and prepared replay.
  HYLE_GRAPH_API static Status bind_configuration(Invocation &, ValueId,
      compute::TensorDesc, compute::ReadLease) noexcept;
  // Trusted adapters supply the exact footprint and qualify every reader.
  // Rejects copies, ordinary operations, aliases and published capacity views.
  HYLE_GRAPH_API static Status bind_projected(Invocation&,ValueId,
      compute::ConstTensorView,std::span<const compute::ReadLease>,
      std::span<const ProjectedReader>) noexcept;
  // No ready permission is manufactured. The trusted adapter owns the typed
  // pending frame and supplies all exact guards in OrderedGraphSubmission.
  HYLE_GRAPH_API static Status bind_pending_projected(Invocation&,ValueId,
      compute::ConstTensorView,std::span<const ProjectedReader>) noexcept;
  // Original pending projected binding checks, without changing any binding.
  HYLE_GRAPH_API static Status validate_pending_projected(Invocation&,ValueId,
      compute::ConstTensorView,std::span<const ProjectedReader>) noexcept;
  // Actual retired outcome only, including failed/closed cleanup. Withdraws
  // this ready projected input, its transparent aliases and page guards;
  // pending/source controls and all other caller/configuration bindings stay.
  // Native input owners use the same qualified reverse release as unbind.
  HYLE_GRAPH_API static Status release_read_projection(Invocation&,ValueId) noexcept;
  // Cold fact query, separate from admission/retirement. False after the
  // original projection was reset/released; a different live scope or ready
  // footprint is an error. Rechecks the exact original guards, including
  // subview bounds. The stable state survives public Invocation moves.
  HYLE_GRAPH_API static Result<StoreProjectionInputs> store_source_projection(
      const InvocationState *, ResourceProjection) noexcept;
  HYLE_GRAPH_API static Result<bool> has_read_projection(const InvocationState*,
      ValueId,std::span<const compute::ReadLease>) noexcept;
  // Cold fact query of the original external input, under the caller's normal
  // Invocation synchronization. Checks the current description and exact
  // storage range, independently of lease authority. False after unbind/reset
  // or replacement; a foreign/non-input ValueId is an error. Does not grant
  // permission or prove submission success/retirement.
  HYLE_GRAPH_API static Result<bool> has_input_binding(const InvocationState*,
      ValueId,compute::ConstTensorView) noexcept;
  // Same-Graph write then read. Checks the original Graph control edges and
  // exact selected artifacts; only the supplied independent page writers are
  // admitted to the original resource episode. No whole-bank write is granted.
  // requires_source also retains a predecessor contract with no physical copy
  // (for example when an append starts exactly at a page boundary).
  HYLE_GRAPH_API static Status bind_written_projection(Invocation&,const Graph&,
      ValueId,compute::ConstTensorView,std::span<const compute::ReadLease>,
      std::span<const compute::WriteLease*const>,
      std::span<const compute::detail::AppendLease*const>,ProjectedWriter,
      std::span<const ProjectedReader>,std::span<const compute::ReadLease> controls,
      ResourceOptions,std::span<const mem::ConstBufferView> pending_reads = {},
      bool requires_source = false) noexcept;
  // Actual retired outcome only, including failure/closed recovery. Revokes
  // the original preparation before an adapter takes back its page owners.
  HYLE_GRAPH_API static Status release_written_projection(Invocation&,ValueId) noexcept;
  // Standalone typed write effect. Capacity is only a range bound, never an
  // external input or whole-bank permission; the exact node owns anonymous
  // ready/prefix, mutable-page and program-control uses in the same episode.
  HYLE_GRAPH_API static Status bind_write_effect(Invocation&,const Graph&,
      compute::ConstTensorView,std::span<const compute::ReadLease>,
      std::span<const compute::WriteLease*const>,
      std::span<const compute::detail::AppendLease*const>,ProjectedWriter,
      std::span<const compute::ReadLease> controls,ResourceOptions,
      std::span<const mem::ConstBufferView> pending_reads = {},bool requires_source = false) noexcept;
  // Pure exact registration query, including an explicitly empty read set.
  // The ranges carry no authority by
  // themselves; the original ordered submission must admit their source.
  HYLE_GRAPH_API static Result<bool> has_pending_effect(const InvocationState*,
      ProjectedWriter,std::span<const mem::ConstBufferView>) noexcept;
  // Cold synchronized query of the original resource episode, never admission
  // or retirement proof. The stable parent state survives moving Invocation.
  // False after reset/release; a foreign artifact or mismatched live authority
  // is an error. Only exact ordinary writer pointers qualify this narrow path.
  HYLE_GRAPH_API static Result<bool> has_write_effect(const InvocationState*,
      ProjectedWriter,std::span<const compute::WriteLease*const>) noexcept;
  // Actual retirement, including failed/closed recovery. Withdraws only this
  // original node's anonymous uses, after revoking its current preparation.
  HYLE_GRAPH_API static Status release_write_effect(Invocation&,NodeId) noexcept;
  // One retired permission withdrawal for original projected writes and
  // writer-only effects. Nodes must be distinct, selected original writers.
  // Validate all before revoking preparation, then compact anonymous guards
  // once. Uses the existing cold ResourceEpisode scratch; no caller cache.
  HYLE_GRAPH_API static Status release_projected_writes(Invocation&,
      std::span<const NodeId>) noexcept;
  HYLE_GRAPH_API static bool successful_outcome(const Invocation&) noexcept;
  HYLE_GRAPH_API static Result<compute::Completion> submit(Invocation&,
      std::span<const QueueBinding>,OrderedGraphSubmission) noexcept;
  // Observation/result gate only; failed or closed episodes may still return
  // their original source controls after actual retirement.
  HYLE_GRAPH_API static bool outcome_available(const Invocation&) noexcept;
  // Typed modules prove the logical access chain. Core checks this same plan,
  // each selected implementation/physical port, transparent aliases and bytes.
  HYLE_GRAPH_API static Status bind_read_projection(Invocation&,ReadProjection,
      compute::ConstTensorView,std::span<const compute::ReadLease>) noexcept;
  // No whole-capacity lease is minted. Only the actual selected prepared
  // implementation may authorize its projected physical input/empty footprint.
  HYLE_GRAPH_API static Result<mem::MemoryResourceRef> projection_resource(
      const Executable&,ResourceProjection,
      ExecutionLaneId* reader_lane=nullptr)noexcept;
  HYLE_GRAPH_API static Status bind_projection(Invocation&,
      ResourceProjection,
      compute::ConstTensorView,std::span<const compute::ReadLease>,bool pending=false)noexcept;
  // The same exact selected-reader and pending-capacity checks as
  // bind_projection, without binding or granting permission. Cold composite
  // adapters then use the original atomic InputAccess bundle commit.
  HYLE_GRAPH_API static Status validate_pending_projection(Invocation&,
      ResourceProjection,compute::ConstTensorView) noexcept;
  HYLE_GRAPH_API static Status validate_ready_projection(Invocation&,
      ResourceProjection,compute::ConstTensorView,std::span<const compute::ReadLease>) noexcept;
private:
  static Status validate_projection_binding(Invocation&,ResourceProjection,
      compute::ConstTensorView,std::span<const compute::ReadLease>,bool) noexcept;
public:
  // Cold successful idle caller output only; empty after readonly publication.
  HYLE_GRAPH_API static Result<compute::WriteLease> take_write(
      Invocation&,ValueId caller_output) noexcept;
private:
  friend class SubmissionSource;
  friend HYLE_GRAPH_API Status release_outputs(Invocation &, std::span<const OutputBinding>) noexcept;
  static Status visit_pinned(InvocationState *,NodeId,RegionAccessFn,void *) noexcept;
  static Status withdraw_bindings_impl(Invocation&,std::span<const ValueId>,
      bool ready_projection, const InputSource *source_owner = nullptr) noexcept;
  static Status validate_projected_writer(InvocationState&,const Graph&,
      ProjectedWriter) noexcept;
  static Status qualify_ordered_writer(InvocationState&,std::uint32_t,
      const compute::detail::CompletionEndpoint*) noexcept;
  static bool writer_preserves_commands(const InvocationState&,std::uint32_t) noexcept;
  static Status release_writer_preparation(Invocation&,std::uint32_t) noexcept;
  static Result<RegionNodeAccess>
  inspect_projection_node(const PlanState &, const InvocationState *, NodeId,
                          compute::ProviderExtensionId) noexcept;
  static Result<RegionNodeAccess>
  projection_access(const PlanState &, const InvocationState *,
                    ResourceProjection) noexcept;
  static Status bind_projected_impl(Invocation&,ValueId,
      compute::ConstTensorView,std::span<const compute::ReadLease>,
      std::span<const ProjectedReader>,bool pending,bool validate_only=false) noexcept;
};
} // namespace hyle::graph::detail
