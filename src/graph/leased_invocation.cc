#include "hyle/graph/leased_invocation.h"
#include "hyle/graph/detail/region_access.h"

#include <atomic>
#include <algorithm>
#include <array>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"
#include "plan_state.h"
#include "submission_internal.h"

namespace hyle::graph {
namespace detail {
struct LeaseBinding final {
  ValueId value{};
  compute::TensorDesc description{};
  compute::ReadLease read{};
  compute::WriteLease write{};
  bool exclusive = false;
  bool input = false;
};

struct LeasedInvocationState final : util::RefCounted<LeasedInvocationState> {
  explicit LeasedInvocationState(mem::AllocatorRef selected) noexcept
      : allocator(selected), bindings(mem::StdAllocator<LeaseBinding>(selected)),
        permissions(selected),
        retentions(mem::StdAllocator<compute::detail::CompletionRetention>(selected)) {}
  void destroy() noexcept {
    const auto selected = allocator;
    std::destroy_at(this);
    selected.deallocate(this, sizeof(LeasedInvocationState), alignof(LeasedInvocationState));
  }
  mem::AllocatorRef allocator;
  // Native invocation destruction precedes its external payload owners.
  std::vector<LeaseBinding, mem::StdAllocator<LeaseBinding>> bindings;
  LeasePermissionSet permissions;
  Invocation invocation{};
  compute::LeaseUsePool uses{};
  std::vector<compute::detail::CompletionRetention,
              mem::StdAllocator<compute::detail::CompletionRetention>> retentions;
  compute::LeaseUseBatch batch{};
  compute::detail::RetainedCompletion completion{};
  compute::detail::CompletionLifetime completion_retirement{};
  util::SpinLock lock{};
  Status first_error{};
  std::atomic<std::size_t> remaining{0}, results{0};
  std::atomic<bool> busy{false};
  bool accepted = false, closed = false;
};
Status RegionAccess::visit(LeasedInvocation&invocation,RegionAccessFn callback,void*user)noexcept{
  auto*state=invocation.state_;
  if(!state)return Status::FailedPrecondition("leased region invocation is empty");
  if(state->closed || state->busy.load(std::memory_order_acquire) || state->results.load(std::memory_order_acquire) ||
     !state->completion.available())return Status::Unavailable("leased region has closed/busy/old output/completion ownership");
  return visit(state->invocation,callback,user);
}

Status RegionAccess::visit_bound(LeasedInvocation&invocation,RegionAccessFn callback,void*user)noexcept{
  auto*state=invocation.state_;
  if(!state)return Status::FailedPrecondition("leased region invocation is empty");
  if(state->closed || state->busy.load(std::memory_order_acquire) || state->results.load(std::memory_order_acquire) ||
     !state->completion.available())return Status::Unavailable("leased region has closed/busy/old output/completion ownership");
  return visit_bound(state->invocation,callback,user);
}

void release_owner(LeasedInvocationState *state) noexcept {
  bool last;
  {
    std::lock_guard guard(state->lock);
    last = state->is_unique();
    if (!last) state->decref();
  }
  if (last) state->decref();
}

LeaseBinding *find_binding(LeasedInvocationState *state, ValueId value) noexcept {
  for (auto &binding : state->bindings) if (binding.value == value) return &binding;
  return nullptr;
}
Status control_ready(LeasedInvocationState *state) noexcept {
  if (!state) return Status::FailedPrecondition("leased graph invocation is empty");
  if (state->closed) return Status::FailedPrecondition("leased graph invocation is closed");
  if (state->busy.load(std::memory_order_acquire) || state->invocation.in_flight())
    return Status::Unavailable("leased graph invocation is in flight");
  if (state->results.load(std::memory_order_acquire) != 0)
    return Status::Unavailable("leased graph results still retain this invocation");
  if (!state->completion.available())
    return Status::Unavailable("old leased graph completion retains its slot");
  return Status::Ok();
}
Status RegionAccess::reserve_reads(LeasedInvocation&invocation,std::uint32_t extra,
    LeasedInvocationOptions options)noexcept{
  auto*state=invocation.state_;HYLE_RETURN_IF_ERROR(control_ready(state));
  for(const auto&binding:state->bindings)if(binding.read.valid() || binding.write.valid())
    return Status::FailedPrecondition("projected capacity must be reserved before bindings");
  if(extra>UINT32_MAX-1 || state->bindings.size()>UINT32_MAX-1-extra)
    return Status::ResourceExhausted("projected graph permission capacity overflow");
  HYLE_ASSIGN_OR_RETURN(auto uses,compute::LeaseUsePool::create({.max_batches=1,
    .max_resources_per_batch=std::uint32_t(state->bindings.size()+extra+1),
    .allocator=options.allocator,.resource_budget=options.resource_budget}));
  try{state->bindings.reserve(state->bindings.size()+extra);}
  catch(...){return Status::ResourceExhausted("projected graph fixed permission metadata");}
  if(extra)HYLE_RETURN_IF_ERROR(state->permissions.reserve(state->bindings.size()+extra));
  state->uses=std::move(uses);return Status::Ok();
}
Status bind_projected(LeasedInvocationState*state,ValueId value,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease> pages,
    compute::ProviderId provider,std::span<const std::uint32_t> readers,
    bool pending)noexcept{
  HYLE_RETURN_IF_ERROR(control_ready(state));
  auto*binding=find_binding(state,value);
  if(!binding || !binding->input || !tensor.valid() || (!pending && pages.empty()) || readers.empty())
    return Status::InvalidArgument("projected graph requires an input and actual page permissions");
  if(binding->read.valid() || binding->write.valid())
    return Status::FailedPrecondition("projected graph input needs an explicit reset before rebinding");
  const auto&plan=*InvocationAccess::state(state->invocation)->plan;
  const auto logical=value.ordinal();const auto physical=plan.external_instances[logical];
  if((plan.binding_roles[logical]&kExternalOutput)!=0)
    return Status::InvalidArgument("projected capacity cannot be a graph output");
  for(std::size_t i=0;i<plan.alias_sources.size();++i){
    // Every descendant chain must contain a direct child of this root.
    if(i!=physical && plan.alias_sources[i]==physical)
      return Status::InvalidArgument("projected capacity cannot have a physical alias descendant");
  }
  bool consumed=false;
  for(std::size_t i=0;i<plan.steps.size();++i){const auto&step=plan.steps[i];
    for(std::size_t port=0;port<step.input_count;++port){
      const auto input=plan.node_inputs[step.input_begin+port];
      if(plan.instance_values[input]!=logical)continue;
      consumed=true;
      if(input!=physical || step.kind!=PlanStepKind::kRegion ||
         step.region.provider().id!=provider || std::find(readers.begin(),readers.end(),i)==readers.end())
        return Status::InvalidArgument("projected capacity has an unqualified physical reader");
    }
  }
  if(!consumed)return Status::InvalidArgument("projected capacity has no qualified reader");
  if(!pages.empty() && pages.size()-1>state->bindings.capacity()-state->bindings.size())
    return Status::ResourceExhausted("projected graph permission capacity exhausted");
  const auto storage=tensor.storage();
  for(const auto&page:pages){HYLE_ASSIGN_OR_RETURN(auto view,page.view());
    if(view.resource()!=storage.resource() || view.native_handle()!=storage.native_handle() ||
       view.offset()<storage.offset() || view.offset()-storage.offset()>storage.size() ||
       view.size()>storage.size()-(view.offset()-storage.offset()))
      return Status::InvalidArgument("projected page permission is outside its native capacity");
  }
  HYLE_RETURN_IF_ERROR(state->invocation.bind(value,tensor));
  if(!pages.empty())binding->read=pages.front();
  binding->description=tensor.description();binding->exclusive=false;
  for(std::size_t i=1;i<pages.size();++i)state->bindings.push_back({.read=pages[i]});
  return Status::Ok();
}
Status RegionAccess::bind_projected(LeasedInvocation&invocation,ValueId value,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease> pages,
    compute::ProviderId provider,std::span<const std::uint32_t> readers)noexcept{
  return detail::bind_projected(invocation.state_,value,tensor,pages,provider,readers,false);
}
Status RegionAccess::bind_pending_projected(LeasedInvocation&invocation,ValueId value,
    compute::ConstTensorView tensor,compute::ProviderId provider,
    std::span<const std::uint32_t> readers)noexcept{
  return detail::bind_projected(invocation.state_,value,tensor,{},provider,readers,true);
}
bool RegionAccess::outcome_available(const LeasedInvocation&invocation)noexcept{
  const auto*state=invocation.state_;
  return state && !state->busy.load(std::memory_order_acquire) &&
      !state->invocation.in_flight() && !state->results.load(std::memory_order_acquire) &&
      state->completion.available();
}
Status RegionAccess::bind_gather(LeasedInvocation&invocation,const Graph&graph,
    ValueId capacity,ValueId indices,compute::ConstTensorView tensor,
    std::span<const compute::ReadLease> pages)noexcept{
  auto*state=invocation.state_;HYLE_RETURN_IF_ERROR(control_ready(state));
  auto*binding=find_binding(state,capacity);const auto*source=GraphAccess::state(graph);
  const auto&plan=*InvocationAccess::state(state->invocation)->plan;
  if(!source || source->generation!=plan.generation || capacity==indices ||
     GraphIdentity::generation(capacity)!=source->generation || GraphIdentity::generation(indices)!=source->generation ||
     capacity.kind()!=ValueKind::kInput || indices.kind()!=ValueKind::kInput ||
     !binding || !binding->input || !tensor.valid() || !tensor.description().is_dense() || pages.empty())
    return Status::InvalidArgument("gather projection requires the exact graph/input/footprint");
  if(binding->read.valid() || binding->write.valid())
    return Status::FailedPrecondition("gather projection needs reset before rebinding");
  if(std::find(source->inputs.begin(),source->inputs.end(),indices.ordinal())==source->inputs.end())
    return Status::InvalidArgument("gather indices are not an input of this graph");
  NodeId reshape{},gather{};ValueId alias{};
  for(std::size_t i=0;i<source->nodes.size();++i){const auto node=GraphIdentity::node(source->generation,std::uint32_t(i));
    HYLE_ASSIGN_OR_RETURN(auto count,graph.node_input_count(node));
    for(std::size_t port=0;port<count;++port){HYLE_ASSIGN_OR_RETURN(auto input,graph.node_input(node,port));if(input!=capacity)continue;
      HYLE_ASSIGN_OR_RETURN(auto operation,graph.operation(node));const auto*view=operation.get_if<compute::View>();
      if(reshape.valid() || !view || view->kind()!=compute::ViewKind::kReshape || count!=1)
        return Status::InvalidArgument("capacity has a reader outside its unique reshape");
      HYLE_ASSIGN_OR_RETURN(auto outputs,graph.node_output_count(node));if(outputs!=1)return Status::InvalidArgument("capacity reshape arity");
      reshape=node;HYLE_ASSIGN_OR_RETURN(alias,graph.result(node));
    }
  }
  if(!reshape.valid())return Status::InvalidArgument("capacity has no unique reshape");
  for(std::size_t i=0;i<source->nodes.size();++i){const auto node=GraphIdentity::node(source->generation,std::uint32_t(i));
    HYLE_ASSIGN_OR_RETURN(auto count,graph.node_input_count(node));
    for(std::size_t port=0;port<count;++port){HYLE_ASSIGN_OR_RETURN(auto input,graph.node_input(node,port));if(input!=alias)continue;
      HYLE_ASSIGN_OR_RETURN(auto operation,graph.operation(node));const auto*op=operation.get_if<compute::Gather>();
      if(gather.valid() || !op || op->axis!=0 || op->bounds!=compute::IndexBoundsPolicy::kChecked || count!=2 || port!=0)
        return Status::InvalidArgument("capacity alias has a reader outside its unique checked Gather");
      HYLE_ASSIGN_OR_RETURN(auto supplied_indices,graph.node_input(node,1));if(supplied_indices!=indices)
        return Status::InvalidArgument("capacity Gather uses foreign metadata");
      gather=node;
    }
  }
  if(!gather.valid() || std::find(source->outputs.begin(),source->outputs.end(),capacity.ordinal())!=source->outputs.end() ||
     std::find(source->outputs.begin(),source->outputs.end(),alias.ordinal())!=source->outputs.end())
    return Status::InvalidArgument("capacity/reshape cannot be published");
  const auto root=plan.external_instances[capacity.ordinal()];bool consumed=false;
  for(const auto&step:plan.steps){const auto nodes=std::span<const std::uint32_t>(plan.step_nodes).subspan(step.node_set_begin,step.node_count);
    for(std::size_t port=0;port<step.input_count;++port){auto input=plan.node_inputs[step.input_begin+port];
      auto origin=input;while(plan.alias_sources[origin]!=kNoInPlaceSource)origin=plan.alias_sources[origin];if(origin!=root)continue;
      const auto logical=plan.instance_values[input];
      if(logical!=capacity.ordinal() && logical!=alias.ordinal())return Status::InvalidArgument("capacity has an extra physical alias reader");
      const auto has=[&](NodeId node){return std::find(nodes.begin(),nodes.end(),node.ordinal())!=nodes.end();};
      if(step.kind==PlanStepKind::kView && !step.materializes && has(reshape))continue;
      if(!has(gather) || (step.kind!=PlanStepKind::kRegion &&
         !(step.kind==PlanStepKind::kOperation && step.operation.kind()==compute::OperationKind::kGather)))
        return Status::InvalidArgument("capacity physical reader materializes/copies outside Gather");
      consumed=true;
    }
  }
  if(!consumed)return Status::InvalidArgument("capacity has no physical Gather reader");
  if(pages.size()-1>state->bindings.capacity()-state->bindings.size())
    return Status::ResourceExhausted("gather projection guard capacity exhausted");
  const auto storage=tensor.storage();for(const auto&page:pages){HYLE_ASSIGN_OR_RETURN(auto view,page.view());
    if(view.resource()!=storage.resource() || view.native_handle()!=storage.native_handle() ||
       view.offset()<storage.offset() || view.offset()-storage.offset()>storage.size() ||
       view.size()>storage.size()-(view.offset()-storage.offset()))return Status::InvalidArgument("gather page guard outside native capacity");
  }
  HYLE_RETURN_IF_ERROR(state->invocation.bind(capacity,tensor));
  binding->read=pages.front();binding->description=tensor.description();binding->exclusive=false;
  for(std::size_t i=1;i<pages.size();++i)state->bindings.push_back({.read=pages[i]});return Status::Ok();
}
Result<compute::WriteLease> RegionAccess::take_write(LeasedInvocation&invocation,ValueId value)noexcept{
  auto*state=invocation.state_;HYLE_RETURN_IF_ERROR(control_ready(state));
  if(InvocationAccess::state(state->invocation)->phase.load(std::memory_order_acquire)!=InvocationPhase::kIdle)
    return std::unexpected(Status::FailedPrecondition("failed graph output cannot be reclaimed"));
  auto*binding=find_binding(state,value);if(!binding || binding->input)
    return std::unexpected(Status::InvalidArgument("only caller output writer can be reclaimed"));
  return std::move(binding->write);
}
void record_error(LeasedInvocationState *state, const Status &error) noexcept {
  if (!error.ok()) {
    std::lock_guard guard(state->lock);
    state->first_error.Update(error);
  }
}
void child_terminal(void *opaque, const Status &terminal) noexcept {
  record_error(static_cast<LeasedInvocationState *>(opaque), terminal);
}
void release_episode(void *opaque) noexcept {
  auto *state = static_cast<LeasedInvocationState *>(opaque);
  if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  Status terminal;
  {
    std::lock_guard guard(state->lock);
    terminal = state->first_error;
  }
  if (state->accepted) {
    InvocationAccess::retire(state->invocation, terminal);
    auto retired = compute::Completion::completed(terminal);
    HYLE_FATAL_ASSERT(state->batch.attach(retired).ok());
  } else {
    state->batch = {}; // rejection before native acceptance does not poison
  }
  const auto observation = std::exchange(state->completion_retirement, {});
  observation.terminal(observation.state, terminal);
  observation.release(observation.state);
  bool last;
  {
    // Include callback cleanup in drain's lock barrier. Its caller still has
    // a root reference; only an abandoned adapter can take the last branch.
    std::lock_guard guard(state->lock);
    last = state->is_unique();
    if (!last) state->decref();
    state->busy.store(false, std::memory_order_release);
    state->busy.notify_all();
  }
  if (last) state->decref();
}
void retain_accepted(void *opaque, std::size_t segment,
                     compute::Completion &completion) noexcept {
  auto *state = static_cast<LeasedInvocationState *>(opaque);
  auto &node = state->retentions[segment];
  HYLE_FATAL_ASSERT(!node.own.valid());
  state->accepted = true;
  state->remaining.fetch_add(1, std::memory_order_relaxed);
  node.own = {state, &child_terminal, &release_episode};
  HYLE_FATAL_ASSERT(compute::detail::CompletionAccess::attach_retention(completion, node).ok());
}

Result<compute::ReadLease> output_storage(LeasedInvocationState *state,
    compute::ConstTensorView tensor) noexcept {
  const auto wanted = tensor.storage();
  HYLE_ASSIGN_OR_RETURN(const auto bytes, tensor.description().required_bytes());
  for (auto &binding : state->bindings) {
    if (!binding.read.valid()) continue;
    HYLE_ASSIGN_OR_RETURN(auto source, binding.read.view());
    if (source.resource() != wanted.resource() || source.native_handle() != wanted.native_handle() ||
        wanted.offset() < source.offset()) continue;
    const auto offset = wanted.offset() - source.offset();
    if (offset <= source.size() && bytes <= source.size() - offset)
      return binding.read.subview(offset, bytes);
  }
  return std::unexpected(Status::FailedPrecondition("output uses original private Graph storage"));
}
} // namespace detail

LeasedGraphOutput::LeasedGraphOutput(detail::LeasedInvocationState *state,
    compute::ConstTensorView tensor) noexcept : state_(state), tensor_(tensor) {
  state_->incref();
  state_->results.fetch_add(1, std::memory_order_relaxed);
}
LeasedGraphOutput::LeasedGraphOutput(const LeasedGraphOutput &other) noexcept
    : state_(other.state_), tensor_(other.tensor_) {
  if (state_) {
    state_->incref(); state_->results.fetch_add(1, std::memory_order_relaxed);
  }
}
LeasedGraphOutput &LeasedGraphOutput::operator=(const LeasedGraphOutput &other) noexcept {
  if (this != &other) { LeasedGraphOutput copy(other); *this = std::move(copy); }
  return *this;
}
LeasedGraphOutput::LeasedGraphOutput(LeasedGraphOutput &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)), tensor_(other.tensor_) {}
LeasedGraphOutput &LeasedGraphOutput::operator=(LeasedGraphOutput &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); tensor_ = other.tensor_; }
  return *this;
}
LeasedGraphOutput::~LeasedGraphOutput() { reset(); }
void LeasedGraphOutput::reset() noexcept {
  if (auto *state = std::exchange(state_, nullptr)) {
    state->results.fetch_sub(1, std::memory_order_release);
    detail::release_owner(state);
  }
  tensor_ = {};
}
Result<compute::ConstTensorView> LeasedGraphOutput::tensor() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased graph output is empty"));
  return tensor_;
}
Result<compute::ReadLease> LeasedGraphOutput::storage_lease() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased graph output is empty"));
  return detail::output_storage(state_, tensor_);
}

LeasedInvocation::LeasedInvocation(LeasedInvocation &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
LeasedInvocation &LeasedInvocation::operator=(LeasedInvocation &&other) noexcept {
  if (this != &other) {
    if (state_) { close().IgnoreError(); state_->completion.reset(); detail::release_owner(state_); }
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
LeasedInvocation::~LeasedInvocation() {
  if (state_) { close().IgnoreError(); state_->completion.reset(); detail::release_owner(state_); }
}
Result<LeasedInvocation> LeasedInvocation::create(const Executable &executable,
    LeasedInvocationOptions options, std::span<const ShapeBinding> shape) noexcept {
  if (!executable.valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid leased graph/allocator"));
  HYLE_ASSIGN_OR_RETURN(auto invocation, executable.create_invocation(shape));
  const auto actual = invocation.executable();
  const auto inputs = actual.input_count(), outputs = actual.output_count();
  if (inputs > UINT32_MAX - 1 || outputs > UINT32_MAX - 1 - inputs)
    return std::unexpected(Status::ResourceExhausted("leased graph binding capacity"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto uses, compute::LeaseUsePool::create({.max_batches = 1,
      .max_resources_per_batch = static_cast<std::uint32_t>(inputs + outputs + 1),
      .allocator = options.allocator, .resource_budget = options.resource_budget}));
  auto *storage = allocator.allocate(sizeof(detail::LeasedInvocationState), alignof(detail::LeasedInvocationState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("leased graph state metadata"));
  auto *state = std::construct_at(static_cast<detail::LeasedInvocationState *>(storage), allocator);
  LeasedInvocation result(state);
  HYLE_ASSIGN_OR_RETURN(state->completion, compute::detail::RetainedCompletion::create(allocator));
  state->invocation = std::move(invocation); state->uses = std::move(uses);
  try {
    state->bindings.resize(inputs);
    state->bindings.reserve(inputs + outputs);
    state->retentions.resize(actual.segment_count() + 1);
    for (std::size_t index = 0; index < inputs; ++index) {
      HYLE_ASSIGN_OR_RETURN(state->bindings[index].value, actual.input_id(index));
      state->bindings[index].input = true;
    }
    for (std::size_t index = 0; index < outputs; ++index) {
      HYLE_ASSIGN_OR_RETURN(auto ownership, actual.output_ownership(index));
      if (ownership != OutputOwnership::kCaller) continue;
      HYLE_ASSIGN_OR_RETURN(auto value, actual.output_id(index));
      if (!detail::find_binding(state, value)) state->bindings.push_back({.value = value});
    }
  } catch (...) {
    return std::unexpected(Status::ResourceExhausted("leased graph fixed metadata"));
  }
  return result;
}
bool LeasedInvocation::in_flight() const noexcept {
  return state_ && (state_->busy.load(std::memory_order_acquire) || state_->invocation.in_flight());
}
Status LeasedInvocation::bind(ValueId value, compute::TensorDesc description,
                             compute::ReadLease lease) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  auto *binding = detail::find_binding(state_, value);
  if (!binding || !binding->input) return Status::InvalidArgument("read lease must bind a graph input");
  HYLE_ASSIGN_OR_RETURN(auto tensor, lease.tensor(description));
  HYLE_RETURN_IF_ERROR(state_->invocation.bind(value, tensor));
  binding->write.reset(); binding->read = std::move(lease);
  binding->description = description; binding->exclusive = false;
  return Status::Ok();
}
Status LeasedInvocation::bind(ValueId value, compute::TensorDesc description,
                             compute::WriteLease &&lease) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  auto *binding = detail::find_binding(state_, value);
  if (!binding) return Status::InvalidArgument("write lease must bind an external graph value");
  HYLE_ASSIGN_OR_RETURN(auto tensor, lease.tensor(description));
  HYLE_RETURN_IF_ERROR(state_->invocation.bind(value, tensor));
  binding->read.reset(); binding->write = std::move(lease);
  binding->description = description; binding->exclusive = true;
  return Status::Ok();
}
Status LeasedInvocation::prepare_replay(std::span<const QueueBinding> queues,
                                      ReplayOptions options) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  return state_->invocation.prepare_replay(queues, options);
}
ReplayRoute LeasedInvocation::replay_route() const noexcept {
  return state_ ? state_->invocation.replay_route() : ReplayRoute::kUnprepared;
}
Result<ReplayReport> LeasedInvocation::replay_report() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased graph is empty"));
  return state_->invocation.replay_report();
}
Status LeasedInvocation::reset() noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  HYLE_RETURN_IF_ERROR(state_->invocation.reset());
  std::erase_if(state_->bindings,[](const auto&binding){return !binding.value.valid();});
  for (auto &binding : state_->bindings) {
    binding.read.reset(); binding.write.reset(); binding.exclusive = false;
  }
  return Status::Ok();
}
Result<compute::Completion> LeasedInvocation::submit() noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  if (state_->invocation.replay_route() == ReplayRoute::kUnprepared)
    HYLE_RETURN_IF_ERROR(state_->invocation.prepare_replay({}));
  return submit(std::span<const QueueBinding>{});
}
Result<compute::Completion> LeasedInvocation::submit(compute::Queue &queue) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  const auto executable = state_->invocation.executable();
  if (executable.required_queue_count() != 1)
    return std::unexpected(Status::InvalidArgument("leased graph needs its full queue bindings"));
  for (std::size_t index = 0; index < executable.lane_count(); ++index) {
    const auto lane = ExecutionLaneId{static_cast<std::uint32_t>(index)};
    HYLE_ASSIGN_OR_RETURN(auto required, executable.requires_queue(lane));
    if (!required) continue;
    const std::array bindings{QueueBinding{lane, &queue}};
    if (state_->invocation.replay_route() == ReplayRoute::kUnprepared)
      HYLE_RETURN_IF_ERROR(state_->invocation.prepare_replay(bindings));
    return submit(bindings);
  }
  return std::unexpected(Status::Internal("leased graph active queue is missing"));
}
Result<compute::Completion> LeasedInvocation::submit(std::span<const QueueBinding> queues) noexcept {
  return detail::RegionAccess::submit(*this,queues,{});
}
Result<compute::Completion> detail::RegionAccess::submit(LeasedInvocation&invocation,
    std::span<const QueueBinding> queues,OrderedGraphSubmission ordered)noexcept{
  auto*state_=invocation.state_;
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  if(bool(ordered.admit)!=bool(ordered.map) || bool(ordered.admit)!=bool(ordered.pending_input) ||
      (ordered.admit && !ordered.owner))
    return std::unexpected(Status::InvalidArgument("ordered Graph submission requires both admission and mapping"));
  for (auto &binding : state_->bindings) {
    if (binding.exclusive && binding.read.valid()) {
      HYLE_ASSIGN_OR_RETURN(binding.write, binding.read.try_write());
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto batch, state_->uses.begin());
  auto &permissions = state_->permissions;
  permissions.reset();
  for (const auto &binding : state_->bindings) {
    if(!binding.exclusive && !binding.read.valid() && binding.input &&
        ordered.pending_input && ordered.pending_input(ordered.owner,binding.value))continue;
    if (permissions.enabled()) {
      HYLE_RETURN_IF_ERROR(binding.exclusive ? permissions.add(binding.write) : permissions.add(binding.read));
    } else {
      HYLE_RETURN_IF_ERROR(binding.exclusive ? batch.write(binding.write) : batch.read(binding.read));
    }
  }
  if (permissions.enabled()) HYLE_RETURN_IF_ERROR(permissions.admit(batch));
  HYLE_ASSIGN_OR_RETURN(auto cursor, detail::SubmissionCursor::begin(state_->invocation, queues));
  state_->completion_retirement = state_->completion.reserve_retirement();
  {
    std::lock_guard guard(state_->lock);
    state_->first_error = Status::Ok(); state_->accepted = false;
    state_->batch = std::move(batch);
    state_->remaining.store(1, std::memory_order_relaxed);
    state_->busy.store(true, std::memory_order_release);
    state_->incref();
  }
  struct Admission final {LeasedInvocationState* state;OrderedGraphSubmission ordered;};
  Admission admission{state_,ordered};
  const SharedSequenceFn before=ordered.admit?+[](void*opaque,ExecutionLaneId lane,
      compute::detail::QueueSequence&sequence)noexcept->Status{
    auto&call=*static_cast<Admission*>(opaque);
    return call.ordered.admit(call.ordered.owner,lane,sequence,call.state->batch);
  }:nullptr;
  while (!cursor.done()) {
    auto advanced = cursor.advance(true, &detail::retain_accepted, state_,before,&admission);
    if (!advanced) cursor.reject_successors(std::move(advanced).error());
  }
  detail::record_error(state_, cursor.submission_error());
  auto completion = cursor.finish(true);
  if (completion) {
    if(ordered.map)*completion=ordered.map(ordered.owner,std::move(*completion));
    detail::retain_accepted(state_, state_->retentions.size() - 1, *completion);
  } else {
    detail::record_error(state_, completion.error());
  }
  detail::release_episode(state_); // admission sentinel, never GPU preemption
  if (!completion) return std::unexpected(std::move(completion).error());
  return state_->completion.publish(std::move(*completion), true);
}
Result<LeasedGraphOutput> LeasedInvocation::output(ValueId value) noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased graph is empty"));
  if (in_flight()) return std::unexpected(Status::Unavailable("leased graph output is not retired"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, state_->invocation.output(value));
  for (auto &binding : state_->bindings) {
    if (binding.write.valid()) {
      HYLE_ASSIGN_OR_RETURN(binding.read, binding.write.publish());
    }
  }
  return LeasedGraphOutput(state_, tensor);
}
Status LeasedInvocation::close() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  return in_flight() ? Status::Unavailable("leased graph still has native work") : Status::Ok();
}
Status LeasedInvocation::drain() noexcept {
  if (!state_) return Status::Ok();
  close().IgnoreError();
  const auto observed = state_->completion.wait();
  // An observed error can already have handed a prefix to its native reaper.
  // Drain still waits the actual-retirement latch, preserving that error;
  // observing the public terminal is not proof that all children retired.
  while (state_->busy.load(std::memory_order_acquire)) state_->busy.wait(true, std::memory_order_acquire);
  std::lock_guard guard(state_->lock);
  state_->first_error.Update(observed);
  return state_->first_error;
}
} // namespace hyle::graph
