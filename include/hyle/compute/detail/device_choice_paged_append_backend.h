#pragma once
#include "hyle/compute/detail/paged_append_backend.h"

namespace hyle::compute::detail {
// Separate capability and call ABI: the ordinary paged-append ABI is unchanged.
inline constexpr BackendExtensionId kDeviceChoicePagedAppendExtensionId{
    0x48594c4550414745ULL,0x43484f4943453031ULL,1};
struct DeviceChoicePagedAppendNativeBindings final {
  mem::ConstBufferView arena,rows,metadata,selector;
};
// Uses the existing mirror/wire allocation. Instruction zero is a validated
// directory: source_kind=count0, source_page=count1, source_row=1+count0.
// Remaining records are the two original instruction streams in that order.
// Only this independent choice ABI also accepts source_kind=2: a validated,
// disjoint raw-envelope suffix zero fill. The ordinary append ABI stays 0/1.
struct DeviceChoicePagedAppendBackendVTable final {
  Status (*prepare)(void*,PagedAppendNativeSpec,mem::AllocatorRef,void**)noexcept=nullptr;
  void (*destroy)(void*,mem::AllocatorRef)noexcept=nullptr;
  EnqueueResult (*enqueue)(void*,void*,void*,DeviceChoicePagedAppendNativeBindings)noexcept=nullptr;
  bool supports_command_capture=false;
};
class DeviceChoicePagedAppendProgram final {
public:
  DeviceChoicePagedAppendProgram()noexcept=default;
  DeviceChoicePagedAppendProgram(const DeviceChoicePagedAppendProgram&)=delete;
  DeviceChoicePagedAppendProgram&operator=(const DeviceChoicePagedAppendProgram&)=delete;
  DeviceChoicePagedAppendProgram(DeviceChoicePagedAppendProgram&&other)noexcept
      :extension_(std::move(other.extension_)),table_(other.table_),program_(std::exchange(other.program_,nullptr)),allocator_(other.allocator_){}
  DeviceChoicePagedAppendProgram&operator=(DeviceChoicePagedAppendProgram&&other)noexcept{
    if(this!=&other){reset();extension_=std::move(other.extension_);table_=other.table_;
      program_=std::exchange(other.program_,nullptr);allocator_=other.allocator_;}return *this;
  }
  ~DeviceChoicePagedAppendProgram(){reset();}
  static Result<DeviceChoicePagedAppendProgram> prepare(BackendExtensionRef extension,
      PagedAppendNativeSpec spec,mem::AllocatorRef allocator)noexcept{
    if(!extension.valid() || extension.id()!=kDeviceChoicePagedAppendExtensionId || !allocator.valid())
      return std::unexpected(Status::InvalidArgument("device-choice append requires its own backend capability"));
    DeviceChoicePagedAppendProgram result;result.extension_=std::move(extension);result.allocator_=allocator;
    struct Call{DeviceChoicePagedAppendProgram*p;PagedAppendNativeSpec spec;}call{&result,spec};
    HYLE_RETURN_IF_ERROR(BackendExtensionAccess::call(result.extension_,[](void*context,const void*opaque,void*user)noexcept->Status{
      auto&call=*static_cast<Call*>(user);const auto*t=static_cast<const DeviceChoicePagedAppendBackendVTable*>(opaque);
      if(!t || !t->prepare || !t->destroy || !t->enqueue || !t->supports_command_capture)
        return Status::Unimplemented("backend has no capturable device-choice append");
      call.p->table_=t;return t->prepare(context,call.spec,call.p->allocator_,&call.p->program_);
    },&call));
    if(!result.program_)return std::unexpected(Status::Internal("device-choice append returned no program"));
    return result;
  }
  bool valid()const noexcept{return program_!=nullptr;}
  EnqueueResult enqueue(QueueSequence&sequence,DeviceChoicePagedAppendNativeBindings bindings)const noexcept{
    if(!valid())return EnqueueResult::rejected(Status::FailedPrecondition("device-choice append is unprepared"));
    struct Call{const DeviceChoicePagedAppendProgram*p;DeviceChoicePagedAppendNativeBindings bindings;}call{this,bindings};
    return BackendExtensionAccess::enqueue_admitted(extension_,sequence,[](void*context,const void*,void*queue,void*user)noexcept{
      auto&call=*static_cast<Call*>(user);return call.p->table_->enqueue(context,queue,call.p->program_,call.bindings);
    },&call);
  }
private:
  void reset()noexcept{if(program_){table_->destroy(program_,allocator_);program_=nullptr;}extension_={};table_=nullptr;}
  BackendExtensionRef extension_;const DeviceChoicePagedAppendBackendVTable*table_=nullptr;
  void*program_=nullptr;mem::AllocatorRef allocator_{};
};
} // namespace hyle::compute::detail
