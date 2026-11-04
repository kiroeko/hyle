#pragma once
#include <utility>
#include "hyle/compute/detail/device_lengths_paged_append_wire.h"
#include "hyle/compute/detail/paged_append_backend.h"
namespace hyle::compute::detail {
inline constexpr BackendExtensionId kDeviceLengthsPagedAppendExtensionId{
    0x48594c4550414745ULL,0x4c454e4754483031ULL,1};
struct DeviceLengthsPagedAppendBindings final {
  mem::ConstBufferView arena,rows,metadata,query_lengths,kv_lengths;
};
struct DeviceLengthsPagedAppendVTable final {
  Status (*prepare)(void*,DeviceLengthsPagedAppendSpec,mem::AllocatorRef,void**)noexcept=nullptr;
  void (*destroy)(void*,mem::AllocatorRef)noexcept=nullptr;
  EnqueueResult (*enqueue)(void*,void*,void*,DeviceLengthsPagedAppendBindings)noexcept=nullptr;
  bool supports_command_capture=false;
};
// Only a native program handle. Original Graph Append owns all page/source
// rights and its ordinary mirror; the admitted Queue owns work and retirement.
class DeviceLengthsPagedAppendProgram final {
public:
  DeviceLengthsPagedAppendProgram()noexcept=default;
  DeviceLengthsPagedAppendProgram(const DeviceLengthsPagedAppendProgram&)=delete;
  DeviceLengthsPagedAppendProgram&operator=(const DeviceLengthsPagedAppendProgram&)=delete;
  DeviceLengthsPagedAppendProgram(DeviceLengthsPagedAppendProgram&&other)noexcept
      :extension_(std::move(other.extension_)),table_(other.table_),program_(std::exchange(other.program_,nullptr)),allocator_(other.allocator_){}
  DeviceLengthsPagedAppendProgram&operator=(DeviceLengthsPagedAppendProgram&&other)noexcept{
    if(this!=&other){reset();extension_=std::move(other.extension_);table_=other.table_;
      program_=std::exchange(other.program_,nullptr);allocator_=other.allocator_;}return *this;
  }
  ~DeviceLengthsPagedAppendProgram(){reset();}
  static Result<DeviceLengthsPagedAppendProgram> prepare(BackendExtensionRef extension,
      DeviceLengthsPagedAppendSpec spec,mem::AllocatorRef allocator)noexcept{
    if(!extension.valid() || extension.id()!=kDeviceLengthsPagedAppendExtensionId || !allocator.valid())
      return std::unexpected(Status::InvalidArgument("device-length append requires its original backend capability"));
    DeviceLengthsPagedAppendProgram result;result.extension_=std::move(extension);result.allocator_=allocator;
    struct Call{DeviceLengthsPagedAppendProgram*p;DeviceLengthsPagedAppendSpec spec;}call{&result,spec};
    HYLE_RETURN_IF_ERROR(BackendExtensionAccess::call(result.extension_,[](void*context,const void*table,void*opaque)noexcept->Status{
      auto&c=*static_cast<Call*>(opaque);const auto*t=static_cast<const DeviceLengthsPagedAppendVTable*>(table);
      if(!t || !t->prepare || !t->destroy || !t->enqueue || !t->supports_command_capture)
        return Status::Unimplemented("backend has no captured device-length append");
      c.p->table_=t;return t->prepare(context,c.spec,c.p->allocator_,&c.p->program_);
    },&call));
    if(!result.program_)return std::unexpected(Status::Internal("device-length append returned no native program"));
    return result;
  }
  bool valid()const noexcept{return program_!=nullptr;}
  EnqueueResult enqueue(QueueSequence&sequence,DeviceLengthsPagedAppendBindings bindings)const noexcept{
    if(!valid())return EnqueueResult::rejected(Status::FailedPrecondition("device-length append is not prepared"));
    struct Call{const DeviceLengthsPagedAppendProgram*p;DeviceLengthsPagedAppendBindings bindings;}call{this,bindings};
    return BackendExtensionAccess::enqueue_admitted(extension_,sequence,[](void*context,const void*,void*queue,void*opaque)noexcept{
      auto&c=*static_cast<Call*>(opaque);return c.p->table_->enqueue(context,queue,c.p->program_,c.bindings);
    },&call);
  }
private:
  void reset()noexcept{if(program_){table_->destroy(program_,allocator_);program_=nullptr;}extension_={};table_=nullptr;}
  BackendExtensionRef extension_;const DeviceLengthsPagedAppendVTable*table_=nullptr;
  void*program_=nullptr;mem::AllocatorRef allocator_{};
};
} // namespace hyle::compute::detail
