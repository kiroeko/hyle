#include <gtest/gtest.h>
#include "../../src/compute_route_apple/apple_device_append_internal.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/model_ops.h"

#include <mlx/ops.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/event.h>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace c=hyle::compute;namespace a=c::apple;namespace d=a::detail;namespace m=hyle::mem;
template<class T>T take(hyle::Result<T> value){if(!value)throw std::runtime_error(std::string(value.error().message()));return std::move(*value);}
void check(hyle::Status value){if(!value.ok())throw std::runtime_error(std::string(value.message()));}
constexpr std::uint32_t page_rows=5,pages=8,row_words=3,input_rows=7;
constexpr std::size_t row_bytes=row_words*2,bank_bytes=pages*page_rows*row_bytes;
struct Cell final {
  a::Context context=take(a::Context::create());
  a::MemoryResource memory=take(a::MemoryResource::create(context));
  m::Buffer bank=take(m::Buffer::create(memory.ref(),bank_bytes+4));
  m::Buffer rows=take(m::Buffer::create(memory.ref(),input_rows*row_bytes+4));
  m::Buffer queries=take(m::Buffer::create(memory.ref(),16)),lengths=take(m::Buffer::create(memory.ref(),16));
  m::BufferView capacity=take(bank.view().subview(2,bank_bytes));
  c::TensorDesc row_desc=take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{input_rows,row_words}));
  c::TensorDesc control_desc=take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{2}));
  const std::array<c::detail::DeviceAppendRequest,2> requests{{{3,1,4,7},{4,2,3,6}}};
  std::array<d::DeviceAppendPageBinding,4> bindings;
  std::vector<std::uint16_t> original=std::vector<std::uint16_t>(bank_bytes/2+2,0xdead);
  std::array<std::uint16_t,input_rows*row_words> payload;
  explicit Cell(c::DataType type=c::DataType::kBFloat16){
    row_desc=take(c::TensorDesc::dense(type,std::array<std::int64_t,2>{input_rows,row_words}));
    for(std::uint32_t page=0;page<4;++page)for(std::uint32_t row=0;row<page_rows;++row)
      for(std::uint32_t j=0;j<row_words;++j)original[1+(page*page_rows+row)*row_words+j]=std::uint16_t(0x3f80+page*32+row*row_words+j);
    for(std::size_t i=0;i<payload.size();++i)payload[i]=std::uint16_t(0x4000+i);
    check(bank.write(0,std::as_bytes(std::span(original))));check(rows.fill(0,rows.size(),std::byte{0xee}));
    check(rows.write(2,std::as_bytes(std::span(payload))));
    const std::array<c::detail::DeviceAppendPage,4> descriptions{{{0,0,4,5,0,4},{0,1,5,2,UINT32_MAX,0},{1,0,6,5,1,3},{1,1,7,1,UINT32_MAX,0}}};
    for(std::size_t i=0;i<4;++i){
      const auto p=descriptions[i];bindings[i].page=p;
      bindings[i].destination=take(capacity.subview(std::size_t(p.destination_page)*page_rows*row_bytes,p.initialized_rows*row_bytes));
      if(p.source_rows)bindings[i].prefix=take(capacity.subview(std::size_t(p.source_page)*page_rows*row_bytes,p.source_rows*row_bytes));
    }
  }
  ~Cell(){context.close().IgnoreError();}
  std::shared_ptr<d::CopyRecording> slot(){
    auto result=take(d::prepare_copy_recording_runtime(d::ContextAccess::runtime(context),16,m::system_allocator(),bank_bytes,d::CopyRecordingMode::kReplayable));
    check(d::prepare_device_append_recording(*result,{{page_rows,pages,input_rows,16,row_bytes},2},row_desc));
    configure(*result);return result;
  }
  void configure(d::CopyRecording &slot){check(d::configure_device_append_recording(slot,capacity,requests,bindings));}
  mlx::core::array array(m::ConstBufferView view,const c::TensorDesc &desc){return take(d::MemoryAccess::borrow_tensor(take(c::ConstTensorView::create(desc,view))));}
  std::array<d::InputBinding,3> inputs(std::array<std::int32_t,2> q,std::array<std::int32_t,2> k){
    check(queries.write(4,std::as_bytes(std::span(q))));check(lengths.write(4,std::as_bytes(std::span(k))));
    std::array<d::InputBinding,3> result;
    result[0].value=array(take(rows.view().subview(2,payload.size()*2)),row_desc);
    result[1].value=array(take(queries.view().subview(4,8)),control_desc);
    result[2].value=array(take(lengths.view().subview(4,8)),control_desc);return result;
  }
  void bind(d::CopyRecording &slot,std::array<std::int32_t,2> q,std::array<std::int32_t,2> k){check(d::bind_device_append_inputs(slot,inputs(q,k)));}
  std::vector<std::uint16_t> read(){auto result=original;check(bank.read(0,std::as_writable_bytes(std::span(result))));return result;}
  void expect(std::array<std::int32_t,2> q,std::array<std::int32_t,2> k){
    auto expected=original;
    const std::array<std::uint32_t,2> packed{0,std::uint32_t(q[0])};
    for(const auto &binding:bindings){const auto p=binding.page;const auto old=std::uint32_t(k[p.request]-q[p.request]);
      for(std::uint32_t row=0;row<p.initialized_rows;++row)for(std::uint32_t j=0;j<row_words;++j){
        const auto logical=p.logical_page*page_rows+row;std::uint16_t value=0;
        if(logical<old)value=original[1+(p.source_page*page_rows+row)*row_words+j];
        else if(logical<std::uint32_t(k[p.request]))value=payload[(packed[p.request]+logical-old)*row_words+j];
        expected[1+(p.destination_page*page_rows+row)*row_words+j]=value;
      }
    }
    EXPECT_EQ(read(),expected);
  }
};
struct Hold final:d::Job {
  std::binary_semaphore entered{0},release{0},done{0};
  explicit Hold(const std::shared_ptr<d::Runtime> &runtime,bool wait=true){execute=[](d::Job *opaque)noexcept{
    auto &self=*static_cast<Hold *>(opaque);self.entered.release();self.release.acquire();self.done.release();};
    check(runtime->enqueue(*this));if(wait)entered.acquire();}
  void finish(){if(!finished){release.release();done.acquire();finished=true;}}
  ~Hold(){finish();}bool finished=false;
};
struct GpuGate final:d::Job {
  std::shared_ptr<d::Runtime> runtime;
  mlx::core::Event event;
  std::binary_semaphore prepared{0};
  explicit GpuGate(std::shared_ptr<d::Runtime> owner):runtime(std::move(owner)){
    execute=[](d::Job *opaque)noexcept{auto &self=*static_cast<GpuGate *>(opaque);
      self.event=mlx::core::Event(*self.runtime->stream);self.event.set_value(1);self.event.wait(*self.runtime->stream);
      auto &encoder=mlx::core::metal::get_command_encoder(*self.runtime->stream);encoder.end_encoding();encoder.commit();self.prepared.release();};
    check(runtime->enqueue(*this));prepared.acquire();
  }
  void open(){event.cast<mlx::core::metal::EventImpl>().signal(1);}
  ~GpuGate(){open();}
};

TEST(AppleDeviceAppend, OriginalCowPagesUsePackedInputsAndZeroOnlyTheirPrivateEnvelope){
  Cell cell;auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});
  auto completed=take(d::submit_copy_recording(*slot));check(completed.wait());cell.expect({2,3},{5,5});
  EXPECT_EQ(d::copy_recording_report(*slot).native_calls,1u);
}
TEST(AppleDeviceAppend, Float16UsesTheSameOriginalTwoByteCopyContract){
  Cell cell(c::DataType::kFloat16);auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});
  auto completed=take(d::submit_copy_recording(*slot));check(completed.wait());cell.expect({2,3},{5,5});
}
TEST(AppleDeviceAppend, SameRecordingReplaysDifferentIndependentLengthsAndTwoByteOrigins){
  Cell cell;auto slot=cell.slot();
  const std::array<std::array<std::int32_t,4>,4> cases{{{2,3,5,5},{0,4,3,6},{3,1,7,3},{2,0,3,2}}};
  for(const auto &v:cases){
    check(d::clear_copy_recording(*slot));check(cell.bank.write(0,std::as_bytes(std::span(cell.original))));cell.configure(*slot);
    cell.bind(*slot,{v[0],v[1]},{v[2],v[3]});auto completed=take(d::submit_copy_recording(*slot));check(completed.wait());
    cell.expect({v[0],v[1]},{v[2],v[3]});
  }
}
TEST(AppleDeviceAppend, EveryInvalidControlLeavesAllOriginalPagesUntouchedAndPublishesFailure){
  Cell cell;const std::array<std::array<std::int32_t,4>,6> cases{{{4,1,5,3},{2,3,8,5},{2,3,2,5},{0,1,5,3},{-1,1,3,3},{0,0,0,2}}};
  for(const auto &v:cases){auto slot=cell.slot();cell.bind(*slot,{v[0],v[1]},{v[2],v[3]});
    auto completed=take(d::submit_copy_recording(*slot));EXPECT_TRUE(completed.wait().IsInvalidArgument());EXPECT_EQ(cell.read(),cell.original);
    std::int32_t verdict=0;check(take(d::device_append_status(*slot)).storage().read(std::as_writable_bytes(std::span(&verdict,1))));EXPECT_NE(verdict,0);
  }
  auto good=cell.slot();cell.bind(*good,{2,3},{5,5});auto completed=take(d::submit_copy_recording(*good));check(completed.wait());cell.expect({2,3},{5,5});
}
TEST(AppleDeviceAppend, ColdAuthorityMismatchAndInPlaceCowAreRefusedBeforeAnyWrite){
  Cell cell;auto slot=cell.slot();auto wrong=cell.bindings;wrong[0].page.destination_page=3;
  EXPECT_TRUE(d::configure_device_append_recording(*slot,cell.capacity,cell.requests,wrong).IsInvalidArgument());
  wrong=cell.bindings;wrong[0].page.destination_page=0;wrong[0].destination=take(cell.capacity.subview(0,page_rows*row_bytes));
  EXPECT_TRUE(d::configure_device_append_recording(*slot,cell.capacity,cell.requests,wrong).IsInvalidArgument());
  EXPECT_EQ(cell.read(),cell.original);cell.configure(*slot);cell.bind(*slot,{2,3},{5,5});
  auto completed=take(d::submit_copy_recording(*slot));check(completed.wait());cell.expect({2,3},{5,5});
}
TEST(AppleDeviceAppend, OriginalRecordingMetadataAndByteBudgetsRemainAuthoritative){
  Cell cell;const auto runtime=d::ContextAccess::runtime(cell.context);
  auto small=take(d::prepare_copy_recording_runtime(runtime,1,m::system_allocator(),bank_bytes,d::CopyRecordingMode::kReplayable));
  EXPECT_TRUE(d::prepare_device_append_recording(*small,{{page_rows,pages,input_rows,16,row_bytes},2},cell.row_desc).IsResourceExhausted());
  auto bytes=take(d::prepare_copy_recording_runtime(runtime,16,m::system_allocator(),row_bytes,d::CopyRecordingMode::kReplayable));
  check(d::prepare_device_append_recording(*bytes,{{page_rows,pages,input_rows,16,row_bytes},2},cell.row_desc));
  EXPECT_TRUE(d::configure_device_append_recording(*bytes,cell.capacity,cell.requests,cell.bindings).IsResourceExhausted());
  EXPECT_EQ(cell.read(),cell.original);
}
TEST(AppleDeviceAppend, UnobservedNativeProducerPinsRowsAndBothDeviceControlPorts){
  Cell cell;auto runtime=d::ContextAccess::runtime(cell.context);auto slot=cell.slot();
  const std::array specs{take(a::input_spec(cell.row_desc)),take(a::input_spec(cell.control_desc)),take(a::input_spec(cell.control_desc))};
  auto program=take(cell.context.compile([](const std::vector<mlx::core::array> &args,mlx::core::Stream stream)->hyle::Result<std::vector<mlx::core::array>>{
    return std::vector{mlx::core::add(args[0],mlx::core::array(0,args[0].dtype()),stream),
        mlx::core::add(args[1],mlx::core::array(1),stream),mlx::core::add(args[2],mlx::core::array(1),stream)};
  },specs));auto producer=take(program.create_invocation());
  auto bound=cell.inputs({1,2},{4,4});std::array<mlx::core::array,3> examples{*bound[0].value,*bound[1].value,*bound[2].value};check(producer.bind(examples));
  std::array<d::InputBinding,3> sources;for(std::size_t i=0;i<3;++i)sources[i].source=take(d::InvocationAccess::source(producer,i));
  check(d::bind_device_append_inputs(*slot,sources));check(d::reserve_copy_dependencies(*slot,1));Hold pending(runtime);
  auto produced=take(producer.submit());const std::array dependencies{c::detail::CompletionAccess::dependency(produced.completion())};
  auto copied=take(d::submit_copy_recording(*slot,dependencies));EXPECT_FALSE(d::InvocationAccess::reuse_gate(producer).ok());
  pending.finish();check(copied.wait());check(produced.completion().wait());cell.expect({2,3},{5,5});
}
TEST(AppleDeviceAppend, CancellationRetiresTheOriginalRecordingBeforeAnotherSlotReusesPages){
  Cell cell;auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});Hold pending(d::ContextAccess::runtime(cell.context));
  auto completion=take(d::submit_copy_recording(*slot));check(completion.request_cancel());pending.finish();
  EXPECT_TRUE(completion.wait().IsCancelled());completion={};EXPECT_EQ(cell.read(),cell.original);
  auto next=cell.slot();cell.bind(*next,{2,3},{5,5});auto completed=take(d::submit_copy_recording(*next));check(completed.wait());cell.expect({2,3},{5,5});
}
TEST(AppleDeviceAppend, FailedNativeControlProducerPreservesItsErrorAndLeavesEveryCowPageUntouched){
  Cell cell;auto slot=cell.slot();auto values=cell.inputs({2,3},{5,2});
  const std::array specs{take(a::input_spec(cell.control_desc)),take(a::input_spec(cell.control_desc))};
  auto program=take(cell.context.prepare(c::Gather{},specs));auto producer=take(program.create_invocation());
  auto indices=take(m::Buffer::create(cell.memory.ref(),8));const std::array<std::int32_t,2> bad{0,2};check(indices.write(0,std::as_bytes(std::span(bad))));
  const std::array args{*values[1].value,cell.array(indices.view(),cell.control_desc)};check(producer.bind(args));
  values[1].value.reset();values[1].source=take(d::InvocationAccess::source(producer,0));
  check(d::bind_device_append_inputs(*slot,values));check(d::reserve_copy_dependencies(*slot,1));auto produced=take(producer.submit());
  const std::array dependencies{c::detail::CompletionAccess::dependency(produced.completion())};auto copied=d::submit_copy_recording(*slot,dependencies);
  if(copied)EXPECT_TRUE(copied->wait().IsOutOfRange());else EXPECT_TRUE(copied.error().IsOutOfRange());
  EXPECT_TRUE(produced.completion().wait().IsOutOfRange());EXPECT_EQ(cell.read(),cell.original);
}
TEST(AppleDeviceAppend, OriginalGpuEventFeedsAnUnobservedReaderWhileTheFifoOwnerContinues){
  Cell cell;auto runtime=d::ContextAccess::runtime(cell.context);auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});
  constexpr auto words=bank_bytes/2;
  const auto desc=take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{1,words}));
  const auto weight_desc=take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{words,words}));
  const auto output_desc=take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{1,words}));
  auto weights=take(m::Buffer::create(cell.memory.ref(),words*words*2));
  std::vector<std::uint16_t> identity(words*words,0);for(std::size_t i=0;i<words;++i)identity[i*words+i]=0x3f80;
  check(weights.write(0,std::as_bytes(std::span(identity))));
  const std::array specs{take(a::input_spec(desc)),take(a::input_spec(weight_desc))};
  auto program=take(cell.context.prepare(c::model_ops::dense_linear_f32(),specs));
  auto reader=take(program.create_invocation());const std::array inputs{cell.array(cell.capacity,desc),cell.array(weights.view(),weight_desc)};check(reader.bind(inputs));
  check(d::InvocationAccess::reserve_dependencies(reader,1));GpuGate gpu(runtime);
  auto copied=take(d::submit_copy_recording(*slot));const std::array dependencies{c::detail::CompletionAccess::dependency(copied)};
  auto consumed=take(d::InvocationAccess::submit_ordered(reader,{},nullptr,nullptr,dependencies));Hold witness(runtime,false);
  const bool encoded=witness.entered.try_acquire_for(std::chrono::seconds(2));
  if(encoded){EXPECT_EQ(take(copied.poll()),c::CompletionPoll::kPending);EXPECT_EQ(take(consumed.completion().poll()),c::CompletionPoll::kPending);}
  gpu.open();witness.finish();EXPECT_TRUE(encoded)<<"original FIFO waited for the GPU append instead of importing its native event";
  check(consumed.completion().wait());check(copied.wait());cell.expect({2,3},{5,5});
  const auto output=take(d::InvocationAccess::output(reader,0));auto read=take(cell.memory.import_readonly(output,output_desc));
  auto observed=std::vector<float>(words);check(take(read.view()).read(std::as_writable_bytes(std::span(observed))));
  const auto actual=cell.read();for(std::size_t i=0;i<words;++i)
    EXPECT_EQ(observed[i],std::bit_cast<float>(std::uint32_t(actual[i+1])<<16));
  EXPECT_EQ(d::copy_recording_report(*slot).host_waits,0u);
}
TEST(AppleDeviceAppend, OriginalGpuEventAlsoOrdersTheExistingMlxReader){
  Cell cell;auto runtime=d::ContextAccess::runtime(cell.context);auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});
  const auto desc=take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,1>{bank_bytes/2}));
  const std::array specs{take(a::input_spec(desc))};
  auto program=take(cell.context.compile([](const std::vector<mlx::core::array>&args,mlx::core::Stream stream)->hyle::Result<std::vector<mlx::core::array>>{
    return std::vector{mlx::core::add(args[0],mlx::core::array(0,args[0].dtype()),stream)};
  },specs));auto reader=take(program.create_invocation());const std::array inputs{cell.array(cell.capacity,desc)};check(reader.bind(inputs));
  check(d::InvocationAccess::reserve_dependencies(reader,1));GpuGate gpu(runtime);
  auto copied=take(d::submit_copy_recording(*slot));const std::array dependencies{c::detail::CompletionAccess::dependency(copied)};
  auto consumed=take(d::InvocationAccess::submit_ordered(reader,{},nullptr,nullptr,dependencies));gpu.open();
  check(consumed.completion().wait());check(copied.wait());cell.expect({2,3},{5,5});
  const auto output=take(d::InvocationAccess::output(reader,0));auto read=take(cell.memory.import_readonly(output,desc));
  auto observed=std::vector<std::uint16_t>(bank_bytes/2);check(take(read.view()).read(std::as_writable_bytes(std::span(observed))));
  const auto actual=cell.read();EXPECT_TRUE(std::equal(observed.begin(),observed.end(),actual.begin()+1));
}
TEST(AppleDeviceAppend, OriginalFenceOrdersBlitThenAppendThenBlitWithoutAnOwnerWait){
  Cell cell;auto runtime=d::ContextAccess::runtime(cell.context);auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});
  const auto upload_bytes=2+cell.payload.size()*2;
  auto source=take(m::Buffer::create(cell.memory.ref(),upload_bytes));
  auto result=take(m::Buffer::create(cell.memory.ref(),cell.bank.size()));
  std::vector<std::uint16_t> words(1+cell.payload.size(),0xeeee);
  std::copy(cell.payload.begin(),cell.payload.end(),words.begin()+1);
  check(source.write(0,std::as_bytes(std::span(words))));check(cell.rows.fill(0,upload_bytes,std::byte{0}));
  auto upload=take(d::prepare_copy_recording_runtime(runtime,1,m::system_allocator(),upload_bytes,d::CopyRecordingMode::kReplayable));
  auto download=take(d::prepare_copy_recording_runtime(runtime,1,m::system_allocator(),cell.bank.size(),d::CopyRecordingMode::kReplayable));
  const std::array<c::detail::CopyBackendRange,1> input{{{take(cell.rows.view().subview(0,upload_bytes)),source.view()}}};
  const std::array<c::detail::CopyBackendRange,1> output{{{result.view(),cell.bank.view()}}};
  check(d::bind_copy_recording(*upload,input));check(d::bind_copy_recording(*download,output));
  GpuGate gpu(runtime);
  auto first=take(d::submit_copy_recording(*upload));auto appended=take(d::submit_copy_recording(*slot));
  auto last=take(d::submit_copy_recording(*download));Hold witness(runtime,false);
  const bool encoded=witness.entered.try_acquire_for(std::chrono::seconds(2));
  if(encoded){EXPECT_EQ(take(appended.poll()),c::CompletionPoll::kPending);EXPECT_EQ(take(last.poll()),c::CompletionPoll::kPending);}
  gpu.open();witness.finish();EXPECT_TRUE(encoded);
  check(last.wait());check(appended.wait());check(first.wait());cell.expect({2,3},{5,5});
  auto observed=cell.original;check(result.read(0,std::as_writable_bytes(std::span(observed))));
  EXPECT_EQ(observed,cell.read());
  for(const auto &recording:{upload,slot,download})EXPECT_EQ(d::copy_recording_report(*recording).host_waits,0u);
}
} // namespace
