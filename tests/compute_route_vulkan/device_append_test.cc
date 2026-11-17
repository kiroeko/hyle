#include <gtest/gtest.h>
#include "../../src/compute_route_vulkan/vulkan_device_append_internal.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"
#include "../../src/compute_route_vulkan/vulkan_composition_access.h"
#include "hyle/compute/vulkan_memory.h"
#include "fixture.h"

#include <array>
#include <stdexcept>
#include <string>

namespace {
namespace c=hyle::compute;
namespace v=c::vulkan;
namespace d=v::detail;
namespace m=hyle::mem;
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::move(*value);
}
void check(hyle::Status value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.message()));
}
constexpr std::uint32_t page_rows=5,pages=8,input_rows=7;
struct Cell final {
  v::Context context=take(vulkan_test::context());
  v::MemoryResource memory=take(v::MemoryResource::create(context,v::MemoryKind::kDevice));
  const std::uint32_t columns;
  const std::size_t row_bytes,bank_bytes,origin;
  m::Buffer bank,rows;
  m::Buffer queries=take(m::Buffer::create(memory.ref(),16));
  m::Buffer lengths=take(m::Buffer::create(memory.ref(),16));
  m::BufferView capacity;
  c::TensorDesc row_desc;
  c::TensorDesc control_desc=take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{2}));
  const std::array<c::detail::DeviceAppendRequest,2> requests{{{3,1,4,7},{4,2,3,6}}};
  std::array<d::DeviceAppendPageBinding,4> bindings;
  std::vector<std::uint16_t> original,payload;
  explicit Cell(c::DataType type=c::DataType::kBFloat16,bool whole_words=false)
      : columns(whole_words?4:3),row_bytes(columns*2),bank_bytes(pages*page_rows*row_bytes),origin(whole_words?4:2),
        bank(take(m::Buffer::create(memory.ref(),bank_bytes+2*origin))),
        rows(take(m::Buffer::create(memory.ref(),(input_rows*row_bytes+origin+3)/4*4))),
        capacity(take(bank.view().subview(origin,bank_bytes))),
        row_desc(take(c::TensorDesc::dense(type,std::array<std::int64_t,2>{input_rows,columns}))),
        original(bank.size()/2,0xdead),payload(input_rows*columns) {
    EXPECT_FALSE(context.environment().software_device);
    for (std::uint32_t page=0;page<4;++page) for(std::uint32_t row=0;row<page_rows;++row)
      for(std::uint32_t j=0;j<columns;++j)
        original[origin/2+(page*page_rows+row)*columns+j]=std::uint16_t(0x3f80+page*32+row*columns+j);
    for (std::size_t i=0;i<payload.size();++i) payload[i]=std::uint16_t(0x4000+i);
    reset();
    std::vector<std::uint16_t> carrier(rows.size()/2,0xeeee);
    std::copy(payload.begin(),payload.end(),carrier.begin()+origin/2);
    check(rows.write(0,std::as_bytes(std::span(carrier))));
    const std::array<c::detail::DeviceAppendPage,4> descriptions{{
      {0,0,4,5,0,4},{0,1,5,2,UINT32_MAX,0},{1,0,6,5,1,3},{1,1,7,1,UINT32_MAX,0}}};
    for(std::size_t i=0;i<bindings.size();++i) {
      const auto p=descriptions[i];bindings[i].page=p;
      bindings[i].destination=take(capacity.subview(std::size_t(p.destination_page)*page_rows*row_bytes,p.initialized_rows*row_bytes));
      if(p.source_rows) bindings[i].prefix=take(capacity.subview(std::size_t(p.source_page)*page_rows*row_bytes,p.source_rows*row_bytes));
    }
  }
  ~Cell() { context.close().IgnoreError(); }
  c::detail::DeviceLengthsPagedAppendSpec geometry() const { return {{page_rows,pages,input_rows,16,row_bytes},2}; }
  void reset() { check(bank.write(0,std::as_bytes(std::span(original)))); }
  std::shared_ptr<d::CopyRecording> slot() {
    auto result=take(d::prepare_copy_recording_runtime(d::Access::runtime(context),16,m::system_allocator(),bank_bytes,d::CopyRecordingMode::kReplayable));
    check(d::prepare_device_append_recording(*result,geometry(),row_desc));
    configure(*result);return result;
  }
  void configure(d::CopyRecording &slot) { check(d::configure_device_append_recording(slot,capacity,requests,bindings)); }
  v::Tensor tensor(m::ConstBufferView view,const c::TensorDesc &description) {
    return take(d::MemoryAccess::tensor_view(d::Access::runtime(context),
        take(c::ConstTensorView::create(description,view)),m::system_allocator()));
  }
  std::array<v::Tensor,3> inputs(std::array<std::int32_t,2> q,std::array<std::int32_t,2> k) {
    check(queries.write(4,std::as_bytes(std::span(q))));
    check(lengths.write(4,std::as_bytes(std::span(k))));
    return {tensor(take(rows.view().subview(origin,payload.size()*2)),row_desc),
        tensor(take(queries.view().subview(4,8)),control_desc),tensor(take(lengths.view().subview(4,8)),control_desc)};
  }
  void bind(d::CopyRecording &slot,std::array<std::int32_t,2> q,std::array<std::int32_t,2> k) {
    check(d::bind_device_append_inputs(slot,inputs(q,k)));
  }
  std::vector<std::uint16_t> read() {
    auto result=original;check(bank.read(0,std::as_writable_bytes(std::span(result))));return result;
  }
  void expect(std::array<std::int32_t,2> q,std::array<std::int32_t,2> k) {
    auto expected=original;
    const std::array<std::uint32_t,2> packed{0,std::uint32_t(q[0])};
    for (const auto &binding:bindings) {
      const auto p=binding.page;const auto old=std::uint32_t(k[p.request]-q[p.request]);
      for(std::uint32_t row=0;row<p.initialized_rows;++row) for(std::uint32_t j=0;j<columns;++j) {
        const auto logical=p.logical_page*page_rows+row;std::uint16_t value=0;
        if(logical<old) value=original[origin/2+(p.source_page*page_rows+row)*columns+j];
        else if(logical<std::uint32_t(k[p.request])) value=payload[(packed[p.request]+logical-old)*columns+j];
        expected[origin/2+(p.destination_page*page_rows+row)*columns+j]=value;
      }
    }
    EXPECT_EQ(read(),expected);
  }
};

TEST(VulkanDeviceAppend, ExactHalfwordCowPreservesPrefixesAndAllOutsideCanaries) {
  Cell cell;auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});
  auto done=take(d::submit_copy_recording(*slot));check(done.wait());cell.expect({2,3},{5,5});
  EXPECT_EQ(d::copy_recording_report(*slot).native_calls,2u);
  auto verdict=take(d::device_append_status(*slot));
  EXPECT_FALSE(verdict.storage().resource().properties().has(m::MemoryProperty::kHostAccessible));
  std::int32_t status=-1;check(verdict.storage().read(std::as_writable_bytes(std::span(&status,1))));EXPECT_EQ(status,0);
}
TEST(VulkanDeviceAppend, Float16RowsRetainTheirOriginalBits) {
  Cell cell(c::DataType::kFloat16);auto slot=cell.slot();cell.bind(*slot,{3,1},{7,3});
  auto done=take(d::submit_copy_recording(*slot));check(done.wait());cell.expect({3,1},{7,3});
}
TEST(VulkanDeviceAppend, AlignedWholeWordsNeedNoStorage16Payload) {
  Cell cell(c::DataType::kBFloat16,true);auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});
  auto done=take(d::submit_copy_recording(*slot));check(done.wait());cell.expect({2,3},{5,5});
}
TEST(VulkanDeviceAppend, SameRetiredRecordingReplaysEmptyAndChangedDeviceLengths) {
  Cell cell;auto slot=cell.slot();
  const std::array<std::array<std::int32_t,4>,4> cases{{{2,3,5,5},{0,4,3,6},{3,1,7,3},{2,0,3,2}}};
  for(const auto &item:cases) {
    cell.reset();cell.configure(*slot);cell.bind(*slot,{item[0],item[1]},{item[2],item[3]});
    {auto done=take(d::submit_copy_recording(*slot));check(done.wait());}
    cell.expect({item[0],item[1]},{item[2],item[3]});check(d::clear_copy_recording(*slot));
  }
}
TEST(VulkanDeviceAppend, InvalidControlsWriteNoPageAndFailedRecordingRemainsTerminal) {
  Cell cell;
  const std::array<std::array<std::int32_t,4>,6> cases{{{4,1,5,3},{2,3,8,5},{2,3,2,5},{0,1,5,3},{-1,1,3,3},{0,0,0,2}}};
  for(const auto &item:cases) {
    auto slot=cell.slot();cell.bind(*slot,{item[0],item[1]},{item[2],item[3]});
    {auto done=take(d::submit_copy_recording(*slot));EXPECT_TRUE(done.wait().IsInvalidArgument());}
    EXPECT_EQ(cell.read(),cell.original);
    auto verdict=take(d::device_append_status(*slot));std::int32_t status=0;
    check(verdict.storage().read(std::as_writable_bytes(std::span(&status,1))));EXPECT_NE(status,0);
    EXPECT_FALSE(d::bind_device_append_inputs(*slot,cell.inputs({2,3},{5,5})).ok());
  }
  auto next=cell.slot();cell.bind(*next,{2,3},{5,5});
  auto done=take(d::submit_copy_recording(*next));check(done.wait());cell.expect({2,3},{5,5});
}
TEST(VulkanDeviceAppend, ColdPageAuthorityAndFreshCowRemainMandatory) {
  Cell cell;auto slot=cell.slot();auto wrong=cell.bindings;
  wrong[0].destination=take(cell.capacity.subview(3*page_rows*cell.row_bytes,5*cell.row_bytes));
  EXPECT_TRUE(d::configure_device_append_recording(*slot,cell.capacity,cell.requests,wrong).IsInvalidArgument());
  wrong=cell.bindings;wrong[0].page.destination_page=0;
  wrong[0].destination=take(cell.capacity.subview(0,5*cell.row_bytes));
  EXPECT_TRUE(d::configure_device_append_recording(*slot,cell.capacity,cell.requests,wrong).IsInvalidArgument());
  cell.configure(*slot);auto input=cell.inputs({2,3},{5,5});
  input[0]=cell.tensor(take(cell.capacity.subview(4*page_rows*cell.row_bytes,input_rows*cell.row_bytes)),cell.row_desc);
  EXPECT_TRUE(d::bind_device_append_inputs(*slot,input).IsInvalidArgument());
  EXPECT_EQ(cell.read(),cell.original);
}
TEST(VulkanDeviceAppend, OriginalMetadataAndByteBudgetsRejectBeforeWriting) {
  Cell cell;const auto runtime=d::Access::runtime(cell.context);
  auto small=take(d::prepare_copy_recording_runtime(runtime,1,m::system_allocator(),cell.bank_bytes,d::CopyRecordingMode::kReplayable));
  EXPECT_TRUE(d::prepare_device_append_recording(*small,cell.geometry(),cell.row_desc).IsResourceExhausted());
  auto bytes=take(d::prepare_copy_recording_runtime(runtime,16,m::system_allocator(),cell.row_bytes,d::CopyRecordingMode::kReplayable));
  check(d::prepare_device_append_recording(*bytes,cell.geometry(),cell.row_desc));
  EXPECT_TRUE(d::configure_device_append_recording(*bytes,cell.capacity,cell.requests,cell.bindings).IsResourceExhausted());
  EXPECT_EQ(cell.read(),cell.original);
}
TEST(VulkanDeviceAppend, NativeProducedRowsAndBothControlPortsNeedNoHostObservation) {
  Cell cell;auto slot=cell.slot();auto source=cell.inputs({2,3},{5,5});
  d::Plan plan;plan.inputs={0,1,2};plan.outputs={3,4,5};
  for(const auto &description:std::array{cell.row_desc,cell.control_desc,cell.control_desc,cell.row_desc,cell.control_desc,cell.control_desc})
    plan.values.push_back({description,{}});
  plan.steps={{c::Operation(c::Copy{}),{0},{3},{},{}},
      {c::Operation(c::Copy{}),{1},{4},{},{}},{c::Operation(c::Copy{}),{2},{5},{},{}}};
  auto program=take(d::prepare_plan(cell.context,plan));auto producer=take(program.create_invocation());
  check(producer.bind(source));
  for(std::size_t i=0;i<source.size();++i) source[i]=take(d::CompositionAccess::output(producer,i));
  check(d::bind_device_append_inputs(*slot,source));check(d::CompositionAccess::seal(producer));
  auto produced=take(producer.submit());auto copied=take(d::submit_copy_recording(*slot));
  check(copied.wait());check(produced.completion().wait());cell.expect({2,3},{5,5});
}
TEST(VulkanDeviceAppend, ActualProducerErrorGuardsPagesEvenWhenAllLengthValuesAreLegal) {
  Cell cell;auto slot=cell.slot();auto source=cell.inputs({2,3},{5,5});
  const std::array<std::int32_t,2> bad{0,2};
  auto indices=take(cell.context.upload(cell.control_desc,std::as_bytes(std::span(bad))));
  const std::array<v::Argument,2> arguments{cell.control_desc,cell.control_desc};
  auto program=take(v::prepare(cell.context,c::Operation(c::Gather{}),arguments));
  auto producer=take(program.create_invocation());check(producer.bind(std::array{source[1],indices}));
  const auto run=d::Access::invocation(producer)->run;
  ASSERT_TRUE(run->checked);ASSERT_EQ(run->checked->buffers.size(),1u);
  const auto scalar=take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{1}));
  const std::array checks{d::Access::tensor(scalar,run->checked->buffers[0])};
  // This is the actual original producer word, not a fabricated failing
  // control. The consumer's numeric lengths are independently valid.
  check(d::bind_device_append_inputs(*slot,source,checks));
  auto produced=take(producer.submit());auto copied=take(d::submit_copy_recording(*slot));
  EXPECT_TRUE(copied.wait().IsInvalidArgument());
  EXPECT_TRUE(produced.completion().wait().IsOutOfRange());
  EXPECT_EQ(cell.read(),cell.original);
  EXPECT_EQ(d::copy_recording_report(*slot).native_calls,3u);
}
TEST(VulkanDeviceAppend, AcceptedOriginalCompletionExcludesAllEarlyReconfiguration) {
  Cell cell;auto slot=cell.slot();auto inputs=cell.inputs({2,3},{5,5});
  check(d::bind_device_append_inputs(*slot,inputs));auto copied=take(d::submit_copy_recording(*slot));
  // Before any fence observation the original Run still owns accepted work,
  // independently of whether this small GPU dispatch has physically finished.
  EXPECT_TRUE(d::copy_recording_retired_gate(*slot).IsUnavailable());
  EXPECT_TRUE(d::clear_copy_recording(*slot).IsUnavailable());
  EXPECT_TRUE(d::configure_device_append_recording(*slot,cell.capacity,cell.requests,cell.bindings).IsUnavailable());
  EXPECT_TRUE(d::bind_device_append_inputs(*slot,inputs).IsUnavailable());
  EXPECT_FALSE(d::submit_copy_recording(*slot));
  check(copied.wait());copied={};check(d::copy_recording_retired_gate(*slot));
  check(d::clear_copy_recording(*slot));cell.expect({2,3},{5,5});
}
TEST(VulkanDeviceAppend, CancellationKeepsNativeOwnershipUntilRetirementAndAllowsFreshWork) {
  Cell cell;auto slot=cell.slot();cell.bind(*slot,{2,3},{5,5});
  auto copied=take(d::submit_copy_recording(*slot));check(copied.request_cancel());
  EXPECT_TRUE(d::clear_copy_recording(*slot).IsUnavailable());
  EXPECT_TRUE(copied.wait().IsCancelled());copied={};
  check(d::copy_recording_retired_gate(*slot));
  // Vulkan cancellation does not undo an already accepted GPU dispatch. Its
  // private COW bytes are not a published result. Do not infer page retirement
  // or reset the terminal error merely from a cancellation request.
  cell.expect({2,3},{5,5});
  EXPECT_FALSE(d::bind_device_append_inputs(*slot,cell.inputs({2,3},{5,5})).ok());
  cell.reset();auto next=cell.slot();cell.bind(*next,{0,4},{3,6});
  auto complete=take(d::submit_copy_recording(*next));check(complete.wait());cell.expect({0,4},{3,6});
}
TEST(VulkanDeviceAppend, AbandonedWrapperAndCompletionRetainNativeBuffersWithoutAnOwnershipCycle) {
  Cell cell;std::weak_ptr<d::Buffer> input_owner;
  {
    auto slot=cell.slot();auto inputs=cell.inputs({2,3},{5,5});
    input_owner=d::Access::buffer(inputs[0]);check(d::bind_device_append_inputs(*slot,inputs));
    auto accepted=take(d::submit_copy_recording(*slot));
    inputs={};cell.rows={};slot.reset();EXPECT_FALSE(input_owner.expired());
    accepted={}; // Abandonment queues original retirement; it never frees GPU IO.
    EXPECT_FALSE(input_owner.expired());
  }
  check(cell.context.drain());EXPECT_TRUE(input_owner.expired());cell.expect({2,3},{5,5});
}
} // namespace
