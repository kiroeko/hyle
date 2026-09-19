#include <gtest/gtest.h>

#include <algorithm>

#include "reduction_test_support.h"
#include "vulkan_internal.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_reduction_test;

void expect_managed(const v::Program &program) {
  const auto &dispatches = v::detail::Access::program(program)->dispatches;
  EXPECT_EQ(std::ranges::count_if(dispatches,
                                  [](const auto &dispatch) {
                                    return dispatch.region != nullptr;
                                  }),
            1);
}

TEST(VulkanReduction, MatureRowsCoverBothDispatchBoundariesAndHalfRounding) {
  auto context = vulkan_test::context().value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16})
    for (int rows : {511, 512, 513, 262145}) {
      const int width = rows > 513 ? 1 : 3;
      const auto description = dense(type, {rows, width});
      const auto data = pattern(*description.element_count());
      const auto input = make(context, description, data);
      for (auto kind : {c::ReductionKind::kSum, c::ReductionKind::kMean}) {
        SCOPED_TRACE(::testing::Message()
                     << static_cast<int>(type) << '/' << rows << '/'
                     << static_cast<int>(kind));
        const c::Reduce operation{kind, c::Axes::from(std::array{-1}).value(),
                                  true, c::DataType::kFloat32};
        auto program = v::prepare(context, operation, description);
        ASSERT_TRUE(program) << program.error().message();
        expect_managed(*program);
        auto slot = program->create_invocation().value();
        ASSERT_TRUE(slot.bind(input).ok());
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        expect(
            context, result->get()->tensors()[0],
            reduce_reference(values(context, input), description, operation));
      }
    }
}

TEST(VulkanReduction, MatureLongProbabilityRetainsLastBlockPeak) {
  auto context = vulkan_test::context().value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16})
    for (int width : {1, 16384, 16385, 32769}) {
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(type) << '/' << width);
      const auto description = dense(type, {2, width});
      std::vector<double> data(2 * width, 0.0);
      data[width - 1] = 1000.0;
      data[2 * width - 1] = -1000.0;
      const auto input = make(context, description, data);
      auto program = v::prepare(context, c::Softmax{}, description);
      ASSERT_TRUE(program) << program.error().message();
      expect_managed(*program);
      auto slot = program->create_invocation().value();
      ASSERT_TRUE(slot.bind(input).ok());
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      expect(context, result->get()->tensors()[0],
             probability_reference(data, description, c::Softmax{}));
    }
}

TEST(VulkanReduction, MatureMasksPreserveScaleOrderAndF32IntermediateRounding) {
  auto context = vulkan_test::context().value();
  const auto description = dense(c::DataType::kFloat32, {1, 2});
  const auto input =
      make(context, description, std::array{10000.0009765625, 10000.0});
  const auto mask =
      make(context, description, std::array{-100000016.0, -100000008.0});
  const c::Softmax additive{.scale = 10000.0009765625F,
                            .mask = c::MaskKind::kAdditive};
  auto program = v::prepare(context, additive, description, description);
  ASSERT_TRUE(program) << program.error().message();
  expect_managed(*program);
  auto slot = program->create_invocation().value();
  ASSERT_TRUE(slot.bind(input, mask).ok());
  auto work = slot.submit().value();
  auto result = work.wait().value();
  expect(context, result.get()->tensors()[0], std::array{0.5, 0.5});

  const auto bad =
      make(context, dense(c::DataType::kFloat32, {1, 3}),
           std::array{std::numeric_limits<double>::quiet_NaN(), 2.0, -1.0});
  const auto flags_desc = dense(c::DataType::kBool, {3});
  const std::array<std::uint64_t, 3> flags{0, 1, 1};
  const auto flags_tensor =
      context.upload(flags_desc, payload(flags_desc, flags)).value();
  for (float scale : {0.0F, -2.0F}) {
    const c::Softmax boolean{.scale = scale, .mask = c::MaskKind::kBoolean};
    auto output =
        execute(context, c::Operation(boolean), std::array{bad, flags_tensor});
    ASSERT_TRUE(output) << output.error().message();
    expect(context, *output,
           probability_reference(values(context, bad), bad.description(),
                                 boolean, mask_values(flags)));
  }
}

TEST(VulkanReduction, MatureReductionRespectsColdGridLimits) {
  auto context = vulkan_test::context().value();
  auto &limits = v::detail::Access::runtime(context)->device->properties.limits;
  const auto original = limits;
  const auto description = dense(c::DataType::kFloat32, {3, 32769});
  for (const auto operation :
       {c::Operation(c::Reduce{.axes = c::Axes::from(std::array{-1}).value(),
                               .keep_dimensions = true}),
        c::Operation(c::Softmax{})}) {
    limits.maxComputeWorkGroupCount[0] = 2;
    auto program = v::prepare(context, operation, description);
    limits = original;
    ASSERT_TRUE(program) << program.error().message();
    const auto &dispatches = v::detail::Access::program(*program)->dispatches;
    EXPECT_TRUE(std::ranges::all_of(dispatches, [](const auto &dispatch) {
      return dispatch.region == nullptr;
    }));
  }
}

TEST(VulkanReduction, AllTypesAxesLayoutsIdentitiesAndSignedZero) {
  auto context = vulkan_test::context().value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    for (auto shape : {std::vector<std::int64_t>{3, 129},
                       {2, 17, 3},
                       {1, 2, 1, 3, 1, 17, 1, 1},
                       {2, 0, 3},
                       {0, 17},
                       {},
                       {3, 2049}}) {
      auto desc = c::TensorDesc::dense(type, shape).value();
      if (*desc.element_count() != 0) {
        std::vector<std::int64_t> strides(shape.size());
        std::int64_t stride =
            static_cast<std::int64_t>(c::data_type_size(type)) + 1;
        for (std::size_t axis = shape.size(); axis-- > 0;) {
          strides[axis] = stride;
          stride = stride * shape[axis] + 3;
        }
        desc = c::TensorDesc::strided_bytes(type, shape, strides).value();
      }
      auto data = pattern(*desc.element_count());
      for (auto &x : data)
        x = rounded(1 + x * 0.1, type);
      const auto input = make(context, desc, data);
      std::vector<c::Axes> axes{c::Axes::all()};
      if (shape.size() == 3)
        axes.push_back(c::Axes::from(shape[1] == 0
                                         ? std::vector<std::int32_t>{1}
                                         : std::vector<std::int32_t>{0, -1})
                           .value());
      else if (shape.size() == 8)
        axes.push_back(c::Axes::from(std::array{1, 3, 5}).value());
      else if (!shape.empty())
        axes.push_back(c::Axes::from(std::array{-1}).value());
      for (const auto &selected : axes)
        for (bool keep : {false, true})
          for (auto kind :
               {c::ReductionKind::kSum, c::ReductionKind::kMean,
                c::ReductionKind::kMinimum, c::ReductionKind::kMaximum}) {
            SCOPED_TRACE(::testing::Message()
                         << static_cast<int>(type) << '/' << shape.size() << '/'
                         << static_cast<int>(kind) << '/' << keep << '/'
                         << selected.is_all());
            const c::Reduce operation{kind, selected, keep,
                                      c::DataType::kFloat32};
            auto result =
                execute(context, c::Operation(operation), std::array{input});
            ASSERT_TRUE(result) << result.error().message();
            expect(context, *result, reduce_reference(data, desc, operation));
          }
      expect(context, input, data, true);
    }
    const auto inf = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> data{0, -0.0, 0, 0,   -0.0, -0.0,
                                   1, nan,  3, inf, -inf, 1};
    const auto desc = dense(type, {4, 3});
    const auto input = make(context, desc, data);
    for (auto kind : {c::ReductionKind::kSum, c::ReductionKind::kMean,
                      c::ReductionKind::kMinimum, c::ReductionKind::kMaximum}) {
      const c::Reduce operation{kind, c::Axes::from(std::array{-1}).value(),
                                false, c::DataType::kFloat32};
      auto result =
          execute(context, c::Operation(operation), std::array{input});
      ASSERT_TRUE(result) << result.error().message();
      expect(context, *result, reduce_reference(data, desc, operation));
    }
    const auto huge = c::TensorDesc::strided_bytes(
                          type,
                          std::array<std::int64_t, 2>{
                              0, (std::numeric_limits<std::int64_t>::max)()},
                          std::array<std::int64_t, 2>{0, 0})
                          .value();
    const auto empty = context.allocate(huge).value();
    for (auto kind : {c::ReductionKind::kSum, c::ReductionKind::kMean,
                      c::ReductionKind::kMinimum, c::ReductionKind::kMaximum}) {
      const c::Reduce operation{.kind = kind,
                                .accumulation_type = c::DataType::kFloat32};
      auto result =
          execute(context, c::Operation(operation), std::array{empty});
      ASSERT_TRUE(result) << result.error().message();
      expect(context, *result, reduce_reference({}, huge, operation));
    }
  }
}

TEST(VulkanReduction, SoftmaxMasksAxesCacheBoundaryAndOddViewOrigins) {
  auto context = vulkan_test::context().value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    for (int width : {0, 1, 127, 129, 2048, 2049, 8193, 32769})
      for (int inner : {1, 2, 3}) {
        const auto parent_desc =
            c::TensorDesc::strided_bytes(
                type, std::array<std::int64_t, 3>{4, width, inner},
                std::array<std::int64_t, 3>{size + 1,
                                            inner * (4 * (size + 1) + 3) + 5,
                                            4 * (size + 1) + 3})
                .value();
        const auto parent =
            make(context, parent_desc, pattern(*parent_desc.element_count()));
        const auto input = parent.view(c::View::slice(0, 1, 3).value()).value();
        const auto original = bits(context, input);
        const auto data = values(context, input);
        for (auto mask : {c::MaskKind::kNone, c::MaskKind::kBoolean,
                          c::MaskKind::kAdditive}) {
          std::vector<double> mask_data(width);
          std::vector<std::uint64_t> flags(width);
          for (int col = 0; col < width; ++col) {
            flags[col] = col % 3 != 1;
            mask_data[col] =
                mask == c::MaskKind::kBoolean
                    ? flags[col]
                    : rounded(flags[col]
                                  ? (col % 7) * 0.3
                                  : -std::numeric_limits<double>::infinity(),
                              type);
          }
          const auto mask_desc =
              dense(mask == c::MaskKind::kBoolean ? c::DataType::kBool : type,
                    {1, width, 1});
          std::vector<v::Tensor> inputs{input};
          if (mask == c::MaskKind::kBoolean)
            inputs.push_back(
                context.upload(mask_desc, payload(mask_desc, flags)).value());
          else if (mask == c::MaskKind::kAdditive)
            inputs.push_back(make(context, mask_desc, mask_data));
          for (auto kind :
               {c::SoftmaxKind::kSoftmax, c::SoftmaxKind::kLogSoftmax}) {
            SCOPED_TRACE(::testing::Message()
                         << static_cast<int>(type) << '/' << width << '/'
                         << inner << '/' << static_cast<int>(mask) << '/'
                         << static_cast<int>(kind));
            const c::Softmax operation{
                .axis = -2, .scale = -0.7F, .kind = kind, .mask = mask};
            auto result = execute(context, c::Operation(operation), inputs);
            ASSERT_TRUE(result) << result.error().message();
            expect(context, *result,
                   probability_reference(data, input.description(), operation,
                                         mask_data));
          }
        }
        EXPECT_EQ(bits(context, input), original);
      }
  }
}

TEST(VulkanReduction, NonfiniteRowsTinyLogTailBroadcastAndRejection) {
  auto context = vulkan_test::context().value();
  const auto inf = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto desc = dense(type, {4, 5});
    const auto input =
        make(context, desc,
             std::array{nan,  1.0,  2.0,  -inf, 4.0,  0.0, inf, 2.0, 3.0, 4.0,
                        -inf, -inf, -inf, -inf, -inf, 0.0, 1.0, nan, 3.0, 4.0});
    for (auto flags : {std::array<std::uint64_t, 5>{0, 1, 1, 1, 1},
                       {0, 0, 0, 0, 0},
                       {1, 1, 1, 1, 1}}) {
      const auto mask_desc = dense(c::DataType::kBool, {5});
      const auto mask =
          context.upload(mask_desc, payload(mask_desc, flags)).value();
      for (auto kind :
           {c::SoftmaxKind::kSoftmax, c::SoftmaxKind::kLogSoftmax}) {
        const c::Softmax operation{.kind = kind, .mask = c::MaskKind::kBoolean};
        auto result =
            execute(context, c::Operation(operation), std::array{input, mask});
        ASSERT_TRUE(result) << result.error().message();
        expect(context, *result,
               probability_reference(values(context, input), desc, operation,
                                     mask_values(flags)));
      }
    }
    const auto tiny_desc = dense(type, {4, 5});
    const auto tiny =
        make(context, tiny_desc,
             std::array{0.0,  -16.0, -17.0,  -18.0, -19.0,   1.0,  1.0,
                        1.0,  -15.0, -16.0,  0.0,   -4.1875, -inf, -inf,
                        -inf, 0.0,   -4.125, -inf,  -inf,    -inf});
    const c::Softmax logarithm{.kind = c::SoftmaxKind::kLogSoftmax};
    auto log = execute(context, c::Operation(logarithm), std::array{tiny});
    ASSERT_TRUE(log) << log.error().message();
    const auto expected =
        probability_reference(values(context, tiny), tiny_desc, logarithm);
    expect(context, *log, expected);
    const auto observed = values(context, *log);
    // The latter two rows straddle the log1p polynomial/native boundary.
    for (const auto index : {0u, 10u, 15u}) {
      EXPECT_LT(observed[index], 0.0);
      EXPECT_NEAR(observed[index], rounded(expected[index], type),
                  std::abs(rounded(expected[index], type)) *
                      (type == c::DataType::kFloat32 ? 6e-6 : 0.008));
    }
    const auto one = make(context, dense(type, {1}), std::array<double, 1>{2});
    const auto broadcast =
        one.view(c::View::broadcast_to(
                     c::TensorShape::create(
                         std::array<std::int64_t, 8>{1, 1, 2, 1, 3, 1, 1, 129})
                         .value())
                     .value())
            .value();
    auto uniform =
        execute(context, c::Operation(c::Softmax{}), std::array{broadcast});
    ASSERT_TRUE(uniform) << uniform.error().message();
    expect(context, *uniform, std::vector<double>(6 * 129, 1.0 / 129));
    EXPECT_FALSE(v::prepare(
        context, c::Softmax{.accumulation_type = c::DataType::kFloat64}, desc));
    EXPECT_FALSE(v::prepare(context, c::Softmax{.axis = 2}, desc));
    EXPECT_FALSE(v::prepare(
        context, c::Softmax{.scale = std::numeric_limits<float>::infinity()},
        desc));
    EXPECT_FALSE(v::prepare(context, c::Softmax{.mask = c::MaskKind::kBoolean},
                            desc, dense(c::DataType::kBool, {4})));
    if (type != c::DataType::kFloat32) {
      EXPECT_FALSE(v::prepare(context, c::Reduce{}, desc));
      EXPECT_FALSE(v::prepare(context,
                              c::Softmax{.mask = c::MaskKind::kAdditive}, desc,
                              dense(c::DataType::kFloat32, {5})));
    } else
      EXPECT_TRUE(v::prepare(context, c::Reduce{}, desc));
  }
  const auto offset_desc = dense(c::DataType::kFloat32, {1, 4});
  const auto offset =
      make(context, offset_desc, std::array{1e8, 1e8, 1e8, 1e8});
  auto log = execute(
      context, c::Operation(c::Softmax{.kind = c::SoftmaxKind::kLogSoftmax}),
      std::array{offset});
  ASSERT_TRUE(log) << log.error().message();
  expect(context, *log, std::vector<double>(4, -std::log(4.0)));
}

TEST(VulkanReduction, FixedRecorderPreservesGuardBytesAndRejectsAlias) {
  auto context = vulkan_test::context().value();
  const auto alignment =
      std::max(VkDeviceSize{4},
               v::detail::Access::runtime(context)
                   ->device->properties.limits.minStorageBufferOffsetAlignment);
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16})
    for (int inner : {1, 2, 3}) {
      const auto desc = dense(type, {3, 129, inner});
      const auto input = make(context, desc, pattern(*desc.element_count()));
      const c::Reduce reduce{.axes = c::Axes::from(std::array{1}).value(),
                             .accumulation_type = c::DataType::kFloat32};
      const c::Softmax softmax{.axis = 1};
      for (auto operation : {c::Operation(reduce), c::Operation(softmax)}) {
        const auto output_desc =
            c::infer_single_output(operation, std::array{desc}).value();
        const auto capacity =
            (*output_desc.required_bytes() + 3) & ~std::size_t{3};
        std::vector<std::byte> guard(alignment + capacity + 8, std::byte{0x5a});
        const auto backing =
            context
                .upload(dense(c::DataType::kBool,
                              {static_cast<std::int64_t>(guard.size())}),
                        guard)
                .value();
        const auto output =
            context
                .borrow_ready(output_desc, backing.native_buffer(), alignment,
                              capacity, backing.native_owner())
                .value();
        const auto program = v::prepare(context, operation, desc).value();
        auto recorder = v::Recorder::create(context).value();
        ASSERT_TRUE(
            recorder.append(program, std::array{input}, std::array{output})
                .ok());
        auto recording = recorder.finish().value();
        auto work = recording.submit().value();
        ASSERT_TRUE(work.wait().ok());
        const auto data = values(context, input);
        expect(context, output,
               operation.kind() == c::OperationKind::kReduce
                   ? reduce_reference(data, desc, reduce)
                   : probability_reference(data, desc, softmax));
        const auto actual = bytes(context, backing);
        for (std::size_t i = 0; i < actual.size(); ++i)
          if (i < alignment || i >= alignment + capacity)
            EXPECT_EQ(actual[i], guard[i]) << i;
      }
      const auto program = v::prepare(context, softmax, desc).value();
      auto rejected = v::Recorder::create(context).value();
      EXPECT_FALSE(
          rejected.append(program, std::array{input}, std::array{input}).ok());
    }
}

struct ReplayProbe {
  inline static ReplayProbe *active = nullptr;
  std::shared_ptr<v::detail::Device> device;
  v::detail::DeviceFunctions original;
  int buffers = 0, begins = 0;
  explicit ReplayProbe(const v::Context &context)
      : device(v::detail::Access::runtime(context)->device),
        original(device->fn) {
    active = this;
    device->fn.CreateBuffer = buffer;
    device->fn.BeginCommandBuffer = begin;
  }
  ~ReplayProbe() {
    device->fn = original;
    active = nullptr;
  }
  static VKAPI_ATTR VkResult VKAPI_CALL
  buffer(VkDevice device, const VkBufferCreateInfo *info,
         const VkAllocationCallbacks *callbacks, VkBuffer *output) {
    ++active->buffers;
    return active->original.CreateBuffer(device, info, callbacks, output);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL
  begin(VkCommandBuffer command, const VkCommandBufferBeginInfo *info) {
    ++active->begins;
    return active->original.BeginCommandBuffer(command, info);
  }
};

TEST(VulkanReduction,
     StatisticsBuffersAreReusedAndIndependentResultsSurviveClose) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kBFloat16, {3, 129, 3});
  const c::Softmax operation{.axis = 1, .kind = c::SoftmaxKind::kLogSoftmax};
  auto program = v::prepare(context, operation, desc).value();
  ReplayProbe probe(context);
  auto first = program.create_invocation().value(),
       second = program.create_invocation().value();
  EXPECT_EQ(probe.buffers, 4);
  EXPECT_EQ(probe.begins, 2);
  v::Tensor retained;
  std::vector<std::uint64_t> previous;
  for (int phase = 0; phase < 3; ++phase) {
    const auto input =
        make(context, desc, pattern(*desc.element_count(), phase));
    const auto before = probe.buffers, recordings = probe.begins;
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait().value();
    EXPECT_EQ(probe.buffers, before);
    EXPECT_EQ(probe.begins, recordings);
    expect(context, result.get()->tensors()[0],
           probability_reference(values(context, input), desc, operation));
    if (phase == 0) {
      retained = result.get()->tensors()[0];
      previous = bits(context, retained);
    } else
      EXPECT_EQ(bits(context, retained), previous);
  }
  first = {};
  second = {};
  program = {};
  EXPECT_TRUE(context.close().ok());
  EXPECT_EQ(bits(context, retained), previous);
}
} // namespace
