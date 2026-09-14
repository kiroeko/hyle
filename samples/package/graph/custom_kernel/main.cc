#include "kernel.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>

#include <hyle/compute/backends/cpu.h>
#include <hyle/graph/executable.h>
#include <hyle/mem/buffer.h>

int main() {
  using namespace hyle::compute;
  auto provider = example::make_provider();
  if (!provider)
    return 1;
  auto context = Context::create(cpu::device(), std::move(*provider));
  if (!context)
    return 2;
  auto queue = context->create_queue();
  auto desc =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{4});
  auto source = hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64);
  auto target = hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64);
  if (!queue || !desc || !source || !target)
    return 3;
  const std::array<float, 4> values{1, 2, 3, 4};
  if (!source->view().write(std::as_bytes(std::span(values))).ok())
    return 4;
  auto input = ConstTensorView::create(*desc, source->view());
  auto output = TensorView::create(*desc, target->view());
  if (!input || !output)
    return 5;
  const auto operation = Operation::custom(example::Affine{3});
  auto prepared =
      prepare(*context, operation, std::array{*input}, std::array{*output});
  if (!prepared) {
    const auto message = prepared.error().message();
    std::fprintf(stderr, "custom prepare: %.*s\n",
                 static_cast<int>(message.size()), message.data());
    return 6;
  }
  auto invocation = prepared->create_invocation();
  if (!invocation)
    return 7;
  auto done = invocation->submit(*queue, *input, *output);
  if (!done || !done->wait().ok())
    return 8;
  std::array<float, 4> actual{};
  if (!target->view().read(std::as_writable_bytes(std::span(actual))).ok() ||
      actual != std::array<float, 4>{5, 7, 9, 11})
    return 9;

  auto builder = hyle::graph::GraphBuilder::create();
  if (!builder)
    return 10;
  auto in = builder->input(*desc);
  if (!in)
    return 11;
  auto out = builder->add(operation, *in);
  if (!out || !builder->publish(*out).ok())
    return 12;
  auto graph = std::move(*builder).finish();
  if (!graph)
    return 13;
  auto compiled = hyle::graph::compile(*graph, *context);
  if (!compiled)
    return 14;
  auto run = compiled->executable.create_invocation();
  if (!run || !run->bind(*in, *input).ok() || !run->bind(*out, *output).ok())
    return 15;
  compiled->executable = {}; // Invocation retains the kernel/factory owner.
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto complete = run->submit(*queue);
    if (!complete || !complete->wait().ok())
      return 16;
    if (!target->view().read(std::as_writable_bytes(std::span(actual))).ok() ||
        actual != std::array<float, 4>{5, 7, 9, 11})
      return 17;
  }
  return 0;
}
