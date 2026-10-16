#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <hyle/compute/vulkan.h>

#include "../compute_models/sampling_math_fixture.h"

TEST(VulkanSamplingMath, ActualProductionTransformAtOpen64EndpointsAndCorpus) {
  namespace t=sampling_test;namespace v=hyle::compute::vulkan;namespace c=hyle::compute;
  bool validation=false;
#if defined(_WIN32)
  char *environment=nullptr;std::size_t size=0;
  if(_dupenv_s(&environment,&size,"HYLE_VULKAN_VALIDATION")==0&&environment)
    validation=environment[0]=='1';
  std::free(environment);
#else
  if(const auto *environment=std::getenv("HYLE_VULKAN_VALIDATION"))validation=environment[0]=='1';
#endif
  auto context=t::take(v::Context::create({.validation=validation}));
  std::ifstream file(HYLE_G08_VULKAN_MATH_SHADER,std::ios::binary|std::ios::ate);
  ASSERT_TRUE(file);const auto bytes=file.tellg();ASSERT_GT(bytes,0);ASSERT_EQ(bytes%4,0);
  std::vector<std::uint32_t> spirv(std::size_t(bytes)/4);file.seekg(0);
  file.read(reinterpret_cast<char *>(spirv.data()),std::streamsize(bytes));ASSERT_TRUE(file);
  const auto words=t::endpoint_words();const auto count=std::uint32_t(words.size()/2);
  const auto input=t::take(context.upload(t::dense(c::DataType::kUInt32,{count,2}),std::as_bytes(std::span(words))));
  const auto output=t::take(context.allocate(t::dense(c::DataType::kFloat32,{count})));
  const auto kernel=t::take(v::Kernel::create(context,{.code=spirv,.storage_bindings=2,.push_constant_bytes=4}));
  auto recorder=t::take(v::Recorder::create(context));
  const std::array bindings{input,output};
  t::require(recorder.dispatch(kernel,bindings,{(count+127)/128,1,1},std::as_bytes(std::span(&count,1))));
  auto program=t::take(recorder.finish());auto work=t::take(program.submit());t::require(work.wait());
  std::vector<float> actual(count);t::require(context.download(output,std::as_writable_bytes(std::span(actual))));
  EXPECT_NO_THROW(t::check_endpoints(words,actual));
  t::require(context.close());
}
