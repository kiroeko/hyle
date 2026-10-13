#if defined(HYLE_NATIVE_MEMORY_APPLE)
#include <hyle/compute/apple_memory.h>
namespace native=hyle::compute::apple;
#else
#include <hyle/compute/vulkan_memory.h>
namespace native=hyle::compute::vulkan;
#endif
#include <type_traits>
#include <vector>

static_assert(std::is_nothrow_default_constructible_v<native::MemoryResource>);
static_assert(std::is_nothrow_move_constructible_v<native::MemoryResource>);

int main(){
  native::MemoryResource empty;
  if(empty.valid() || empty.ref().valid())return 1;
  std::vector<native::MemoryResource> resources;
  resources.resize(3);
  for(const auto &resource:resources)if(resource.valid())return 2;
  auto made=native::Context::create();if(!made)return 3;
  auto context=std::move(*made);
#if defined(HYLE_NATIVE_MEMORY_APPLE)
  auto created=native::MemoryResource::create(context);
#else
  auto created=native::MemoryResource::create(context,native::MemoryKind::kDevice);
#endif
  if(!created)return 4;
  empty=std::move(*created);if(!empty.valid() || created->valid())return 5;
  resources[1]=std::move(empty);if(empty.valid() || !resources[1].valid())return 6;
  resources.clear();
  return context.close().ok()?0:7;
}
