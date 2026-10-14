#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <utility>

namespace {
struct Object {
  unsigned refs=1;
  Object* retain(){++refs;return this;}
  void release(){--refs;}
};
struct Derived : Object {
  Derived* retain(){Object::retain();return this;}
};
bool check(bool value,const char* message){
  if(!value)std::fprintf(stderr,"SharedPtr lifetime: %s\n",message);
  return value;
}
bool lifecycle(){
  using Ptr=NS::SharedPtr<Derived>;using Base=NS::SharedPtr<Object>;
  {
    Ptr empty,explicit_null(nullptr),copy(empty),moved(std::move(copy));
    Base base_copy(empty),base_move(std::move(empty));
    empty=copy;empty=std::move(moved);base_copy=empty;base_move=std::move(empty);
    empty.reset();base_copy.reset();
    auto retained=NS::RetainPtr<Derived>(nullptr);
    if(!check(!retained && !empty && !base_move,"empty copy/move/reset/retain changed identity"))return false;
  }
  Derived object;
  {
    auto first=NS::RetainPtr(&object);
    if(!check(object.refs==2,"RetainPtr must add exactly one reference"))return false;
    Ptr copy(first);Base converted(first);
    Ptr assigned;assigned=copy;Base base_assigned;base_assigned=copy;
    if(!check(object.refs==6,"copy construction/assignment counts differ"))return false;
    Ptr moved(std::move(assigned));Base converted_move(std::move(copy));
    if(!check(object.refs==6 && !assigned && !copy,"move construction retained or lost references"))return false;
    Ptr destination;destination=std::move(moved);
    Base base_destination;base_destination=std::move(first);
    if(!check(object.refs==6 && !moved && !first,"move assignment into empty owner differs"))return false;
    auto alias=NS::RetainPtr(&object);destination=std::move(alias);
    if(!check(object.refs==6 && !alias,"same-object move assignment did not drop exactly one alias"))return false;
    auto base_alias=NS::RetainPtr(&object);base_destination=std::move(base_alias);
    if(!check(object.refs==6 && !base_alias,"converted same-object move assignment differs"))return false;
    const auto* same=&destination;destination=*same;
    if(!check(object.refs==6,"self copy changed reference count"))return false;
    destination.reset();converted.reset();converted_move.reset();base_assigned.reset();base_destination.reset();
    if(!check(object.refs==1,"reset did not release all retained references"))return false;
  }
  if(!check(object.refs==1,"empty/moved-from destruction changed original reference"))return false;
  {
    Ptr empty;Base base;
    auto owner=NS::RetainPtr(&object);owner=empty;
    auto converted=NS::RetainPtr(&object);base=std::move(converted);base=empty;
    if(!check(object.refs==1,"assigning empty copy did not release the previous nonnull owner"))return false;
    auto moved=NS::RetainPtr(&object);moved=Ptr{};
    auto base_moved=NS::RetainPtr(&object);base=std::move(base_moved);base=Ptr{};
    if(!check(object.refs==1,"assigning empty move did not release previous owner"))return false;
  }
  return check(object.refs==1,"final destructors did not preserve the external reference");
}
}
int main(){
  if(!lifecycle())return 1;
  auto* pool=NS::AutoreleasePool::alloc()->init();
  bool available=false;
  {
    NS::SharedPtr<MTL::Device> device;
    device=NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    available=bool(device);
    if(available){auto copy=device;device.reset();available=bool(copy);}
  }
  pool->release();
  if(!check(available,"actual Metal device ownership journey failed"))return 1;
  std::puts("SharedPtr empty/default/retain/copy/move/reset/destruction, reference counts and actual Metal device PASS");
  return 0;
}
