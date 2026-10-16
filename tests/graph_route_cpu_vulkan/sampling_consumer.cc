#include "../compute_models/sampling_native_joint_consumer.h"

int main() {
  try {sampling_test::native_joint();return 0;}
  catch(const std::exception &error) {
    std::fprintf(stderr,"FAIL CPU+Vulkan sampling: %s\n",error.what());return 1;
  }
}
