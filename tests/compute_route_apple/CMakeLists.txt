hyle_add_header_self_contained_test(compute_route_apple
  HEADER_DIR compute HEADERS apple.h)
if(HYLE_COMPUTE_APPLE_ENABLED)
  hyle_add_test(compute_route_apple SOURCES context_test.cc)
endif()
