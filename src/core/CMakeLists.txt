hyle_add_module(core SOURCES version.cc c_core.cc)

# The checked-in public version definitions are usable without a configured
# build tree. The root reads project(... VERSION ...) from this same header;
# compile-time assertions in version.cc keep the runtime compilation path
# honest without making public headers depend on generated/install-only files.
target_compile_definitions(hyle_core PRIVATE
  HYLE_CMAKE_PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  HYLE_CMAKE_PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  HYLE_CMAKE_PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})
