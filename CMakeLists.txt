# CMakeList.txt : CMake project for MANNI, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("MANNI")

# Add source to this project's executable.
add_executable (MANNI "MANNI.cpp" "MANNI.h")

# Add include directory
include_directories("${PROJECT_SOURCE_DIR}/include")

# Link with OpenBLAS library
target_link_libraries(MANNI PRIVATE "${PROJECT_SOURCE_DIR}/lib/OpenBLAS/lib/libopenblas.dll.a")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET MANNI PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.

