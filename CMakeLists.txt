cmake_minimum_required(VERSION 3.8)

project("MANNI")

# Add source to this project's executable.
add_executable(MANNI "MANNI.cpp" "MANNI.h")

# Add include directory
include_directories("${PROJECT_SOURCE_DIR}/include")

if(WIN32)
  # Link with OpenBLAS library
  target_link_libraries(MANNI PRIVATE "${PROJECT_SOURCE_DIR}/lib/OpenBLAS/lib/libopenblas.dll.a")
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(MANNI PRIVATE "${PROJECT_SOURCE_DIR}/lib/linux/OpenBLAS-0.3.23/libopenblas.a")
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET MANNI PROPERTY CXX_STANDARD 20)
endif()
