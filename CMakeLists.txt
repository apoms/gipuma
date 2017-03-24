cmake_minimum_required (VERSION 2.8)
project (gipuma)

find_package(CUDA 6.0 REQUIRED ) # For Cuda Managed Memory and c++11
find_package(OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)
set(PUBLIC_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_30,code=sm_30 -gencode arch=compute_52,code=sm_52)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -lineinfo --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_60,code=sm_60)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(HEADERS
  gipuma.h
  files.h
  imageinfo.h
  config.h
  managed.h
  helper_cuda.h
  helper_string.h
  mathUtils.h
  cameraGeometryUtils.h
  vector_operations.h
  camera.h
  globalstate.h
  algorithmparameters.h
  cameraparameters.h
  linestate.h
  groundTruthUtils.h
  displayUtils.h
  fileIoUtils.h)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_library(gipuma SHARED gipuma.cu)
cuda_add_executable(gipuma_main main.cpp)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(gipuma ${OpenCV_LIBS})
target_link_libraries(gipuma_main gipuma)

install(FILES ${HEADERS} DESTINATION include/gipuma)

install(TARGETS gipuma
  EXPORT GipumaTarget
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

set(GIPUMA_LIBRARY "${CMAKE_INSTALL_PREFIX}/lib/libgipuma.so")

install(FILES ${CONFIG_FILE} DESTINATION lib/cmake)

set(CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/GipumaConfig.cmake")
configure_file(${PROJECT_SOURCE_DIR}/cmake/GipumaConfig.cmake.in
  ${CONFIG_FILE} @ONLY)
install(FILES ${CONFIG_FILE} DESTINATION lib/cmake)
install(EXPORT GipumaTarget
  DESTINATION lib/cmake)
