# CMakeLists.txt — AdaptiLLM native build with llama.cpp
#
# Builds the JNI bridge library 'llamabridge' which connects
# LlamaCppBridge.kt to the real llama.cpp inference runtime.

cmake_minimum_required(VERSION 3.22.1)
project("llamabridge" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# --------------------------------------------------------------------------
# llama.cpp build options for Android
# --------------------------------------------------------------------------
set(GGML_OPENMP    OFF CACHE BOOL "" FORCE)  # Not available on Android NDK
set(GGML_CUDA      OFF CACHE BOOL "" FORCE)  # No CUDA on mobile
set(GGML_VULKAN    OFF CACHE BOOL "" FORCE)  # Disable for now
set(GGML_METAL     OFF CACHE BOOL "" FORCE)  # No Metal on Android
set(GGML_HIPBLAS   OFF CACHE BOOL "" FORCE)
set(GGML_SYCL      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

# Include llama.cpp as a subdirectory — it handles all ggml deps internally
add_subdirectory(llama.cpp)

# --------------------------------------------------------------------------
# JNI bridge library
# --------------------------------------------------------------------------
add_library(llamabridge SHARED
    llama_interface.cpp
)

target_include_directories(llamabridge PRIVATE
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/common
)

target_compile_definitions(llamabridge PRIVATE LLAMA_AVAILABLE)

target_link_libraries(llamabridge
    llama
    android
    log
)
