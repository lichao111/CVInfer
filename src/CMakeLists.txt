file(GLOB_RECURSE ENGINE_SRC_FILES engine/*cpp)
file(GLOB_RECURSE CUDA_SRC engine/*cu)
aux_source_directory(node NODE_SRC_FILES)
aux_source_directory(pipeline PIPELINE_SRC_FILES)
aux_source_directory(tools TOOLS_SRC_FILES)
aux_source_directory(signal SIGNAL_SRC_FILES)
aux_source_directory(model MODEL_SRC_FILES)

find_package(CUDA REQUIRED)

message(STATUS "ENGINE_SRC_FILES: ${ENGINE_SRC_FILES}")
cuda_add_library(plugin_list SHARED ${CUDA_SRC})
target_link_libraries(plugin_list ${CUDA_LIBRARIES})
target_include_directories(plugin_list PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_sources(${LIB_CVINFER} PRIVATE ${ENGINE_SRC_FILES} ${NODE_SRC_FILES} ${PIPELINE_SRC_FILES} ${TOOLS_SRC_FILES} ${SIGNAL_SRC_FILES} ${MODEL_SRC_FILES})
target_include_directories(${LIB_CVINFER} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
