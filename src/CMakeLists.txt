aux_source_directory(engine ENGINE_SRC_FILES)
aux_source_directory(node NODE_SRC_FILES)
aux_source_directory(pipeline PIPELINE_SRC_FILES)
aux_source_directory(tools TOOLS_SRC_FILES)
aux_source_directory(signal SIGNAL_SRC_FILES)
aux_source_directory(model MODEL_SRC_FILES)

target_sources(${LIB_CVINFER} PRIVATE ${ENGINE_SRC_FILES} ${NODE_SRC_FILES} ${PIPELINE_SRC_FILES} ${TOOLS_SRC_FILES} ${SIGNAL_SRC_FILES} ${MODEL_SRC_FILES})
target_include_directories(${LIB_CVINFER} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})