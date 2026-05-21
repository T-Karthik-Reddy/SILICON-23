# Define minimal required version of CMake.
cmake_minimum_required(VERSION "3.25")

# Project definition
project(
	ml_audio_classifier_2
	VERSION 1.0
	LANGUAGES C CXX ASM
)

# Include the definition of the slc_ml_audio_classifier_2 target,
# which contains the content of the SLC project
include(ml_audio_classifier_2.cmake)

add_executable(ml_audio_classifier_2
    # Add additional sources here
)

target_include_directories(ml_audio_classifier_2 PUBLIC
    # Add additional include paths here
)

target_compile_definitions(ml_audio_classifier_2 PUBLIC
    # Add additional macros here
)

target_compile_options(ml_audio_classifier_2 PUBLIC
    # Set additional compiler flags here
)

target_link_options(ml_audio_classifier_2 PUBLIC
    # Set additional linker flags here
)

# Link with the content defined in the SLC project
target_link_libraries(ml_audio_classifier_2 PRIVATE
    slc
)

# Include managed project content if available
include(ml_audio_classifier_2_project.cmake OPTIONAL RESULT_VARIABLE managed_project)
if(managed_project)
    message(STATUS "Using managed project content from ${managed_project}")
endif()

# Force the gcc linker command
set_target_properties(ml_audio_classifier_2 PROPERTIES LINKER_LANGUAGE C)

# Create .bin, .hex and .s37 artifacts after building the project
add_custom_command(TARGET ml_audio_classifier_2
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_SREC_CMD} "$<TARGET_FILE:ml_audio_classifier_2>" "$<TARGET_FILE_DIR:ml_audio_classifier_2>/$<TARGET_FILE_BASE_NAME:ml_audio_classifier_2>.s37"
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_IHEX_CMD} "$<TARGET_FILE:ml_audio_classifier_2>" "$<TARGET_FILE_DIR:ml_audio_classifier_2>/$<TARGET_FILE_BASE_NAME:ml_audio_classifier_2>.hex"
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_BIN_CMD}  "$<TARGET_FILE:ml_audio_classifier_2>" "$<TARGET_FILE_DIR:ml_audio_classifier_2>/$<TARGET_FILE_BASE_NAME:ml_audio_classifier_2>.bin" 
)

# Run post-build pipeline to perform additional post-processing
if(post_build_command)
add_custom_command(TARGET ml_audio_classifier_2
    POST_BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    COMMAND ${post_build_command}
)
endif()