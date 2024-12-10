cmake_minimum_required(VERSION 3.5)

project("COSI_Labs")

list(APPEND CMAKE_PREFIX_PATH "./build/build/Release/generators")

find_package(OpenCV REQUIRED )

set(SOURCES
    src/main.cpp
)

set(HEADERS
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${OpenCV_INCLUDE_DIRS}>
)

message("Hello:")
message(${OpenCV_LIBS})

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
