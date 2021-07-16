cmake_minimum_required(VERSION 3.1)

project(bm_tracker VERSION 3.9.1 LANGUAGES CXX)

add_library(bm_tracker INTERFACE) 

option(BM_TRACKER_FEATURES "Enable support for feature tracking controlled by C_Color (requires OpenCV)." ON)
option(BM_TRACKER_JSON "Enable support for config loading from JSON." ON)
option(BM_TRACKER_DOWNLOAD_JSON "Download nlohmann::json includes from repository." ON)

if (BM_TRACKER_JSON AND BM_TRACKER_DOWNLOAD_JSON)

    include(FetchContent)

    FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
    GIT_TAG v3.9.1)

    FetchContent_GetProperties(json)
    if(NOT json_POPULATED)
        FetchContent_Populate(json)
        add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    target_link_libraries(bm_tracker INTERFACE nlohmann_json::nlohmann_json)
endif()
if (NOT BM_TRACKER_JSON)
    target_compile_definitions( bm_tracker INTERFACE BM_TRACKER_DISABLE_JSON )
endif ()

if (BM_TRACKER_FEATURES)
    find_package( OpenCV REQUIRED )
    
    target_include_directories(bm_tracker INTERFACE ${OpenCV_INCLUDE_DIRS} )
    target_link_libraries(bm_tracker INTERFACE ${OpenCV_LIBS})
else()
    target_compile_definitions( bm_tracker INTERFACE BM_TRACKER_DISABLE_FEATURES )
endif ()

target_include_directories(bm_tracker INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include" )