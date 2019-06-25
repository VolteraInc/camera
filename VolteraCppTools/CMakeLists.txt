CMAKE_MINIMUM_REQUIRED(VERSION 3.0 )
PROJECT(VolteraCppTools)

set (CMAKE_CXX_STANDARD 14)

#Setup PyBind11
add_subdirectory(${PROJECT_SOURCE_DIR}/libs/pybind11)
#include_directories("${PROJECT_SOURCE_DIR}/libs/pybind11/include")
set(PYBIND11_CPP_STANDARD -std=c++14)


#Setup paths
set(LIB_PATH                                   "${PROJECT_BINARY_DIR}/lib")
set(LIB_NAME                                   "${PROJECT_NAME}")
set(BIN_PATH                                   "${PROJECT_BINARY_DIR}/bin")
set(INTRINSICS_MAIN_NAME                       "IntrinsicsFitter")
set(TEST_PATH                                  "${PROJECT_BINARY_DIR}/test")
set ( TEST_NAME ${LIB_NAME}_tests)

set (BUILD_MAIN ON)
set (BUILD_TESTS ON)
set (BUILD_PYTHON ON)
set (PYBIND11_PYTHON_VERSION 3.5)

#Source File Listing
SET ( SRC_DIR "${PROJECT_SOURCE_DIR}/src" )
SET ( TEST_SRC_DIR "${PROJECT_SOURCE_DIR}/test" )
SET ( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

set(LIBRARY_SRC_FILES
  "${SRC_DIR}/LoadCSV.cpp"
  "${SRC_DIR}/Intrinsics.cpp"
  "${SRC_DIR}/PreviewResiduals.cpp"
  "${SRC_DIR}/LaserExtrinsics.cpp"
  "${SRC_DIR}/AnalysisImage.cpp"
)

set (HEADERS
  "${INCLUDE_DIR}/VolteraCppTools/LoadCSV.h"
  "${INCLUDE_DIR}/VolteraCppTools/ReprojectionError.h"
  "${INCLUDE_DIR}/VolteraCppTools/Intrinsics.h"
  "${INCLUDE_DIR}/VolteraCppTools/PreviewResiduals.h"
  "${INCLUDE_DIR}/VolteraCppTools/LaserExtrinsics.h"
  "${INCLUDE_DIR}/VolteraCppTools/LaserReprojectionError.h"
  "${INCLUDE_DIR}/VolteraCppTools/AnalysisImage.h"

)

set (PYTHON_BINDING_FILES
  "${SRC_DIR}/Intrinsics.cpp"
  "${SRC_DIR}/LaserExtrinsics.cpp"
  "${SRC_DIR}/VolteraCppToolsPythonBindings.cpp"
)

set(INTRINSICS_MAIN_SRC
  "${SRC_DIR}/intrinsics_fitting_main.cpp"
)

SET (TEST_DATA_DIR "${TEST_SRC_DIR}/test_data/")
CONFIGURE_FILE ("${TEST_SRC_DIR}/test_path.h.in" "${TEST_SRC_DIR}/test_path.h")
set(TEST_SRC
  "${TEST_SRC_DIR}/testLoadCSV.cpp"
  "${TEST_SRC_DIR}/testLineFinder.cpp"
)

#External Libraries
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )
find_package (Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
find_package (OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( include )

if (BUILD_PYTHON)
#Set up python library
pybind11_add_module( ${LIB_NAME}_Python ${PYTHON_BINDING_FILES})
TARGET_LINK_LIBRARIES (${LIB_NAME}_Python PUBLIC Eigen3::Eigen ${CERES_LIBRARIES} ${OpenCV_LIBS})
endif()

if (BUILD_MAIN OR BUILD_TESTS)
#Set up library and executables
ADD_LIBRARY(${LIB_NAME} ${LIBRARY_SRC_FILES} ${HEADERS})
TARGET_LINK_LIBRARIES (${LIB_NAME} Eigen3::Eigen ${CERES_LIBRARIES} ${OpenCV_LIBS})
endif()

if (BUILD_MAIN)
ADD_EXECUTABLE ( ${INTRINSICS_MAIN_NAME} ${INTRINSICS_MAIN_SRC})
TARGET_LINK_LIBRARIES ( ${INTRINSICS_MAIN_NAME} ${LIB_NAME})
endif()

if (BUILD_TESTS)
ADD_EXECUTABLE(${TEST_NAME} ${TEST_SRC})
target_link_libraries(${TEST_NAME} ${LIB_NAME} gtest_main)
add_test(NAME ${TEST_NAME} COMMAND "${TEST_PATH}/${TEST_NAME}")


#Set up gtest
if ( NOT TARGET gtest )

# Download and unpack googletest at configure time
CONFIGURE_FILE(GoogleTest.txt.in googletest-download/CMakeLists.txt)
EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/googletest-download )
IF(result)
  MESSAGE(FATAL_ERROR "CMake step for googletest failed: ${result}")
ENDIF()
EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/googletest-download )
IF(result)
  MESSAGE(FATAL_ERROR "Build step for googletest failed: ${result}")
ENDIF()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
ADD_SUBDIRECTORY(${CMAKE_BINARY_DIR}/googletest-src
               ${CMAKE_BINARY_DIR}/googletest-build
               EXCLUDE_FROM_ALL)

endif()
endif()