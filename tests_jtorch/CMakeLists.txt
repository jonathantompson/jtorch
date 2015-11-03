PROJECT(tests_jtorch)
IF(DEFINED root_VERBOSE_MAKEFILE)
  SET(CMAKE_VERBOSE_MAKEFILE root_VERBOSE_MAKEFILE)
else()
  SET(CMAKE_VERBOSE_MAKEFILE false) # Default to verbose
ENDIF()

IF(NOT DEFINED BUILD)
  IF(DEFINED CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE MATCHES DEBUG OR CMAKE_BUILD_TYPE MATCHES DebugOR CMAKE_BUILD_TYPE MATCHES debug))
    SET(BUILD debug)
  else()
    SET(BUILD release) # default to release
  ENDIF()
ENDIF()

cmake_minimum_required(VERSION 2.8.10)

MESSAGE("*************************************************")
MESSAGE("************ TESTS_JTORCH CMAKELISTS ************")
MESSAGE("*************************************************")

IF(BUILD MATCHES debug)
  MESSAGE("cmake compilation is in debug mode, target will be tests_jtorch_d") 
  SET(TARGET_NAME tests_jtorch_d)
else()
  MESSAGE("cmake compilation is in release mode, target will be tests_jtorch")
  SET(TARGET_NAME tests_jtorch)
ENDIF()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# EXTRA LIBS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
IF (WIN32)
  # nothing to do.
ELSE()
  FIND_PACKAGE(Threads)
ENDIF()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# COMPILER FLAGS AND PREDEFINES
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Use: "cmake -DCMAKE_BUILD_TYPE=Debug" for debug
IF(BUILD MATCHES debug)
  MESSAGE(STATUS "adding debug flags")
  IF (WIN32)
    # nothing to do.
  ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -g -Wextra -Wno-ignored-qualifiers -Wno-unused-parameter -pthread -std=c++11 -Wl,--no-as-needed")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -g -Wextra -Wno-ignored-qualifiers -Wno-unused-parameter -pthread")
  ENDIF()
  ADD_DEFINITIONS(-DDEBUG)
  ADD_DEFINITIONS(-D_DEBUG)
  ADD_DEFINITIONS(-DBREAK_ON_EXCEPTION)
ELSE()
  MESSAGE(STATUS "adding release flags")
  IF (WIN32)
    # nothing to do.
  ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -g -Wextra -Wno-ignored-qualifiers -Wno-unused-parameter -pthread -std=c++11 -O3 -msse -msse2 -Wl,--no-as-needed")
    SET(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS_RELEASE} -fopenmp -g -Wextra -Wno-ignored-qualifiers -Wno-unused-parameter -O3 -msse -msse2 -pthread")
  ENDIF()
ENDIF()
IF (WIN32)
  # nothing to do
ELSE()
  SET(CMAKE_EXE_LINKER_FLAGS "-pthread")
ENDIF()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# TESTS_JTORCH SOURCE AND HEADERS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
SET(ROOT_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/headers/")
SET(ROOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# TESTS_JTORCH TARGET
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

file(GLOB_RECURSE TESTS_JTORCH_CXX_SOURCE ${ROOT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE TESTS_JTORCH_CC_SOURCE ${ROOT_SOURCE_DIR}/*.c)
file(GLOB_RECURSE TESTS_JTORCH_HEADER ${ROOT_HEADER_DIR}/*.h)

MESSAGE( "C++ SOURCE FILES:" )
MESSAGE( STATUS ${TESTS_JTORCH_CXX_SOURCE} )
MESSAGE( "C SOURCE FILES:" )
MESSAGE( STATUS ${TESTS_JTORCH_CC_SOURCE} )
MESSAGE( "HEADER FILES:" )
MESSAGE( STATUS ${TESTS_JTORCH_HEADER} )

INCLUDE_DIRECTORIES(
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/headers/"
)

ADD_EXECUTABLE(${TARGET_NAME} ${TESTS_JTORCH_CXX_SOURCE} ${TESTS_JTORCH_CC_SOURCE} ${TESTS_JTORCH_HEADER})

IF(BUILD MATCHES debug)
  TARGET_LINK_LIBRARIES(${TARGET_NAME} jtorch_d)
else()
  TARGET_LINK_LIBRARIES(${TARGET_NAME} jtorch)
ENDIF()
TARGET_LINK_LIBRARIES(${TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT})
