PROJECT(jtorch)
IF(DEFINED root_VERBOSE_MAKEFILE)
  SET(CMAKE_VERBOSE_MAKEFILE root_VERBOSE_MAKEFILE)
ELSE()
  SET(CMAKE_VERBOSE_MAKEFILE false) # Default to verbose
ENDIF()

IF(NOT DEFINED BUILD)
  IF(DEFINED CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE MATCHES DEBUG OR CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES debug))
    SET(BUILD debug)
  ELSE()
    SET(BUILD release) # default to release
  ENDIF()
ENDIF()

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.10)

MESSAGE("*******************************************************")
MESSAGE("*************** JTORCH + JCL CMAKELISTS ***************")
MESSAGE("*******************************************************")

IF(BUILD MATCHES debug)
  MESSAGE("cmake compilation is in debug mode, target will be jtorch_d") 
  SET(TARGET_NAME jtorch_d)
ELSE()
  MESSAGE("cmake compilation is in release mode, target will be jtorch")
  SET(TARGET_NAME jtorch)
ENDIF()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Extra libraries.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")
IF (WIN32)
  # nothing to do.
ELSE()
  FIND_PACKAGE(Threads)
ENDIF()
FIND_PACKAGE(OpenCL REQUIRED)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# COMPILER FLAGS AND PREDEFINES
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# -g adds debugging symbols
# -Wall turns on all warnings
# -Wextra turns on a lot of warnings (but not too pedantic)

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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# JTORCH + JCL TARGET
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

FILE(GLOB_RECURSE JTORCH_CXX_SOURCE ${CMAKE_SOURCE_DIR}/src/*.cpp)
FILE(GLOB_RECURSE JTORCH_CC_SOURCE ${CMAKE_SOURCE_DIR}/src/*.c)
FILE(GLOB_RECURSE JTORCH_HEADER ${CMAKE_SOURCE_DIR}/include/*.h)

MESSAGE( "SOURCE C++ FILES:" )
MESSAGE( STATUS ${JTORCH_CXX_SOURCE} )
MESSAGE( "SOURCE C FILES:" )
MESSAGE( STATUS ${JTORCH_CC_SOURCE} )
MESSAGE( "HEADER FILES:" )
MESSAGE( STATUS ${JTORCH_HEADER} )

ADD_LIBRARY(${TARGET_NAME} STATIC ${JTORCH_CXX_SOURCE} ${JTORCH_CC_SOURCE} ${JTORCH_HEADER})

INCLUDE_DIRECTORIES(
  "${CMAKE_SOURCE_DIR}/include/" 
  ${OPENCL_INCLUDE_DIRS}
  "${CMAKE_SOURCE_DIR}/../clBLAS/src/"
  "${CMAKE_SOURCE_DIR}/opencl_cpp_header/"
)

LINK_DIRECTORIES("${CMAKE_SOURCE_DIR}/../clBLAS/src/build/library/" "${CMAKE_SOURCE_DIR}/../clBLAS/build/staging/Release")

TARGET_LINK_LIBRARIES(
  ${TARGET_NAME}
  ${OPENCL_LIBRARIES}
  clBLAS
  ${CMAKE_THREAD_LIBS_INIT}
)

IF (WIN32)
  # nothing to do.
ELSE()
  TARGET_LINK_LIBRARIES(${TARGET_NAME} pthread)
ENDIF()

IF(DEFINED BUILD_JCL_TESTS AND (BUILD_JCL_TESTS MATCHES false))
  MESSAGE("Not building the jcl tests...")
ELSE()
  MESSAGE("building the jcl tests...")
  ADD_SUBDIRECTORY(tests_jcl) # default to building the tests
ENDIF()

IF(DEFINED BUILD_JTORCH_TESTS AND (BUILD_JTORCH_TESTS MATCHES false))
  MESSAGE("Not building the jtorch tests...")
ELSE()
  MESSAGE("building the jtorch tests...")
  ADD_SUBDIRECTORY(tests_jtorch) # default to building the tests
ENDIF()

