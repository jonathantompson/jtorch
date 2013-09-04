project(jtorch)
if(DEFINED root_VERBOSE_MAKEFILE)
  set(CMAKE_VERBOSE_MAKEFILE root_VERBOSE_MAKEFILE)
else()
  set(CMAKE_VERBOSE_MAKEFILE true) # Default to verbose
endif()

if(NOT DEFINED BUILD)
  if(DEFINED CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE MATCHES DEBUG OR CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES debug))
    set(BUILD debug)
  else()
    set(BUILD release) # default to release
  endif()
endif()

cmake_minimum_required(VERSION 2.8.10)

message("*************************************************")
message("*************** JTORCH CMAKELISTS ***************")
message("*************************************************")

if(BUILD MATCHES debug)
  message("cmake compilation is in debug mode, target will be jtorch_d") 
  set(TARGET_NAME jtorch_d)
else()
  message("cmake compilation is in release mode, target will be jtorch")
  set(TARGET_NAME jtorch)
endif()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# CMAKE MODULES PATH
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# EXTRA LIBS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
if(APPLE)
    message("Detected Apple")
    SET(EXTRA_LIBS "-framework Cocoa -framework OpenCL -framework IOKit")
elseif(MSVC)
    message( FATAL_ERROR, "Windows cmake is not supported.  Use visual studio." )
endif(APPLE)

set(LIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../lib/MAC_OS_X)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# INCLUDE DIRECTORIES
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../include/MAC_OS_X)

set(JTIL_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../PRenderer2/jtil/include/)
set(JCL_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../PRenderer2/jcl/include/)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# COMPILER FLAGS AND PREDEFINES
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# check the c and C++ compiler versions
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
if (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
    message(STATUS "gcc version >= 4.7")
else()
    message( FATAL_ERROR, "GCC version 4.7 or greater is required, use 'export CC=path/to/gcc-mp-4.9'" )
    message( "on mac you need 'sudo port install gcc49' and add 'export CXX=/opt/local/bin/g++-mp-4.9'")
endif()

execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GXX_VERSION)
if (GXX_VERSION VERSION_GREATER 4.7 OR GXX_VERSION VERSION_EQUAL 4.7)
    message(STATUS "g++ Version >= 4.7")
else()
    message( FATAL_ERROR, "GCC version 4.7 or greater is required" )
    message( "on mac you need 'sudo port install gcc49' and add 'export CC=/opt/local/bin/gcc-mp-4.9'")
endif()

# -g adds debugging symbols
# -Wall turns on all warnings
# -Wextra turns on a lot of warnings (but not too pedantic)
add_definitions(-DGLFW_INCLUDE_GLCOREARB)
add_definitions(-DASSIMP_BUILD_BOOST_WORKAROUND)

# Use: "cmake -DCMAKE_BUILD_TYPE=Debug" for debug
if(BUILD MATCHES debug)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -g -Wextra -Wno-ignored-qualifiers -Wno-unused-parameter -std=c++11")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -g -Wextra -Wno-ignored-qualifiers -Wno-unused-parameter")
    message(STATUS "adding debug flags")
    add_definitions(-DDEBUG)
    add_definitions(-D_DEBUG)
    add_definitions(-DBREAK_ON_EXCEPTION)
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -g -Wextra -Wno-ignored-qualifiers -Wno-unused-parameter -std=c++11 -O3 -msse -msse2")
    SET(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS_RELEASE} -fopenmp -g -Wextra -Wno-ignored-qualifiers -Wno-unused-parameter -O3 -msse -msse2")
    message(STATUS "adding release flags")
endif()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# JTORCH SOURCE AND HEADERS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
set(ROOT_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set(ROOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# JTORCH TARGET
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

file(GLOB_RECURSE JTORCH_CXX_SOURCE ${ROOT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE JTORCH_CC_SOURCE ${ROOT_SOURCE_DIR}/*.c)
file(GLOB_RECURSE JTORCH_HEADER ${ROOT_HEADER_DIR}/*.h)

if(root_PRINT_SOURCE_FILES MATCHES true)
    message( "SOURCE C++ FILES:" )
    message( STATUS ${JTORCH_CXX_SOURCE} )
    message( "SOURCE C FILES:" )
    message( STATUS ${JTORCH_CC_SOURCE} )
    message( "HEADER FILES:" )
    message( STATUS ${JTORCH_HEADER} )
endif()

message( "INCLUDE DIRECTORIES:" )
message( STATUS "jtil: " ${JTIL_INC_DIR} )
message( STATUS "jcl: " ${JCL_INC_DIR} )

include_directories(
    ${ROOT_HEADER_DIR} 
    ${JTIL_INC_DIR}
    ${JCL_INC_DIR}
)

message( "EXTRA_LIBS to be linked in: " ${EXTRA_LIBS} )

add_library(${TARGET_NAME} STATIC ${JTORCH_CXX_SOURCE} ${JTORCH_CC_SOURCE} ${JTORCH_HEADER})

if(NOT DEFINED jtil)
  set(BUILD_JTIL_TESTS false)
  add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/../jtil/" "${CMAKE_CURRENT_BINARY_DIR}/jtil")
endif()
if(NOT DEFINED jcl)
  SET(BUILD_JCL_TESTS false)
  add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/../jcl/" "${CMAKE_CURRENT_BINARY_DIR}/jcl")
endif()

target_link_libraries(${TARGET_NAME} ${EXTRA_LIBS})

if(BUILD MATCHES debug)
    target_link_libraries(${TARGET_NAME} jtil_d)
    target_link_libraries(${TARGET_NAME} jcl_d)
else()
    target_link_libraries(${TARGET_NAME} jtil)
    target_link_libraries(${TARGET_NAME} jcl)
endif()

if(DEFINED BUILD_JTORCH_TESTS AND (BUILD_JTORCH_TESTS MATCHES false))
  message("Not building the jtorch tests...")
else()
  message("building the jtorch tests...")
  add_subdirectory(tests_jtorch) # default to building the tests
endif()



