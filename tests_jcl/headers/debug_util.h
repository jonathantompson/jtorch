//
//  debug_util.h
//
//  Created by Jonathan Tompson on 6/2/12.
//
//  A layer for interfacing with visual studio and XCode's debug utilities
//  
//  NOTE: This header should be included last
//
//  ****** Originally from my jtil library (but pulled out for jcl to reduce
//  compilation dependencies).  ******
//

#pragma once

#ifdef _DEBUG
  #ifdef _WIN32
    #include <Windows.h>
    #include <crtdbg.h>  // for _CrtSetDbgFlag
  #endif
  #ifdef _APPLE__
    #error Debug Utilities are not yet implemented for Mac OS X 
  #endif
#endif

namespace jcl {
namespace debug {

  // EnableMemoryLeakChecks - On exit report memory leaks, should be the first
  // line in main() function.
  void EnableMemoryLeakChecks();

  void EnableAggressiveMemoryLeakChecks();  // Check on EVERY allocation

  // SetBreakPointOnAlocation - Breakpoint once the allocation count hits 
  // alloc_num.  Set in main() function after EnableMemoryLeakChecks()
  void SetBreakPointOnAlocation(int alloc_num);

};  // namespace debug
};  // namespace jcl
