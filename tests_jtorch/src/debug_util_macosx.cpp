#ifdef __APPLE__

#include <stdio.h>  // printf
#include "debug_util.h"

namespace jcl {
namespace debug {

  void EnableMemoryLeakChecks() {
    printf("WARNING: EnableMemoryLeakChecks not implemented yet for Mac\n");
  }

  void SetBreakPointOnAlocation(int alloc_num) {
    static_cast<void>(alloc_num);
    printf("WARNING: SetBreakPointOnAlocation not implemented yet for Mac\n");
  }

}  // namespace debug
}  // namespace jcl

#endif  // __APPLE__
