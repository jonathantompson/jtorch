#if defined(WIN32) || defined(_WIN32)

#include "debug_util.h"

namespace jcl {
namespace debug {

  void EnableMemoryLeakChecks() {
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
  }

  void EnableAggressiveMemoryLeakChecks() {
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | 
      _CRTDBG_CHECK_ALWAYS_DF );
#endif
  }

  void SetBreakPointOnAlocation(int alloc_num) {
#ifdef _DEBUG
    _CrtSetBreakAlloc(alloc_num);
#endif
  }

}  // namespace debug
}  // namespace jcl

#endif  // WIN32 || _WIN32
