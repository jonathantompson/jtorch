#include <sstream>
#include <string>
#include "file_io.h"
#if defined(WIN32) || defined(_WIN32)
#include <Windows.h>
#endif

namespace jcl {
namespace file_io {
bool fileExists(const std::string& filename) {
  // TODO: This is a pretty stupid way to check if a file exists.  I think
  // opening a file handler is probably slow.  Rethink this.
  std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
  bool ret_val = false;
  if (file.is_open()) {
    ret_val = true;
    file.close();
  }
  return ret_val;
}

}  // namespace file_io
}  // namespace jtil
