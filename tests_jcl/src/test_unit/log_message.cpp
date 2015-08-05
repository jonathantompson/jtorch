//
//  log_message.cpp
//
//  Created by Alberto Lerner, then edited by Jonathan Tompson on 4/26/12.
//

#include <stdlib.h>  // exit()
#include <iomanip>

#include "test_unit/log_message.h"
#include "test_unit/log_writer.h"

bool TESTS_exit_on_fatal = true;
bool TESTS_has_fatal_message = false;

namespace tests {

  LogMessage::LogMessage(const char* file, const int line, Severity severity)
    : severity_(severity) {
      char labels[MAX_SEVERITY] = { ' ', 'W', 'E', 'F' };
      msg_stream_ << labels[severity_] << " " << file << ":" << line << " ";
  }

  LogMessage::~LogMessage() {
    flush();
    if (severity_ == FATAL) {
      if (TESTS_exit_on_fatal) {
        exit(1);
      } else {
        TESTS_has_fatal_message = true;
      }
    }
  }

  void LogMessage::flush() {
    msg_stream_ << std::endl;
    LogWriter::instance()->write(msg_stream_.str());
  }

  ostream& LogMessage::stream() {
    return msg_stream_;
  }
}  // namespace tests
