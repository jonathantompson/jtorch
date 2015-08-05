//
//  log_writer.cpp
//
//  Created by Alberto Lerner, then edited by Jonathan Tompson on 4/26/12.
//

#include <string>  // for string
#include "test_unit/log_writer.h"

namespace tests {

  // LogWriter static state
  bool LogWriter::init_control_ = true;
  LogWriter*     LogWriter::instance_ = NULL;

  LogWriter::LogWriter() : log_file_(NULL) {}

  LogWriter::~LogWriter() {
    fclose(log_file_);
  }

  void LogWriter::init() {
    instance_ = new LogWriter;
  }

  LogWriter* LogWriter::instance() {
    // pthread_once(&init_control_, &LogWriter::init);
    if (init_control_) {
      init();
      init_control_ = false;      
    }
    return instance_;
  }

  bool LogWriter::createFile() {
#ifdef _WIN32
    fopen_s(&log_file_, "log.txt", "w");
#else
    log_file_ = fopen("log.txt", "w");
#endif
    if (log_file_ == NULL) {
      perror("Could not create a log file");
      return false;
    }
    return true;
  }

  void LogWriter::write(const string& msg) {
    m_.lock();

    if ((log_file_ == NULL) && !createFile()) {
      // If the log file is not there, there is no way to log a
      // message. We'll try to create a new one in the next write() in
      // the hopes that this is a transient error.
      m_.unlock();
      return;
    }
    fwrite(msg.c_str(), 1, msg.size(), log_file_);
    m_.unlock();
  }

}  // namespace tests
