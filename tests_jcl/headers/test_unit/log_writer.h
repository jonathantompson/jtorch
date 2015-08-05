//
//  log_writer.h
//
//  Created by Alberto Lerner, then edited by Jonathan Tompson on 4/26/12.
//
//  Used to be multi-threaded, but pthreads stripped out
//
//  ****** Originally from my jtil library (but pulled out for jcl to reduce
//  compilation dependencies).  ******
//

#pragma once

// #include <pthreads.h>
#include <stdio.h>
#include <mutex>
#include <string>
// #include "lock.h"

namespace tests {

  using std::ofstream;
  using std::string;

  class LogWriter {
  public:
    ~LogWriter();

    static LogWriter* instance();
    void write(const string& msg);

  private:
    LogWriter();
    static void init();

    // REQUIRES: m_ is locked
    bool createFile();

    // static pthread_once_t  init_control_;
    static bool               init_control_;
    static LogWriter*         instance_;      // not owned here
    std::mutex                m_;             // protects below
    FILE*                     log_file_;
  };

}  // namespace base
