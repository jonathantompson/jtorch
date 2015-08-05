//
//  test_util.h
//
//  Created by Alberto Lerner, then edited by Jonathan Tompson on 7/10/12.
//

#pragma once

#include <mutex>

namespace tests {
  
  using std::mutex;
  
  // A simple counter class used often in unit test (and not supposed to
  // be used outside *_test.cpp files.
  
  class Counter {
  public:
    Counter()          { reset(); }
    virtual ~Counter() { }
    
    int  count() const { return count_; }
    void set(int i)    { count_ = i; }
    void reset()       { count_ = 0; }
    void inc()         { ++count_; }
    void incBy(int i)  { count_ += i; }
    void incOneThousand();
    
    bool between(int i, int j) {
      if (i > count_) return false;
      if (j < count_) return false;
      return true;
    }
    
  private:
    int  count_;
  };
  
  // Counter method with non-trivial work used in thread_pool_benchmark.cpp
  void Counter::incOneThousand() {
    for (int i = 0; i < 1000; i ++)
      count_++;
  };
  
  // A thread-safe version of the counter, only writes take the lock to ensure
  // that number of thread inc() results in the correct count value, while 
  // count() is only a snapshot of the current value.
  class CounterThreadSafe {
  public:
    CounterThreadSafe()          { reset(); }
    virtual ~CounterThreadSafe() { }
    
    int  count() const { return count_; }
    void set(int i);
    void reset();
    void inc();
    void incBy(int i);
    
    bool between(int i, int j) {
      if (i > count_) return false;
      if (j < count_) return false;
      return true;
    }
    
  private:
    int  count_;
    mutex lock_;
  };
  
  // These should be moved to test_util.cpp, but I don't want to edit the class
  // wscript (I'm worried it wont compile on Prof. Lerner's machine).
  void CounterThreadSafe::set(int i) {
    lock_.lock();
    count_ = i;
    lock_.unlock();
  };
  
  void CounterThreadSafe::reset() {
    lock_.lock();
    count_ = 0;
    lock_.unlock();
  };
  
  void CounterThreadSafe::inc() {
    lock_.lock();
    ++count_;
    lock_.unlock();
  };
  
  void CounterThreadSafe::incBy(int i) {
    lock_.lock();
    count_ += i;
    lock_.unlock();
  };
  
};  // namespace tests
