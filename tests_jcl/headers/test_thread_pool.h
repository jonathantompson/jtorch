//
//  test_thread_pool.cpp
//
//  Created by Jonathan Tompson on 7/10/12.
//
//  There's a lot of code borrowed from Prof Alberto Lerner's code repository 
//  (I took his distributed computing class, which was amazing), particularly 
//  the test unit stuff.

#include <stdlib.h>  // exit
#include <thread>
#include <sstream>
#include "test_unit/test_unit.h"
#include "test_unit/test_util.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

// Create a thread pool, add one task and then request a stop
TEST(ThreadPool, CreateAddOnceAndStop) {
  static const uint32_t kNumWorkers = 4;
  static const int kCountStride = 11;

  jcl::threading::ThreadPool tp(kNumWorkers);
  tests::CounterThreadSafe c;

  // Create the callback associated with counter increments and add the task
  jcl::threading::Callback<void>* threadBody = 
    jcl::threading::MakeCallableOnce(&tests::CounterThreadSafe::incBy, 
                                     &c, kCountStride);
  tp.addTask(threadBody);

  // Wait until the task has been sent for execution
  while (tp.count()>0) {
    std::this_thread::yield();
  }

  // Request a stop (which will return once the task has finished)
  tp.stop();

  // Make sure the thread ran:
  // a) There should be no waiting tasks left (since we waited for tp.count=0)
  EXPECT_EQ(tp.count(), 0);
  // b) The counter should have incremented.
  EXPECT_EQ(c.count(), kCountStride);
}

// Create a thread pool, add many tasks (similar to before but more of them),
// and make sure we're adding tasks with different methods.
TEST(ThreadPool, CreateAddManyAndStop) {
  static const uint32_t kNumWorkers = 4;
  static const uint32_t kNumTaskRequests = 1001;
  static const int kCountStride = 11;
  static const int kNumTestRepeats = 11;

  tests::CounterThreadSafe c;

  // Repeat the test many times, each time with a new TP.  This will catch any
  // race conditions that might happen between the stop thread and destructor.
  for (int i = 0; i < kNumTestRepeats; i ++) {
    jcl::threading::ThreadPool* tp = new jcl::threading::ThreadPool(kNumWorkers);

    // Create the callback associated with one counter increment and add it
    jcl::threading::Callback<void>* threadBodyInc = 
      jcl::threading::MakeCallableMany(&tests::CounterThreadSafe::incBy, 
                                       &c, kCountStride);
    jcl::threading::Callback<void>* threadBody = 
      jcl::threading::MakeCallableMany(&tests::CounterThreadSafe::inc, &c);
    for ( uint32_t i = 0; i < kNumTaskRequests; i ++)
      tp->addTask(threadBodyInc);
    for ( uint32_t i = 0; i < kNumTaskRequests; i ++)
      tp->addTask(threadBody);

    // Request a stop by placing a stop request on the queue (to check that
    // recursive stops aren't an issue)
    jcl::threading::Callback<void>* p_StopTask = 
      jcl::threading::MakeCallableOnce(&jcl::threading::ThreadPool::stop, tp);
    tp->addTask(p_StopTask);

    // Wait until the tasks are sent for execution (including stop task)
    while (tp->count()>0) {
      std::this_thread::yield();
    }

    // Make sure the thread ran:
    // a) There should be no waiting tasks left (since we waited for tp.count=0)
    EXPECT_EQ(tp->count(), 0);

    // Note: There is an important test here: Destructor may be called before
    // stop task has finished.
    delete tp;  // destructor will wait for stop to finish

    // Since we created the tasks with MakeCallableMany we need to delete them
    delete threadBodyInc;
    delete threadBody;

    // b) The counter should have incremented
    EXPECT_EQ(c.count(), (kCountStride + 1) * kNumTaskRequests);
    c.reset();
  }
}

// Create a thread pool, request a stop, then add tasks and make sure they
// don't execute
TEST(ThreadPool, CreateAfterStopped) {
  static const uint32_t kNumWorkers = 4;
  static const int kCountStride = 11;

  jcl::threading::ThreadPool tp(kNumWorkers);
  tests::CounterThreadSafe c;

  tp.stop();

  // Create the callback associated with one counter increment and add the task
  jcl::threading::Callback<void>* threadBody = 
    jcl::threading::MakeCallableOnce(&tests::CounterThreadSafe::incBy, &c, 
                                     kCountStride);
  tp.addTask(threadBody);

  // We want to test that threadBody isn't executed.  Try yielding this thread
  // a few times to make sure that we give the TP enough time to make a mistake.
  for (int i = 0; i < 101; i ++) {
    std::this_thread::yield();
  }

  // Make sure the thread DIDN'T run:
  // a) There should be no waiting tasks left
  EXPECT_EQ(tp.count(), 1);
  // b) The counter shouldn't have incremented.
  EXPECT_EQ(c.count(), 0);
}

