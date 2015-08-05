//
//  test_thread.cpp
//
//  Created by Jonathan Tompson on 7/10/12.
//
//  There's a lot of code borrowed from Prof Alberto Lerner's code repository 
//  (I took his distributed computing class, which was amazing), particularly 
//  the test unit stuff.

#include <thread>
#include <sstream>
#include "test_unit/test_unit.h"
#include "test_unit/test_util.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"

#define NUM_EXECUTE_THREADS 101  // A non-trivial number of threads
#define COUNT_STRIDE 11

using std::thread;
using tests::Counter;
using tests::CounterThreadSafe;
using jcl::threading::Callback;
using jcl::threading::MakeCallableOnce;
using jcl::threading::MakeCallableMany;
using jcl::threading::MakeThread;
using jcl::threading::GetThreadID;

int count = 0;
void DumbCounter() {
  count += 314159;
}

// Just a quick test to make sure that the current compiler supports C++11
// threads and that the debug tools are compatible with them.
TEST(Cpp11, ThreadCreation) {
  std::thread th(DumbCounter);
  th.join();
  EXPECT_EQ(count, 314159);
}

TEST(Creation, MakeAndExecuteOnce) {
  // Use a counter to provide the task.
  Counter c;

  // Create the callback associated with one counter increment
  Callback<void>* threadBody = MakeCallableOnce(&Counter::inc, &c);

  // Run the callback on a new thread
  thread th = MakeThread(threadBody);
  EXPECT_NEQ(GetThreadID(&th), 0);

  // Wait for it to execute and finish
  th.join();

  // Check that the counter incremented
  EXPECT_EQ(1, c.count());
}

TEST(Creation, MakeAndExecuteOnceWithParams) {
  // Use a counter to provide the task.
  Counter c;

  // Create the callback associated with one counter increment
  Callback<void>* threadBody = MakeCallableOnce(&Counter::incBy, &c,
                                                COUNT_STRIDE);

  // Run the callback on a new thread
  thread th = MakeThread(threadBody);
  EXPECT_NEQ(GetThreadID(&th), 0);

  // Wait for it to execute and finish
  th.join();

  // Check that the counter incremented
  EXPECT_EQ(COUNT_STRIDE, c.count());
}


TEST(Creation, MakeAndExecuteMany) {
  CounterThreadSafe c;
  std::thread tid[NUM_EXECUTE_THREADS];

  // Create the callback associated with a counter increment
  Callback<void>* threadBody = MakeCallableMany(&CounterThreadSafe::inc, &c);

  // Run the callback on many threads
  for (int i = 0; i < NUM_EXECUTE_THREADS; i ++) {
    tid[i] = MakeThread(threadBody);
    EXPECT_NEQ(GetThreadID(&tid[i]), 0);
  }

  // Wait for them to execute
  for (int i = 0; i < NUM_EXECUTE_THREADS; i ++) {
    tid[i].join();
  }

  // Clean up the callback
  delete threadBody;

  // Check that the counter incremented
  EXPECT_EQ(NUM_EXECUTE_THREADS, c.count());
}
