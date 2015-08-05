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
  tests::Counter c;

  // Create the callback associated with one counter increment
  jcl::threading::Callback<void>* threadBody = 
    jcl::threading::MakeCallableOnce(&tests::Counter::inc, &c);

  // Run the callback on a new thread
  std::thread th = jcl::threading::MakeThread(threadBody);
  EXPECT_NEQ(jcl::threading::GetThreadID(&th), 0);

  // Wait for it to execute and finish
  th.join();

  // Check that the counter incremented
  EXPECT_EQ(1, c.count());
}

TEST(Creation, MakeAndExecuteOnceWithParams) {
  static const int kCountStride = 11;

  // Use a counter to provide the task.
  tests::Counter c;

  // Create the callback associated with one counter increment
  jcl::threading::Callback<void>* threadBody = 
    jcl::threading::MakeCallableOnce(&tests::Counter::incBy, &c, 
                                     kCountStride);

  // Run the callback on a new thread
  std::thread th = jcl::threading::MakeThread(threadBody);
  EXPECT_NEQ(jcl::threading::GetThreadID(&th), 0);

  // Wait for it to execute and finish
  th.join();

  // Check that the counter incremented
  EXPECT_EQ(kCountStride, c.count());
}


TEST(Creation, MakeAndExecuteMany) {
  static const uint32_t kNumExecuteThreads = 101;
  tests::CounterThreadSafe c;
  std::thread tid[kNumExecuteThreads];

  // Create the callback associated with a counter increment
  jcl::threading::Callback<void>* threadBody = 
    jcl::threading::MakeCallableMany(&tests::CounterThreadSafe::inc, &c);

  // Run the callback on many threads
  for (int i = 0; i < kNumExecuteThreads; i ++) {
    tid[i] = MakeThread(threadBody);
    EXPECT_NEQ(jcl::threading::GetThreadID(&tid[i]), 0);
  }

  // Wait for them to execute
  for (int i = 0; i < kNumExecuteThreads; i ++) {
    tid[i].join();
  }

  // Clean up the callback
  delete threadBody;

  // Check that the counter incremented
  EXPECT_EQ(kNumExecuteThreads, c.count());
}
