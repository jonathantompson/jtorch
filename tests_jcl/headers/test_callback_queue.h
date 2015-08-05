//
//  test_callback_queue.cpp
//
//  Created by Jonathan Tompson on 7/10/12.
//
//  There's a lot of code borrowed from Prof Alberto Lerner's code repository 
//  (I took his distributed computing class, which was amazing), particularly 
//  the test unit stuff.

#include "jcl/threading/callback_queue.h"
#include "test_unit/test_unit.h"
#include "test_unit/test_util.h"

#define TEST_BUFFER_SIZE 11

using jcl::threading::CallbackQueue;

// TEST 1: enqueue values and then dequeue them
TEST(Simple, EnqueueAndDequeue) {
  CallbackQueue<int> q;
  
  for (int i = TEST_BUFFER_SIZE-1; i >= 0; i--) {
    q.enqueue(i);
    EXPECT_EQ(q.size(), TEST_BUFFER_SIZE - i);
    EXPECT_EQ(q.peak(), 10);  // enqueuing on end, while peak gets head
  }
  
  // Check that print queue works (and doesn't cause seg faults)
  // q.printQueue();
  EXPECT_EQ(q.size(), TEST_BUFFER_SIZE);
  
  for (int i = TEST_BUFFER_SIZE-1; i >= 0; i--) {
    EXPECT_EQ(q.peak(), i);
    EXPECT_EQ(q.dequeue(), i);
    EXPECT_EQ(q.size(), i);
  }
  
  EXPECT_TRUE(q.empty());  // Check that we've dequeued all the values
  EXPECT_EQ(q.size(), 0);
}

// TEST 2: Enqueue values, call clear and then check integrity
TEST(Simple, ClearAndEnqueue) {
  CallbackQueue<int> q;
  
  // Insert TEST_BUFFER_SIZE values
  for (int i = TEST_BUFFER_SIZE-1; i >= 0; i--)
    q.enqueue(i);
  
  // Call clear and then check that it is actually now empty
  q.clear();
  EXPECT_TRUE(q.empty());
  EXPECT_EQ(q.size(), 0);
  
  // Make sure the data structure isn't corrupt by re-insertion of new values
  for (int i = TEST_BUFFER_SIZE-1; i >= 0; i--) {
    q.enqueue(i);
    EXPECT_EQ(q.size(), TEST_BUFFER_SIZE - i);
  }
  
  for (int i = TEST_BUFFER_SIZE-1; i >= 0; i--) {
    EXPECT_EQ(q.dequeue(), i);
    EXPECT_EQ(q.size(), i);
  }
}
