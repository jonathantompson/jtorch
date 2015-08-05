//
//  test_callback.cpp
//
//  Created by Jonathan Tompson on 7/10/12.
//
//  There's a lot of code borrowed from Prof Alberto Lerner's code repository 
//  (I took his distributed computing class, which was amazing), particularly 
//  the test unit stuff.

#include "jcl/threading/callback.h"
#include "test_unit/test_unit.h"
#include "test_unit/test_util.h"

using jcl::threading::Callback;
using jcl::threading::MakeCallableOnce;
using jcl::threading::MakeCallableMany;
using tests::Counter;

TEST(Once, Simple) {
  Counter c;
  Callback<void>* cb = MakeCallableOnce(&Counter::inc, &c);
  EXPECT_TRUE(cb->once());
  (*cb)();
  EXPECT_EQ(c.count(), 1);
}

TEST(Once, Binding) {
  // early
  Counter c;
  Callback<void>* cb1 = MakeCallableOnce(&Counter::incBy, &c, 2);
  EXPECT_TRUE(cb1->once());
  (*cb1)();
  EXPECT_EQ(c.count(), 2);

  // late
  c.reset();
  Callback<void, int>* cb2 = MakeCallableOnce(&Counter::incBy, &c);
  EXPECT_TRUE(cb2->once());
  (*cb2)(3);
  EXPECT_EQ(c.count(), 3);
}

TEST(Once, Currying) {
  Counter c;
  Callback<void, int>* cb1 = MakeCallableOnce(&Counter::incBy, &c);
  Callback<void>* cb2 =
    MakeCallableOnce(&Callback<void, int>::operator(), cb1, 4);
  (*cb2)();
  EXPECT_EQ(c.count(), 4);
}

TEST(Once, ReturnType) {
  Counter c;
  c.set(7);
  Callback<bool, int, int>* cb1 = MakeCallableOnce(&Counter::between, &c);
  EXPECT_TRUE((*cb1)(5, 10));

  Callback<bool, int>* cb2 = MakeCallableOnce(&Counter::between, &c, 5);
  EXPECT_TRUE(cb2->once());
  EXPECT_TRUE((*cb2)(10));

  Callback<bool>* cb3 = MakeCallableOnce(&Counter::between, &c, 5, 10);
  EXPECT_TRUE((*cb3)());
}

TEST(Many, Simple) {
  Counter c;
  Callback<void>* cb = MakeCallableMany(&Counter::inc, &c);
  EXPECT_FALSE(cb->once());
  (*cb)();
  (*cb)();
  EXPECT_EQ(c.count(), 2);
  delete cb;
}
