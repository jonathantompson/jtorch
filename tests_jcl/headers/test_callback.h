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

TEST(Once, Simple) {
  tests::Counter c;
  jcl::threading::Callback<void>* cb = 
    jcl::threading::MakeCallableOnce(&tests::Counter::inc, &c);
  EXPECT_TRUE(cb->once());
  (*cb)();
  EXPECT_EQ(c.count(), 1);
}

TEST(Once, Binding) {
  // early
  tests::Counter c;
  jcl::threading::Callback<void>* cb1 = 
    jcl::threading::MakeCallableOnce(&tests::Counter::incBy, &c, 2);
  EXPECT_TRUE(cb1->once());
  (*cb1)();
  EXPECT_EQ(c.count(), 2);

  // late
  c.reset();
  jcl::threading::Callback<void, int>* cb2 = 
    jcl::threading::MakeCallableOnce(&tests::Counter::incBy, &c);
  EXPECT_TRUE(cb2->once());
  (*cb2)(3);
  EXPECT_EQ(c.count(), 3);
}

TEST(Once, Currying) {
  tests::Counter c;
  jcl::threading::Callback<void, int>* cb1 = 
    jcl::threading::MakeCallableOnce(&tests::Counter::incBy, &c);
  jcl::threading::Callback<void>* cb2 =
    jcl::threading::MakeCallableOnce(&jcl::threading::Callback<void, int>::operator(), 
                                     cb1, 4);
  (*cb2)();
  EXPECT_EQ(c.count(), 4);
}

TEST(Once, ReturnType) {
  tests::Counter c;
  c.set(7);
  jcl::threading::Callback<bool, int, int>* cb1 = 
    jcl::threading::MakeCallableOnce(&tests::Counter::between, &c);
  EXPECT_TRUE((*cb1)(5, 10));

  jcl::threading::Callback<bool, int>* cb2 = 
    jcl::threading::MakeCallableOnce(&tests::Counter::between, &c, 5);
  EXPECT_TRUE(cb2->once());
  EXPECT_TRUE((*cb2)(10));

  jcl::threading::Callback<bool>* cb3 = 
    jcl::threading::MakeCallableOnce(&tests::Counter::between, &c, 5, 10);
  EXPECT_TRUE((*cb3)());
}

TEST(Many, Simple) {
  tests::Counter c;
  jcl::threading::Callback<void>* cb = 
    jcl::threading::MakeCallableMany(&tests::Counter::inc, &c);
  EXPECT_FALSE(cb->once());
  (*cb)();
  (*cb)();
  EXPECT_EQ(c.count(), 2);
  delete cb;
}
