//
//  test_circular_buffer.h
//
//  Created by Jonathan Tompson on 4/26/12.
//

#include <sstream>

#include "jcl/data_str/hash_funcs.h"
#include "jcl/data_str/hash_map_managed.h"
#include "test_unit/test_unit.h"

// TEST 1: Create a hash table, insert items from 1:N
TEST(HashMapManaged, CreationAndInsertion) {
  static const uint32_t kTestHMMStartSize = 101;  // Must be prime
  static const uint32_t kTestHMMNumValues = 52;  // Enough to force at least one re-hash
  jcl::data_str::HashMapManaged<uint32_t, uint32_t> ht(kTestHMMStartSize, 
    &jcl::data_str::HashUInt);

  for (uint32_t i = 0; i < kTestHMMNumValues; i += 1) {
    uint32_t val = i*2;
    EXPECT_EQ(ht.insert(i, val), true);  // Insert key = i, value = i*2
  }
  // Now check that they're all there
  uint32_t val;
  for (uint32_t i = 0; i < kTestHMMNumValues; i += 1) {
    EXPECT_TRUE(ht.lookup(i, val));  // find key = i (value = i)
    EXPECT_EQ(i*2, val);
  }
  for (uint32_t i = kTestHMMNumValues; i < 2*kTestHMMNumValues; i += 1) {
    EXPECT_FALSE(ht.lookup(i, val));  // find key = i (value = i)
  }
  EXPECT_EQ(ht.count(), kTestHMMNumValues);
}

// TEST 3: Duplicate insertion
TEST(HashMapManaged, DuplicateInsertion) {
  static const uint32_t kTestHMMStartSize = 101;  // Must be prime
  static const uint32_t kTestHMMNumValues = 52;  // Enough to force at least one re-hash
  jcl::data_str::HashMapManaged<uint32_t, uint32_t> ht(kTestHMMStartSize, 
    &jcl::data_str::HashUInt);
  EXPECT_TRUE(ht.insert(0, 1));
  EXPECT_FALSE(ht.insert(0, 0));  // Try twice with two different values
  EXPECT_FALSE(ht.insert(0, 1));  // (since value shouldn't matter)
}

// TEST 4: Create a hash table, clear it and make sure it cleared
TEST(HashMapManaged, CreationAndClear) {
  static const uint32_t kTestHMMStartSize = 101;  // Must be prime
  static const uint32_t kTestHMMNumValues = 52;  // Enough to force at least one re-hash
  jcl::data_str::HashMapManaged<uint32_t, uint32_t> ht(kTestHMMStartSize, 
    &jcl::data_str::HashUInt);
  for (uint32_t i = 0; i < kTestHMMNumValues; i += 1) {
    uint32_t val = i*2;
    EXPECT_EQ(ht.insert(i, val), true);  // Insert key = i, value = i*2
  }

  ht.clear();
  EXPECT_EQ(ht.count(), 0);

  // Now check that they're all not there  (TEST 1, ensures that they were)
  uint32_t val;
  for (uint32_t i = 0; i < 2*kTestHMMNumValues; i += 1) {
    EXPECT_FALSE(ht.lookup(i, val));  // find key = i (value = i)
  }
}

// TEST 5: String hashing
TEST(HashMapManaged, StringCreationAndInsertion) {
  static const uint32_t kTestHMMStartSize = 101;  // Must be prime
  static const uint32_t kTestHMMNumValues = 52;  // Enough to force at least one re-hash
  jcl::data_str::HashMapManaged<std::string, uint32_t> ht(kTestHMMStartSize, 
    &jcl::data_str::HashString);
  std::stringstream ss;
    
  // Try a constant hash (check assembly to make sure hashing was constant)
  uint32_t cur_val;
  cur_val = 0;
  ht.insertPrehash(CONSTANT_HASH(ht.size(), "0"), "0", cur_val);

  for (uint32_t i = 1; i < kTestHMMNumValues; i += 1) {
    cur_val = i*2;
    ss.str("");
    ss << i;
    // Insert key = i, value = i*2
    EXPECT_EQ(ht.insert(ss.str().c_str(), cur_val), true);  
  }
  // Now check that they're all there
  uint32_t val;
  for (uint32_t i = 0; i < kTestHMMNumValues; i += 1) {
    ss.str("");
    ss << i;
    // find key = i (value = i)
    EXPECT_TRUE(ht.lookup(ss.str().c_str(), val));  
    EXPECT_EQ(i*2, val);
  }
  for (uint32_t i = kTestHMMNumValues; i < 2*kTestHMMNumValues; i += 1) {
    ss.str("");
    ss << i;
    // find key = i (value = i)
    EXPECT_FALSE(ht.lookup(ss.str().c_str(), val));  
  }
  EXPECT_EQ(ht.count(), kTestHMMNumValues);
}

// TEST 1Ptr: Create a hash table, insert items from 1:N
TEST(HashMapManagedPtr, CreationAndInsertion) {
  static const uint32_t kTestHMMStartSize = 101;  // Must be prime
  static const uint32_t kTestHMMNumValues = 52;  // Enough to force at least one re-hash
  jcl::data_str::HashMapManaged<uint32_t, uint32_t*> ht(kTestHMMStartSize, 
    &jcl::data_str::HashUInt);
  uint32_t* cur_val;

  for (uint32_t i = 0; i < kTestHMMNumValues; i += 1) {
    cur_val = new uint32_t[1];
    *cur_val = i*2;
    // Insert key = i, value = i*2
    EXPECT_EQ(ht.insert(i, cur_val), true);  
  }
  // Now check that they're all there
  uint32_t* val = nullptr;
  for (uint32_t i = 0; i < kTestHMMNumValues; i += 1) {
    EXPECT_TRUE(ht.lookup(i, val));  // find key = i (value = i)
    EXPECT_EQ(i*2, *val);
  }
  for (uint32_t i = kTestHMMNumValues; i < 2*kTestHMMNumValues; i += 1) {
    EXPECT_FALSE(ht.lookup(i, val));  // find key = i (value = i)
  }
  EXPECT_EQ(ht.count(), kTestHMMNumValues);
}

// TEST 3Ptr: Duplicate insertion
TEST(HashMapManagedPtr, DuplicateInsertion) {
  static const uint32_t kTestHMMStartSize = 101;  // Must be prime
  jcl::data_str::HashMapManaged<uint32_t, uint32_t*> ht(kTestHMMStartSize, 
    &jcl::data_str::HashUInt);
  uint32_t* cur_val;
  cur_val = new uint32_t[1]; 
  *cur_val = 1;
  EXPECT_TRUE(ht.insert(0, cur_val));
  cur_val = new uint32_t[1]; 
  *cur_val = 0;
  EXPECT_FALSE(ht.insert(0, cur_val));  // Try twice with two different values
  delete cur_val;
  cur_val = new uint32_t[1]; 
  *cur_val = 1;
  EXPECT_FALSE(ht.insert(0, cur_val));  // (since value shouldn't matter)
  delete cur_val;
}

// TEST 4Ptr: Create a hash table, clear it and make sure it cleared
TEST(HashMapManagedPtr, CreationAndClear) {
  static const uint32_t kTestHMMStartSize = 101;  // Must be prime
  static const uint32_t kTestHMMNumValues = 52;  // Enough to force at least one re-hash
  jcl::data_str::HashMapManaged<uint32_t, uint32_t*> ht(kTestHMMStartSize, 
    &jcl::data_str::HashUInt);
  uint32_t* cur_val;
  for (uint32_t i = 0; i < kTestHMMNumValues; i += 1) {
    cur_val = new uint32_t[1];
    *cur_val = i*2;
    // Insert key = i, value = i*2
    EXPECT_EQ(ht.insert(i, cur_val), true);  
  }

  ht.clear();
  EXPECT_EQ(ht.count(), 0);

  // Now check that they're all not there  (TEST 1, ensures that they were)
  uint32_t* val = nullptr;
  for (uint32_t i = 0; i < 2*kTestHMMNumValues; i += 1) {
    EXPECT_FALSE(ht.lookup(i, val));  // find key = i (value = i)
  }
}

// TEST 5Ptr: String hashing
TEST(HashMapManagedPtr, StringCreationAndInsertion) {
  static const uint32_t kTestHMMStartSize = 101;  // Must be prime
  static const uint32_t kTestHMMNumValues = 52;  // Enough to force at least one re-hash
  jcl::data_str::HashMapManaged<std::string, uint32_t*> ht(kTestHMMStartSize, 
    &jcl::data_str::HashString);
  std::stringstream ss;
    
  // Try a constant hash (check assembly to make sure hashing was constant)
  uint32_t* cur_val;
  cur_val = new uint32_t[1];
  *cur_val = 0;
  ht.insertPrehash(CONSTANT_HASH(ht.size(), "0"), "0", cur_val);

  for (uint32_t i = 1; i < kTestHMMNumValues; i += 1) {
    cur_val = new uint32_t[1];
    *cur_val = i*2;
    ss.str("");
    ss << i;
    // Insert key = i, value = i*2
    EXPECT_EQ(ht.insert(ss.str().c_str(), cur_val), true);  
  }
  // Now check that they're all there
  uint32_t* val = nullptr;
  for (uint32_t i = 0; i < kTestHMMNumValues; i += 1) {
    ss.str("");
    ss << i;
    // find key = i (value = i)
    EXPECT_TRUE(ht.lookup(ss.str().c_str(), val));  
    EXPECT_EQ(i*2, *val);
  }
  for (uint32_t i = kTestHMMNumValues; i < 2*kTestHMMNumValues; i += 1) {
    ss.str("");
    ss << i;
    // find key = i (value = i)
    EXPECT_FALSE(ht.lookup(ss.str().c_str(), val));  
  }
  EXPECT_EQ(ht.count(), kTestHMMNumValues);
}
