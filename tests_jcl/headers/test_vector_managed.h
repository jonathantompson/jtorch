//
//  test_vector_managed.h
//
//  Created by Jonathan Tompson on 6/1/12.
//

#include "jcl/data_str/vector_managed.h"
#include "test_unit/test_unit.h"

using jcl::data_str::VectorManaged;

TEST(VectorManaged, CreationAndInsertion) {
  VectorManaged<int>* vec1 = new VectorManaged<int>(2);  // capacity = 2
  EXPECT_EQ(vec1->size(), 0);
  vec1->pushBack(1);
  vec1->pushBack(2);
  vec1->pushBack(3);  // should force a resize of vec1's capacity to 4
  EXPECT_EQ(vec1->capacity(), 4);
  EXPECT_EQ(vec1->size(), 3);

  vec1->popBack();
  EXPECT_EQ(vec1->capacity(), 4);
  EXPECT_EQ(vec1->size(), 2);
  vec1->pushBack(3);
  vec1->pushBack(4);
  VectorManaged<int>* vec2 = new VectorManaged<int>(2);
  *vec2 = *vec1;  // should force a resize of vec2's capacity to 4
  EXPECT_EQ(vec2->size(), 4);
  EXPECT_EQ(vec2->capacity(), 4);
  if (vec2->size() == vec1->size()) {
    for (uint32_t i = 0; i < vec2->size(); i ++) {
      EXPECT_EQ(*vec2->at(i), *vec1->at(i));  // Pointers should be the same
    }
  }

  vec2->capacity(8);  // Should grow array
  EXPECT_EQ(vec2->capacity(), 8);
  EXPECT_EQ(vec2->size(), 4);
  vec2->capacity(2);  // Should shrink array
  EXPECT_EQ(vec2->capacity(), 2);
  EXPECT_EQ(vec2->size(), 2);

  *vec2 = *vec1;  // Should grow array again to 4
  EXPECT_TRUE(*vec1 == *vec2);
  int wrong_val = 500;
  vec1->set(3, wrong_val);  // should set the [3] element to 500
  EXPECT_FALSE(*vec1 == *vec2);

  int* last_val = vec1->at(3);
  *last_val = 4;
  EXPECT_TRUE(*vec1 == *vec2);

  delete vec1;
  delete vec2;
}

TEST(VectorManagedPointer, CreationAndInsertion) {
  int* cur_item;
  VectorManaged<int*>* vec1 = new VectorManaged<int*>(2);  // capacity = 2
  EXPECT_EQ(vec1->size(), 0);
  cur_item = new int[1]; 
  *cur_item = 1;
  vec1->pushBack(cur_item);
  cur_item = new int[1]; 
  *cur_item = 2;
  vec1->pushBack(cur_item);
  cur_item = new int[1]; 
  *cur_item = 3;
  vec1->pushBack(cur_item);  // should force a resize of vec1's capacity
  EXPECT_EQ(vec1->capacity(), 4);
  EXPECT_EQ(vec1->size(), 3);

  vec1->popBack();  // Will delete the 3 value  
  EXPECT_EQ(vec1->capacity(), 4);
  EXPECT_EQ(vec1->size(), 2);
  cur_item = new int[1]; 
  *cur_item = 3;
  vec1->pushBack(cur_item);
  cur_item = new int[1]; 
  *cur_item = 4;
  vec1->pushBack(cur_item);
  VectorManaged<int*> * vec2 = new VectorManaged<int*>(2);
  cur_item = new int[1]; 
  *cur_item = 3;
  vec2->pushBack(cur_item);
  cur_item = new int[1]; 
  *cur_item = 4;
  vec2->pushBack(cur_item);

  *vec2 = *vec1;  // should force a resize of vec2's capacity to 4
                  // All vec2's data members should point to vec1's
                  // vec2's members should be deleted
  EXPECT_TRUE(*vec1 == *vec2);
  EXPECT_EQ(vec2->size(), 4);
  EXPECT_EQ(vec2->capacity(), 4);
  if (vec2->size() == vec1->size()) {
    for (uint32_t i = 0; i < vec2->size(); i ++) {
      EXPECT_EQ(*vec2->at(i), *vec1->at(i));  // Pointers should be the same
    }
  }

  delete vec1;
  // Now manually set all vec2's elements to null
  for (uint32_t i = 0; i < vec2->size(); i ++) {
    vec2->set(i, nullptr);
  }
  delete vec2;
}
