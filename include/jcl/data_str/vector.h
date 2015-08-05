//
//  vector.h
//
//  Created by Jonathan Tompson on 4/26/12.
//
//  A very simple templated vector class
//  
//  NOTE: When pushBack is called, Vector will make a copy of the input element
//        and add that to the vector.  Ownership for pointers is NOT
//        transferred.  That is, the vector is not responsible for handling
//        cleanup when Vector<type*> is used.
//

#pragma once

#include <assert.h>
#include "jcl/math/int_types.h"  // for uint

namespace jcl {
namespace data_str {

  template <typename T>
  class Vector {
  public:
    explicit Vector(const uint32_t capacity = 0);
    ~Vector();

    void capacity(const uint32_t capacity);  // Request manual capacity incr
    void clear();
    inline void pushBack(const T& elem);  // Add a copy of the element to back
    inline void popBack(T& elem);  // remove last element and set to elem
    inline void popBack();  // remove last element
    inline void popBackUnsafe(T& elem);  // No bounds checking
    inline void popBackUnsafe();  // No bounds checking
    inline T* at(uint32_t index);  // Get an internal reference
    void deleteAtAndShift(const uint32_t index);  // remove elem and shift down
    inline const T* at(uint32_t index) const;  // Get an internal reference
    inline void set(const uint32_t index, const T& elem);
    void resize(const uint32_t size_);
    inline const uint32_t& size() const { return size_; }
    inline const uint32_t& capacity() const { return capacity_; }
    bool operator==(const Vector<T>& a) const;  // O(n) - linear search
    Vector<T>& operator=(const Vector<T>& other);  // O(n) - copy
    T operator[](const uint32_t index) const;
    T & operator[](const uint32_t index);

  private:
    uint32_t size_;
    uint32_t capacity_;  // will only grow or shrink by a factor of 2
    std::unique_ptr<T[]> pvec_;
  };

  template <typename T>
  Vector<T>::Vector(const uint32_t capacity) {  // capacity = 0
    pvec_.reset(nullptr);
    capacity_ = 0;
    size_ = 0;
    if (capacity != 0) {
      this->capacity(capacity);
    }
  };

  template <typename T>
  Vector<T>::~Vector() {
    clear();
  };

  template <typename T>
  void Vector<T>::capacity(const uint32_t capacity) {
    if (capacity != capacity_ && capacity != 0) {
      std::unique_ptr<T[]> pvec_new(new T[capacity]);

      // Use placement new to call the constructors for the array
      T* pnew = pvec_new.get();
      for (uint32_t i = 0; i < capacity; i ++) {
        pnew[i] = *(new(pnew + i) T());  // Call placement new item
      }

      if (pvec_ != nullptr) {
        for (uint32_t i = 0; i < std::min<uint32_t>(capacity, capacity_); i ++) {
          pnew[i] = pvec_[i];
        }
      }

      pvec_ = std::move(pvec_new);

      capacity_ = capacity;
      if (size_ > capacity_) {  // If we've truncated the array then resize_
        size_ = capacity_;
      }
    } else if (capacity == 0) {  // Clear array if desired capacity is zero
      clear();
    }
  };

  template <typename T>
  void Vector<T>::clear() {
    size_ = 0;
    pvec_.reset(nullptr);
    capacity_ = 0;
  };

  template <typename T>
  void Vector<T>::pushBack(const T& elem) {
    if (capacity_ == 0)
      capacity(1);
    else if (size_ == capacity_)
      capacity(capacity_ * 2);  // Grow the array by size_ 2
    pvec_[size_] = elem;
    size_ += 1;
  };
  
  template <typename T>
  void Vector<T>::popBack(T& elem) {
    assert(size_ > 0);
    elem = pvec_[size_-1];
    size_ -= 1;  // just reduce the size_ by 1
  };

  template <typename T>
  void Vector<T>::popBack() {
    assert(size_ > 0);
    size_ -= 1;  // just reduce the size_ by 1
  };

  template <typename T>
  void Vector<T>::popBackUnsafe(T& elem) {
    elem = pvec_[size_-1];
    size_ -= 1;  // just reduce the size_ by 1
  };

  template <typename T>
  void Vector<T>::popBackUnsafe() {
    size_ -= 1;  // just reduce the size_ by 1
  };

  template <typename T>
  T* Vector<T>::at(const uint32_t index ) {
    assert(index < size_);
    return &pvec_[index];
  };

  template <typename T>
  const T* Vector<T>::at(const uint32_t index ) const {
    assert(index < size_);
    return &pvec_[index];
  };

  template <typename T>
  T Vector<T>::operator[](const uint32_t index) const { 
    assert(index < size_);
    return pvec_[index]; 
  };

  template <typename T>
  T& Vector<T>::operator[](const uint32_t index) { 
    assert(index < size_);
    return pvec_[index]; 
  };

  template <typename T>
  void Vector<T>::set(const uint32_t index, const T& val ) {
    assert(index < size_);
    pvec_[index] = val;
  };

  template <typename T>
  void Vector<T>::resize(const uint32_t size) { 
    assert(size <= capacity_);
    size_ = size; 
  };   

  template <typename T>
  bool Vector<T>::operator==(const Vector& a) const {
    if (this == &a) {  // if both point to the same memory
      return true; 
    }
    if (size_ != a.size_) { 
      return false; 
    }
    for (uint32_t i = 0; i <= (size_-1); i++) {
      if (pvec_[i] != a.pvec_[i]) { 
        return false; 
      }
    }
    return true;
  };

  template <typename T>
  Vector<T>& Vector<T>::operator=(const Vector<T>& other) {
    if (this != &other) {  // protect against invalid self-assignment
      this->clear();
      this->capacity(other.capacity_);
      for (uint32_t i = 0; i < other.size_; i ++) {
        this->pvec_[i] = other.pvec_[i];
      }
      this->size_ = other.size_;
    }
    // by convention, always return *this
    return *this;
  };

  template <typename T>
  void Vector<T>::deleteAtAndShift(const uint32_t index) {
    assert(index < size_);
    for (uint32_t i = index; i < size_-1; i++) {
      pvec_[i] = pvec_[i+1];
    }
    size_--;
  };

};  // namespace data_str
};  // namespace jcl
