//
//  callback_queue.h
//
//  Created by Jonathan Tompson on 7/10/12.
//
//  This is a NON-THREADSAFE templatized queue of pointers used for storing the
//  pending callbacks for execution in the class ThreadPool
//

#pragma once

#include <assert.h>
#include <iostream>  // For std::cout
#include <string>
#include "jcl/threading/callback_queue_item.h"

namespace jcl {
namespace threading {
  
  // This is a non-threadsafe queue of pointers.  Synchronization should be 
  // taken care of by the top level class (owner of the queue).
  template <class T>
  class CallbackQueue {
  public:
    // Default constructor and destructor
    CallbackQueue();
    ~CallbackQueue();
    
    // Add to the tail of the queue
    void enqueue(const T& newItem);
    
    // Remove from the head of the queue, return false if dequeue fails (empty)
    T dequeue();
    
    // Return the head of the queue but don't dequeue it
    T peak();
    
    // Check for empty
    bool empty() const;
    
    // Clear the queue
    void clear();
    
    // Print queue values to std out
    void printQueue() const;
    
    // Return the number of elements in the queue
    int size() const;
    
  private:
    CallbackQueueItem<T>* head_;
    CallbackQueueItem<T>* tail_;
    int num_elements_;
  };
  
  template <typename T>
  CallbackQueue<T>::CallbackQueue() {
    head_ = nullptr;
    tail_ = nullptr;
    num_elements_ = 0;
  }
  
  template <typename T>
  CallbackQueue<T>::~CallbackQueue() {
    clear();
  }
  
  template <typename T>
  void CallbackQueue<T>::clear() {
    CallbackQueueItem<T>* old_head;
    while (head_ != nullptr) {
      old_head = head_;
      head_ = head_->next;
      delete old_head;
    }
    tail_ = nullptr;
    num_elements_ = 0;
  }
  
  template <typename T>
  int CallbackQueue<T>::size() const {
    return num_elements_;
  }
  
  template <typename T>
  void CallbackQueue<T>::enqueue(const T& newItem) {
    CallbackQueueItem<T>* p_cur_item = new CallbackQueueItem<T>(newItem);
    if (head_ == nullptr)
      head_ = p_cur_item;
    if (tail_ != nullptr)
      tail_->next = p_cur_item;
    tail_ = p_cur_item;
    num_elements_++;
  }
  
  template <typename T>
  T CallbackQueue<T>::dequeue() {
    assert(head_ != nullptr);  //  Trying to dequeue from an empty queue
    T ret_val = head_->data;
    if (head_ == tail_) {  // one element left in the queue
      delete head_;
      head_ = nullptr;
      tail_ = nullptr;
    } else {  // more than one element left in the queue
      CallbackQueueItem<T>* old_head = head_;
      head_ = head_->next;
      delete old_head;
    }
    num_elements_--;
    return ret_val;
  }
  
  template <typename T>
  T CallbackQueue<T>::peak() {
    assert(head_ != nullptr);  //  Trying to peak from an empty queue
    return head_->data;
  }
  
  template <typename T>
  bool CallbackQueue<T>::empty() const {
    return head_ == nullptr;
  }
  
  template <typename T>
  void CallbackQueue<T>::printQueue() const {
    CallbackQueueItem<T>* p_cur_item = head_;
    std::cout << "Queue contents: ";
    while (p_cur_item != nullptr) {
      std::cout << p_cur_item->data << " ";
      p_cur_item = p_cur_item->next;
    }
  }
  
};  // namespace threading
};  // namespace jcl
