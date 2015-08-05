//
//  test_unit.h
//
//  Created by Alberto Lerner, then edited by Jonathan Tompson on 4/26/12.
//
//  ****** Originally from my jtil library (but pulled out for jcl to reduce
//  compilation dependencies).  ******
//

#pragma once

#include <unordered_map>
#include <iostream>
#include <string>
#include <vector>
#include "test_unit/param_map.h"

extern bool TESTS_exit_on_fatal;
extern bool TESTS_has_fatal_message;

namespace tests {
  
using std::string;
#ifdef _WIN32
using std::tr1::unordered_map;
#else
using std::unordered_map;
#endif
using std::vector;

class TestCase {
 public:
  virtual ~TestCase() {}

  virtual void testBody() = 0;
  virtual void clear() = 0;

  const string& group() const { return group_; }
  const string& name() const { return name_; }
  string fullName() const { return group_ + "(" + name_ + ")"; }
  int errors() const { return errors_; }

 protected:
  TestCase(const string& group, const string& name)
    : group_(group), name_(name), errors_(0) {}

  void incErrors()     { ++errors_; }

 private:
  string group_;
  string name_;
  int    errors_;
};

class TestRegistry {
 public:
  ~TestRegistry() {}

  static TestRegistry* getInstance();

  TestCase* registerCase(TestCase* test);
  int runAndResetWithArgs(int argc, char* argv[]);

 private:
  typedef vector<TestCase*> Tests;

  static TestRegistry*      instance_;
  Tests                     tests_;    // owned by this
  ParamMap                  args_;     // --case to be run

  void runListCases(int* errors);
  void runSomeCases(int* errors, const string& cases);

  TestRegistry();
};

TestRegistry::TestRegistry() {
  args_.addParam("case", "all", "csv list of test cases to run");
  args_.addParam("listcases", "false", "list the test cases for this test");
}

TestRegistry* TestRegistry::instance_ = NULL;

TestRegistry* TestRegistry::getInstance() {
  if (instance_ == NULL) instance_ = new TestRegistry;
  return instance_;
}

TestCase* TestRegistry::registerCase(TestCase* test) {
  tests_.push_back(test);
  return test;
}

void TestRegistry::runListCases(int* errors) {
  for (Tests::const_iterator it = tests_.begin(); it != tests_.end(); ++it) {
    std::cout << (*it)->fullName() << std::endl;
  }
}

void TestRegistry::runSomeCases(int* errors, const string& cases) {
  unordered_map<string, bool> filter;
  if (cases != "all") {
    string::size_type pos;
    string curr = cases;
    do {
      pos = curr.find(",");
      filter.insert(make_pair(curr.substr(0, pos), false /* executed yet? */));
      curr = curr.substr(pos+1);
    } while (pos != string::npos);
  }

  for (Tests::const_iterator itt = tests_.begin(); itt != tests_.end(); ++itt) {
    TestCase* test = *itt;

    // If a filter is set, check if this test should be skipped. Also
    // keep track that all requested test cases execute.
    unordered_map<string, bool>::iterator iti = filter.find(test->name());
    if (!filter.empty()) {
      if (iti == filter.end()) {
        continue;
      } else {
        iti->second = true /* executed */;
      }
    }

    std::cout << "Running " << test->fullName() << ": " << std::flush;
    test->testBody();
    if (test->errors()) {
      (*errors) += test->errors();
      std::cout << std::endl;
    } else {
      std::cout << "PASSED" << std::endl;
    }
  }

  // Any test in --case that was not executed?
  for (unordered_map<string, bool>::iterator iti = filter.begin();
       iti != filter.end();
       ++iti) {
    if (iti->second == false /* have not executed */) {
      std::cout << "Wrong test case name: " << iti->first << std::endl;
      (*errors)++;
    }
  }
}

int TestRegistry::runAndResetWithArgs(int argc, char* argv[]) {
  int errors = 0;
  bool exit = false;
  TESTS_exit_on_fatal = false;

  if (argv != NULL && !args_.parseArgv(argc, argv)) {
    args_.printUsage();
    errors += 1 /* one error, parsing */;
    exit = true;
  }

  // Check if we just want to learn the names of the test's cases.
  string list_only;
  if (!exit && args_.getParam("listcases", &list_only)) {
    if (list_only == "true") {
      runListCases(&errors);
      exit = true;
    }
  }

  // Check if we'd like to execute only some of the tests cases, as
  // indicated by the --cases parameter.
  string cases;
  if (!exit && args_.getParam("case", &cases)) {
    runSomeCases(&errors, cases);
    exit = true;
  }

  for (Tests::const_iterator it = tests_.begin(); it != tests_.end(); ++it) {
    (*it)->clear();
  }
  tests_.clear();
  TestRegistry* me = instance_;
  instance_ = NULL;
  delete me;

  return errors;
}

}  // namespace base

#define EXPECT_TRUE(a) \
  if (!(a)) { \
    if (errors() == 0) std::cout << "FAILED"; \
    std::cout << "\n    Line " << __LINE__ \
              << " Expected true " << # a << ": " <<  (a); \
    incErrors(); \
  }

#define EXPECT_FALSE(a) \
  if (a) { \
    if (errors() == 0) std::cout << "FAILED"; \
    std::cout << "\n    Line " << __LINE__ \
              << " Expected false " << # a << ": " <<  (a); \
    incErrors(); \
  }

#define EXPECT_EQ(a, b) \
  if ((a) != (b)) { \
    if (errors() == 0) std::cout << "FAILED"; \
    std::cout << "\n    Line " << __LINE__ \
              << " Expected equal " << # a << " and " << # b << ": (" \
              << (a) << " " << (b) << ")";  \
    incErrors(); \
  }

#define EXPECT_APPROX_EQ(a, b) \
  if (abs((a) - (b)) > 0.000001) { \
    if (errors() == 0) std::cout << "FAILED"; \
    std::cout << "\n    Line " << __LINE__ \
              << " Expected approx equal " << # a << " and " << # b << ": (" \
              << (a) << " " << (b) << ")";  \
    incErrors(); \
  }

#define EXPECT_NEQ(a, b) \
  if ((a) == (b)) { \
    if (errors() == 0) std::cout << "FAILED"; \
    std::cout << "\n    Line " << __LINE__ \
              << " Expected different " << # a << " and " << # b << ": (" \
              << (a) << " " << (b) << ")";  \
    incErrors(); \
  }

#define EXPECT_GT(a, b) \
  if ((a) <= (b)) { \
    if (errors() == 0) std::cout << "FAILED"; \
    std::cout << "\n    Line " << __LINE__ \
              << " Expected greater " << # a << " and " << # b << ": (" \
              << (a) << " " << (b) << ")"; \
    incErrors(); \
  }

#define EXPECT_FATAL(a) \
  MCP_BASE_has_fatal_message = false; \
  (a); \
  if (!MCP_BASE_has_fatal_message) { \
    if (errors() == 0) std::cout << "FAILED"; \
    std::cout << "\n    Line " << __LINE__ << " Expected fatal " << # a; \
    incErrors(); \
  }

#define TEST_CLASS_NAME(test_group, test_name) \
  test_group ## _ ## test_name ## _Test

#define TEST(test_group, test_name) \
class TEST_CLASS_NAME(test_group, test_name) : public tests::TestCase { \
public: \
  TEST_CLASS_NAME(test_group, test_name)(); \
  virtual void testBody(); \
  virtual void clear(); \
private: \
  static TestCase* me_; \
}; \
  \
  tests::TestCase* TEST_CLASS_NAME(test_group, test_name)::me_ = \
  tests::TestRegistry::getInstance()->registerCase( \
  new TEST_CLASS_NAME(test_group, test_name)); \
  \
  TEST_CLASS_NAME(test_group, test_name)::\
  TEST_CLASS_NAME(test_group, test_name)() \
  : TestCase(# test_group, # test_name) {} \
  \
  void TEST_CLASS_NAME(test_group, test_name)::clear() { \
  TestCase* me = me_; \
  me_ = NULL; \
  delete me; \
} \
  \
  void TEST_CLASS_NAME(test_group, test_name)::testBody()

#define RUN_TESTS(argc, argv) \
  (tests::TestRegistry::getInstance()->runAndResetWithArgs(argc, argv))

