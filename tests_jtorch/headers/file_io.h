//
//  file_io.h
//  KinectHands
//
//  Created by Jonathan Tompson on 6/5/12.
//
//  ****** Originally from my jtil library (but pulled out for jcl to reduce
//  compilation dependencies).  ******
//

#pragma once

#include <stdexcept>
#include <iostream>
#include <fstream>

namespace jcl {
namespace file_io {

template <class T>
void SaveArrayToFile(const T* arr, const int size,
                     const std::string& filename) {
  std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(std::string("file_io::SaveArrayToFile() - "
                                         "ERROR: Cannot open output file:") +
                             filename);
  }
  file.write(reinterpret_cast<const char*>(arr), size * sizeof(arr[0]));
  file.flush();
  file.close();
}

template <class T>
void LoadArrayFromFile(T* arr, const int size, const std::string& filename) {
  std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(std::string("file_io::LoadArrayFromFile() - "
                                         "ERROR: Cannot open output file:") +
                             filename);
  }
  std::streampos fsize = file.tellg();
  file.seekg(0, std::ios::end);
  fsize = file.tellg() - fsize;
  if ((int)fsize < (int)sizeof(arr[0]) * size) {
    throw std::runtime_error(
        "jtil::LoadArrayFromFile() - ERROR: "
        "File is too small for data request!");
  }
  file.seekg(0);

  file.read(reinterpret_cast<char*>(arr), size * sizeof(arr[0]));
  file.close();
}

bool fileExists(const std::string& filename);

};  // namespace file_io
};  // namespace jcl
