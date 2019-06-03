// LoadCSV.cpp
// Copyright Voltera Inc., 2019
#include "VolteraCalibrationFitting/LoadCSV.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace voltera {

std::string LoadCSV::c_comment_delimiters = "#";
char LoadCSV::c_delimiter = ',';

bool LoadCSV::load(const std::string &filename,
                   std::vector<std::vector<double>> &loaded_data) {

  std::ifstream input_file;
  input_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  try {
    input_file.open(filename.c_str());
    std::string line;
    loaded_data.clear();
    while (std::getline(input_file, line)) {
      if (line.empty() ||
          (c_comment_delimiters.find(line[0]) != std::string::npos)) {
        continue;
      }
      std::stringstream line_stream(line);
      std::string cell;
      std::vector<double> line_vector;
      while (std::getline(line_stream, cell, c_delimiter)) {
        if (!cell.empty()) {
          double cell_double;
          cell_double = strtod(cell.c_str(), nullptr);
          line_vector.push_back(cell_double);
        } else {
          line_vector.push_back(0.0);
        }
      }
      loaded_data.push_back(line_vector);
    }
    input_file.close();
  } catch (std::ifstream::failure) {
    if (input_file.eof()) {
      input_file.close();
      return true;
    }
    std::cerr << "ERROR: Failed to load CSV file " << filename << std::endl;
    return false;
  }

  return true;
}

} // namespace voltera