// testLoadCSV.cpp
// Copyright Voltera Inc., 2019
#include "VolteraCppTools/LoadCSV.h"
#include "test_path.h"
#include "gtest/gtest.h"

#include <vector>

TEST(LoadCSV, non_existent_files) {
  std::string filename = "null.file";
  std::vector<std::vector<double>> out_data;

  ASSERT_FALSE(voltera::LoadCSV::load(filename, out_data));
}

TEST(LoadCSV, proper_loading) {
  std::string file1 = test_data_path + "test1.csv";
  std::string file2 = test_data_path + "test2.csv";

  std::vector<std::vector<double>> data1;
  std::vector<std::vector<double>> data2;

  ASSERT_TRUE(voltera::LoadCSV::load(file1, data1));
  ASSERT_TRUE(voltera::LoadCSV::load(file2, data2));

  EXPECT_EQ(data1.size(), 215);
  EXPECT_DOUBLE_EQ(data1[0][0], 1892.714321692411);
  EXPECT_EQ(data1[45].size(), 5);

  EXPECT_EQ(data2.size(), 4);
  EXPECT_EQ(data2[0].size(), 5);
  EXPECT_EQ(data2[1].size(), 6);
  EXPECT_EQ(data2[2].size(), 2);
  EXPECT_EQ(data2[3].size(), 6);
  EXPECT_DOUBLE_EQ(data2[0][4], 5.0);
  EXPECT_DOUBLE_EQ(data2[1][0], 0.0);
  EXPECT_DOUBLE_EQ(data2[1][3], -6.0);
  EXPECT_DOUBLE_EQ(data2[1][4], 1.0);
  EXPECT_DOUBLE_EQ(data2[3][4], -0.0);
}