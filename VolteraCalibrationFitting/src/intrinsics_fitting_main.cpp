/**
 * intrinsics_fiting_main.cpp
 *
 * Ryan Wicks
 * 8 May 2019
 * Copyright Voltera Inc., 2019
 *
 * Main entry point for the intrinsics program.
 *
 */

#include "cxxopts/cxxopts.h"

#include <ceres/ceres.h>
#include <vector>

#include "VolteraCalibrationFitting/Intrinsics.h"
#include "VolteraCalibrationFitting/LoadCSV.h"
#include "VolteraCalibrationFitting/PreviewResiduals.h"
#include "VolteraCalibrationFitting/ReprojectionError.h"

cxxopts::ParseResult parse(int argc, char *argv[]);

int main(int argc, char *argv[]) {

  google::InitGoogleLogging(argv[0]);

  std::string input_file;
  std::string output_file("out.json");
  std::string parameter_file;
  bool calibrate_laser(false);
  std::string laser_input_file;
  std::vector<std::vector<double>> parameters;

  auto parse_result = parse(argc, argv);

  input_file = parse_result["input_file"].as<std::vector<std::string>>()[0];

  if (parse_result.count("output")) {
    output_file = parse_result["output"].as<std::string>();
  }
  if (parse_result.count("laser")) {
    calibrate_laser = true;
    laser_input_file = parse_result["laser"].as<std::string>();
  }

  parameters = {std::vector<double>({6325, 6325, 1640, 1232}),
                std::vector<double>({0, 0, 0, 0, 0}),
                std::vector<double>({-0.11, -0.035, 0, 0.005, -0.004, 0.0237})};

  if (parse_result.count("parameters")) {
    if (!voltera::LoadCSV::load(parse_result["parameters"].as<std::string>(),
                                parameters)) {
      std::cerr << "Failed to load parameters, using defaults." << std::endl;
    }
  }

  std::cout << "Loading data." << std::endl;
  std::vector<std::vector<double>> data;

  if (!voltera::LoadCSV::load(input_file, data)) {
    std::cerr << "ERROR: Could not load data file: " << input_file << std::endl;
    return -1;
  }
  std::cout << "Loaded " << data.size() << " points." << std::endl;

  const double stage_offset[3]{parameters[2][3], parameters[2][4],
                               parameters[2][5]};
  const double stage_rotation[3]{parameters[2][0], parameters[2][1],
                                 parameters[2][2]};
  const double fx(parameters[0][0]), fy(parameters[0][1]);
  const double cx(parameters[0][2]), cy(parameters[0][3]);

  double cam_matrix[voltera::ReprojectionError::SIZE_CAMMATRIX] = {fx, fy, cx,
                                                                   cy};
  double distortion[voltera::ReprojectionError::SIZE_DISTORTION] = {0, 0, 0, 0,
                                                                    0};
  double extrinsics[voltera::ReprojectionError::SIZE_EXTRINSICS] = {
      stage_rotation[0], stage_rotation[1], stage_rotation[2],
      stage_offset[0],   stage_offset[1],   stage_offset[2]};

  voltera::previewIntrinsics(data, cam_matrix, distortion, extrinsics);
  std::cout << "-------------------------------\nTotal Refinement" << std::endl;
  voltera::runIntrinsics(data, cam_matrix, distortion, extrinsics, false,
                         false);
  voltera::previewIntrinsics(data, cam_matrix, distortion, extrinsics);
  return 0;
}

cxxopts::ParseResult parse(int argc, char *argv[]) {
  try {
    cxxopts::Options options(
        argv[0], "This program calibrates the Voltera Camera/Laser system.");
    options.positional_help("[optional args]").show_positional_help();

    options.add_options()("l,laser", "Calibrate Laser Plane")(
        "o,output_file", "Output file name", cxxopts::value<std::string>())(
        "p,parameters", "Input parameters guess file",
        cxxopts::value<std::string>())(
        "input_file", "Input filename",
        cxxopts::value<std::vector<std::string>>())("h,help", "Print help");
    options.parse_positional({"input_file"});

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }

    if (!result.count("input_file")) {
      std::cerr << "ERROR: Must provide an input file." << std::endl;
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(-1);
    }

    return result;
  } catch (const cxxopts::OptionException &e) {
    std::cerr << "ERROR: Could not parse command line arguments." << std::endl
              << e.what() << std::endl;
    exit(-1);
  }
}