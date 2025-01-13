#include "classifier.h"
#include "json.hpp"
#include <torch/script.h>
#include <torch/torch.h>

using json = nlohmann::json;

std::vector<float> applyMinMaxScaling(const std::vector<float> &input,
                                      const std::vector<float> &minValues,
                                      const std::vector<float> &scaleValues) {
  if (input.size() != minValues.size() || input.size() != scaleValues.size()) {
    throw std::invalid_argument("input size does not match scaler dimensions.");
  }

  std::vector<float> scaleInput(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    scaleInput[i] = (input[i] - minValues[i]) * scaleValues[i];
  }

  return scaleInput;
}

class Model {
  torch::jit::script::Module model;
  torch::Device device =
      torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  static std::vector<float> minValues;
  static std::vector<float> scaleValues;
  static bool scalerLoaded;
  static std::mutex scalerMutex;

  static void loadScalerOnce(const std::string &filePath) {
    std::lock_guard<std::mutex> lock(scalerMutex);
    if (!scalerLoaded) {
      std::ifstream file(filePath);
      if (!file.is_open()) {
        throw std::runtime_error("could not open scaler file.");
      }

      json scalerData;
      file >> scalerData;
      minValues = scalerData["min"].get<std::vector<float>>();
      scaleValues = scalerData["scale"].get<std::vector<float>>();
      scalerLoaded = true;
    }
  }

public:
  Model(const std::string &modelFile);
  bool PredicIsPositiveBinary(float *inputData, int numFeatures);
};

std::vector<float> Model::minValues;
std::vector<float> Model::scaleValues;
bool Model::scalerLoaded = false;
std::mutex Model::scalerMutex;

Model::Model(const std::string &modelFile) {
  if (torch::cuda::is_available()) {
    std::cout << "cuda is available" << std::endl;
  } else {
    std::cout << "cuda is not available" << std::endl;
  }

  try {
    this->loadScalerOnce("../../data/scaler.json");
  } catch (const std::exception &e) {
    std::cerr << "Error loading scaler: " << e.what() << std::endl;
    throw;
  }

  try {
    model = torch::jit::load(modelFile, this->device);
  } catch (const c10::Error &e) {
    std::cerr << "could not load model. Erorr: " << e.what() << std::endl;
    throw std::invalid_argument("Invalid model file.");
  }
}

bool Model::PredicIsPositiveBinary(float *inputData, int numFeatures) {
  std::vector<float> inputVec(inputData, inputData + numFeatures);

  std::vector<float> scaleInput =
      applyMinMaxScaling(inputVec, minValues, scaleValues);

  torch::Tensor inputTensor =
      torch::from_blob(scaleInput.data(), {1, numFeatures}, torch::kFloat32)
          .to(this->device);

  this->model.eval();
  torch::Tensor output = this->model.forward({inputTensor}).toTensor();
  output = torch::round(torch::sigmoid(output));

  return output.item<float>() == 1.0;
}

mModel NewModel(const char *modelFile) {
  try {
    const auto model = new Model(modelFile);
    return (void *)model;
  } catch (const std::invalid_argument &ex) {
    std::cerr << "could not create new model. Error: " << ex.what()
              << std::endl;
    return nullptr;
  }
}

int PredictIsPositiveBinary(mModel model, float *inputData, int numFeatures) {
  auto initializedModel = (Model *)model;
  if (initializedModel == nullptr) {
    return -1;
  }
  return initializedModel->PredicIsPositiveBinary(inputData, numFeatures) ? 1
                                                                          : 0;
}

void DeleteModel(mModel model) {
  auto initializedModel = (Model *)model;
  if (initializedModel == nullptr) {
    return;
  }
  delete initializedModel;
}
