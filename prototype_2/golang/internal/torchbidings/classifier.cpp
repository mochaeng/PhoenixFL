#include "classifier.h"
#include <torch/script.h>
#include <torch/torch.h>

class Model {
  torch::jit::script::Module model;
  torch::Device device =
      torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

public:
  Model(const std::string &modelFile);
  bool PredicIsPositiveBinary(float *inputData, int numFeatures);
};

Model::Model(const std::string &modelFile) {
  if (torch::cuda::is_available()) {
    std::cout << "cuda is available" << std::endl;
  } else {
    std::cout << "cuda is not available" << std::endl;
  }

  try {
    model = torch::jit::load(modelFile, this->device);
  } catch (const c10::Error &e) {
    std::cerr << "could not load model. Erorr: " << e.what() << std::endl;
    throw std::invalid_argument("Invalid model file.");
  }
}

bool Model::PredicIsPositiveBinary(float *inputData, int numFeatures) {
  torch::Tensor input =
      torch::from_blob(inputData, {1, numFeatures}, torch::kFloat32)
          .to(this->device);

  this->model.eval();
  torch::Tensor output = this->model.forward({input}).toTensor();
  output = torch::round(torch::sigmoid(output));

  std::cout << output.item<float>() << std::endl;

  // return output.item<float>() >= 0.5;
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
