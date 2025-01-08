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
  try {
    model = torch::jit::load(modelFile);
    model.to(this->device);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model. Erorr: " << e.what() << std::endl;
    throw std::invalid_argument("Invalid model file.");
  }
}

bool Model::PredicIsPositiveBinary(float *inputData, int numFeatures) {
  torch::Tensor input =
      torch::from_blob(inputData, {1, numFeatures}, torch::kFloat32)
          .to(this->device);

  this->model.eval();
  torch::Tensor output = this->model.forward({input}).toTensor();
  output = torch::sigmoid(output);

  return output.item<float>() >= 0.5;
}

mModel NewModel(const char *modelFile) {
  try {
    const auto model = new Model(modelFile);
    return (void *)model;
  } catch (const std::invalid_argument &ex) {
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
