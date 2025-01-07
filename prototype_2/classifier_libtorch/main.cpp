#include "classifier.h"
#include <iostream>
#include <torch/script.h>

int main() {
  // torch::jit::script::Module module;
  // try {
  //   module = torch::jit::load("model.pt");
  // } catch (const c10::Error &e) {
  //   std::cerr << "error loading the model" << std::endl;
  //   return -1;
  // }
  const char *modelFile = "../../data/fedmedian_model.pt";
  float inputData[] = {6, 0.0,  48, 1, 40,     1,      22, 2,  20,   4294952,
                       0, 0,    0,  0, 48,     40,     40, 48, 48.0, 40.0,
                       0, 0,    0,  0, 384000, 320000, 2,  0,  0,    0,
                       0, 4096, 0,  0, 0,      0,      0,  0,  0};
  int numFeatures = 39;

  mModel model = NewModel(modelFile);
  if (model == nullptr) {
    std::cerr << "Failed to load model!" << std::endl;
    return -1;
  }

  int prediction = PredictIsPositiveBinary(model, inputData, numFeatures);
  if (prediction == -1) {
    std::cerr << "Error in prediction." << std::endl;
  } else {
    std::cout << "Prediction: " << (prediction == 1 ? "Positive" : "Negative")
              << std::endl;
  }

  DeleteModel(model);
  return 0;
}
