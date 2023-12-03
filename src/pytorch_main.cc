#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <torch/script.h>

int main(int argc, char *argv[]) {
  // from the docs: https://pytorch.org/tutorials/advanced/cpp_export.html

  // check for arguments
  if (argc != 2) {
    std::cerr << "usage: pytorch_main.cc <path-to-exported-script-module>\n";
    return -1;
  }

  // Attempt to Deserialize the Model
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "Failed to load model. Exiting.\n";
    return -1;
  }
  
  // Create a vector of inputs for test data and threshold
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randn({1, 3, 224, 224}));
  inputs.push_back(torch::tensor(0.5));

  // 1. Attempt to Perform Inference
  auto output = module.forward(inputs).toTuple();
  try {
    output = module.forward(inputs).toTuple();
  }
  catch (const c10::Error& e) {
    std::cerr << "Error during inference. Exiting.\n";
    return -1;
  }

  //2. Report the number of parameters
  size_t num_parameters = 0;
  for (const auto& parameter : module.parameters()) {
      num_parameters += parameter.numel();
  }
  std::cout << "Total number of parameters: " << num_parameters << std::endl;  
  std::cout << "finished fine!" << std::endl;
}
