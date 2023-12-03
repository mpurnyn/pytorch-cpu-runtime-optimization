import torch
import torchvision.models as models
import torchvision
import torch.jit # for torchscript generation
import time
import torch.quantization # for speedup through quantization


class TestModel(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super(TestModel, self).__init__()
        self.resnet18 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet18.eval()

    def forward(self, x, threshold):
        with torch.no_grad():
            outputs = self.resnet18(x)
            probabilities, indices = torch.max(outputs.data, 1)
            probabilities[probabilities < threshold] = -1
            indices[probabilities < threshold] = -1
        return probabilities, indices
    
def generate_ts_model(model: TestModel) -> None:
    # Generates a TorchScript model from the input model
    # from the docs: https://pytorch.org/tutorials/advanced/cpp_export.html
    # create some sample input
    example = torch.rand(1, 3, 224, 224)
    threshold = torch.tensor([0.5])
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing and save locally.
    traced_script_module = torch.jit.trace(model, (example, threshold))
    traced_script_module.save("model.pt")

def measure_runtime_speedup(model: TestModel) -> None:
    # From the Docs I used oneDNN and Dynamic Quanization 
    # https://pytorch.org/docs/stable/quantization.html
    
    # Measure a baseline runtime
    example = torch.rand(1, 3, 224, 224)
    threshold = torch.tensor([0.5])
    start_time = time.time()
    _ = model(example, threshold)
    end_time = time.time()
    before_speedup_time = (end_time - start_time) * 1000  # Convert to milliseconds

    
    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Time the quantized model
    start = time.time()
    _ = quantized_model(example, threshold)
    end = time.time()
    after_speedup_time = end - start

    print(f"Runtime before speedup: {before_speedup_time} ms")
    print(f"Runtime after oneDNN: {after_speedup_time} ms")
    print(f"Total speedup: {before_speedup_time / after_speedup_time}x")

def main() -> None:
    print("Intatiating Resnet-18 Model")
    model = TestModel()
    print("Generating TorchScript Model")
    generate_ts_model(model)
    print("Measuring Runtime Speedup")
    measure_runtime_speedup(model)

if __name__ == "__main__":
    main()
