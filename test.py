import torch

print(torch.__version__)
print("CUDA Available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version: ", torch.version.cuda)
    print("Device Name: ", torch.cuda.get_device_name(0))