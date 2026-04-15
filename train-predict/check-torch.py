"""
检查 PyTorch 和 CUDA 是否正确安装，并显示相关信息。"""
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))          # 应该显示 Tesla P100
print(torch.cuda.get_arch_list())