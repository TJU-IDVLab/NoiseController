import torch
import torch.nn as nn
import torch.nn.functional as F
import os
class MLP(torch.nn.Module):

    def __init__(self,num_i,num_h,num_o):
        super().__init__()
        self.linear1=torch.nn.Linear(num_i,num_h)

    def forward(self, x):
        x1 = self.linear1(x)
        return x1

    def save_pretrained(self, file_path):
        if not os.path.isdir(file_path):
            os.makedirs(file_path, exist_ok=True)

        torch.save(self.state_dict(), f"{file_path}/mmodel.bin")
