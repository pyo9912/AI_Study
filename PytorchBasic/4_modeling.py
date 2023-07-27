'''
torch.nn 네임스페이스는 신경망을 구성하는데 필요한 모든 요소 제공
pytorch의 모든 모듈은 nn.Module의 하위 클래스
'''
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 학습을 위한 디바이스 얻기
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device = device)
logits = model(X)  # 모델의 출력값
pred_probab = nn.Softmax(dim=1)(logits)  # 출력값에 대해 softmax 적용
y_pred = pred_probab.argmax(1)  # 열을 기준으로 최대값의 위치 반환
print(f"Predicted class: {y_pred}")


