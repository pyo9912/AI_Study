import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

"""
Dataset 준비

Fashion-MNIST: Zalando의 기사 이미지 데이터셋
60,000개의 학습 예제와 10,000개의 테스트 예제로 구성
각 예제는 흑백의 28 * 28 이미지와 10개의 분류 class 중 하나인 정답으로 구성
"""

# 공개 데이터셋에서 학습 데이터 다운로드
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

# 공개 데이터셋에서 테스트 데이터 다운로드
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)


'''
하이퍼 파라미터 설정

epoch: 데이터셋을 반복하는 횟수
batch size: 매개변수가 갱신되기 전에 신경망을 통해 전파된 데이터 샘플 수
learning rate: 각 batch/epoch에서 모델의 매개변수를 조절하는 비율, 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있음
'''
epochs = 5
batch_size = 64


'''
Dataloader 생성
'''

# Dataloader란 dataset을 순회가능한 객체로 감싸 샘플에 쉽게 접근할 수 있도록 하는 역할 수행
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # [batch size, channels, height, width]
    print(f"Shpae of y: {y.shape} {y.dtype}")
    break


'''
모델 만들기
'''

# 학습에 사용할 CPU나 GPU, MPS 장치를 얻음
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 모델 정의
class Model_Name(nn.Module):
    def __init__(self):     # 모델에서 사용될 module, activation function 등 정의
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self, x):    # 모델에서 실행되어야 하는 계산 정의
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = Model_Name().to(device)  # device(cuda)에 최적화된 모델로 변환
print(model)


'''
모델 매개변수 최적화하기
'''

# 모델 학습을 위한 손실함수와 옵티마이저 설정
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        #역전파
        optimizer.zero_grad()  # gradient 초기화
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


'''
모델 저장 및 이용
'''

# 모델 저장하기
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.path")

# 모델 불러오기
model = Model_Name().to(device)
model.load_state_dict(torch.load("model.pth"))

# 모델을 사용하여 task 수행
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')