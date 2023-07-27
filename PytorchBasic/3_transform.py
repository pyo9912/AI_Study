'''
데이터는 항상 머신러닝 알고리즘 학습에 필요한 최종 처리가 된 형태로 제공되지 않음
그래서 Transform (변형)을 통해 데이터를 조작하고 학습에 적합하게 만들고자 함
torchvision.transform 모듈은 주로 사용하는 몇가지 변형을 제공
'''
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
)

# ToTensor는 PIL image나 numpy array를 FloatTensor로 변환하고, 이미지의 픽셀의 크기값을 [0.,1.] 범위로 비례하여 조정함
# Lambda는 사용자 정의 람다 함수 적용. 여기서는 정수를 one-hot으로 부호화된 tensor로 바꾸는 함수 사용

