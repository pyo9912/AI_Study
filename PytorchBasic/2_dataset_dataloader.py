import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''
Dataset 생성
'''
training_data = datasets.FashionMNIST(
    root = "data",          # 학습/테스트 데이터가 저장되는 경로
    train = True,           # 학습용/테스트용 여부 결정
    download = True,        # root에 데이터가 없는 경우 인터넷에서 다운로드
    transform = ToTensor()  # feature와 label의 변형(transform)을 지정
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)


'''
Dataset 순회 및 시각화
'''
labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


'''
파일에서 사용자 정의 데이터셋 만들기
'''

import pandas as pd
from torchvision.io import read_image

# 사용자 정의 Dataset은 반드시 3개의 함수를 구현해야 함
## __init__ 함수는 dataset 객체가 초기화될 때 한 번 수행되며 image, annotation file 디렉토리와 data 및 label에 대한 tranform 초기화
## __len__ 함수는 dataset 샘플의 총 개수 리턴
## __getitem__ 함수는 dataset으로부터 주어진 인덱스 idx의 샘플 반환
class CustomImageDataset(Dataset):
    def __init__(self, annotations_File, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_File, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

'''
Dataloader로 학습용 데이터 준비하기
'''

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Dataloader를 통해 순회하기
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")