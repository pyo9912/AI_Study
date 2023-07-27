import torch
import numpy as np

'''
Tensor
Tensor란? 데이터의 배열
Tensor의 Rank란? 배열의 차원
'''

# tensor 초기화
## 데이터로부터 직접 tensor 생성
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

## Numpy 배열로부터 tensor 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

## 다른 tensor로부터 tensor 생성
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # tensor 속성 덮어주기
print(f"Random Tensor: \n {x_rand} \n")

## 무작위 random 값 또는 상수값 사용하기
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


'''
Tensor의 속성 (Attribute): tensor의 모양(shape), 자료형(datatpe) 및 어느 장치에 저장되는지 나타냄
'''
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatpe of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


'''
Tensor 연산
'''
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First col: {tensor[:,0]}")
print(f"Last col: {tensor[...,-1]}")
tensor[:,1] = 0
print(tensor)

# Tensor 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 산술 연산
## 행렬 곱
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

## 요소 곱
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

## 요소 집계
agg = tensor.sum()
agg_item = agg.item()  # tensor에 하나의 값만 존재할 경우 해당 scalar 값 반환
print(agg_item, type(agg.item))

# in-place 연산 (바꿔치기 연산)
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


'''
NumPy 변환
'''

# Tensor -> Numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Numpy -> Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")