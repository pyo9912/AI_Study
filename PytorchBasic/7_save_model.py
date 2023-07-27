import torch
import torchvision.models as models

'''
모델 저장하기

Pytorch 모델은 학습한 매개변수를 state_dict라고 불리는 내부 상태 사전에 저장
이 상태 값들은 torch.save 메소드를 사용하여 저장할 수 있음
'''
model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights.pth")

model = models.vgg16()  # 여기서는 weights를 지정하지 않았으므로, 학습되지 않은 모델 생성
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()