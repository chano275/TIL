import torch
import random
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ------------------------- 로컬 파일에서 데이터셋 불러오기 --------------------------------------------
# pickle 파일 경로를 지정하여 저장된 데이터셋을 로드합니다.
# pickle은 Python의 객체를 파일로 저장하고 다시 불러올 수 있는 직렬화/역직렬화 도구입니다.
save_path = '../data/gtsrb_train_dataset_100_random.pkl'  # pickle 파일 경로
with open(save_path, 'rb') as f:  # 파일을 'rb' 모드로 열어 바이너리 읽기 방식으로 불러옵니다.
    train_dataset = pickle.load(f)  # pickle로 데이터를 불러와 train_dataset에 저장합니다.


# ------------------------- 기본 전처리 설정 --------------------------------------------
# torchvision.transforms 모듈의 Compose 함수를 사용하여 일련의 이미지 전처리 변환을 적용합니다.
# https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose 참고
# Compose는 여러 전처리 단계를 순차적으로 적용할 수 있도록 해주는 유용한 함수입니다.
# 예를 들어, 이미지를 텐서로 변환하고 정규화하는 과정을 한 번에 처리할 수 있습니다.

# V1 API 기준으로 파악
transform = transforms.Compose([
    # 참고 https://pytorch.org/vision/stable/transforms.html#v1-api-reference
    
    # transforms.Resize() 함수는 torchvision.transforms의 Resize 함수를 사용하여 
    # 이미지의 크기를 224x224로 조정합니다.
    transforms.____________________,

    # transforms.ToTensor() 함수는 torchvision.transforms의 ToTensor 함수를 사용하여 
    # 이미지를 PyTorch 텐서로 변환합니다.
    transforms.____________________,
    
    # transforms.Normalize(mean, std)는 이미지의 각 채널에 대해 지정된 평균(mean)과 표준편차(std)로 정규화합니다. 모두 0.5로 맞춰주세요.
    transforms.____________________  
    # 각 채널(R, G, B)의 평균을 0.5, 표준편차를 0.5로 설정하여 정규화. 정규화는 학습을 더 빠르고 안정적으로 진행하게 도와줍니다.

])


# ------------------------- 커스텀 Dataset 클래스 정의 --------------------------------------------
# PyTorch의 Dataset 클래스를 상속받아 데이터셋을 정의합니다.
# 이 클래스는 '__len__' 메서드를 통해 데이터셋의 크기를 반환하고,
# '__getitem__' 메서드를 통해 특정 인덱스의 이미지와 라벨을 반환합니다.
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 참고
class GTSRBDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # pickle로 불러온 데이터를 저장
        self.transform = transform  # 전처리 설정을 저장

    def __len__(self):
        return len(self.data)  # 데이터셋의 크기 반환

    def __getitem__(self, idx):
        # i번째 이미지 데이터를 self.data['image'] 리스트에서 가져옵니다.
        image = ____________________  # i번째 image를 저장된 self.data attribute에서 가져올 수 있습니다.

        # i번째 이미지의 라벨을 self.data['label'] 리스트에서 가져옵니다.
        label = ____________________  # i번째 label을 저장된 self.data attribute에서 가져올 수 있습니다.

        # 전처리가 설정된 경우, 이미지에 지정된 변환(transform)을 적용합니다.
        if self.transform:
            image = ____________________  # 이미지 전처리 수행
        return image, label  # 전처리된 이미지와 해당 이미지의 라벨을 반환합니다.


# ------------------------- DataLoader 설정 --------------------------------------------
# DataLoader는 배치 단위로 데이터를 로드하며, 학습 시 데이터를 효율적으로 불러올 수 있도록 합니다.
# batch_size=32는 한 번에 32개의 이미지를 로드하겠다는 의미이며, shuffle=True는 데이터의 순서를 무작위로 섞습니다.
train_dataset = GTSRBDataset(train_dataset, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ------------------------- 첫 번째 배치 확인 --------------------------------------------
# 데이터 로더에서 첫 번째 배치를 확인하는 함수
# 데이터를 제대로 로드하고 전처리가 잘 적용되었는지 확인하는 데 유용합니다.
def check_data_loader(loader):
    data_iter = iter(loader)  # 데이터 로더에서 반복자(iterator) 생성
    images, labels = next(data_iter)  # 첫 번째 배치의 이미지와 라벨을 가져옴
    print(f"Image batch shape: {images.shape}")  # 배치의 이미지 크기 확인
    print(f"Labels : {labels}")  # 배치의 라벨 크기 확인

# 첫 번째 배치를 출력하여 확인합니다.
check_data_loader(train_loader)
