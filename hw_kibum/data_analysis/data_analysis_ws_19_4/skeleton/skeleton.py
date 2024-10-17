import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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
        image = ___________________  # i번째 image를 저장된 self.data attribute에서 가져올 수 있습니다.

        # i번째 이미지의 라벨을 self.data['label'] 리스트에서 가져옵니다.
        label = ___________________  # i번째 label을 저장된 self.data attribute에서 가져올 수 있습니다.

        # 전처리가 설정된 경우, 이미지에 지정된 변환(transform)을 적용합니다.
        if self.transform:
            image = ___________________  # 이미지 전처리 수행
        return image, label  # 전처리된 이미지와 해당 이미지의 라벨을 반환합니다.


# ------------------------- Augmentation 전처리 설정 --------------------------------------------
# 다양한 이미지 증강 기법을 사용하여 데이터의 다양성을 증가시킵니다.
# transforms.Compose()는 여러 변환을 차례로 적용할 수 있게 합니다.
# https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose

# V1 API 기준으로 파악
transform_augmentation = transforms.Compose([
    # 참고 https://pytorch.org/vision/stable/transforms.html#v1-api-reference
    
    # transforms.Resize() 함수는 torchvision.transforms의 Resize 함수를 사용하여 
    # 이미지의 크기를 224x224로 조정합니다.
    transforms.___________________,  # 이미지 크기를 224x224로 조정 
    # 이미지의 크기를 일정하게 맞춰주기 위해 사용. 모델에 일관된 입력 크기를 제공함으로써 학습의 효율성을 높입니다.

    # transforms.RandomHorizontalFlip() 함수는 지정된 확률(p)로 이미지를 좌우 반전합니다.
    transforms.___________________,  # 100% 확률로 좌우 반전 
    # 이미지가 좌우 대칭인 경우 (예: 교통 표지판) 좌우 반전이 의미가 있을 수 있음. 데이터 증강을 통해 모델이 다양한 이미지를 학습하도록 돕습니다.

    # transforms.RandomRotation() 함수는 지정된 각도 범위 내에서 이미지를 무작위로 회전시킵니다.
    transforms.___________________,  # 이미지를 30도 이내로 무작위 회전 
    # 이미지가 회전된 상태에서도 잘 인식되도록 회전을 적용. 최대 30도 이내로 무작위로 회전시킵니다.

    # transforms.RandomResizedCrop() 함수는 이미지를 무작위로 자르고 지정된 크기로 리사이즈합니다.
    transforms.___________________,  # 랜덤으로 잘라내고 다시 224x224로 리사이즈
    # 이미지의 특정 부분을 무작위로 자르고, 다시 지정된 크기(224x224)로 리사이즈하여 학습 데이터에 변화를 줌. `scale=(0.8, 1.0)`은 80%에서 100% 크기의 이미지를 자르는 범위를 지정합니다.
    
    # transforms.ToTensor() 함수는 PIL 이미지를 PyTorch 텐서로 변환합니다.
    transforms.___________________,  # PIL 이미지를 텐서로 변환 

    # transforms.Normalize(mean, std) 함수는 각 채널의 평균(mean)과 표준편차(std)를 지정하여 
    # 이미지 값을 정규화합니다.
    transforms.___________________
    # 각 채널(R, G, B)의 평균을 0.5, 표준편차를 0.5로 설정하여 정규화. 정규화는 학습을 더 빠르고 안정적으로 진행하게 도와줍니다.
])


# ------------------------- 데이터 시각화 함수 --------------------------------------------
# 이미지를 시각화하는 함수입니다. 이미지의 정규화를 해제한 후 numpy 배열로 변환하고,
# Matplotlib을 사용하여 이미지를 시각화합니다.
def imshow(img, title="Image"):
    img = img / 2 + 0.5  # 정규화 해제
    npimg = img.numpy()  # 텐서를 numpy 배열로 변환
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 채널 순서를 맞춰서 출력
    plt.title(title)  # 제목 설정
    plt.show()


# ------------------------- Augmentation 전과 후 비교 함수 --------------------------------------------
# 원본 이미지와 Augmentation 적용 후의 이미지를 시각적으로 비교하는 함수입니다.
def show_augmentation_comparison(data, idx):
    # 원본 이미지
    original_image = data['image'][idx]

    # 전처리 전 이미지 출력
    plt.imshow(original_image)  # 원본 이미지를 출력
    plt.title('Before Augmentation')
    plt.show()

    # Augmentation 적용 후 이미지 출력
    augmented_image = transform_augmentation(original_image)  # Augmentation 전처리를 적용
    imshow(augmented_image, title='After Augmentation')  # 전처리 후 이미지를 출력


# ------------------------- 로컬 파일에서 데이터셋 불러오기 --------------------------------------------
# pickle 파일 경로를 지정하여 저장된 데이터셋을 로드합니다.
# pickle은 Python의 객체를 파일로 저장하고 다시 불러올 수 있는 직렬화/역직렬화 도구입니다.
save_path = '../data/gtsrb_train_dataset_100_random.pkl'  # pickle 파일 경로
with open(save_path, 'rb') as f:  # 파일을 'rb' 모드로 열어 바이너리 읽기 방식으로 불러옵니다.
    train_dataset = pickle.load(f)  # pickle로 데이터를 불러와 train_dataset에 저장합니다.


# ------------------------- 전처리 전과 후 비교 시각화 --------------------------------------------
# 첫 번째 이미지를 사용하여 Augmentation 전과 후의 차이를 시각적으로 확인합니다.
show_augmentation_comparison(train_dataset, 0)  # 0번째 인덱스의 이미지를 사용