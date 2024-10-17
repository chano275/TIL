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
        image = _____________________  # i번째 image를 저장된 self.data attribute에서 가져올 수 있습니다.

        # i번째 이미지의 라벨을 self.data['label'] 리스트에서 가져옵니다.
        label = _____________________  # i번째 label을 저장된 self.data attribute에서 가져올 수 있습니다.

        # 전처리가 설정된 경우, 이미지에 지정된 변환(transform)을 적용합니다.
        if self.transform:
            image = _____________________  # 이미지 전처리 수행
        return image, label  # 전처리된 이미지와 해당 이미지의 라벨을 반환합니다.


# ------------------------- 전처리 설정 --------------------------------------------
# torchvision.transforms 모듈의 Compose 함수를 사용하여 일련의 이미지 전처리 변환을 적용할 수 있습니다.
# 이 예시에서는 기본적인 전처리와 흑백(그레이스케일) 전처리를 적용합니다.
# https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose 참고

# V1 API 기준으로 파악
transform = transforms.Compose([
    # 참고 https://pytorch.org/vision/stable/transforms.html#v1-api-reference
    
    # transforms.Resize() 함수는 torchvision.transforms의 Resize 함수를 사용하여 
    # 이미지의 크기를 224x224로 조정합니다.
    transforms._____________________,

    # transforms.ToTensor() 함수는 torchvision.transforms의 ToTensor 함수를 사용하여 
    # 이미지를 PyTorch 텐서로 변환합니다.
    transforms._____________________,
    
    # transforms.Normalize(mean, std)는 이미지의 각 채널에 대해 지정된 평균(mean)과 표준편차(std)로 정규화합니다. 모두 0.5로 맞춰주세요.
    transforms._____________________
    # 각 채널(R, G, B)의 평균을 0.5, 표준편차를 0.5로 설정하여 정규화. 정규화는 학습을 더 빠르고 안정적으로 진행하게 도와줍니다.

])

# 그레이스케일 적용 전처리 설정 (이미지를 흑백으로 변환)
transform_grayscale = transforms.Compose([
    # transforms.Resize() 함수는 torchvision.transforms의 Resize 함수를 사용하여 
    # 이미지의 크기를 224x224로 조정합니다.
    transforms._____________________,

    # transforms.Grayscale() 함수는 torchvision.transforms의 Grayscale 함수를 사용하여 
    # 이미지를 흑백으로 변환합니다. num_output_channels는 1로 설정하여 채널 수를 1로 만듭니다.
    transforms._____________________,

    # transforms.ToTensor() 함수는 torchvision.transforms의 ToTensor 함수를 사용하여 
    # 이미지를 PyTorch 텐서로 변환합니다.
    transforms._____________________,

    # transforms.Normalize(mean, std)는 흑백 이미지의 한 채널(mean=[0.5])에 대해 지정된 평균(mean)과 표준편차(std)로 정규화합니다.
    transforms._____________________  

])


# ------------------------- 이미지 시각화 함수 --------------------------------------------
# 이미지를 시각화하는 함수입니다. 이미지를 정규화에서 원래 값으로 복원한 뒤, numpy 형식으로 변환하고
# 이를 Matplotlib을 사용하여 출력합니다.
def imshow(img, title="Image"):
    img = img / 2 + 0.5  # 정규화 해제
    npimg = img.numpy()  # 텐서를 numpy 배열로 변환
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray" if img.shape[0] == 1 else None)  # 흑백 이미지이면 cmap을 'gray'로 설정
    plt.title(title)
    plt.show()


# ------------------------- 전처리 전과 후 이미지 출력 함수 --------------------------------------------
# 이 함수는 전처리 전과 후의 이미지를 시각적으로 비교하는 데 사용됩니다.
def show_preprocessing_comparison(data, idx):
    # 원본 이미지 (이미 PIL 이미지로 되어있으므로 Image.open을 사용하지 않음)
    original_image = data['image'][idx]

    # 전처리 전 이미지 출력
    plt.imshow(original_image)  # 전처리 전의 원본 이미지를 출력
    plt.title('Before Preprocessing')
    plt.show()

    # 전처리 후 이미지 출력 (기본 전처리)
    transformed_image = transform(original_image)  # 기본 전처리를 적용
    imshow(transformed_image, title='After Basic Preprocessing')

    # 전처리 후 이미지 출력 (그레이스케일 전처리)
    transformed_image_grayscale = transform_grayscale(original_image)  # 그레이스케일 전처리를 적용
    imshow(transformed_image_grayscale, title='After Grayscale Preprocessing')


# ------------------------- 로컬 파일에서 데이터셋 불러오기 --------------------------------------------
# pickle 파일 경로를 지정하여 저장된 데이터셋을 로드합니다.
# pickle은 Python의 객체를 파일로 저장하고 다시 불러올 수 있는 직렬화/역직렬화 도구입니다.
save_path = '../data/gtsrb_train_dataset_100_random.pkl'  # pickle 파일 경로
with open(save_path, 'rb') as f:  # 파일을 'rb' 모드로 열어 바이너리 읽기 방식으로 불러옵니다.
    train_dataset = pickle.load(f)  # pickle로 데이터를 불러와 train_dataset에 저장합니다.


# ------------------------- 전처리 전과 후 비교 시각화 --------------------------------------------
# 첫 번째 이미지를 사용하여 전처리 전과 후의 차이를 시각적으로 확인합니다.
show_preprocessing_comparison(train_dataset, 0)  # 0번째 인덱스의 이미지를 사용
