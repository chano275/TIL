# practice_2를 마치고 풀어주세요!!!!!!!!!
import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.utils.data import Dataset

# Dataset 클래스는 PyTorch에서 데이터셋을 정의할 때 사용하는 기본 클래스입니다.
# __len__() 메서드를 구현하여 데이터셋의 길이를 반환하고,
# __getitem__() 메서드를 구현하여 특정 인덱스의 데이터를 반환하는 방식으로 데이터셋을 정의할 수 있습니다.
# PyTorch의 DataLoader와 함께 사용하여 데이터를 쉽게 배치 단위로 처리할 수 있습니다.
# 참고 페이지: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# Dataset을 상속받아 교통데이터에 맞게 사용할 수 있도록 수정합니다.

### 뉴럴넷 학습 : 데이터 / 데이터 로더 / 모델 필요 > 학습

class TrafficDataset(Dataset):  ### 가지고 있는 데이터를 idx로 접근할 수 있게 해주는 class
    def __init__(self, excel_file, sheet_name=0):
        self.data = pd.read_excel(excel_file, sheet_name=sheet_name)          # 데이터를 로드 (엑셀 파일을 읽어와 pandas DataFrame으로 저장)
        self.data['혼잡'] = self.data['혼잡'].astype(int)          # 범주형 데이터를 숫자형으로 변환 (혼잡 여부: 0 또는 1로 변환)
        self.features = self.data[['8시', '9시', '10시']]          # 필요한 열만 선택하여 features에 저장 (8시, 9시, 10시 시간대의 교통량)
        self.labels = self.data['혼잡']          # 혼잡 여부를 라벨로 설정 (이 데이터셋에서 타겟 값으로 사용)

    def __len__(self):  return len(self.data)        # 데이터셋의 길이 반환 (전체 샘플 개수)

    ### 데이터 로더 : 학습을 시킬 때에는 SGD로 시킴 & 아래 설명에서 RANDOM하게 데이터 뽑아주는 애
    # GD = Gradient Descent : 역전파로 계산된 loss를 가지고 loss가 줄어드는 방향으로 param을 업데이트 하는 것
    # GD를 할 때에 DATA를 batch size(4)로 쪼개서 해당 4개 를 보고 loss를 계산 => 그 loss에 따라 update
    # 이 4개로 쪼개는걸 Stocastic 하게 한다 => SGD

    ### 모델 : 뉴럴넷

    def __getitem__(self, idx):
        # 1. 특징(feature) 데이터를 가져옴
        sample_data = self.features.iloc[idx].values      # 특징 데이터를 가져옴 (NumPy 배열)
        sample_data = sample_data.astype(float)          # 2. 특징 데이터를 float 형으로 변환
        sample_tensor = torch.tensor(sample_data)        # 3. 특징 데이터를 PyTorch 텐서로 변환 & 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        label_data = self.labels.iloc[idx]               # 4. 라벨(label)을 가져옴
        label_tensor = torch.tensor(label_data)          # 5. 라벨 데이터를 PyTorch 텐서로 변환 & 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        return sample_tensor, label_tensor               # 6. 텐서 형태의 특징과 라벨을 반환

# 데이터셋 생성
# TrafficDataset 클래스를 이용해 엑셀 파일에서 데이터를 읽어와 데이터셋을 생성
traffic_dataset = TrafficDataset('../data/weekday_traffic.xlsx')

# DataLoader 설정
# DataLoader는 데이터셋을 배치 단위로 로드하도록 돕는 클래스입니다.
# batch_size는 8로 설정
# shuffle 인자를 이용해서 데이터를 무작위로 섞어서 학습해야합니다.
# num_workers는 데이터를 로드하는 동안 병렬로 처리할 수 있도록 하는 프로세스 수를 지정 (기본값 0)
batch_size = 8
# 참고: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
traffic_loader = DataLoader(traffic_dataset,batch_size=batch_size, shuffle=True)

# DataLoader를 통해 데이터 불러오기
# DataLoader는 배치 단위로 데이터를 불러오며, 이 예시에서는 각 배치의 특징 데이터와 라벨 데이터를 출력
for batch_idx, (data, target) in enumerate(traffic_loader):
    # 각 배치에서 가져온 데이터와 라벨의 크기를 출력
    print(f"배치 {batch_idx+1} - 데이터 크기: {data.size()}, 레이블 크기: {target.size()}")