import torch
import pandas as pd
from torch.utils.data import Dataset


# Dataset 클래스는 PyTorch에서 데이터셋을 정의할 때 사용하는 기본 클래스입니다.
# __len__() 메서드를 구현하여 데이터셋의 길이를 반환하고,
# __getitem__() 메서드를 구현하여 특정 인덱스의 데이터를 반환하는 방식으로 데이터셋을 정의할 수 있습니다.
# PyTorch의 DataLoader와 함께 사용하여 데이터를 쉽게 배치 단위로 처리할 수 있습니다.
# 참고 페이지: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# Dataset을 상속받아 교통데이터에 맞게 사용할 수 있도록 수정합니다.
class TrafficDataset(Dataset):
    def __init__(self, excel_file, sheet_name=0):
        self.data = pd.read_excel(excel_file, sheet_name=sheet_name)        # 데이터를 로드 (엑셀 파일을 읽어와 pandas DataFrame으로 저장)
        self.data['혼잡'] = self.data['혼잡'].astype(int)        # 범주형 데이터를 숫자형으로 변환 (혼잡 여부: 0 또는 1로 변환)
        self.features = self.data[['7시', '8시', '9시', '10시']]        # 필요한 열만 선택하여 features에 저장 (7시, 8시, 9시, 10시 시간대의 교통량)
        self.labels = self.data['혼잡']        # 혼잡 여부를 라벨로 설정 (이 데이터셋에서 타겟 값으로 사용)

    def __len__(self):        return len(self.data)        # 데이터셋의 길이 반환 (전체 샘플 개수)

    def __getitem__(self, idx):
        # 1. 특징(feature) 데이터를 가져옴
        sample_data = self.features.iloc[idx].values  # 특징 데이터를 가져옴 (NumPy 배열)

        # 2. 특징 데이터를 float 형으로 변환
        sample_data = sample_data.astype(float)

        # 3. 특징 데이터를 PyTorch 텐서로 변환
        # 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        sample_tensor = torch.tensor(sample_data, dtype=torch.float)

        # 4. 라벨(label)을 가져옴
        label_data = self.labels.iloc[idx]

        # 5. 라벨 데이터를 PyTorch 텐서로 변환
        # 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        label_tensor = torch.tensor(label_data, dtype=torch.long)

        # 6. 텐서 형태의 특징과 라벨을 반환
        return sample_tensor, label_tensor


if __name__ == '__main__':
    # 데이터셋 예시 사용
    traffic_dataset = TrafficDataset('../data/weekday_traffic.xlsx')
    print(f"데이터셋 개수: {len(traffic_dataset)}")

    # 데이터셋에서 특정 인덱스의 데이터를 가져오기
    sample_data, sample_label = traffic_dataset[0]  # 첫 번째 데이터
    print(f"첫 번째 데이터 (특징): {sample_data}")
    print(f"첫 번째 데이터 (레이블): {sample_label}")

    sample_data, sample_label = traffic_dataset[5]  # 두 번째 데이터
    print(f"두 번째 데이터 (특징): {sample_data}")
    print(f"두 번째 데이터 (레이블): {sample_label}")
