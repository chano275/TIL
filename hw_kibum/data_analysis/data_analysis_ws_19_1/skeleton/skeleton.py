import torch
import pickle

# 장치 설정 (CUDA 또는 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------- 로컬 pickle 파일에서 데이터셋 불러오기 --------------------------------------------

# pickle 파일 경로를 지정하여 저장된 데이터셋을 로드합니다.
# pickle은 Python의 객체를 파일로 저장하고 다시 불러올 수 있는 직렬화/역직렬화 도구입니다.
# 이 예시에서는 로컬 파일에 저장된 교통 데이터셋을 불러옵니다.
save_path = '../data/gtsrb_train_dataset_100_random.pkl'  # pickle 파일 경로
with open(save_path, 'rb') as f:  # 파일을 'rb' 모드로 열어 바이너리 읽기 방식으로 불러옵니다.
    train_dataset = __________________  # pickle로 데이터를 불러와 train_dataset에 저장합니다.

# 로컬에서 불러온 데이터셋의 샘플 수를 출력합니다.
print(f"Number of samples loaded(Local File): {len(train_dataset)}")


# ------------------------- Huggingface에서 데이터셋 불러오기 --------------------------------------------

# Huggingface Datasets 라이브러리를 통해 공개된 데이터셋을 불러옵니다.
# Huggingface는 머신러닝 및 AI 모델 및 데이터셋을 공유하는 플랫폼입니다.
# 이 코드에서는 Huggingface에 공개된 'tanganke/gtsrb' 데이터셋을 불러옵니다.
from datasets import load_dataset  # Huggingface에서 제공하는 load_dataset 함수를 임포트합니다.

# Hugging Face에서 GTSRB 데이터셋 로드
# GTSRB(German Traffic Sign Recognition Benchmark)는 독일의 교통 표지판을 인식하는 데이터셋입니다.
# 'tanganke/gtsrb'는 Huggingface에 등록된 GTSRB 데이터셋의 경로입니다.
# https://huggingface.co/datasets/tanganke/gtsrb 에서 데이터셋을 확인할 수 있습니다.
dataset = __________________

# 학습 데이터(train dataset)와 테스트 데이터(test dataset)로 분리합니다.
# Huggingface에서 제공되는 데이터셋은 기본적으로 'train', 'test' 등으로 나누어져 있을 수 있습니다.
# 여기서는 학습용 데이터(train dataset)만 불러옵니다.
train_dataset = dataset['train']

# Huggingface에서 불러온 데이터셋의 샘플 수를 출력합니다.
print(f"Number of samples loaded(Huggingface): {len(train_dataset)}")

