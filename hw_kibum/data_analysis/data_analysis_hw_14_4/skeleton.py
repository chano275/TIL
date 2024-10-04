import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
# 선형 회귀 모델을 만들기 위해 X와 Y 데이터를 정의합니다.
# np.array() 함수를 사용하여 리스트 [1, 2, 3, 4, 5]와 [2, 4, 5, 4, 5]를 NumPy 배열 X, Y로 변환합니다.
# 이렇게 해야 데이터를 쉽게 처리할 수 있기 때문에 np.array() 함수를 사용합니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.array.html
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# 선형 회귀 구현
# np.vstack() 함수는 X 데이터와 상수 1을 결합하여 행렬 A를 만듭니다. / Transpose(T)를 적용해 각 데이터 포인트가 행(row)으로 구성된 형식으로 변환합니다.
# np.linalg.lstsq() 함수는 최소 제곱법을 사용해 주어진 데이터에 맞는 기울기(m)와 절편(c)를 계산합니다.

# np.linalg.lstsq() 함수를 사용해 선형 회귀를 구현하기 위해 입력 데이터를 준비합니다.
# rcond=None은 최신 NumPy 버전에서 작은 특이값을 무시하지 않도록 설정하는 옵션입니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
A = np.vstack([X, np.ones(len(X))]).T   # X 데이터와 상수 1을 결합해 행렬 A 생성
m, c = np.linalg.lstsq(A, Y, rcond=None)[0]  # lstsq() 함수로 기울기(m)와 절편(c) 계산

# 결과 출력
# 계산된 기울기(m)와 절편(c)를 확인하기 위해 print() 함수를 사용하여 결과를 출력합니다.
# 이 값들을 통해 데이터와 가장 잘 맞는 직선의 방정식을 알 수 있습니다.
print(f"기울기(m): {m}, 절편(c): {c}")


# 그래프 그리기
# plt.plot()을 사용하여 원본 데이터를 시각화합니다.
# 첫 번째 plt.plot() 함수는 X와 Y 데이터를 'o' 마커로 표시하여 원본 데이터를 시각적으로 나타냅니다.
# label='Original data'는 범례에 데이터가 원본임을 표시하기 위해 사용됩니다.
plt.plot(X, Y, 'o', label='Original data', markersize=10)

# 두 번째 plt.plot() 함수는 선형 회귀로 계산된 기울기(m)와 절편(c)를 사용하여 회귀선을 그립니다.
# m*X + c는 직선 방정식 y = mx + c를 의미하며, X 값에 대응하는 회귀선을 계산해 시각화합니다.
# 'r'은 회귀선을 빨간색으로 그리기 위한 옵션입니다.
# label='Fitted line'은 범례에 회귀선임을 표시하기 위해 사용됩니다.
plt.plot(X, m*X + c, 'r', label='Fitted line')

# plt.legend()는 범례를 추가하여 원본 데이터와 회귀선이 어떤 것인지 쉽게 구분할 수 있도록 합니다.
plt.legend()

# plt.show()는 그래프를 화면에 표시하는 명령어로, 데이터와 회귀선을 시각적으로 확인할 수 있습니다.
# 참고: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
plt.show()
