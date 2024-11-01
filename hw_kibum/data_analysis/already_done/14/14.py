import numpy as np

# np.array() 함수를 사용하여 2x2 행렬 A를 정의합니다. - [[1, 2], [3, 4]]로 설정 / 각 요소는 행렬의 원소를 나타냄 - 참고: https://numpy.org/doc/stable/reference/generated/numpy.array.html
# 스칼라 값 정의 - 스칼라 값은 숫자 하나로, 행렬과의 곱셈에서 사용할 값입니다.
# 전치 행렬 [ np.transpose() ] 전치 행렬은 행과 열을 바꾸는 작업 - 참고: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
A = np.array([[1, 2], [3, 4]])
scalar = 3
result_transpose = np.transpose(A)

# 2. 행렬식 계산 [ np.linalg.det() ] (determinant) : 행렬이 공간을 어떻게 변형하는지를 나타내는 값
# 2x2 행렬의 경우, 행렬식은 대각선 원소의 곱에서 반대 대각선 원소의 곱을 뺀 값으로 계산  =>  A = [[a, b], [c, d]]일 때, 행렬식은 (a*d) - (b*c)로 계산됩니다.
# 행렬식이 0이 아닌 경우에만 역행렬이 존재 / 참고: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
result_determinant = np.linalg.det(A)

# 3. 역행렬 구하기 [ np.linalg.inv() ] 역행렬 : 행렬 A와 곱했을 때 단위 행렬(identity matrix)을 만드는 행렬 / 역행렬이 존재하려면 행렬식이 0이 아니어야 / 참고: https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
# 4. 스칼라 곱셈 - 스칼라와 행렬 A의 곱을 계산 / 행렬의 각 원소에 스칼라 값을 곱하는 연산 - 참고: https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
result_inverse = np.linalg.inv(A)
result_scalar_multiplication = scalar * A  # np.multiply(scalar, A)로도 계산 가능

print("\n전치 행렬:\n", result_transpose)
print("\n행렬식:\n", result_determinant)
print("\n역행렬:\n", result_inverse)
print("\n스칼라 곱셈:\n", result_scalar_multiplication)

#####

# np.array() 함수를 사용하여 두 벡터 v1과 v2를 정의 / 벡터 v1은 [1, 2, 3], 벡터 v2는 [4, 5, 6] / 참고: https://numpy.org/doc/stable/reference/generated/numpy.array.html
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 벡터 내적 계산 [ np.dot() ] (dot product) : 두 벡터의 대응하는 원소끼리 곱한 후, 그 결과를 모두 더한 값 / 두 벡터가 얼마나 같은 방향을 향하는지를 나타내는 값 / 참고: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
# 벡터 외적 계산 [ np.cross() ] (cross product) : 두 벡터에 수직인 새로운 벡터를 생성 / 그 벡터의 방향과 크기는 두 벡터의 평면에서 정의 / 참고: https://numpy.org/doc/stable/reference/generated/numpy.cross.html
# 벡터 크기(길이) 계산 (L2 노름) [ np.linalg.norm() ] => 각 원소의 제곱을 더한 후 그 합의 제곱근으로 계산 / 벡터가 공간에서 얼마나 긴지를 나타냄 / 참고: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
dot_product = np.dot(v1, v2)
cross_product = np.cross(v1, v2)
v1_magnitude = np.linalg.norm(v1)
v2_magnitude = np.linalg.norm(v2)

# 단위 벡터 계산 (크기가 1인 벡터) [ np.divide() ] 벡터의 각 원소를 그 벡터의 크기(길이)로 나누어 단위 벡터를 계산 / 단위 벡터는 크기가 1, 원래 벡터와 같은 방향 / 참고: https://numpy.org/doc/stable/reference/generated/numpy.divide.html
v1_unit_vector = v1 / v1_magnitude
v2_unit_vector = v2 / v2_magnitude

# 두 벡터 사이의 각도 계산 (코사인 법칙) 
# np.dot() 함수와 np.linalg.norm()을 사용하여 두 벡터 사이의 코사인 각도를 계산합니다.
# np.arccos() 함수를 사용해 코사인의 역수(arc cosine)를 구하여 두 벡터 사이의 각도를 라디안 단위로 반환합니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.arccos.html
cos_theta = np.dot(v1, v2) / (v1_magnitude * v2_magnitude)
angle_between_vectors = np.arccos(cos_theta)  # 라디안 단위로 반환

# 결과 출력
print("1. 벡터 내적 (dot product):", dot_product)
print("2. 벡터 외적 (cross product):\n", cross_product)
print("3. 벡터 v1의 크기:", v1_magnitude)
print("   벡터 v2의 크기:", v2_magnitude)
print("4. 벡터 v1의 단위 벡터:\n", v1_unit_vector)
print("   벡터 v2의 단위 벡터:\n", v2_unit_vector)
print("5. 두 벡터 사이의 각도 (라디안):", angle_between_vectors)
# A와 B 사건을 정의합니다.
# numpy와 같은 라이브러리를 사용하지 않고, 파이썬 집합 연산을 사용하여 구현하세요.

#####

A, B, U = {1, 2, 3, 4}, {3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}  # 전체집합 (U: Universe)

intersection = A.intersection(B)  # 1. 교집합 (A ∩ B)
union = A.union(B)  # 2. 합집합 (A ∪ B)
complement_A = U.difference(A)  # 3. 여집합 (A^c = U - A)

# 전체 확률 법칙 (P(A) + P(A^c) = 1)
P_A = len(A) / len(U)  # 사건 A가 일어날 확률
P_Ac = len(complement_A) / len(U)  # 사건 A가 여집합일 확률

print("1. 사건 A와 B의 교집합 (A ∩ B):", intersection)
print("2. 사건 A와 B의 합집합 (A ∪ B):", union)
print("3. 사건 A의 여집합 (A^c):", complement_A)
print("4. 전체 확률 법칙 (P(A) + P(A^c)):", P_A + P_Ac)

##### 

# 1. 사건 A와 B에 대한 데이터 정의
# A: 사람이 비가 올 것이라고 예측한 사건 (예측)
# B: 실제로 비가 온 사건 (실제 결과)

# 예측 (A): 0은 비가 안 온다고 예측, 1은 비가 온다고 예측
predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])

# 실제 결과 (B): 0은 비가 안 왔음, 1은 비가 옴
actuals = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

# 2. P(A) 계산: 비가 올 것이라고 예측한 사건의 확률 (A=1)
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
P_A = np.mean(predictions)

# 3. P(A ∩ B) 계산: 비가 올 것이라고 예측했고 실제로 비가 온 사건의 확률 (A=1, B=1)
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
P_A_and_B = np.mean((predictions == 1) & (actuals == 1))

# 4. P(B | A) 계산: 조건부 확률 P(B | A) = P(A ∩ B) / P(A)
P_B_given_A = P_A_and_B / P_A

print("1. P(A): 비가 올 것이라고 예측한 확률 =", P_A)
print("2. P(A ∩ B): 비가 올 것이라고 예측했고 실제로 비가 온 사건의 확률 =", P_A_and_B)
print("3. P(B | A): 조건부 확률 (비가 온다고 예측했을 때 실제로 비가 올 확률) =", P_B_given_A)

##### 

# 베이즈 정리를 사용하여 확률을 계산하는 함수
def bayes_theorem(P_X_given_C, P_C, P_X):
    """
    P_X_given_C: 특정 조건에서 단어가 나올 확률
    P_C: 스팸일 확률 또는 스팸이 아닐 확률
    P_X: 단어가 나타날 전체 확률
    
    return: 최종 확률 (새로운 이메일이 스팸일 확률 또는 스팸이 아닐 확률)
    """
    return (P_X_given_C * P_C) / P_X

# 주어진 데이터를 정의 (이메일에 단어가 포함되었는지 여부)
# X는 각각의 이메일에 단어 1, 2가 포함되었는지 나타냄
X = np.array([[1, 0],  # 단어 1은 있고, 단어 2는 없음
              [1, 1],  # 단어 1과 단어 2가 둘 다 있음
              [0, 1],  # 단어 1은 없고, 단어 2는 있음
              [1, 0]]) # 단어 1만 있고, 단어 2는 없음

# 각 이메일이 스팸인지 아닌지 여부 (0: 스팸 아님, 1: 스팸)
y = np.array([0, 1, 0, 1])

# 스팸일 확률과 스팸이 아닐 확률을 계산
P_spam = np.mean(y == 1)  # 전체 이메일 중 스팸일 확률
P_not_spam = np.mean(y == 0)  # 전체 이메일 중 스팸이 아닐 확률

# 스팸인 경우, 단어 1과 단어 2가 각각 나타날 확률 계산
P_word1_given_spam = np.mean(X[y == 1][:, 0])  # 단어 1이 스팸에서 나타날 확률
P_word1_given_not_spam = np.mean(X[y == 0][:, 0])  # 단어 1이 스팸 아님에서 나타날 확률
P_word2_given_spam = np.mean(X[y == 1][:, 1])  # 단어 2가 스팸에서 나타날 확률
P_word2_given_not_spam = np.mean(X[y == 0][:, 1])  # 단어 2가 스팸 아님에서 나타날 확률

# 새로운 이메일 (단어 1은 있고, 단어 2는 없음)
new_email = np.array([1, 0])

# 새 이메일이 스팸일 경우의 확률 계산
P_X_given_spam = P_word1_given_spam * (1 - P_word2_given_spam)  # 단어 1은 있고, 단어 2는 없을 확률
P_X_given_not_spam = P_word1_given_not_spam * (1 - P_word2_given_not_spam)  # 단어 1은 있고, 단어 2는 없을 확률

# 전체 확률 계산 (스팸일 확률과 스팸이 아닐 확률을 모두 고려)
P_X = (P_X_given_spam * P_spam) + (P_X_given_not_spam * P_not_spam)

# 새 이메일이 스팸일 확률과 스팸이 아닐 확률 계산
P_spam_given_X = bayes_theorem(P_X_given_spam, P_spam, P_X)
P_not_spam_given_X = bayes_theorem(P_X_given_not_spam, P_not_spam, P_X)

# 결과 출력
print(f"스팸일 확률: {P_spam_given_X:.2f}")  # 새 이메일이 스팸일 확률
print(f"스팸이 아닐 확률: {P_not_spam_given_X:.2f}")  # 새 이메일이 스팸이 아닐 확률
import numpy as np

# 행렬 A를 정의
# 2x2 행렬 A를 정의하기 위해 np.array() 함수를 사용합니다.
# np.array() 함수는 리스트를 NumPy 배열로 변환해 주는 함수입니다.
# 리스트 [[1, 2], [2, 3]]는 행렬의 각 행을 나타내며, 이를 통해 2x2 크기의 행렬을 생성합니다.
# 이 행렬은 이후 고유값과 고유벡터를 계산하기 위해 필요합니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.array.html
A = np.array([[1, 2], [2, 3]])

# 고유값과 고유벡터 계산
# 행렬 A의 고유값과 고유벡터를 계산하기 위해 np.linalg.eig() 함수를 사용합니다.
# 고유값(eigenvalues)은 행렬의 크기 변화를 나타내는 값이고, 고유벡터(eigenvectors)는 변환 후에도 방향이 유지되는 벡터입니다.
# np.linalg.eig()는 이 두 가지 값을 반환하며, 행렬의 성질을 분석하는 데 유용합니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
eigenvalues, eigenvectors = np.linalg.eig(A)

# 결과 출력
# 계산된 고유값과 고유벡터를 확인하기 위해 print() 함수를 사용하여 출력합니다.
# 이 결과를 통해 행렬 A의 성질을 파악할 수 있습니다.
print(eigenvalues, eigenvectors)

import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
# 선형 회귀 모델을 만들기 위해 X와 Y 데이터를 정의합니다.
# np.array() 함수를 사용하여 리스트 [1, 2, 3, 4, 5]와 [2, 4, 5, 4, 5]를 NumPy 배열 X, Y로 변환합니다.
# np.array()를 사용하는 이유는 행렬 연산을 제공해주기 때문입니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.array.html
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# 선형 회귀 구현
# 선형회귀의 해를 찾기위해 least-squares solution을 사용합니다.
# numpy에서는 lstsq() 함수를 제공하여 least-squares solution을 찾을 수 있습니다.
# lstsq() 함수는 Ax = B 형태의 선형 방정식을 푸는 함수로, A는 입력 데이터, B는 결과 데이터를 의미합니다.
# 이를 통해 선형 회귀 모델의 기울기와 절편을 찾을 수 있습니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
# 또한 least-square연산을 수행할 때, y절편 값을 추가하기 위해 X 데이터에 상수 1을 추가합니다.
# 만약 상수 1을 추가하지 않으면, 선형회귀의 해는 항상 원점을 지나게 됩니다.
# 따라서, np.vstack()를 사용하여 X 데이터와 상수 1을 결합하여 행렬 A를 만듭니다.
# Transpose(T)를 적용해 각 데이터 포인트가 행(row)으로 구성된 형식으로 변환합니다.
# np.linalg.lstsq() 함수는 최소 제곱법을 사용해 주어진 데이터에 맞는 기울기(m)와 절편(c)를 계산합니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
A = np.vstack([X, np.ones(len(X))]).T  # X 데이터와 상수 1을 결합해 행렬 A 생성
m, c = np.linalg.lstsq(A, Y)[0]  # lstsq() 함수로 기울기(m)와 절편(c) 계산


# 결과 출력
# 계산된 기울기(m)와 절편(c)를 확인하기 위해 print() 함수를 사용하여 결과를 출력합니다.
# 이 값들을 통해 데이터와 가장 잘 맞는 직선의 방정식을 알 수 있습니다.
print(f"기울기(m): {m}, 절편(c): {c}")

# 그래프 그리기
# plt.plot()을 사용하여 원본 데이터를 시각화합니다.
# 첫 번째 plt.plot() 함수는 X와 Y 데이터를 'o' 마커로 표시하여 원본 데이터를 시각적으로 나타냅니다.
# label='Original data'는 범례에 데이터가 원본임을 표시하기 위해 사용됩니다.
plt.plot(X, Y, 'o', label='Original data')

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
