import numpy as np

# 행렬 A를 정의
# 2x2 행렬 A를 정의하기 위해 np.array() 함수를 사용합니다.
# np.array() 함수는 리스트를 NumPy 배열로 변환해 주는 함수입니다.
# 리스트 [[1, 2], [2, 3]]는 행렬의 각 행을 나타내며, 이를 통해 2x2 크기의 행렬을 생성합니다.
# 이 행렬은 이후 고유값과 고유벡터를 계산하기 위해 필요합니다.
# 참고: https://numpy.org/doc/stable/reference/generated/numpy.array.html
A = np.array([[1, 2], [2, 3]])
# print(A)

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


