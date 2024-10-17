import cv2

# Haar Cascade Classifier 파일 경로
# 얼굴을 탐지하기 위해 미리 학습된 모델 파일을 불러온다.
# Haar Cascade Classifier는 얼굴의 특징을 감지하기 위한 도구이다.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 1. 이미지 불러오기
# 얼굴을 탐지할 대상을 메모리로 불러오기 위해 이미지를 읽는다.
img = cv2.imread('../data/human_face.png')

# 2. 흑백 이미지로 변환
# 컬러 이미지를 흑백 이미지로 변환합니다.
# cvtColor 함수를 사용하여 이미지를 흑백으로 변환합니다.
# 간단한 모델의 경우 흑백 이미지로 변환하여 처리하는 것이 효율적일 수 있습니다.
# 따라서, 컬러 이미지를 흑백 이미지로 변환하겠습니다.
# 참고: https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#gaf86c09fe702ed037c03c2bc603ceab14
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 얼굴 탐지 수행
# 이미지에서 얼굴을 탐지하기 위해 detectMultiScale 함수를 사용한다.
# scaleFactor를 설정하여 다양한 크기의 얼굴을 탐지할 수 있게 하고,
# minNeighbors를 통해 얼굴 후보군을 더 엄격하게 확인하여 실제 얼굴을 찾는다.
# minSize는 탐지할 얼굴의 최소 크기를 지정하여 너무 작은 얼굴을 무시하기 위해 사용한다.
# detectMultiScal 함수 참고: https://docs.opencv.org/4.x/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# 4. 탐지된 얼굴 주위에 사각형 그리기
# 탐지된 얼굴 위치를 시각적으로 표시하기 위해 얼굴 주위에 사각형을 그린다.
# cv2.rectangle 함수를 사용하여 (x, y, w, h) 좌표를 기준으로 사각형을 그린다.
# (255, 0, 0)은 파란색을 나타내며, 선의 두께를 2로 설정한다.
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 5. 결과 이미지 출력
# 탐지된 얼굴 결과를 화면에 보여주기 위해 이미지 창을 연다.
# imshow 함수로 'Detected Faces'라는 이름의 창에 결과 이미지를 출력하고,
# waitKey(0)를 사용하여 사용자가 키를 누를 때까지 창을 유지한다.
# destroyAllWindows 함수로 모든 창을 닫는다.
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
