# OpenCV 라이브러리 불러오기
import cv2

def basic_image_processing():
    # 1. 이미지 읽기
    # 주어진 경로에서 컬러 이미지를 불러옵니다.
    # 이미지를 처리하기 위해 먼저 읽어와야 하며, 이미지가 없으면 오류 메시지를 반환합니다.
    # imread 함수를 사용해 이미지를 읽어옵니다.
    # 참고: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gacbaa02cffc4ec2422dfa2e24412a99e2
    image_path = '../data/color_image.png'
    image = cv2.imread(image_path)

    if image is None:
        return "이미지를 불러오지 못했습니다. 경로를 확인하세요."

    # 2. 흑백 이미지로 변환
    # 컬러 이미지를 흑백 이미지로 변환합니다.
    # cvtColor 함수를 사용하여 이미지를 흑백으로 변환합니다.
    # 간단한 모델의 경우 흑백 이미지로 변환하여 처리하는 것이 효율적일 수 있습니다.
    # 따라서, 컬러 이미지를 흑백 이미지로 변환하겠습니다.
    # 참고: https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#gaf86c09fe702ed037c03c2bc603ceab14
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 반환 이미지 저장
    # 변환된 흑백 이미지를 저장할 경로를 설정합니다.
    # image_path 변수를 활용하여 이미지 파일명을 활용하거나, 새로운 파일명을 지정할 수 있습니다.
    # imwrite 함수를 사용하여 이미지를 저장합니다.
    # 참고: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga8ac397bd09e48851665edbe12aa28f25
    save_path = 'result2.png'
    cv2.imwrite(save_path, grayscale_image)

    # 변환 결과를 알려주는 메시지를 반환합니다.
    return f"이미지를 흑백으로 변환하여 '{save_path}'에 저장했습니다."

# 실행
# 이미지 처리 함수 호출 및 결과 출력
result = basic_image_processing()
print(result)

####################################################



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
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
