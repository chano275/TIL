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
    save_path = '../answer/result2.png'
    cv2.imwrite(save_path, grayscale_image)

    # 변환 결과를 알려주는 메시지를 반환합니다.
    return f"이미지를 흑백으로 변환하여 '{save_path}'에 저장했습니다."

# 실행
# 이미지 처리 함수 호출 및 결과 출력
result = basic_image_processing()
print(result)