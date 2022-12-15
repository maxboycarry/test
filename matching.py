# BFMatcher와 SIFT로 매칭 (match_bf_sift.py)

import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)  # 해상도

img1 = cv2.imread('template.png')
#img2 = cv2.imread('sample2.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # 프레임 읽기 메서드(capture.read)를 이용하여 카메라의 상태 및 프레임을 받아옵니다.
# ret, frame = capture.read()
# # ret은 카메라의 상태가 저장되며 정상 작동할 경우 True를 반환합니다. 작동하지 않을 경우 False를 반환합니다.
# # frame에 현재 시점의 프레임이 저장됩니다.
# templateimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# ret, dst = cv2.threshold(templateimg, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow("VideoFrame", dst)  # 지속적 프레임 캡쳐


while cv2.waitKey(33) < 0:
    # 프레임 읽기 메서드(capture.read)를 이용하여 카메라의 상태 및 프레임을 받아옵니다.
    ret, frame = capture.read()
# ret은 카메라의 상태가 저장되며 정상 작동할 경우 True를 반환합니다. 작동하지 않을 경우 False를 반환합니다.
# frame에 현재 시점의 프레임이 저장됩니다.
    templateimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(templateimg, 200, 255, cv2.THRESH_BINARY)
    # SIFT 서술자 추출기 생성 ---①
    detector = cv2.xfeatures2d.SIFT_create()
# 각 영상에 대해 키 포인트와 서술자 추출 ---②
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(dst, None)
# BFMatcher 생성, L1 거리, 상호 체크 ---③
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# 매칭 계산 ---④
try:
    matches = matcher.match(desc1, desc2)
    res = cv2.drawMatches(img1, kp1, dst, kp2, matches, None,
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    print("매칭값1 : ", desc1)
    print("매칭값2 : ", desc2)
    cv2.imshow("VideoFrame", res)  # 지속적 프레임 캡쳐
except:
    print('예외 발생')
# 매칭 결과 그리기 ---⑤

# 결과 출력
cv2.imshow('BFMatcher + SIFT', res)
cv2.waitKey()
cv2.destroyAllWindows()
