import cv2
import numpy as np
import matplotlib.pyplot as plt

# 비디오
capture = cv2.VideoCapture(0)  # 비디오
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 해상도

while cv2.waitKey(33) < 0:
    # 프레임 읽기 메서드(capture.read)를 이용하여 카메라의 상태 및 프레임을 받아옵니다.
    ret, frame = capture.read()
    # ret은 카메라의 상태가 저장되며 정상 작동할 경우 True를 반환합니다. 작동하지 않을 경우 False를 반환합니다.
    # frame에 현재 시점의 프레임이 저장됩니다.
    templateimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(templateimg, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow("VideoFrame", dst)  # 지속적 프레임 캡쳐

# 입력이미지와 템플릿 이미지 읽기
img = cv2.imread('template_sample.png')
template = cv2.imread(dst)
th, tw = template.shape[:2]
plt.figure(figsize=(20, 6))

# 3가지 매칭 메서드 순회
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF_NORMED']  # 매칭 메서드
for i, method_name in enumerate(methods):
    img_draw = img.copy()
    method = eval(method_name)
    # 템플릿 매칭   ---①
    res = cv2.matchTemplate(img, template, method)
    # 최대, 최소값과 그 좌표 구하기 ---②
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(method_name, min_val, max_val, min_loc, max_loc)

    # TM_SQDIFF의 경우 최소값이 좋은 매칭, 나머지는 그 반대 ---③
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val
    # 매칭 좌표 구해서 사각형 표시   ---④
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(img_draw, top_left, bottom_right, (0, 0, 255), 2)
    # 매칭 포인트 표시 ---⑤
    cv2.putText(img_draw, str(match_val), top_left,
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)

    plt.subplot(1, 3, i+1)
    plt.imshow(img_draw[:, :, (2, 1, 0)])
    plt.xticks([]), plt.yticks([])
plt.show()
