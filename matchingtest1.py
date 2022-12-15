import cv2
import matplotlib.pyplot as plt

#영상 읽어서 그레이 스케일로 변환
img = cv2.imread('template.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 16x16 크기로 축소 ---①
gray = cv2.resize(gray, (16,16))
# 영상의 평균값 구하기 ---②
avg = gray.mean()
# 평균값을 기준으로 0과 1로 변환 ---③
bin = 1 * (gray > avg)
print(bin)

# 2진수 문자열을 16진수 문자열로 변환 ---④
dhash = []
for row in bin.tolist():
    s = ''.join([str(i) for i in row])
    dhash.append('%02x'%(int(s,2)))
dhash = ''.join(dhash)
print("dhash : ",dhash)

plt.figure(figsize = (10,6))
imgs = {'pistol':img}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()