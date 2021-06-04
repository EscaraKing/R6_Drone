import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sample_img = cv2.imread('result_gray.png', cv2.IMREAD_COLOR)
#print(sample_img.shape)
#print(sample_img.size)

#cv2.imshow('sliced_img', sample_img)
#cv2.waitKey(0)




#히스토그램 평활화 과정

img_range = sample_img.shape
N = img_range[0]*img_range[1]

INTENSITY = 256
img_histogram = np.zeros((2,256))


for i in range(img_histogram.shape[1]):
    img_histogram[0,i] = i


for num_row in range(img_range[0]):
    for num_col in range(img_range[1]):
        #print("finish column %s" %j)
        a = sample_img[num_row,num_col][0]
        img_histogram[1,a] += 1

    #print("Finish for equation %s" %num_row)
'''
x = img_histogram[0]
y = img_histogram[1]

plt.plot(x,y)
plt.title("Histogram")
plt.xlabel("N")
plt.ylabel("sum")
'''

#print(img_histogram)

temp_sum = 0
equalized_histogram = np.zeros((2,256))
for i in range(equalized_histogram.shape[1]):
    equalized_histogram[0,i] = i


for i in range(img_histogram.shape[1]):
    temp_sum = temp_sum + img_histogram[1][i]
    n = temp_sum/N*255
    equalized_histogram[1,i] = n
    #print(temp_sum, n)

'''
x = equalized_histogram[0]
y = equalized_histogram[1]

plt.plot(x,y)
plt.title("Histogram")
plt.xlabel("N")
plt.ylabel("sum")

plt.show()
'''

equalized_image = np.zeros((1000,1000,3), np.uint8)
print(equalized_image[1,1,1])

for num_row in range(img_range[0]):
    for num_col in range(img_range[1]):
        for i in range(equalized_histogram.shape[1]):
            n = equalized_histogram[1][i]

            if sample_img[num_row,num_col][0] == i:
                equalized_image[num_row, num_col, 0] = n
                equalized_image[num_row, num_col, 1] = n
                equalized_image[num_row, num_col, 2] = n
    print("Finish %s row" %num_row)

cv2.imshow('result', equalized_image)
cv2.waitKey(0)

'''아래는 기본 활용예시'''
#########################################################################

'''특정부분 선택방법'''

'''
#사각형 부분 추출
sample_img[500:600,500:600] = [210,20,50]
cv2.imshow('result', sample_img)
cv2.waitKey(0)
'''

'''
#자른 이미지 특정부위에 집어넣기
sliced_img = sample_img[500:600, 0:400]
sample_img[100:200, 300:700] = sliced_img
cv2.imshow('sliced_img', sample_img)
cv2.waitKey(0)
'''

'''
#전체 색조 바꾸기
sample_img[:,:,2] = 250
#전체 이미지에서 BGR 중 3번째(BGR 중 R에 해당) 값을 2로 바꾼다
#2인데 3번째인 이유는 index에서 0이 처음이기 때문
cv2.imshow('result', sample_img)
cv2.waitKey(0)
'''
######################################################################

'''이미지 변형
    1. 크기 변형
    
변형에 필요한 행렬(matrix)가 필요한게 특징이다'''


'''
#확대
expanded_img = cv2.resize(sample_img, None, fx=2.0,fy=2.0, interpolation=cv2.INTER_CUBIC)
cv2.imshow('result', expanded_img)
cv2.waitKey(0)
#축소
expanded_img = cv2.resize(sample_img, None, fx=0.5,fy=0.8, interpolation=cv2.INTER_CUBIC)
cv2.imshow('result', expanded_img)
cv2.waitKey(0)
'''

heigt, width = sample_img.shape[:2]

'''
#이미지 이동
Matrix_for_move = np.float32([[1,0,100],[0,1,10]])
Moved_img = cv2.warpAffine(sample_img, Matrix_for_move,(width,heigt))
cv2.imshow('result', Moved_img)
cv2.waitKey(0)
'''

'''#이미지 회전
#회전을 위한 변환행렬 함수
Matrix_for_rotation = cv2.getRotationMatrix2D((width/2, heigt/2), 90, 0.5)
#                                                  중심점          각도  scale
Rotated_img = cv2.warpAffine(sample_img,Matrix_for_rotation,(width,heigt))
cv2.imshow('result',Rotated_img)
cv2.waitKey(0)
'''
'''
#이미지합치기
add_img = cv2.imread('diva_sample2.png', cv2.IMREAD_COLOR)
add_img_height = add_img.shape[0]
add_img_width = add_img.shape[1]

h_parameter = heigt/add_img_height
w_parameter = heigt/add_img_width

expanded_add_img = cv2.resize(add_img, None, fx=w_parameter,fy=h_parameter, interpolation=cv2.INTER_CUBIC)
#크기가 같아야함을 기억하자
print(expanded_add_img.shape)

result = cv2.add(sample_img, expanded_add_img)
cv2.imshow('result', result)
cv2.waitKey(0)

result2 = sample_img + expanded_add_img
#결과를 보면 별로 효과적인 합치기 기술은 아니다
#RGB갑이 255를 넘어가면 0으로 되돌아가는 현상 때문이다
#이미지가 단순한 행렬임을 기억하자
cv2.imshow('result', result2)
cv2.waitKey(0)
'''
'''
#임계값 처리

#THRESH_BINARY : 임계점을 넘으면 흰색으로 작으면 검은색
#THRESH_BINARY_INV : 임계점을 넘으면 검은색으로 크면 흰색
#THRESH_TRUNC : 임계값이 넘어가면 회색으로 작으면 그대로
#THRESH_BINARY_TOZERO : 임계값이 크면 그대로 작으면 검은색으로
#THRESH_BINARY_TOZERO_INV : 임계값보다 크면 검은색으로 작으면 그대로

sample_img_gray = cv2.imread('diva_sample.png', cv2.IMREAD_GRAYSCALE)

thres_images = []
ret, thres1 = cv2.threshold(sample_img_gray, 127, 255, cv2.THRESH_BINARY)
#                                            임계값 바꿀값   조건
ret, thres2 = cv2.threshold(sample_img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thres3 = cv2.threshold(sample_img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thres4 = cv2.threshold(sample_img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thres5 = cv2.threshold(sample_img_gray, 127, 255, cv2.THRESH_TOZERO_INV)


cv2.imshow('result',thres5)
cv2.waitKey(0)
'''

'''
#적응임계값 처리
#adaptiveThreshold
#그냥 임계값으로 사라지는 정보가 많을 떄 사용하기 좋다
#전체 이미지를 block으로 나누어 적응임계값 적용
#ADAPTIVE_THRESH_MEAN_C
#ADAPTIVE_THRESH_GAUSSIAN_C

sample_img_gray = cv2.imread('diva_sample.png', cv2.IMREAD_GRAYSCALE)
adap_thres = cv2.adaptiveThreshold(sample_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
cv2.imshow('result', adap_thres)
cv2.waitKey(0)

sample_img_gray = cv2.imread('diva_sample.png', cv2.IMREAD_GRAYSCALE)
adap_thres = cv2.adaptiveThreshold(sample_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3)
cv2.imshow('result', adap_thres)
cv2.waitKey(0)
'''

'''
#Tracker
#cv2.createTrackbar

def change_color(x):
'''
    #트랙바를 추적하며
    #값이 변할 때마다
    #트랙바의 값을 가져오는 함수
'''
    r = cv2.getTrackbarPos("R", "Image")
    g = cv2.getTrackbarPos("G", "Image")
    b = cv2.getTrackbarPos("B", "Image")
    #getTrackbarPos : 트랙바의 위치를 가져온다
    #여기선 RGB값이 된다
    image[:] = [b,g,r]
    #이미지 전체에 RGB값 대입
    cv2.imshow('Image', image)

#임의의 이미지 생성
image = np.zeros((1000,1000,3), np.uint8)
#윈도우창으로 나올 수 있게
cv2.namedWindow("Image")
#트랙바 생성
cv2.createTrackbar("R", "Image", 0, 255, change_color)
cv2.createTrackbar("G", "Image", 0, 255, change_color)
cv2.createTrackbar("B", "Image", 0, 255, change_color)

cv2.imshow('Image', image)
cv2.waitKey(0)
'''

'''
#직선그리기
image = np.full((512, 512, 3), 255, np.uint8)
image = cv2.line(image, (0,0),(255,255),(255,0,0),10)
#              바탕이미지  시작    끝좌표   선의 색깔  두께
cv2.imshow('result', image)
cv2.waitKey(0)
'''
'''
#사각형그리기
image = np.full((512,512,3),255,np.uint8)
image = cv2.rectangle(image, (20,20),(255,255),(255,0,0),3)
#직선과 파라미터 위치는 동일
cv2.imshow('result', image)
cv2.waitKey(0)
'''

'''
#원 그리기
image = np.full((512,512,3),255,np.uint8)
image = cv2.circle(image,(255,255), 30 ,(255,0,0), 3)
cv2.imshow('result', image)
cv2.waitKey(0)
'''

'''
#다각형 그리기
image = np.full((512,512,3),255,np.uint8)
points = np.array([[2,4],[500,200],[400,200],[120,20]])
image = cv2.polylines(image, [points], True, (255,0,0), 5)
cv2.imshow('result', image)
cv2.waitKey(0)
'''

'''
#텍스트그리기
image = np.full((512,512,3),255,np.uint8)
image = cv2.putText(image, 'Hello World', (0,200), cv2.FONT_ITALIC, 2, (255,0,0))
#                  바탕이미지 입력글씨          위치       글씨모양     두께    색깔
cv2.imshow('result', image)
cv2.waitKey(0)
'''

