import cv2
import numpy as np
import math

def sp_noise(image, ratio):
    output = np.zeros(image.shape, np.uint8)
    ratio = ratio / 2
    thres = 1 - ratio
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < ratio:
                output[i][j][0] = 0
                output[i][j][1] = 0
                output[i][j][2] = 0
            elif rdn > thres:
                output[i][j][0] = 255
                output[i][j][1] = 255
                output[i][j][2] = 255
            else:
                output[i][j][0] = image[i][j][0]
                output[i][j][1] = image[i][j][1]
                output[i][j][2] = image[i][j][2]
    return output

def median_filter(image, ksize):
    output = np.zeros(image.shape, np.uint8)
    border = ksize // 2
    image_border = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REFLECT_101)
    height, width = image.shape[:2]
    
    
    for k in range(3):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                member = []
                
                member.append(image_border[i-1][j-1][k])
                member.append(image_border[i-1][j][k])
                member.append(image_border[i-1][j+1][k])
                member.append(image_border[i][j-1][k])
                member.append(image_border[i][j][k])
                member.append(image_border[i][j+1][k])
                member.append(image_border[i+1][j-1][k])
                member.append(image_border[i+1][j][k])
                member.append(image_border[i+1][j+1][k])

                member.sort()
                output[i-1][j-1][k] = member[4]
    
    return output

def getPSNR(orig, proc):
    height, width = orig.shape[:2]
    mse = 0.0
    
    for k in range(3):
        for i in range(height):
            for j in range(width):
                mse += (float(orig[i][j][k]) - float(proc[i][j][k])) ** 2 
    
    mse = mse / orig.size / 3
    psnr = 10 * math.log(255 * 255 / mse)
    
    return psnr

# source: https://www.pexels.com/photo/woman-looking-at-camera-325531/
original = cv2.imread('beauty.jpg')
cv2.imwrite(r'lab\original.jpg', original)

# 分別加入 10, 20, 30% 的 salt & pepper noise
noise_10 = sp_noise(original, 0.1)
cv2.imwrite(r'lab\noise_10.jpg', noise_10)

noise_20 = sp_noise(original, 0.2)
cv2.imwrite(r'lab\noise_20.jpg', noise_20)

noise_30 = sp_noise(original, 0.3)
cv2.imwrite(r'lab\noise_30.jpg', noise_30)

# 以中值濾波器除去雜訊，邊緣以鏡射處理
median_10 = median_filter(noise_10, 3)
cv2.imwrite(r'lab\median_10.jpg', median_10)

median_20 = median_filter(noise_20, 3)
cv2.imwrite(r'lab\median_20.jpg', median_20)

median_30 = median_filter(noise_30, 3)
cv2.imwrite(r'lab\median_30.jpg', median_30)

# 印出各圖片的 PSNR
noise_10_psnr = getPSNR(original, noise_10)
median_10_psnr = getPSNR(original, median_10)
print('PSNR of noise_10 = %d dB, PSNR of median_10 = %d dB' %(noise_10_psnr, median_10_psnr))

noise_20_psnr = getPSNR(original, noise_20)
median_20_psnr = getPSNR(original, median_20)
print('PSNR of noise_20 = %d dB, PSNR of median_20 = %d dB' %(noise_20_psnr, median_20_psnr))

noise_30_psnr = getPSNR(original, noise_30)
median_30_psnr = getPSNR(original, median_30)
print('PSNR of noise_30 = %d dB, PSNR of median_30 = %d dB' %(noise_30_psnr, median_30_psnr))
