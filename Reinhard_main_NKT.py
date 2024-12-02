import cv2
import numpy as np
import os

#################################################################
def lab_split(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
    I = I.astype(np.float32)
    I1, I2, I3 = cv2.split(I)
    # I1 /= 2.55
    # I2 -= 128.0
    # I3 -= 128.0
    return I1, I2, I3


def merge_back(I1, I2, I3):
    # I1 *= 2.55
    # I2 += 128.0
    # I3 += 128.0
    I = np.clip(cv2.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(I, cv2.COLOR_LAB2BGR)


def get_mean_std(I):
    I1, I2, I3 = lab_split(I)
    m1, sd1 = cv2.meanStdDev(I1)
    m2, sd2 = cv2.meanStdDev(I2)
    m3, sd3 = cv2.meanStdDev(I3)
    means = m1, m2, m3
    stds = sd1, sd2, sd3
    return means, stds

def Reinhard_method(target,I):
    means, stds = get_mean_std(target)
    target_means = means
    target_stds = stds
    I1, I2, I3 = lab_split(I)
    means, stds = get_mean_std(I)
    norm1 = ((I1 - means[0]) * (target_stds[0] / stds[0])) + target_means[0]
    norm2 = ((I2 - means[1]) * (target_stds[1] / stds[1])) + target_means[1]
    norm3 = ((I3 - means[2]) * (target_stds[2] / stds[2])) + target_means[2]
    return merge_back(norm1, norm2, norm3)
def Read_images(path):
    imgs=[]
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))           
        if img is not None:
            imgs.append(img)
            
    return imgs
    
if __name__ == "__main__":
    counter=0
    path="dataset/"
    imgs=Read_images(path)
    target_i=cv2.imread("Template_image/K3180026.jpg")
    for v in range (len(imgs)):
        print(v)
        counter=counter+1
        img=imgs[v]
        img=Reinhard_method(target_i,img)
        savepath="results/testimg_"+str(counter)+".png"
        cv2.imwrite(savepath,img)