import cv2
import numpy as np

def salt(img,n):
    print "img.ndim",img.ndim
    for k in range(n):
        #print "n",n
        #print img.shape,img.shape[1],img.shape[2]
        i = int(np.random.random()*img.shape[1])
        j = int(np.random.random()*img.shape[0])

        #print i,j

        if img.ndim == 2:
            img[j,i] = 255

        elif img.ndim == 3:
            img[j,i,0] = 255
            img[j,i,1] = 255
            img[j,i,2] = 255
    return img


if __name__ == '__main__':
    #img = cv2.imread('/Volumes/Untitled/guo-qiang/kaggle/statefarm/rawdata/train/c0/img_34.jpg')
    img = cv2.imread('./mao.jpg')
    img = img.transpose((2,0,1))
    saltImage = salt(img,500)

    cv2.imshow("Salt",saltImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()