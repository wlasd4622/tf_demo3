import cv2
import numpy as np

import random

if __name__ == '__main__':

    count = 0
    while True:
        # if count < 100:
        im = cv2.imread('./temp/t.png')
        im = cv2.resize(im, (int(100), int(50)))
        blue = (255, 0, 0)
        (height, width) = im.shape[:2]
        x = random.randint(3, width-10)
        y = random.randint(3, height-10)
        print(x, y)
        cv2.rectangle(im, (x, y), (x+9, y+9), blue, -1)
        cv2.imwrite('./data/train/k_'+str(x)+'_'+str(y)+'.png', im)
        count = count+1
