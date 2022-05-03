# import opencv and numpy
import cv2
import numpy as np

# import imgs
off_side_1 = cv2.imread('imgs/off side 1.png')
off_side_2 = cv2.imread('imgs/off side 2.png')
off_side_3 = cv2.imread('imgs/off side 3.png')

# convert img to hsv function
def convert_to_hsv(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# mask of green (36,0,0) ~ (70, 255, 255) function
def mask_green(img):
        lower_green = np.array([33,0,0])
        upper_green = np.array([70,255,255])
        return cv2.inRange(img, lower_green, upper_green)

# slice the green function
def slice(mask, img):
        return cv2.bitwise_and(img, img, mask=mask)

#save the image in result_imgs folder
def save_img(img, name):
        cv2.imwrite('result_imgs/' + name + '.png', img)

#show img function
def show_img(img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# detect green function
def detect_green(img):
        img_hsv = convert_to_hsv(img)
        save_img(img_hsv, 'img_hsv')
        mask = mask_green(img_hsv)
        save_img(mask, 'mask')
        green = slice(mask, img)
        return green

# main function
def main():
        off_side_1_green = detect_green(off_side_1)
        off_side_2_green = detect_green(off_side_2)
        off_side_3_green = detect_green(off_side_3)
        save_img(off_side_1_green, 'off_side_1_green')
        save_img(off_side_2_green, 'off_side_2_green')
        save_img(off_side_3_green, 'off_side_3_green')
        show_img(off_side_1_green)
        show_img(off_side_2_green)
        show_img(off_side_3_green)


# run the main function
if __name__ == '__main__':
        main()
