import cv2
import numpy as np

uint8_img = cv2.imread(img_path)
uint16_img = cv2.imread(img_path, -1)
uint16_img -= uint16_img.min()
uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
uint16_img *= 255
new_uint16_img = uint16_img.astype(np.uint8)
cv2.imshow('UINT8', uint8_img)
cv2.imshow('UINT16', new_uint16_img)
