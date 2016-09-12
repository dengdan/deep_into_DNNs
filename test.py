import cv2
from util import *

img = draw_line(angle = PI / 4)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cv2.imshow('image',img)

cv2.waitKey(0)

cv2.destroyAllWindows()
