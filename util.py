
# coding: utf-8


import cv2
import numpy as np


COLOR_BLACK = (0, 0, 0);
COLOR_WHITE = (255, 255, 255);
PI = np.pi
dft_shape = (100, 100, 3);
dft_color = COLOR_BLACK;
dft_length = 100;


# generate an image object with pure color
def pure_color(color = dft_color, shape = dft_shape):
    img = np.ones(shape, np.uint8)
    img = img * color;
    img = np.uint8(img)
    return img


# draw a straight line in the generated image.
def draw_line(img = None, color = COLOR_WHITE, bg = COLOR_BLACK, shape = dft_shape, start = (0, 0), angle = PI /4, length = dft_length, width = 1):
    """
        draw a straight line in a given or generated image.
        shape: tuple with 3 elements:(width, height, channels)
        start: the start position of the line to be drew. .
        angle: the angle between the line and x-coordinate
        color: the color of the line.
        length: the length of the line. It can be arbitary value. However, if the calculated end point should lay beyond the image, it will be put on the edge, and thus, the final length of the line will be shorter than the given length.

    """
     
    # if an existing image is not given, a new one will be created, using the giving bg color.
    if img == None:
        img = pure_color(color = bg, shape = shape);
        if color == bg:
            color = COLOR_WHITE - bg

    #the end point of the line
    startX, startY = start
    endX = startX + length * np.cos(angle)
    endY = startY + length * np.sin(angle)
    end = (int(round(endX)), int(round(endY)))    
    start = (int(round(startX)), int(round(startY)))

    
    # draw the line
    cv2.line(img, start, end,color,width)
    return img;
