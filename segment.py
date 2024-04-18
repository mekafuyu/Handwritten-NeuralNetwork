import cv2 as cv
import numpy as np
from datetime import datetime

def sort_func(e):
    return(e[0][0])

def segment(im):
    newImage = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    _, newImage = cv.threshold(newImage, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    figures = []
    figures_pixels = []
    for x in range(len(newImage)):
        for y in range(len(newImage[x])):
            if newImage[x][y] == 0:
                currX = x
                currY = y
                upper = im.shape[1]
                bottom = 0
                left = im.shape[0]
                right = 0
                queue = []
                queue.append((x, y))
                figure_pixels = []
                while len(queue) > 0:
                    currX, currY = queue.pop(0)
                    upper, left, bottom, right = expand(newImage, upper, left, bottom, right, currX, currY, queue, figure_pixels)
                    
                figures_pixels.append(figure_pixels)
                
                newfig = np.zeros((right - left + 1, bottom - upper + 1, 1), np.uint8)
                newfig[newfig == 0] = 255

                for x, y in figure_pixels:
                    newfig[x - left][y - upper] = 0
                
                ofcFig = np.zeros((128, 128, 1), np.uint8)
                ofcFig[ofcFig == 0] = 255
                width, heigth, _ = newfig.shape
                
                if heigth > width:
                    width = int(128 / heigth * width)
                    heigth = 128
                    newfig = cv.resize(newfig, (heigth, width))
                    offset = int((128 - width) / 2)
                    for x in range(len(newfig)):
                        for y in range(len(newfig[x])):
                            ofcFig[x + offset - 1][y] = newfig[x][y]
                else:
                    heigth = int(128 / width * heigth)
                    width = 128
                    newfig = cv.resize(newfig, (heigth, width))
                    offset = int((128 - heigth) / 2)
                    for x in range(len(newfig)):
                        for y in range(len(newfig[x])):
                            ofcFig[x][y + offset - 1] = newfig[x][y]
                
                path_newimg = 'upload_folder/{0}.png'.format(len(figures))
                figures.append(((upper, left), (bottom, right), path_newimg))
                _, ofcFig = cv.threshold(ofcFig, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                cv.imwrite(path_newimg, ofcFig)
                
    figures.sort(key=sort_func)
    return figures

def expand(im, u, l, b, r, x, y, queue: list, pixels: list):
    if im[x][y] == 0:
        if x < l:
            l = x
        if x > r:
            r = x
        if y < u:
            u = y
        if y > b:
            b = y
        queue.append((x - 1, y))
        queue.append((x + 1, y))
        queue.append((x, y - 1))
        queue.append((x, y + 1))
        im[x][y] = 128
        pixels.append((x, y))
    return u, l, b, r
