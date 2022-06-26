#!/usr/bin/python3


import cv2 as cv
import numpy as np
import argparse
from typing import Tuple
from math import inf
import math
from itertools import chain
from datetime import datetime
from configure import Configurator
from picamera2 import Picamera2


size=(640, 480)

def detectCactus(img):
    windowWidth = 10
    windowHeight = 10
    start = 40
    threshold = 0.35
    centerColorTh = 200
    edgeColorTh = 128
    res = []
    for i in range(start, 256 - windowWidth):
        currentPart = img[(64 - windowHeight):(64), i : (i + windowWidth)]
        colorTh = edgeColorTh + (1 - (abs((i + windowWidth // 2) - 128) / 128)) * (centerColorTh - edgeColorTh)
        if np.count_nonzero(currentPart < colorTh) / currentPart.size >= threshold:
            res.append((i, 64 - windowHeight, windowWidth, windowHeight))
    return mergeOverlapingObjects(res)

def detectBirds(img):
    # TODO
    return []


def detectObjects(img, templates):
    objects = []
    for template in templates:
        w, h = template.shape[::-1]
        res = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        objects += [(pt[0], pt[1], w, h) for pt in zip(*loc[::-1])]

    return objects


def isOverlap(obj1, obj2):
    if len(obj1) != len(obj2):
        raise Exception("Size mismatch")
    if len(obj1) == 2:
        x1, w1 = obj1
        x2, w2 = obj2
        if x1 > x2:
            x1, x2 = x2, x1
            w1, w2 = w2, w1
        return x1 + w1 > x2
    elif len(obj1) == 4:
        x1,y1,w1,h1 = obj1
        x2,y2,w2,h2 = obj2
        return isOverlap((x1, w1), (x2, w2)) and isOverlap((y1, h1), (y2, h2))
    raise Exception("Unsupported dimension")


def mergeOverlapingObjects(objects):
    merged = [False] * len(objects)
    res = []
    for i in range(len(objects)):
        if merged[i]:
            continue
        merged[i] = True
        x, y, w, h = objects[i]
        r = x + w
        b = y + h
        updated = True
        while updated:
            updated = False
            for j in range(i + 1, len(objects)):
                if merged[j]:
                    continue
                if isOverlap((x, y, w, h), objects[j]):
                    x2, y2, w2, h2 = objects[j]
                    x = min(x, x2)
                    y = min(y, y2)
                    r = max(r, x2 + w2)
                    b = max(b, y2 + h2)
                    merged[j] = True
                    updated = True
        res.append((x, y, r - x, b - y))
    return res


def detectNight(img):
    GRID_DENSITY_X = 20
    GRID_DENSITY_Y = 20
    w, h = img.shape[::-1]
    whites = 0
    blacks = 0
    for x in np.linspace(0, w - 1, GRID_DENSITY_X):
        for y in np.linspace(0, h - 1, GRID_DENSITY_Y):
            if img[int(y), int(x)] < 127:
                blacks += 1
            else:
                whites += 1
    return blacks > whites


def loop(picam2, crop):
    (maxX, maxY, minX, minY) = crop
    start = datetime.now()
    frame = cv.cvtColor(picam2.capture_array(), cv.COLOR_YUV2GRAY_I420)
    playFiledNormalized = cv.resize(frame[minY : maxY, minX: maxX], (258,64))
    if detectNight(playFiledNormalized):
        playFiledNormalized = np.invert(playFiledNormalized)

    dino = mergeOverlapingObjects(detectObjects(playFiledNormalized, [cv.imread('templates/dino.jpg',0)]))
    cactuses = detectCactus(playFiledNormalized)
    birds = detectBirds(playFiledNormalized)
    # isGameOver = len(detectObjects(playFiledNormalized, [cv.imread('templates/gameOver.png',0)])) > 0
    # if isGameOver:
    #     print("Game over")

    playFiledNormalized = cv.cvtColor(playFiledNormalized,cv.COLOR_GRAY2RGB)
    for pt in cactuses:
        cv.rectangle(playFiledNormalized, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (0,255,0), 1)
    for pt in dino:
        cv.rectangle(playFiledNormalized, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (255,0,0), 1)
    for pt in birds:
        cv.rectangle(playFiledNormalized, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (0,0,255), 1)


    cv.imshow("img", cv.resize(playFiledNormalized, (512, 128)))
    cv.waitKey(1) & 0xFF 
    print((datetime.now() - start).total_seconds())
        

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.preview_configuration(main={"format": 'YUV420', "size": size}))
    picam2.start()
    crop = Configurator().configure(picam2)

    while True:
        start = datetime.now()
        loop(picam2, crop)
        while((datetime.now() - start).total_seconds() < 0.166):
            pass

    cv.destroyAllWindows()
    

if __name__ == "__main__":
    main()
