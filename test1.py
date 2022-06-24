#!/usr/bin/python3


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
from typing import Tuple
from math import inf
import math
from itertools import chain
from datetime import datetime


def loadImage(src, size):
    img = cv.imread(cv.samples.findFile(src))
    img = cv.resize(img, size)
    img = cv.medianBlur(img,5)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def extractScreen(img, size):
    _,imgTh = cv.threshold(img,127,255,cv.THRESH_BINARY)

    visited = set()
    w,h = size
    maxX = -inf
    maxY = -inf
    minX = inf
    minY = inf
    toProcess = [(w // 2,h // 2), (4 * w // 10 ,h // 2), (6 * w // 10 ,h // 2)]

    while len(toProcess) > 0:
        x,y = toProcess.pop()
        if (x,y) in visited or x < 0 or y < 0 or x >= w or y >= h or imgTh[y,x] < 200:
            continue
        visited.add((x,y))
        maxX = max(maxX, x)
        maxY = max(maxY, y)
        minX = min(minX, x)
        minY = min(minY, y)
        toProcess.append((x + 1,y + 1))
        toProcess.append((x + 1,y - 1))
        toProcess.append((x - 1,y + 1))
        toProcess.append((x - 1,y - 1))

    return img[minY : maxY, minX: maxX], (maxX, maxY, minX, minY)


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


def main():
    cap = cv.VideoCapture('vid/play1.mp4')
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            continue
        # screenImage, _ = extractScreen(frame, size)
        playFiledNormalized = cv.resize(frame, (600,450))
        playFiledNormalized = cv.cvtColor(playFiledNormalized, cv.COLOR_BGR2GRAY)
        if detectNight(playFiledNormalized):
            playFiledNormalized = np.invert(playFiledNormalized)

        dino = mergeOverlapingObjects(detectObjects(playFiledNormalized, [cv.imread('templates/dino.jpg',0)]))
        cactuses = mergeOverlapingObjects(detectObjects(playFiledNormalized, [
            cv.imread('templates/cactus_pile1.jpg',0), cv.imread('templates/cactus_pile2.jpg',0), cv.imread('templates/cactus_pile3.jpg',0),
            cv.imread('templates/cactus_pile4.jpg',0), cv.imread('templates/z_cactus_big1.jpg',0), cv.imread('templates/z_cactus1.jpg',0)
        ]))
        birds = mergeOverlapingObjects(detectObjects(playFiledNormalized, [cv.imread('templates/bird_low.jpg',0)]))
        isGameOver = len(detectObjects(playFiledNormalized, [cv.imread('templates/gameOver.png',0)])) > 0
        if isGameOver:
            print("Game over")

        for pt in dino + cactuses + birds:
            cv.rectangle(playFiledNormalized, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (0,0,255), 2)

        cv.imshow("img", playFiledNormalized)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


 
    

if __name__ == "__main__":
    main()
