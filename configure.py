#!/usr/bin/python3

import cv2 as cv

from picamera2 import Picamera2
from math import inf
import time
import numpy as np

size=(640, 480)

playgroundRatio = 4


class Configurator():
    def getImage(self, size, picam2):
        time.sleep(1)
        res = cv.cvtColor(picam2.capture_array(), cv.COLOR_YUV2GRAY_I420)
        return res

    def extractScreen(self, img, size):
        # _,imgTh = cv.threshold(img,127,255,cv.THRESH_BINARY)
        # cv.imshow("imageTh", imgTh)
        # cv.waitKey()
        visited = set()
        w,h = size
        maxX = -inf
        maxY = -inf
        minX = inf
        minY = inf
        toProcess = [(w // 2,h // 2), (4 * w // 10 ,h // 2), (6 * w // 10 ,h // 2)]

        while len(toProcess) > 0:
            x,y = toProcess.pop()
            if (x,y) in visited or x < 0 or y < 0 or x >= w or y >= h or img[y,x] < 127:
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

        if minX == inf:
            return [], (0,0,0,0)
        return img[minY : maxY, minX: maxX], (maxX, maxY, minX, minY)

    def findFloor(self, img, screen):
        (maxX, maxY, minX, minY) = screen
        expectedHeight = 2
        start = minY + 2 * (maxY - minY) // 5
        end = minY + 4 * (maxY - minY) // 5
        best = 0
        bestVal = 0
        for i in range(start, end):
            currentPart = img[i:(i+expectedHeight), minX : maxX]
            act = np.count_nonzero(currentPart < 180)
            if act > bestVal:
                best = i
                bestVal = act
        return best

    def configure(self, picam2):
        img = self.getImage(size, picam2)
        screenImage, (maxX, maxY, minX, minY) = self.extractScreen(img, size)
        floorPosition = self.findFloor(img, (maxX, maxY, minX, minY))
        expectedHeight = (maxX - minX) // playgroundRatio
        finalCrop = (maxX, floorPosition, minX, floorPosition - expectedHeight)

        img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
        cv.rectangle(img, (minX, minY), (maxX, maxY), (0,0,255), 2)
        cv.rectangle(img, (minX, floorPosition), (maxX, floorPosition  + 2), (255,0,0), 1)
        cv.rectangle(img, (finalCrop[2], finalCrop[3]), (finalCrop[0], finalCrop[1]), (0,255,0), 2)
        cv.imshow("image", img)
        cv.waitKey()
        cv.destroyAllWindows()
        return finalCrop



if __name__ == "__main__":
    picam2 = Picamera2()
    picam2.configure(picam2.preview_configuration(main={"format": 'YUV420', "size": size}))
    picam2.start()
    Configurator().configure(picam2)
