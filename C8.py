import cv2
import numpy as np

# detect and label shape within an image

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None,  scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2RGB)
        imageBlank = np.zeros((imgArray[0][0].shape[0], imgArray[0][0].shape[1], 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2RGB)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def sidelenRange(coors):
    sides = len(coors) #assume triple nested
    minLen, maxLen = 99999.0, 0.0

    slens = []
    #print(coors)
    for s in range(0, sides):
        ns = s+1 if s +1 < sides else 0
        #print(coors[s][0][0], coors[ns][0][0], coors[s][0][1],coors[ns][0][1])
        slen = ((coors[s][0][0] - coors[ns][0][0])**2 + (coors[s][0][1] - coors[ns][0][1])**2)**0.5
        slens.append(slen)
        minLen = slen if slen < minLen else minLen
        maxLen = slen if slen > maxLen else maxLen
    return maxLen - minLen

def getContours(img, imgContour):
    #contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_SIMPLE

    cCount, rCount, tCount, sCount = 0, 0, 0, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x,y), (x+w,y+h), (0,0,0),1)
            if w < 0.9*img.shape[1] and h < 0.9*img.shape[0]:
                cv2.drawContours(imgContour,cnt,-1, (255,0,0),5)

                if len(approx) == 3:
                    tCount+=1
                    oType = 'T'
                elif len(approx) == 4:
                    rCount+=1
                    oType = 'R'+ str(rCount)
                    aRatio = float(w)/h
                    #print(oType,rCount, approx)

                    if aRatio > 0.95 and aRatio < 1.05 and sidelenRange(approx) < 5:
                        sCount +=1
                        oType = 'S'+str(rCount)
                else:
                    cCount+=1
                    oType = 'C'
                cv2.putText(imgContour, oType, (x+w//2 - 15, y+h//2+5), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,0), 2)

path = 'Resources/shapes.png'
img = cv2.imread(path)
imgContour = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#imbBlank = np.zeros_like(img)
(thresh, mImg) = cv2.threshold(imgGray, 225, 254, cv2.THRESH_BINARY)
getContours(mImg, imgContour)

imgStack = stackImages(0.7, ([img,imgGray], [mImg,imgContour]))
cv2.imshow("S", imgStack)
cv2.imwrite("output.jpg",imgStack)
cv2.waitKey(0)
