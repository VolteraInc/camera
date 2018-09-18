from multiprocessing import Pool
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from gpiozero import PWMLED
import numpy as np
import math

def drawBackground(h,w):
    background = np.zeros((h+10, w+10,3), np.uint8)
    
    cv2.line(background, (10,10), (10,h), (255, 255, 255),1)
    cv2.line(background, (10,h), (w, h), (255,255,255), 1)
    
    for i in range(1, int(h/20)+1):
        cv2.line(background, (11, h-i*20), (w, h-i*20), (50, 50, 50), 1)
        cv2.putText(background, str(i*20*2.5/2)+"um", (12, h-i*20-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
    for i in range(1, int(w/35)+1):
        cv2.line(background, (w-i*35, 10), (w-i*35, h-1), (50, 50, 50), 1)

    return background

def drawGraph(background, values):
    h,w = background.shape[:2]
    graphimg = background.copy()
##    print(str(values))
##    minVal = min(np.array(values)[:,1])
    cols = np.arange(0,w-10, interval)
    minVal = min(values)
    
    for i in range(len(values)-1):
##        print(str(minVal) + ' ' + str(x[1]))
        if values[i] != -1:
            graphimg[h-10-abs(int(values[i]-minVal)*2), 10+cols[i]+1, 1] = 255
            cv2.putText(graphimg, str(abs(int(values[i]-minVal))*2.5), (10+cols[i]+1, h-10-abs(int(values[i]-minVal))*2-3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.3, (255,255,255), 1, cv2.LINE_AA)
##        background[h-10-abs(int(x[1]-minVal)*2), 10+x[0], 1] = 255

    return graphimg
##    return background

def findCenterLine(img):
    h,w = img.shape[:2]
    filter = np.zeros((h,w,1), np.uint8)
    values = []
    
    for v in range(w):
        if v%interval == 0 or v==0:
            
##            cols.append(v)
            
            window = []
            windowLastIndex = []
            windowLastIndex.append(0)
            maxWindowTotal = 0
            
##            print(str(max(img[:,v])))
            threshold = max(img[:,v])*0.6
            
            for u in range(h):
                if len(window) >= windowSize:
                    if img[u,v] > threshold:
                        #window.append(img[u,v])
                        #filter[u,v] = img[u,v]
                        
                        window.append(threshold)
                        filter[u,v] = threshold
                    else:
                        window.append(0.0)
                        filter[u,v] = 0                        
                    window.pop(0)
                    
    ##                print(str(u) + ' ' + str(window) )
                    
                    if sum(window) > maxWindowTotal:
                        windowLastIndex = []
                        windowLastIndex.append(u)
                        maxWindowTotal = sum(window)
                    elif sum(window) == maxWindowTotal:
                        windowLastIndex.append(u) 
                else:
                    if img[u,v] > threshold:
##                        window.append(img[u,v])
##                        filter[u,v] = img[u,v]
                        window.append(threshold)
                        filter[u,v] = threshold
                    else:
                        window.append(0.0)
                        filter[u,v] = 0
                        
                    maxWindowTotal = sum(window)
                    windowLastIndex[0] = u
            
            avgWindowLastIndex = sum(windowLastIndex)/len(windowLastIndex)
##            rows.append(avgWindowLastIndex-(windowSize-1)/2)
            
            if maxWindowTotal > 0:
                values.append([v, avgWindowLastIndex-((windowSize-1)/2)])
##    return rows, cols
    return values, filter

def findCenterLineThreaded(img):
    h,w = img.shape[:2]
    filter = np.zeros((h,w,1), np.uint8)
    cols = np.arange(0,w, interval)
    
    colValues = [img[:,col] for col in cols]
    
    pool = Pool()
    laserPoints = pool.map(findCenterPoint, colValues)
    
    pool.close()
    pool.join()
    
    return laserPoints

def findCenterPoint(colVals):
    filter = np.zeros(len(colVals), np.uint8)
    values = []
    
    window = []
    windowLastIndex = []
    windowLastIndex.append(0)
    maxWindowTotal = 0

    threshold = max(colVals)*0.75
            
    for u in range(len(colVals)):
        if len(window) >= windowSize:
            if colVals[u] > threshold:
                window.append(threshold)
                filter[u] = threshold
            else:
                window.append(0.0)
                filter[u] = 0                        
            
            window.pop(0)
                    
            if sum(window) > maxWindowTotal:
                windowLastIndex = []
                windowLastIndex.append(u)
                maxWindowTotal = sum(window)
            elif sum(window) == maxWindowTotal:
                windowLastIndex.append(u) 
        else:
            if colVals[u] > threshold:
                window.append(threshold)
                filter[u] = threshold
            else:
                window.append(0.0)
                filter[u] = 0
                        
            maxWindowTotal = sum(window)
            windowLastIndex[0] = u
            
    avgWindowLastIndex = sum(windowLastIndex)/len(windowLastIndex)
            
    if maxWindowTotal > 0:
        retVal = avgWindowLastIndex-((windowSize-1)/2)
##        values.append([v, avgWindowLastIndex-((windowSize-1)/2)])
    else:
        retVal = -1

    return retVal
    

def drawResults(img, values):
    h,w = img.shape[:2]
    result = np.zeros((h,w,1), np.uint8)
    
    cols = np.arange(0,w, interval)
    
    for i in range(len(cols)):
        if values[i] != -1:
            result[int(values[i]), int(cols[i])] = 255
    
##    for point in values:
####        print(rows[i])
##        result[int(point[1]), int(point[0])] = 255
        
    return result

##h=720
##w=1280
##blank = np.zeros((h,w,1), np.uint8)
##
##cv2.line(blank, (0, int(h/2)), (w, int(h/2)), 255, 30)
##
####cv2.line(blank, (0, int(h/2)), (w, int(h/2)), 200, 27)
##rows, cols = findCenterLine(blank, 10, 20)
##
##cv2.imshow('line', drawResults(blank, rows, cols))
##cv2.waitKey(0)
##cv2.destroyAllWindows()
index = 1
mode = ['auto', 'night', 'nightpreview', 'backlight', 'sports', 'snow', 'beach', 'verlong', 'fixedfps', 'antishake', 'fireworks']

laser = PWMLED(4)
##laser.value = 1
##time.sleep(5)

camera = PiCamera(sensor_mode = 1)
camera.resolution = (1280,720)
camera.zoom = (320/1920, 180/1080, 1280/1920, 720/1080)
camera.awb_mode = "off"
camera.awb_gains = 1.6
camera.framerate = 30
time.sleep(2)
camera.shutter_speed = 5000
camera.exposure_mode = 'off'
time.sleep(3)
##camera.exposure_mode = 'beach'
##time.sleep(5)
##print(camera.framerate)
##camera.shutter_speed = int(1/(camera.framerate)*1000)


rawCapture = PiRGBArray(camera, size=(1280, 720))


print(camera.shutter_speed)

toggle = True

gbg = drawBackground(400,1280)
laser.value = 1
windowSize = 30
interval = 25

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
    now = time.time()
    
    if toggle:
##        toggle = False
##        imageLaser = frame.array
##        laser.value = 0
##        gray_imageLaser = cv2.cvtColor(imageLaser, cv2.COLOR_BGR2GRAY)
##        blueLaser = cv2.medianBlur(gray_imageLaser, 35)
##        blueLaser = cv2.medianBlur(imageLaser[:,:,0], 39)
        blueLaser = cv2.medianBlur(frame.array[:,:,2], 39)
        
##        blueLaser = cv2.bilateralFilter(blueLaser,9,75,75)
##        retval, blueLaser = cv2.threshold(blueLaser, 150, 255, cv2.THRESH_TOZERO)

##        cv2.imshow('img',cv2.add(blueLaser, findLine(blueLaser)))
        
##        sub = cv2.medianBlur(cv2.subtract(blueLaser, base), 13)
##        points,filtr = findCenterLine(sub, 30, 10)
##        points,filtr = findCenterLine(blueLaser)
##        points = np.array(points)

        points = findCenterLineThreaded(blueLaser)

##        img = cv2.subtract(filtr, drawResults(filtr, points))
        
        img = cv2.subtract(blueLaser, drawResults(blueLaser, points))
        
##        graph = drawGraph(gbg, points)
##        cv2.imshow('graph', graph)
        
        
        try:
            graph = drawGraph(gbg, points)
            cv2.imshow('graph', graph)
        except:
            print('Graphing Error')
            
            
##        print(str(points))
##        cv2.imwrite('/home/pi/Desktop/testimg.jpg', img)
##        cv2.imshow('img', img)

##        cv2.imshow('img', img)
##        cv2.waitKey(1000)


        cv2.imshow('img', img)
##    else:
##        toggle = True
##        imageBase = frame.array
####        base = frame.array
####        base = cv2.cvtColor(imageBase, cv2.COLOR_BGR2GRAY)
##        base = cv2.medianBlur(imageBase[:,:,0],13)
##
####        cv2.imwrite('/home/pi/Desktop/033_12_lens.jpg', base)
##        laser.value = 1
##        retVal, base = cv2.threshold(base, 27, 255, cv2.THRESH_TRUNC)
##        base = (base*4).astype(np.uint8)
####        cv2.imshow('img', base)
    
    key = cv2.waitKey(1) & 0xFF

    
    if key== ord("q"):
        cv2.destroyAllWindows()
        break
    elif key == ord("b"):
        cv2.imwrite('/home/pi/Desktop/baseline.jpg', img)
        print("Baseline saved")
    
    laser.value = 1
##    time.sleep(0.01)
    rawCapture.truncate(0)
    print("in: " + str(time.time() - now) + " seconds")
