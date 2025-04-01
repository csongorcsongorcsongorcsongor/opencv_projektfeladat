import cv2
import numpy as np

# Constants for image width and height
widthImg = 540
heightImg = 640

# Capture video from the default camera
cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)

# Function to process the image (convert to gray, blur, edge detection, dilation, erosion)
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

# Function to get the contours of the image
def getContours(img, imgContour):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # You can adjust this threshold
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:  # Looking for quadrilateral shapes
                biggest = approx
                maxArea = area
    if biggest.size != 0:
        cv2.drawContours(imgContour, [biggest], -1, (255, 0, 0), 20)
    return biggest

# Function to reorder points for perspective warp
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # top-left
    myPointsNew[3] = myPoints[np.argmax(add)]  # bottom-right
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # bottom-left
    return myPointsNew

# Function to apply a perspective warp
def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # Safely crop the image to avoid errors if the image is too small
    if imgOutput.shape[0] > 40 and imgOutput.shape[1] > 40:
        imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
        imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
        return imgCropped
    return imgOutput  # Return the entire image if it's too small to crop

# Function to stack images for visualization
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    # Process the image and extract contours
    imgThres = preProcessing(img)
    biggest = getContours(imgThres, imgContour)

    # Apply perspective warp if a contour is detected
    imgWarped = img
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        imageArray = ([img, imgThres],
                      [imgContour, imgWarped])
    else:
        imageArray = ([img, imgThres],
                      [img, img])

    # Stack images and display
    stackedImages = stackImages(0.6, imageArray)
    cv2.imshow("WorkFlow", stackedImages)
    cv2.imshow("ImageWarped", imgWarped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
