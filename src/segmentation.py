import cv2
import numpy as np

from PIL import Image
import numpy as np

w, h = 512, 512


#import image
image = cv2.imread(r'./data/im2.jpg')
image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second',thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((5,25), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

image_to_detect = []
for i, ctr in enumerate(sorted_ctrs):

    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi=image[y:y+h, x:x+w]
    image_to_detect.append(roi)
    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)

    #os.system("main.py")
    cv2.waitKey(0)
data = image_to_detect
for i in range(0, len(data)):
    img = Image.fromarray(data[i], 'RGB')
    # img.save('my.png')
    img.show()
#data[0:256, 0:256] = [255, 0, 0] # red patch in upper left


print(image_to_detect)
cv2.imshow('marked areas',image)
cv2.waitKey(0)