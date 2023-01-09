import cv2
import PIL

import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

img = cv2.imread('uploads/images/maintainence_log_test.jpg')

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.imshow('Result',img)

print(pytesseract.image_to_string(img))
cv2.imshow('Result',img)
##cv2.waitKey(0)

hImg,wImg,_ = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    print(b)

for b in boxes.splitlines():    
    b = b.split(' ')
    print(b)
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])

##cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),3)
#cv2.putText(img,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)

##detecting words
hImg,wImg,_ = img.shape
boxes = pytesseract.image_to_data(img)
print(boxes)

for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b = b.split()
        if len(b)==12:
            print(b)

vehicle_no = "KAS51P6745"
true_vehicle_no = 1
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b = b.split()
        if len(b)==12:
            ##print (b[11])
            if b[11] == vehicle_no:
                true_vehicle_no=true_vehicle_no*0  
            else:
                true_vehicle_no=true_vehicle_no*1
print (true_vehicle_no)                

valid_upto = "18-03-2013"
true_valid_upto = 1
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b = b.split()
        if len(b)==12:
            ##print (b[11])
            if b[11] == valid_upto:
                true_valid_upto=true_valid_upto*0  
            else:
                true_valid_upto=true_valid_upto*1
print (true_valid_upto)  





