import cv2, os, math
import numpy as np,time


#����������� ���� ������� ����� ������(���������� ������� �����)

eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#�������� ������� ��� ������ �����������

# ���������� ���� � ������� �������� ��������
recognizer = cv2.createLBPHFaceRecognizer(1,8,4,4,100)

#���������/��������� �������
def bright_cng(img1,img2):
	dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
	return dst
#���������� �����������

def add_img(img1,img2):
	rows,cols,channels = img2.shape
	rows1,cols1,channels1 = img1.shape
	#roi = img1[0:rows, 0:cols]
	roi=img1[(rows1-rows)/2:(rows1-rows)/2+rows,(cols1-cols)/2:(cols1-cols)/2+cols]
	img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 230, 255, cv2.THRESH_BINARY_INV)
	mask_inv = cv2.bitwise_not(mask)
	cv2.imshow('1',mask)

	img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)

	img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

	dst = cv2.add(img1_bg,img2_fg)
	#img1[0:rows, 0:cols ] = dst
	img1[(rows1-rows)/2:(rows1-rows)/2+rows,(cols1-cols)/2:(cols1-cols)/2+cols]=dst
	return img1
#�������� ����������� ��� �������������
def set_images(path):
    image_paths = []
    for d,dirs,files in os.walk(path):
        for f in files:
            path_to=os.path.join(d,f)
            if  '10.pgm'in path_to:
                image_paths.append(path_to)
    return image_paths
#�������� ����������� ��� ��������
def get_images(path):
    image_paths = []
    for d,dirs,files in os.walk(path):
        for f in files:
            path_to=os.path.join(d,f)
            if not '10.pgm' in path_to:
                image_paths.append(path_to)
    images = []
    labels = []
    for image_path in image_paths:
        # ��������� �����������, ��������� ��� � �� ������ � �������� � ������� �������
        gray=cv2.imread(image_path)
        gray2 = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        image = np.array(gray2, 'uint8')
        # ���������� ������ �������� �� ������
        subject_number = int(image_path[12:].split('\\')[0][1:])
        #print subject_number
        # ����������� ����
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(70, 70))
        # ������� ���� ��� ��������:������� ����������� � ������ images � ��������������� ��� ����� � labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(subject_number)

    return images, labels
# ���� � �����������
path = './orl_facez' #����� �������� � ��� �� ���������� ��� � ���� ����
# �������� ����������� � �������������� ��� �����
images, labels = get_images(path)
#������� �������� ������������ ������ ����
recognizer.train(images, np.array(labels))
set_paths = set_images(path)
fa1=fa2=fa3=fa4=brg=bbrg= 0.0
#n=1.0

fon = cv2.imread('fon.jpg')
fheight, fwidth = fon.shape[:2]
for br in xrange(0,251,25):
        for n in xrange(1,11):
                for set_path in set_paths:
                    gray=cv2.imread(set_path)   #���������
                    height, width = gray.shape[:2]
                    #� ���� ���� ���������� ����� �������� ���������� � ������ � ������ ������������ ����������� ��� 10 ������������� 0.1 ���� ���� �� ������ �����������, 10/2 - 0.2 � ��� ����� ��������������
                    nheight = height*math.sqrt(10/float(n))
                    nwidth = width*math.sqrt(10/float(n))
                    #white_image = np.zeros((nheight,nwidth,3), np.uint8)
                    #white_image[:,0:nwidth] = (255,255,255)
                    #white_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
                    #black_image = np.zeros((nheight,nwidth,3), np.uint8)
                    #black_image[:,0:nwidth] = (55,55,55)
                    #black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
                    #print wintegral[-1,-1]
                    #res = cv2.resize(fon,(int(nwidth), int(nheight)), interpolation = cv2.INTER_CUBIC)
                    
                    #gray[np.where((gray==[255,255,255]).all(axis=2))] = [br,br,br]
                    
                    blank_image = np.zeros((nheight,nwidth,3), np.uint8)#������� ����������� ��� ����������
                    blank_image[:,0:nwidth] = (br,br,br)
                    #blank_image = bright_cng(blank_image,res)
                    #add_res = bright_cng(blank_image,res)
                    #��������� ���� ��� ������������� � ��������� ����������� � ������ ���������� ��� ��������� ��������� ���� ���� �� ������ �������
                    img_add =add_img(blank_image,gray)
                    #img_add =bright_cng(black_image,img_add)
                    gray2 = cv2.cvtColor(img_add, cv2.COLOR_BGR2GRAY)
                    #gray2 = bright_cng(black_image,gray2) #��������� ���������� ��������� � �� ������ � �������� � ���� �������
                    image = np.array(gray2, 'uint8')
                    #print integral[-1,-1]
                    #white_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
                    #brg+=float(cv2.integral(gray2)[-1,-1])/float(cv2.integral(white_image)[-1,-1])
                    #print float(cv2.integral(gray2)[-1,-1])/float(cv2.integral(white_image)[-1,-1])
                    #image = chng_brightness(image)
                    
                    cv2.imshow('',image)
                    cv2.waitKey(1)
                    #����������� ����: ���� �� ������� �� ������ ���� ����������� ������� ������ �������������� � ������� ��������� �� �����
                    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(70, 70),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
                    if len(faces)<1:
                        print "Found 0 faces in {0} image!".format(int(set_path[12:].split('\\')[0][1:]))
                        fa1+=1
                    if len(faces)>1:
                        fa3+=1
                        fa2-=1
                        if fa1 !=0:
                                fa1-=1
                        #cv2.imshow('',image)
                        #cv2.waitKey(0)
                        #���� ���� ������������� �� �������� �� ����������
                    for (x, y, w, h) in faces:
                        #���� ���� ����������� �� ������� �� ����� ����������� ��������� � ����������� ���� �������(��� ������ number_predicted ��� ���� ����������� ��������� � ������ �������������

                        number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])

                        # �������� �������� ����� ����������� � ���������� ��� � ��������������
                        number_actual = int(set_path[12:].split('\\')[0][1:])
                        print "Found face in image {0}!".format(number_actual) #�������� � ���������� ����
                        if number_predicted==-1:
                                fa4+=1
                        else:
                                if number_actual == number_predicted:
                                    print "{} is Correctly Recognized with confidence {}".format(number_actual, conf) #���� ����������� �����
                                else:
                                    print "{} is Incorrect Recognized as {}".format(number_actual, number_predicted) #���� ����������� �������, ����������� ������� ������ �������������
                                    fa2+=1
                        cv2.imshow("Recognizing Face ", image[y: y + h, x: x + w])#���������� ��������� � ������������ ����
                        #cv2.imwrite('img_detect{}.jpg'.format(number_actual),image[y: y + h, x: x + w])
                        cv2.waitKey(0)
                file_1 = open("file6.txt", "a")
                #file_1.write(str(brg/40.0)+' ')
                file_1.write(str(fa1/len(set_paths))+' ')
                file_1.write(str(fa3/len(set_paths))+' ')
                file_1.write(str(fa2/(len(set_paths)-(fa1+fa3)))+' ')
                file_1.write(str(fa4/(len(set_paths)-(fa1+fa3)))+' ')
                file_1.write(str(1-fa2/(len(set_paths)-(fa1+fa3))-fa4/(len(set_paths)-(fa1+fa3)))+'\n')
                #print brg/40.0
                #bbrg+=brg/40.0
                
                conf=fa1=fa2=fa3=fa4=brg=0.0
                
#file_1.write(str(bbrg/10.0))
        file_1.write('\n')
file_1.close()
cv2.destroyAllWindows()

