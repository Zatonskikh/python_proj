import cv2, os, math
import numpy as np,time


#детектируем лица методом виолы джонса(используем каскады хаара)

eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#выбираем таблицу для записи результатов

# распознаем лица с помощью бинарных шаблонов
recognizer = cv2.createLBPHFaceRecognizer(1,8,4,4,100)

#повышение/понижение яркости
def bright_cng(img1,img2):
	dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
	return dst
#скеливание изображений

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
#загрузка изображений для распознования
def set_images(path):
    image_paths = []
    for d,dirs,files in os.walk(path):
        for f in files:
            path_to=os.path.join(d,f)
            if  '10.pgm'in path_to:
                image_paths.append(path_to)
    return image_paths
#загрузка изображений для обучения
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
        # считываем изображение, переводим его в чб формат и приводим к формату массива
        gray=cv2.imread(image_path)
        gray2 = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        image = np.array(gray2, 'uint8')
        # извлечение номера человека из адреса
        subject_number = int(image_path[12:].split('\\')[0][1:])
        #print subject_number
        # детектируем лица
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(70, 70))
        # создаем базу для обучения:заносим изображение в массив images а соответствующий ему номер в labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(subject_number)

    return images, labels
# Путь к фотографиям
path = './orl_facez' #папка хранится в той же директории что и файл кода
# получаем изображение и соответсвующий ему номер
images, labels = get_images(path)
#обучаем алгоритм распозновать нужные лица
recognizer.train(images, np.array(labels))
set_paths = set_images(path)
fa1=fa2=fa3=fa4=brg=bbrg= 0.0
#n=1.0

fon = cv2.imread('fon.jpg')
fheight, fwidth = fon.shape[:2]
for br in xrange(0,251,25):
        for n in xrange(1,11):
                for set_path in set_paths:
                    gray=cv2.imread(set_path)   #считываем
                    height, width = gray.shape[:2]
                    #в этих двух переменных будет хранится информация о ширине и высоте создаваемого изображения где 10 соответсвтует 0.1 доле лица от общего изображения, 10/2 - 0.2 и так далее соответственно
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
                    
                    blank_image = np.zeros((nheight,nwidth,3), np.uint8)#создаем изображение для склеивания
                    blank_image[:,0:nwidth] = (br,br,br)
                    #blank_image = bright_cng(blank_image,res)
                    #add_res = bright_cng(blank_image,res)
                    #склеиваем лицо для распознования и созданное изображения в нужных пропорциях для получения требуемой доли лица от общего размера
                    img_add =add_img(blank_image,gray)
                    #img_add =bright_cng(black_image,img_add)
                    gray2 = cv2.cvtColor(img_add, cv2.COLOR_BGR2GRAY)
                    #gray2 = bright_cng(black_image,gray2) #переводим полученный результат в чб формат и приводим к виду массива
                    image = np.array(gray2, 'uint8')
                    #print integral[-1,-1]
                    #white_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
                    #brg+=float(cv2.integral(gray2)[-1,-1])/float(cv2.integral(white_image)[-1,-1])
                    #print float(cv2.integral(gray2)[-1,-1])/float(cv2.integral(white_image)[-1,-1])
                    #image = chng_brightness(image)
                    
                    cv2.imshow('',image)
                    cv2.waitKey(1)
                    #детектируем лица: если не найдено ни одного лица увеличиваем счетчик ошибки детектирования и выводим результат на экран
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
                        #если лица детектированы то пытаемся их распознать
                    for (x, y, w, h) in faces:
                        #если лицо распознанно то выводим на экран уверенность алгоритма в соответсвии лица нужному(чем меньше number_predicted тем выше уверенность алгоритма в верном распозновании

                        number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])

                        # получаем истинный номер изображения и сравниваем его с предположенным
                        number_actual = int(set_path[12:].split('\\')[0][1:])
                        print "Found face in image {0}!".format(number_actual) #сообщаем о нахождении лица
                        if number_predicted==-1:
                                fa4+=1
                        else:
                                if number_actual == number_predicted:
                                    print "{} is Correctly Recognized with confidence {}".format(number_actual, conf) #лицо распознанно верно
                                else:
                                    print "{} is Incorrect Recognized as {}".format(number_actual, number_predicted) #лицо распознанно неверно, увеличиваем счетчик ошибки распознования
                                    fa2+=1
                        cv2.imshow("Recognizing Face ", image[y: y + h, x: x + w])#показываем найденное и распознанное лицо
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

