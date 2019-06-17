import cv2
from functools import wraps
from pygame import mixer
import time

lastsave = 0

#函数修饰器，传入函数名作为参数，在
def counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):#*args 为不定长元组参数 **kwargs为不定长字典参数
        tmp.count += 1
        global lastsave
        if time.time() - lastsave > 3:
            # this is in seconds, so 5 minutes = 300 seconds
            lastsave = time.time() # time 函数以秒为精度，获取当前时间，返回浮点类型
            tmp.count = 0
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp

#级联分类器，基于机器学习，通过大量的样本训练得到的分类器，目前推测返回的是一个函数
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)


@counter
def closed():
  print("Eye is Closed")


def openeye():
  print("Eye is Open")


#音频输出函数，可通过耳机口输出音乐
def sound():
    mixer.init()
    mixer.music.load('sound.mp3')
    mixer.music.play()

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
#判断上一函数是否检测到了有效的eyes，若是则有值，若否则返回空的元组值
        if eyes is not (): #is not() 判断前后两个变量是否处于同一地址   //  （）表示空的元组类型
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  #框出眼睛
                openeye()
        else:               #未检测到有效的eyes数据
           closed()
           if closed.count == 3:
               print("driver is sleeping")
               sound()





    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff


cap.release()
cv2.destroyAllWindows()
