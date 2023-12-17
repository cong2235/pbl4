from keras import backend as K 
import imutils
from imutils.video import VideoStream
from keras.models import load_model
import numpy as np
import requests
import keras
import requests
import serial
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import dlib
import cv2, os, sys
import collections
import random
import face_recognition
import pickle
import math
import threading
import tensorflow as tf
import urllib
import firebase_admin
from firebase_admin import db, credentials
from google.cloud import firestore
from datetime import datetime


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"N:\PBL4\test\esp32-cam-3afc4-firebase-adminsdk-nlnol-e5420a89da.json"

# Kiếm cổng của arduino của loa


ip_address = "192.168.1.100"
port = 81 

width, height = 640, 480
fps = 20

#is_app_initialized = False 
is_app_initialized = False
if not is_app_initialized:
    try:
            # Thử khởi tạo ứng dụng với tên "my-app"
        firebase_admin.get_app()  # Change "DEFAULT" to "my-app"
    except ValueError:
            # Nếu ứng dụng chưa tồn tại, thực hiện khởi tạo
        cred = credentials.Certificate(r"N:\PBL4\test\esp32-cam-3afc4-firebase-adminsdk-nlnol-e5420a89da.json")
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://esp32-cam-3afc4-default-rtdb.asia-southeast1.firebasedatabase.app/'})
        is_app_initialized = True

####### DINH NGHIA CLASS, FUNCTION  ########

# class nhan dien mat
class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Du doan trang thai mat(dong, mo)
def predict_eye_state(model,image):
    image = cv2.resize(image, (20,10))
    image = image.astype(dtype=np.float32)
    
    # Chuyen thanh tensor
    image_batch = np.reshape(image, (1,10,20,1))
    image_batch = keras.applications.mobilenet.preprocess_input(image_batch)
    
    return np.argmax(model.predict(image_batch)[0])


#Doc du lieu tu Realtime DB- Lấy id đang đăng nhập
def read_data_from_readtime_db():
    #cred = credentials.Certificate(r"N:\PBL4\test\esp32-cam-3afc4-firebase-adminsdk-nlnol-e5420a89da.json")
    #firebase_admin.initialize_app(cred, {'databaseURL': 'https://esp32-cam-3afc4-default-rtdb.asia-southeast1.firebasedatabase.app/'}, name='my-app')

    # Đọc dữ liệu từ Realtime Database
    rt_db = db.reference()
    data = rt_db.get()
    
    current_userID_value = data['current_userID']
    return current_userID_value

#Ket noi Firestore
def connect_to_firestore(data):
    firestore_db = firestore.Client()

    # Create a dictionary with the timestamp and 'Da ngu gat'
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    data_dict = {formatted_time: 'Da ngu gat'}

    # Specify the document with the name obtained from the user input
    collections_ref = firestore_db.collection('users').document(data)

    # Use the set method to update the document with the dictionary
    doc_ref = collections_ref.update(data_dict)
##### Chuong trinh chinh #########

facial_landmarks_predictor = '68_face_landmarks_predictor.dat'
predictor = dlib.shape_predictor(facial_landmarks_predictor)

#Load model predict xem mat nguoi dang dong hay mo
model = load_model('weights.149-0.01.hdf5')



scale = 0.5
countClose = 0
currState = 0
alarmThreshold = 5
url = 'http://172.20.10.3/cam-lo.jpg'
cap = cv2.VideoCapture(url)
while True:
    c = time.time()
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype= np.uint8)
    
    frame = cv2.imdecode(imgnp, -1)
    sucess, img= cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0,0),fx=scale,fy=scale)
    
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l,_,_ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]
    height_ratio, width_ratio = original_height / resized_height, original_width / resized_width
    
    face_locations = face_recognition.face_locations(l,model='hog')
    
    if len(face_locations):
        top,right,bottom,left = face_locations[0]
        x1,y1,x2,y2 = left,top,right,bottom
        
        x1 = int(x1 * width_ratio)
        y1 = int(y1 * height_ratio)
        x2 = int(x2 * width_ratio)
        y2 = int(y2 * height_ratio)
        
        #trich xuat vi tri 2 mat
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray,dlib.rectangle(x1,y1,x2,y2))
        face_landmarks = face_utils.shape_to_np(shape)
        
        left_eye_indices = face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                          FacialLandMarksPosition.left_eye_end_index]
            
        (x,y,w,h) = cv2.boundingRect(np.array(left_eye_indices))
        left_eye = gray[y:y + h, x:x + w]
        
        right_eye_indices = face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                           FacialLandMarksPosition.right_eye_end_index]
        
        (x,y,w,h) = cv2.boundingRect(np.array(right_eye_indices))
        right_eye = gray[y:y + h, x:x + w]
        
        #Dung mobilenet de xem tung mat mo hay dong
        
        left_eye_open = 'yes' if predict_eye_state(model=model, image=left_eye) else 'no'
        right_eye_open = 'yes' if predict_eye_state(model=model, image=right_eye) else 'no'
        
        print('left eye open: {0}   right eye open: {1}'.format(left_eye_open,right_eye_open))
        
        if left_eye_open == 'yes' or right_eye_open == 'yes':
            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2)
            currState = 0
            countClose = 0
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,0,255),2)
            currState = 1
            countClose +=1
        
        frame = cv2.flip(frame, 1)
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Chọn codec MP4
        #output_file = f"ngugat_{datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')}.mp4"
        #out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        #start_time = time.time()  # Thời gian bắt đầu
        #end_time = start_time + 5  # Ghi video trong 5 giây

        if countClose > alarmThreshold:
            data_save = read_data_from_readtime_db()
            connect_to_firestore(data_save)
            
            # Kết nối đến arduino
            #ser = serial.Serial('COM6', 9600) 
            # Gửi tín hiệu đến arduino
            #ser.write(b'1')
            #Đóng kết nối
            #ser.close()
            try:
                responese = requests.get(url)
                img_array = np.array(bytearray(responese.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, -1)
                '''
                if frame is not None:
                    #out.write(frame) 
                    #print("Tệp video được lưu tại:", os.path.abspath(output_file))
                    cv2.imshow('ESP32-CAM Video', frame)
                    '''
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
            
        cv2.imshow('Sleep Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
       