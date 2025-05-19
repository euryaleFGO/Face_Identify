# scripts/prepare_data.py
import os
import dlib
import numpy as np
import cv2

# 加载预训练模型
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("Data/Models/Pretrained_Models/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("Data/Models/Pretrained_Models/dlib_face_recognition_resnet_model_v1.dat")

def get_face_encodings(directory, role):
    known_face_encodings = []
    known_face_names = []

    for name in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, name)):
            continue

        for filename in os.listdir(os.path.join(directory, name)):
            img_path = os.path.join(directory, name, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = face_detector(gray, 1)
            for face in faces:
                landmarks = shape_predictor(gray, face)
                encoding = np.array(face_recognition_model.compute_face_descriptor(img, landmarks))
                known_face_encodings.append(encoding)
                known_face_names.append((name, role))

    return known_face_encodings, known_face_names

# 获取管理员和普通用户的数据
admin_encodings, admin_names = get_face_encodings('Data/Character/Admins', 'admin')
user_encodings, user_names = get_face_encodings('Data/Character/Users', 'user')

# 合并所有编码和名称
all_encodings = admin_encodings + user_encodings
all_names = admin_names + user_names
# 保存编码和名称
np.save('Data/Models/Face_Data/known_encodings.npy', all_encodings)
np.save('Data/Models/Face_Data/known_names.npy', all_names)