import cv2
import dlib
import numpy as np
from scipy.spatial import distance

def recognize_face(image_path):
    # 加载保存的信息
    known_encodings = np.load('Data/Models/Face_Data/known_encodings.npy')
    known_names = np.load('Data/Models/Face_Data/known_names.npy')

    # 初始化dlib的检测器、预测器和人脸识别模型
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("Data/Models/Pretrained_Models/shape_predictor_68_face_landmarks.dat")
    face_recognition_model = dlib.face_recognition_model_v1("Data/Models/Pretrained_Models/dlib_face_recognition_resnet_model_v1.dat")

    # 创建一次CLAHE对象，在循环中重复使用
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 读取图片
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"无法加载图片: {image_path}")
        return

    # 缩放图片以减少处理时间和内存消耗
    # 这里将图片缩放到最大边为800像素，你可以根据需要调整这个值
    scale_percent = 100  # 缩放比例
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_detector(gray, 1)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # 只对人脸区域进行CLAHE处理
        face_gray = gray[y:y+h, x:x+w]
        face_gray = clahe.apply(face_gray)

        # 使用原始灰度图像来定位特征点
        landmarks = shape_predictor(gray, face)
        encoding = np.array(face_recognition_model.compute_face_descriptor(resized_frame, landmarks))

        # 寻找最近匹配
        distances = [distance.euclidean(encoding, known) for known in known_encodings]
        min_distance_index = np.argmin(distances)

        if distances[min_distance_index] < 0.4:  # 阈值可调
            person_name, role = known_names[min_distance_index]
            identity_label = f"{person_name} ({role})"
        else:
            identity_label = "Unknown"

        # 绘制矩形框和标签
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(resized_frame, identity_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果图像
    cv2.imshow('Image', resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()