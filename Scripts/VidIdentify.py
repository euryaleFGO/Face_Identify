import cv2
import dlib
import numpy as np
from scipy.spatial import distance

def recognize_faces_in_video(video_source=0, scale_percent=50, skip_frames=2):
    # 加载保存的信息
    known_encodings = np.load('Data/Models/Face_Data/known_encodings.npy')
    known_names = np.load('Data/Models/Face_Data/known_names.npy')

    # 初始化dlib的检测器、预测器和人脸识别模型
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("Data/Models/Pretrained_Models/shape_predictor_68_face_landmarks.dat")
    face_recognition_model = dlib.face_recognition_model_v1("Data/Models/Pretrained_Models/dlib_face_recognition_resnet_model_v1.dat")

    # 创建一次CLAHE对象，在循环中重复使用
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 打开视频源
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("无法打开视频源")
        return

    frame_counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法获取帧，可能是视频结束了")
            break

        frame_counter += 1
        if frame_counter % (skip_frames + 1) != 0:
            continue

        # 缩放图片以减少处理时间和内存消耗
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
        cv2.imshow('Video', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
