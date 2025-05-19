import sys
import os
import subprocess
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPainter, QColor
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QMessageBox,QMainWindow,QAction,QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PicIdentify import recognize_face 
from VidIdentify import recognize_faces_in_video

# 加载保存的信息
known_encodings = np.load('Data/Models/Face_Data/known_encodings.npy', allow_pickle=True)
known_names = np.load('Data/Models/Face_Data/known_names.npy', allow_pickle=True)

# 初始化dlib的检测器、预测器和人脸识别模型
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("Data/Models/Pretrained_Models/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("Data/Models/Pretrained_Models/dlib_face_recognition_resnet_model_v1.dat")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.checked_in = set()  # 用于记录已经打卡的人脸


    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 0)

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                landmarks = shape_predictor(gray, face)
                encoding = np.array(face_recognition_model.compute_face_descriptor(frame, landmarks))

                distances = [distance.euclidean(encoding, known) for known in known_encodings]
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]

                identity_label = "Unknown"
                similarity_percentage = (1 - min_distance) * 100 if min_distance < 1 else 0

                if min_distance < 0.4:
                    person_name, role = known_names[min_distance_index]
                    identity_label = f"{person_name} ({role})"
                    if similarity_percentage > 0:
                        identity_label += f" - {similarity_percentage:.2f}%"

                    # 检查这个人是否已经打卡过
                    if person_name not in self.checked_in:
                        self.checked_in.add(person_name)  # 将此人添加到已打卡集合中
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{person_name}已打卡 当前时间为:{current_time}")
                        # 只在第一次打卡时打印信息，之后不再打印
                    else:
                        # 如果已经打卡过，不打印信息，但仍然显示人脸框和标签
                        identity_label = f"{person_name} ({role})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, identity_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.change_pixmap_signal.emit(frame)

        self.cap.release()
    def __del__(self):
        self.cap.release()

class FaceRecognitionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Face Recognition with PyQt')
        self.setGeometry(100, 100, 800, 650)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #005ca3;
            }
            QLabel {
                font-size: 16px;
                color: #333;
            }
        """)

        layout = QVBoxLayout()

        title_label = QLabel("人脸识别", self)
        title_font = QFont("KaiTi", 48, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMinimumHeight(80)
        title_label.setStyleSheet("font-size: 48px;")
        
        layout.addWidget(title_label)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.label)

        button_layout = QHBoxLayout()
        
        self.btnAddFace = QPushButton('抓取人脸', self)
        self.btnAddFace.clicked.connect(self.add_new_face)
        self.btnAddFace.setMinimumSize(150, 40)
        button_layout.addWidget(self.btnAddFace)

        self.btnExit = QPushButton('Exit', self)
        self.btnExit.clicked.connect(self.close)
        self.btnExit.setMinimumSize(100, 40)
        button_layout.addWidget(self.btnExit)

        layout.addLayout(button_layout)
        self.setCentralWidget(QWidget())  # 设置一个中心窗口部件
        self.centralWidget().setLayout(layout)
        self.setLayout(layout)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        
        self.setWindowTitle('Face Recognition with PyQt')
        self.setGeometry(100, 100, 800, 650)
    
    # 创建菜单栏
        menubar = self.menuBar()

    # 创建“选项”菜单
        options_menu = menubar.addMenu('选项')

    # 创建“选项”下的子菜单或动作，并连接到适当的槽
        option_action = QAction('设置...', self)
        options_menu.addAction(option_action)
        import_photo_action = QAction('导入照片', self)
        options_menu.addAction(import_photo_action)
        import_photo_action.triggered.connect(self.import_photo)
        import_video_action = QAction('导入视频', self)  # 新增导入视频的动作
        options_menu.addAction(import_video_action)
        import_video_action.triggered.connect(self.import_video)

    # 创建“帮助”菜单
        help_menu = menubar.addMenu('帮助')

    # 创建“帮助”下的子菜单或动作，并连接到适当的槽
        help_action = QAction('帮助文档', self)
        help_menu.addAction(help_action)

    # 创建“关于”菜单
        about_menu = menubar.addMenu('关于')

    # 创建“关于”下的子菜单或动作，并连接到适当的槽
        about_action = QAction('关于我们...', self)
        about_menu.addAction(about_action)

    # 连接动作到槽函数（假设您已经有了实现这些功能的方法）
        option_action.triggered.connect(self.show_options)
        help_action.triggered.connect(self.show_help)
        about_action.triggered.connect(self.show_about)   
         
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.add_datetime_to_image(qt_img)
        self.label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def add_datetime_to_image(self, qt_img):
    # 获取当前日期和时间
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 创建QPainter对象，用于在图像上绘制文本
        painter = QPainter(qt_img)
        painter.setRenderHint(QPainter.Antialiasing)  # 设置抗锯齿
        painter.setFont(QFont("Times New Roman", 16, QFont.Bold))  # 设置字体为新罗马，大小16，加粗

    # 设置文本颜色为白色
        white_pen = QColor(255, 255, 255)
        painter.setPen(white_pen)
    
    # 计算文本宽度和高度
        text_width = painter.fontMetrics().width(current_datetime)
        text_height = painter.fontMetrics().height()
    
    # 设置文本位置（左上角）
        text_x = 10
        text_y = 20
    
    # 绘制白色文本
        painter.drawText(text_x + 1, text_y + 1, current_datetime)  # 偏移1像素，用于黑色描边效果
    
    # 设置画笔颜色为黑色，用于描边
        black_pen = QColor(0, 0, 0)
        painter.setPen(black_pen)
    
    # 绘制黑色描边文本
        painter.drawText(text_x, text_y, current_datetime)
        painter.end()  # 结束绘制
       # 其他菜单和动作...

    def import_photo(self):
        # 打开文件选择对话框
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image files (*.jpg *.jpeg *.png)", options=options)
        if fileName:
            # 用户选择了文件，调用人脸识别代码
            recognize_face(fileName)
            
    def import_video(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video files (*.mp4 *.avi *.mkv)", options=options)
        if fileName:
            # 调用VidIdentify模块中的函数
           recognize_faces_in_video(fileName)

    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def add_new_face(self):
        self.thread.cap.grab()
        ret, frame = self.thread.cap.retrieve()
        if not ret:
            QMessageBox.critical(self, 'Error', 'Failed to capture image.')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)

        if len(faces) == 0:
            QMessageBox.warning(self, 'Warning', 'No face detected.')
            return

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_image = frame[y:y+h, x:x+w]

            landmarks = shape_predictor(gray, face)
            encoding = np.array(face_recognition_model.compute_face_descriptor(frame, landmarks))

            distances = [distance.euclidean(encoding, known) for known in known_encodings]
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

            if min_distance >= 0.4:
                new_person_name = self.generate_unique_name('User')
                user_dir = os.path.join('Data/Character/Users', new_person_name)
                os.makedirs(user_dir, exist_ok=True)
                
                img_count = len([name for name in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, name)) and name.startswith('image_')])
                img_path = os.path.join(user_dir, f'image_{img_count + 1}.jpg')
                while os.path.exists(img_path):
                    img_count += 1
                    img_path = os.path.join(user_dir, f'image_{img_count + 1}.jpg')

                cv2.imwrite(img_path, face_image)
                QMessageBox.information(self, 'Info', f'New face added as {new_person_name}.')
            else:
                person_name, role = known_names[min_distance_index]
                if role != 'user':
                    QMessageBox.warning(self, 'Warning', 'Cannot add images to Admin or non-user profiles.')
                    return

                user_dir = os.path.join('Data/Character/Users', person_name)
                if not os.path.exists(user_dir):
                    QMessageBox.critical(self, 'Error', f'User directory {user_dir} does not exist.')
                    return
                
                img_count = len([name for name in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, name)) and name.startswith('image_')])
                img_path = os.path.join(user_dir, f'image_{img_count + 1}.jpg')
                while os.path.exists(img_path):
                    img_count += 1
                    img_path = os.path.join(user_dir, f'image_{img_count + 1}.jpg')

                cv2.imwrite(img_path, face_image)
                QMessageBox.information(self, 'Info', f'Face added to existing user {person_name}.')

            self.update_face_data()
            self.restart_program()

    def generate_unique_name(self, prefix):
        existing_names = set([name for name, _ in known_names])
        index = 1
        while True:
            new_name = f"{prefix}{index:02d}"
            if new_name not in existing_names:
                return new_name
            index += 1

    def update_face_data(self):
        try:
            subprocess.run(['python', 'Scripts\Face_Identify.py'], check=True)
            QMessageBox.information(self, 'Info', 'Face data updated successfully.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to update face data: {str(e)}')

    def restart_program(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)
    def show_options(self):
    # 这里放置打开设置对话框或者进行其他设置相关操作的代码
        QMessageBox.information(self, '选项', '开发中')

    def show_help(self):
    # 这里放置打开帮助文档或者显示帮助信息的代码
        help_text = ("<h4>抓取人脸:</h4>"
                     "<p>点击按钮，程序将捕获当前帧并识别人脸。如果未检测到人脸，将显示警告。如果检测到人脸，将询问是否将其添加为新用户。如果是新用户，将要求输入用户名，并将人脸图像保存到相应的用户目录中。如果是现有用户，将人脸图像保存到现有用户的目录中。</p>"
                     "<h4>导入照片:</h4>"
                     "<p>点击导入照片，程序将打开文件选择对话框，选择要导入的图片文件。系统将照片与已保存的人脸特征进行对比</p>"
                     "<h4>导入视频:</h4>"
                     "<p>点击导入视频，程序将打开文件选择对话框，选择要导入的视频文件。系统将视频中的人脸与已保存的人脸特征进行对比</p>"
                     "<h4>Exit:</h4>"
                     "<p>点击退出按钮，程序将关闭。</p>"
                     )
        QMessageBox.information(self, '帮助', help_text)

    def show_about(self):
    # 这里放置显示关于信息的代码
        about_text = (
        "<h2>关于 Face Recognition with PyQt</h2>"
        "<p>这是一个使用PyQt5和dlib开发的人脸识别程序。</p>"
        "<p>版本: v1.0</p>"
        "<p>开发者: 山西大学软工2208班人工智能第四小组 </p>"
        "<p>版权所有 © 2024 软工2208班人工智能第四小组</p>"
        )
        QMessageBox.about(self, '关于我们', about_text)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceRecognitionWindow()
    ex.show()
    sys.exit(app.exec_())