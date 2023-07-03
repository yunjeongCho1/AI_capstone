from pathlib import Path
import subprocess
from tkinter import Tk, Canvas, Button, PhotoImage, Label
from PIL import ImageTk
import numpy as np
import cv2
import tkinter as tk
import time
import os
from threading import Thread
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from tkinter import messagebox

##########################################################
# face.py
def face():
    pro_id_face_dir ='C:/detection_face/pro_ID_face'
    pro_real_face_dir ='C:/detection_face/pro_REAL_face'
    real_face_dir = 'C:/detection_face/REAL_face'
    id_face_dir = 'C:/detection_face/ID_face'

    os.makedirs(real_face_dir, exist_ok=True)
    os.makedirs(id_face_dir, exist_ok=True)
    os.makedirs(pro_id_face_dir,exist_ok=True)
    os.makedirs(pro_real_face_dir,exist_ok=True)

    cap = None
    def ID_camera():
        xml = 'C:/Users/yjcho/Desktop/AI_CAPSTONE/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(xml)

        cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용
        cap.set(3, 720) # 너비
        cap.set(4, 480) # 높이

        save_face = False

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) # 좌우 대칭
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.05, 3)
            print("Number of id_faces detected: " + str(len(faces)))

            if len(faces) == 1:  # 1개의 얼굴만 감지되는 경우
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
                cv2.imshow('id_result',frame)

                k = cv2.waitKey(30) & 0xFF
                if k == ord('q'):  # q 키를 누르면 얼굴 캡처
                    id_image = frame[y:y+h, x:x+w]
                    output_path = os.path.join(id_face_dir, "id_face.jpg")
                    cv2.imwrite(output_path, id_image)
                    print("ori_id_face save")

                # 전처리 및 저장
                    id_extract_and_save_faces(id_image, xml, pro_id_face_dir)

                    break

            k = cv2.waitKey(30) & 0xFF
            if k == 27: # Esc 키를 누르면 종료
                break

        cap.release()
        cv2.destroyAllWindows()
    # if save_face:
    #     real_camera()
        real_camera()

    def real_camera():
        xml = 'C:/Users/yjcho/Desktop/AI_CAPSTONE/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(xml)

        cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용
        cap.set(3, 720) # 너비
        cap.set(4, 480) # 높이

        save_face = False

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) # 좌우 대칭
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.05, 3)
            print("Number of real_faces detected: " + str(len(faces)))
            if len(faces) == 1:  # 1개의 얼굴만 감지되는 경우
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
                cv2.imshow('real_result',frame)

                k = cv2.waitKey(30) & 0xFF
                if k == ord('q'):  # q 키를 누르면 얼굴 캡처
                    real_image = frame[y:y+h, x:x+w]
                    output_path = os.path.join(real_face_dir, "real_face.jpg")
                    cv2.imwrite(output_path, real_image)
                    print("ori_real_face save")

                    # 전처리 및 저장
                    real_extract_and_save_faces(real_image, xml, pro_real_face_dir)

                    break

            k = cv2.waitKey(30) & 0xFF
            if k == 27: # Esc 키를 누르면 종료
                break

        cap.release()
        cv2.destroyAllWindows()


    # if save_face:
    #model()
    
    def ID_button():
        thread = Thread(target=ID_camera)
        thread.start()
        root.withdraw()

    def REAL_button():
        thread = Thread(target=real_camera)
        thread.start()
        root.withdraw()

    def id_extract_and_save_faces(id_img, xml, pro_id_face_dir):
        face_cascade = cv2.CascadeClassifier(xml)
        id_img = np.array(id_img, dtype=np.uint8)
        gray = cv2.cvtColor(id_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    
        for i, (x, y, w, h) in enumerate(faces):
            face_image = id_img[y:y+h, x:x+w]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(face_image, (100, 100))

            output_path = os.path.join(pro_id_face_dir, f"face_{i+1}.jpg")
            cv2.imwrite(output_path, resized)

        print("pro_id_face Save!")

    def real_extract_and_save_faces(real_img, xml, pro_real_face_dir):
        face_cascade = cv2.CascadeClassifier(xml)
        real_img = np.array(real_img, dtype=np.uint8)
        gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        for i, (x, y, w, h) in enumerate(faces):
            face_image = real_img[y:y+h, x:x+w]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(face_image, (100, 100))

            output_path = os.path.join(pro_real_face_dir, f"face_{i+1}.jpg")
            cv2.imwrite(output_path, resized)

        print("pro_real_face Save!")
        
        messagebox.showinfo("FINISH")
    
    def close_window():
        cap.release()  # 카메라 종료
        cv2.destroyAllWindows()  # 창 닫기
        root.destroy()  # Tkinter 창 닫기

    root = tk.Tk()
    root.title("AI_CAPSTONE")
    root.geometry("720x400+100+100")
    root.protocol("WM_DELETE_WINDOW", close_window)

    btn1 = tk.Button(root, text="START", command=ID_button)
    btn1.configure(font=("Verdana", 15), width=20)
    btn1.pack(pady=150)

    root.mainloop()

    
################################################################################
# gui.py

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\yjcho\Desktop\AI_CAPSTONE\assets\frame0")

window = Tk()

current_page = 1  # 현재 페이지 번호
total_pages = 2  # 전체 페이지 개수


def model():
    class FaceNet(nn.Module):
        def __init__(self, embedding_size):
            super(FaceNet, self).__init__()
            self.model = resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, embedding_size)

        def forward(self, x):
            return self.model(x)

    embedding_size = 128  # 임베딩 벡터의 차원
    model_path = 'C:/Users/yjcho/Desktop/AI_CAPSTONE/facenet_model.pt'
    id_dir = 'C:/detection_face/pro_ID_face/face_1.jpg'
    real_dir = 'C:/detection_face/pro_REAL_face/face_1.jpg'
    
    
    threshold = 0.78


    # 모델 로드
    model = FaceNet(embedding_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # 이미지 로드 및 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    id_img = Image.open(id_dir).convert("RGB")
    real_img = Image.open(real_dir).convert("RGB")

    id_tensor = transform(id_img).unsqueeze(0)
    real_tensor = transform(real_img).unsqueeze(0)

    # 임베딩 추출
    with torch.no_grad():
        id_embedding = model(id_tensor)
        real_embedding = model(real_tensor)

# 거리 계산
    distance = torch.dist(id_embedding, real_embedding).item()

# 코사인 유사도 계산
    similarity = F.cosine_similarity(id_embedding, real_embedding).item()

# 동일 인물 여부 판별
    is_same_person = similarity >= threshold

# 결과 생성
    print(f"두 이미지의 유사도: {similarity:.4f}")
    result_text = f"They are: "
     #{similarity:.4f}
    if is_same_person:
        result_text += "\n\nSAME PERSON.\n\nPASS"
    else:
        result_text += "\n\nNOT SAME PERSON.\n\n Check caution and try again."

    return result_text

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def start_button_clicked():
    #subprocess.Popen(["python", "face.py"])
    face()
    print("CAMERA ON")

def next_page():
    global current_page
    current_page += 1
    if current_page > total_pages:
        current_page = 1
    
    update_page()

def update_page():
    global result_label
    # 페이지 전환 시 캔버스 배경 이미지 변경
    if current_page == 1:
        canvas.config(bg="#FFFFFF")
        canvas.itemconfig(image_1, image=image_image_1)
        if result_label:
            result_label.pack_forget()
        result_label.config(text="")
    elif current_page == 2:
        canvas.config(bg="#000000")
        canvas.itemconfig(image_1, image=image_image_1)
        canvas.itemconfigure(info1, state="hidden")
        canvas.itemconfigure(info2, state="hidden")
        canvas.itemconfigure(info3, state="hidden")
        canvas.itemconfigure(info4, state="hidden")
        canvas.itemconfigure(info5, state="hidden")
        canvas.itemconfigure(info6, state="hidden")
        start_button.place_forget()
        next_button.place_forget()
        # 모델 실행 및 결과 출력
        result_text = model()
        result_label = Label(window, text="", font=("Verdana bold", 12))
        result_label.place(x=220,y=180,width=300,height=120)
        result_label.config(text=result_text)

class FaceNet(nn.Module):
    def __init__(self, embedding_size):
        super(FaceNet, self).__init__()
        self.model = resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        return self.model(x)


window.geometry("720x400+100+100")
window.configure(bg="#FFFFFF")

# background
canvas = Canvas(window, bg="#FFFFFF", height=500, width=800, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=-50, y=0)

# page1
image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(439.0, 212.0, image=image_image_1)
button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
# title
title = canvas.create_text(430, 30.0, anchor="n", text="FACE", fill="#FFFFFF",
                   font=("Verdana bold", 48 * -1))
# info
info1 = canvas.create_text(200.0, 150.0, anchor="nw", 
                        text="1. Recognize the face on the ID card\n", fill="#000000",
                        font=("Verdana bold", 17 * -1))
info2 = canvas.create_text(200.0, 200.0, anchor="nw", 
                        text="2. Recognize the actual face\n", fill="#000000",
                        font=("Verdana bold", 17 * -1))
info3 = canvas.create_text(200.0, 250.0, anchor="nw", 
                        text="3. Wait until the faces are saved\n", fill="#000000",
                        font=("Verdana bold", 17 * -1))
info4 = canvas.create_text(200.0, 300.0, anchor="nw", 
                        text="4. Cheak the results\n", fill="#000000",
                        font=("Verdana bold", 17 * -1))
info5 = canvas.create_text(140.0, 370,anchor="nw",
                           text=": Remove your glasses / Bright lighting / Simple background",
                           fill="#000000",font=("Verdana", 14 * -1))
info6 = canvas.create_text(70.0, 370,anchor="nw",
                           text="Caution",
                           fill="red",font=("Verdana bold", 14 * -1))

# Button
start_button = Button(image=button_image_1, borderwidth=0,
                      highlightthickness=0, command=start_button_clicked, relief="flat")
start_button.place(x=575.0, y=160.0, width=100.0, height=100.0)

next_button = Button(text="RESULT",font=("Verdana",12), command=next_page)
next_button.place(x=575,y=300,width=100,height=40)
#next_button.pack()

window.resizable(False, False)
window.mainloop()
