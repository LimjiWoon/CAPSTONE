import torch
import torch.nn.functional as F
import sys
import os
import glob
import subprocess
import time
import firebase_admin
from firebase_admin import credentials, storage
import requests
import serial #(팀장)
from my_models import resnet18, resnet34, resnet50, VGG, make_layers, MobileNet
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from detect_ui import BadProductDetectionSystem
from torchvision import transforms
from PIL import Image

class DetectionThread(QThread):
    update_count = pyqtSignal(int, int)  # 신호 정의
    update_image = pyqtSignal(str)       # 이미지 업데이트 신호 정의

    def __init__(self, model_dir, current_directory, source, output_dir, bucket, ser): #ser 추가(팀장)
        super().__init__()
        self.current_directory = current_directory
        self.source = source
        self.output_dir = output_dir
        self.models = load_models(model_dir)
        self.count = 0
        self.bucket = bucket
        self.ser = ser #팀장

    def run(self):
        while True:
            download_image(self.count, self.source, self.bucket)
            jpg_files = glob.glob(f"{self.current_directory}/{self.source}/*.jpg")
            if jpg_files:
                run_detection(python_path, weights, self.source, self.output_dir)
                self.count += 1
                for file in jpg_files:
                    os.remove(file)
            else:
                print("No files found, sleeping...")
                #추후 시간초 조정
                time.sleep(5)
                continue

            # 결과 폴더에서 파일 처리
            image_directory = f'{self.current_directory}/result/exp'
            for filename in os.listdir(image_directory):
                if filename.endswith('_crop_0.jpg'):  # 특정 파일 필터링 가능
                    image_path = os.path.join(image_directory, filename)
                    image = transform_image(image_path)
                    _, _, final_prediction = predict_with_soft_voting(self.models, image)

                    if final_prediction.item() == 1:
                        self.update_count.emit(1, 0)  # 정품 수 1 증가
                        # 추가 코드 작성 바람(아두이노 연결) #(팀장)
                        self.ser.write(b'0')
                    else:
                        self.update_count.emit(0, 1)  # 불량품 수 1 증가
                        # 추가 코드 작성 바람(아두이노 연결) #(팀장)
                        self.ser.write(b'1')

                    self.update_image.emit(image_path)  # 이미지 업데이트 신호 발생
                    time.sleep(0.1) #신호 발생과 동시에 파일 삭제가 일어나 ui 업데이트가 일어나지 않음
                    # 해당 파일을 삭제한다.
                    os.remove(image_path)

#파이어베이스 다운로드 코드
def download_image(image_number, download_dir, bucket):
    # 파일 이름 생성
    file_name = f"image_{image_number}.jpg"
    
    # Firebase Storage에서 blob 참조 가져오기
    blob = bucket.blob(f"image/{file_name}")
    
    # 다운로드 경로 생성
    download_path = os.path.join(download_dir, file_name)
    
    # 파일 다운로드
    if blob.exists():
        blob.download_to_filename(download_path)
        print(f"Downloaded {file_name} to {download_path}")
    else:
        print(f"{file_name} does not exist in the bucket.")

#탐색 코드
def run_detection(python_path ,weights_path, source, output_dir):
    # detect.py 호출
    subprocess.run([
        'python', python_path,
        '--weights', weights_path,
        '--source', source,
        '--project', output_dir,  # 결과(저장된 이미지)를 저장할 디렉토리
        '--exist-ok',  # 기존 결과 디렉토리가 있어도 덮어쓰기 허용
        '--conf-thres', '0.5'  # 신뢰 임계값을 0.5로 설정
    ], check=True)


def load_models(model_dir):
    models = {}
    
    # ResNet 모델 설정
    #resnet18_model = resnet18()
    resnet34_model = resnet34()
    #resnet50_model = resnet50()
    
    # VGG16 모델 설정
    #cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #vgg16_features = make_layers(cfg, batch_norm=True)
    #vgg16_model = VGG(vgg16_features, num_classes=2)
    
    mobile_net_model = MobileNet(num_classes=2)

    # 모델들의 상태 로드
    #resnet18_model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet18_model.pth')))
    resnet34_model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet34_model.pth')))
    #resnet50_model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet50_model.pth')))
    #vgg16_model.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16_model.pth')))
    mobile_net_model.load_state_dict(torch.load(os.path.join(model_dir, 'mobilenet_model.pth')))

    # 평가 모드 설정
    #resnet18_model.eval()
    resnet34_model.eval()
    #resnet50_model.eval()
    #vgg16_model.eval()
    mobile_net_model.eval()

    # 모델들을 딕셔너리에 저장
    #models['resnet18'] = resnet18_model
    models['resnet34'] = resnet34_model
    #models['resnet50'] = resnet50_model
    #models['vgg16'] = vgg16_model
    models['mobilenet'] = mobile_net_model

    return models


def transform_image(image_path):
    #데이터 증강기법 transform
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]

    # RGB 값의 평균을 계산하여 그레이스케일 채널에 사용
    mean_gray = sum(mean_rgb) / len(mean_rgb)
    std_gray = sum(std_rgb) / len(std_rgb)

    transform = transforms.Compose([
        transforms.Resize((128,128)),  # 이미지 크기를 128*128로 변경
        transforms.Grayscale(),       # 그레이스케일로 변환
        transforms.ToTensor(),        # 텐서로 변환
        transforms.Normalize(mean=[mean_gray], std=[std_gray])  # 정규화
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원 추가
    return image

def predict_with_soft_voting(models, image):
    softmax_outputs = []
    with torch.no_grad():
        for model in models.values():  # 딕셔너리의 값(모델)들을 순회
            logits = model(image)
            probs = F.softmax(logits, dim=1)
            softmax_outputs.append(probs)
    
    ensemble_probs = torch.mean(torch.stack(softmax_outputs), dim=0)
    final_prediction = torch.argmax(ensemble_probs, dim=1)
    return softmax_outputs, ensemble_probs, final_prediction


#앙상블이 잘 됬는지 확인하기 위한 함수
def process_images(image_directory, models):
    count = 0
    true_0 = 0
    true_1 = 0
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_directory, filename)
            image = transform_image(image_path)
            _, _, final_prediction = predict_with_soft_voting(models.values(), image)
            count += 1
            if count < 96 or count > 299:
                if final_prediction.item() == 1:
                    true_1 += 1
            else:
                if final_prediction.item() == 0:
                    true_0 += 1
                    
            print(f"{filename} 결과: {final_prediction}")
    print(f"true 0: {true_0}, true 1: {true_1}")

#앙상블 모델 평가를 위한 함수
def analyze_model_contributions(models, image):
    model_contributions = {}
    with torch.no_grad():
        for model_name, model in models.items():
            logits = model(image)
            probs = F.softmax(logits, dim=1)
            model_contributions[model_name] = probs
    
    return model_contributions


# 평가를 위한 이미지 처리 함수
def analyze_process_images(image_directory, models):
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_directory, filename)
            image = transform_image(image_path)
            model_contributions = analyze_model_contributions(models, image)
            
            for model_name, contribution in model_contributions.items():
                print(f"{filename}의 {model_name} 모델 기여도: {contribution.squeeze().tolist()}")



def check_loaded_models(models):
    print("Loaded models:")
    for name, model in models.items():
        print(f"{name}: {model.__class__.__name__}")

if __name__ == '__main__':
    # 모델의 경로
    data_root = os.getcwd()
    model_dir = f'{data_root}/model'

    # 모델 로드
    models = load_models(model_dir)

    ##########################################################
    #아래 코드는 모델 및 성능 평가를 위한 코드임 테스트 용도

    #경로 지정
    #image_directory = f'{data_root}/img_washer_train'

    #모델이 잘 들어갔는지 확인
    #check_loaded_models(models)

    #각각의 모델들이 이미지를 잘 처리하고 있는지 확인
    #analyze_process_images(image_directory, models)

    #모델들이 잘 예측하고 있는지 확인
    #process_images(image_directory, models)

    ###########################################################

    #아래는 실시간 감지를 위한 코드임 추후
    
    #경로 지정
    image_directory = f'{data_root}/result/exp'

    #yolov5 를 호출하기 위한 경로 지정
    current_directory = data_root.replace('\\','/')
    python_path = f'{current_directory}/yolov5/detect.py'
    weights = f'{current_directory}/train/exp/weights/best.pt'

    # 분석할 이미지 또는 비디오의 경로(yolov5 폴더의 data)
    source = 'yolov5/data/images'
    # 감지 결과를 저장할 디렉토리 (현재 폴더의 새로만든 result 폴더)
    output_dir = 'result'

    #아두이노 통신을 위한 시리얼 선언 #(팀장)
    ser = serial.Serial('COM15', 9600)
    
    #파이어베이스 경로
    cred = credentials.Certificate(f"{data_root}/camerasend-cf6b7-firebase-adminsdk-wzgz6-a1b3b00dbb.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket' : 'camerasend-cf6b7.appspot.com'
    })
    
    bucket = storage.bucket()
    
    
    # UI를 띄워서 실시간으로 확인
    good_count = 0
    defective_count = 0
    
    app = QApplication(sys.argv)
    window = BadProductDetectionSystem(good_count, defective_count)
    
    detection_thread = DetectionThread(model_dir, current_directory, source, output_dir, bucket, ser) #ser 추가(팀장)
    detection_thread.update_count.connect(window.updateUI)
    detection_thread.update_image.connect(window.updateImage)  # 이미지 업데이트 신호 연결
    detection_thread.start()

    sys.exit(app.exec_())