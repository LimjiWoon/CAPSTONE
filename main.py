import torch
import torch.nn.functional as F
import sys
import os
import glob
import subprocess
import time
from my_models import resnet18, resnet34, resnet50, VGG, make_layers, MobileNet
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from detect_ui import BadProductDetectionSystem
from torchvision import transforms
from PIL import Image

class DetectionThread(QThread):
    update_count = pyqtSignal(int, int)  # 신호 정의

    def __init__(self, model_dir, current_directory, source, output_dir):
        super().__init__()
        self.current_directory = current_directory
        self.source = source
        self.output_dir = output_dir
        # 모델 로드를 메인 스레드로 이동
        self.models = load_models(model_dir)

    def run(self):
        while True:
            jpg_files = glob.glob(f"{self.current_directory}/{self.source}/*.jpg")
            if jpg_files:
                run_detection(python_path, weights, self.source, self.output_dir)
                for file in jpg_files:
                    os.remove(file)
            else:
                print("No files found, sleeping...")
                time.sleep(5)
                continue

            # 결과 폴더에서 파일 처리
            image_directory = f'{self.current_directory}/result/exp'
            for filename in os.listdir(image_directory):
                if filename.endswith('_crop_0.jpg'):
                    image_path = os.path.join(image_directory, filename)
                    image = transform_image(image_path)
                    _, _, final_prediction = predict_with_soft_voting(self.models, image)

                    if final_prediction.item() == 1:
                        self.update_count.emit(1, 0)  # 정품 수 1 증가
                        #추가 코드 작성 바람(아두이노 연결)
                    else:
                        self.update_count.emit(0, 1)  # 불량품 수 1 증가
                        #추가 코드 작성 바람(아두이노 연결)

                    #해당 파일을 삭제한다.
                    file_path = os.path.join(image_directory, filename)
                    os.remove(file_path)
                    

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
    # 모델 인스턴스 생성
    resnet18_model = resnet18()
    resnet34_model = resnet34()
    resnet50_model = resnet50()
    
    # VGG16 모델 설정
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    vgg16_features = make_layers(cfg, batch_norm=True)
    vgg16_model = VGG(vgg16_features, num_classes=2)
    
    mobile_net_model = MobileNet(num_classes=2)

    # 모델들의 상태 로드
    resnet18_model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet18_model.pth')))
    resnet34_model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet34_model.pth')))
    resnet50_model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet50_model.pth')))
    vgg16_model.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16_model.pth')))
    mobile_net_model.load_state_dict(torch.load(os.path.join(model_dir, 'mobilenet_model.pth')))

    # 평가 모드 설정
    resnet18_model.eval()
    resnet34_model.eval()
    resnet50_model.eval()
    vgg16_model.eval()
    mobile_net_model.eval()

    return resnet18_model, resnet34_model, resnet50_model,vgg16_model, mobile_net_model

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
    with torch.no_grad():  # 그라디언트 계산 비활성화
        for model in models:
            logits = model(image)  # 모델의 로짓 출력
            probs = F.softmax(logits, dim=1)  # 로짓을 확률로 변환
            softmax_outputs.append(probs)
        
    # 소프트 보팅을 통한 결과 계산
    # 각 모델의 확률 출력을 쌓고 평균을 계산
    ensemble_probs = torch.mean(torch.stack(softmax_outputs), dim=0)
    
    # 최종 예측 클래스 결정
    final_prediction = torch.argmax(ensemble_probs, dim=1)
    return softmax_outputs, ensemble_probs, final_prediction

#hard-voting 계산을 위한 함수.
def predict_with_hard_voting(models, image):
    votes = []
    with torch.no_grad():  # 그라디언트 계산 비활성화
        for model in models:
            logits = model(image)  # 모델의 로짓 출력
            predicted_class = torch.argmax(logits, dim=1)  # 가장 확률이 높은 클래스 선택
            votes.append(predicted_class)
    
    # 하드 보팅을 통한 결과 계산
    # torch.mode를 사용하여 가장 빈번한 값(최빈값)을 찾음
    final_prediction, _ = torch.mode(torch.stack(votes), dim=0)
    return votes, final_prediction

#앙상블이 잘 됬는지 확인하기 위한 함수
def process_images(image_directory, models):
    count = 0
    true_0 = 0
    true_1 = 0
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_directory, filename)
            # 이미지 파일을 전처리하여 텐서로 변환
            image = transform_image(image_path)
            # 이미지를 모델에 전달하여 결과를 출력
            _, _, final_prediction = predict_with_soft_voting(models, image)
            count += 1
            if count < 96 or count > 299:
                if final_prediction.item() == 1:
                    true_1 += 1
            else:
                if final_prediction.item() == 0:
                    true_0 += 1
                    
            print(f"{filename} 결과: {final_prediction}")
    print(f"true 0: {true_0}, true 1: {true_1}")
if __name__ == '__main__':
    # 모델의 경로
    data_root = os.getcwd()
    model_dir = f'{data_root}/model'

    # 모델 로드
    models = load_models(model_dir)

    #image_directory = f'{data_root}/result/exp'
    image_directory = f'{data_root}/img_washer_train'

    #제대로 됬는지 확인하기 위한 코드
    process_images(image_directory, models)

    #아래는 실시간 감지를 위한 코드임 추후 #을 지우면 됨

    #yolov5 를 호출하기 위한 경로 지정
    #current_directory = data_root.replace('\\','/')
    #python_path = f'{current_directory}/yolov5/detect.py'
    #weights = f'{current_directory}/train/exp/weights/best.pt'

    # 분석할 이미지 또는 비디오의 경로(yolov5 폴더의 data)
    #source = 'yolov5/data/images'
    # 감지 결과를 저장할 디렉토리 (현재 폴더의 새로만든 result 폴더)
    #output_dir = 'result'

    #UI를 띄워서 실시간으로 확인
    #good_count = 0
    #defective_count = 0
    #app = QApplication(sys.argv)
    #window = BadProductDetectionSystem(good_count, defective_count)
    
    #detection_thread = DetectionThread(model_dir, current_directory, source, output_dir)
    #detection_thread.update_count.connect(window.updateUI)
    #detection_thread.start()

    #sys.exit(app.exec_())