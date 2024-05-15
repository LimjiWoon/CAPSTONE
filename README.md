detect.py : yolov5의 detect.py을 수정한 파일로 탐색만이 아니라 탐색한 데이터를 잘라 저장하게 수정함 
yolov5에서 git.clone()으로 프로그램을 가져온 후 해당 .py을 yolov5에 덮어 씌우면 됩니다.

main.py : 메인 프로그램, detect.py을 통해 지정된 폴더에 있는 사진에서 washer를 탐색한 후 result 폴더에 탐색한 사진과 crop된 이미지를 저장함

          그 후 crop된 이미지를 학습된 model(ResNet18, ResNet34, VGG16, MobileNet)을 통해 판별을 한 후 앙상블 중 Soft-voting으로 최적의 정답을 도출해네 그 결과를 UI창에 실시간으로 업데이트합니다.
          
