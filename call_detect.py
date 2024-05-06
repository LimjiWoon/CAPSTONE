import subprocess
import os

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

#파일들의 위치 획득
current_directory = os.getcwd().replace('\\','/')
python_path = f'{current_directory}/yolov5/detect.py'
weights = f'{current_directory}/yolov5/runs/train/exp/weights/best.pt'

# 분석할 이미지 또는 비디오의 경로(yolov5 폴더의 data)
source = 'yolov5/data/images'
# 감지 결과를 저장할 디렉토리 (현재 폴더의 새로만든 result 폴더)
output_dir = 'result'  
run_detection(python_path, weights, source, output_dir)
