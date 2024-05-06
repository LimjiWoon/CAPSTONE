import subprocess
import os

def run_detection(weights_path, source, output_dir):
    # detect.py 호출
    subprocess.run([
        'python', 'path/to/detect.py',
        '--weights', weights_path,
        '--source', source,
        '--project', output_dir,  # 결과(저장된 이미지)를 저장할 디렉토리
        '--exist-ok'  # 기존 결과 디렉토리가 있어도 덮어쓰기 허용
    ], check=True)

# 예제 사용
weights = 'C:/github/CAPSTONE/yolov5/runs/train/exp/weights/best.pt'
source = 'data/images'  # 분석할 이미지 또는 비디오의 경로
output_dir = 'runs/detect'  # 감지 결과를 저장할 디렉토리
run_detection(weights, source, output_dir)
