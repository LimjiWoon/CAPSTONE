import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore

class BadProductDetectionSystem(QWidget):
    def __init__(self, good_count, defective_count):
        super().__init__()
        self.is_running = True
        self.total_count = good_count + defective_count
        self.good_count = good_count
        self.defective_count = defective_count
        if self.defective_count == 0:
            self.defective_probability = 0
        else:
            self.defective_probability = self.defective_count / self.total_count * 100
        self.labels = {}  # 레이블 참조를 저장할 딕셔너리
        self.initUI()

    def initUI(self):
        self.setWindowTitle('불량품 검출 시스템')
        self.setStyleSheet('''
            background-color: #f0f0f0;
            font-size: 14px;
            border: 2px solid #333;
            border-radius: 10px;
        ''')

        grid = QGridLayout()
        self.setLayout(grid)

        labels_info = {
            '총 개수': self.total_count,
            '정품 개수': self.good_count,
            '불량품 개수': self.defective_count,
            '불량품 확률': f"{self.defective_probability:.2f}%"
        }

        for i, (label, value) in enumerate(labels_info.items()):
            label_widget = QLabel(f"{label}: {value}")
            label_widget.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(label_widget, i, 0)
            self.labels[label] = label_widget

        icon_label = QLabel()
        pixmap = QPixmap('icon.png')
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        grid.addWidget(icon_label, 0, 1, len(labels_info), 1)

        # 창 크기 조정
        self.setGeometry(100, 100, 600, 300)
        self.show()

    def updateUI(self, additional_good, additional_defective):
        # 누적된 수량을 업데이트
        self.good_count += additional_good
        self.defective_count += additional_defective
        self.total_count = self.good_count + self.defective_count
        if self.total_count > 0:
            self.defective_probability = self.defective_count / self.total_count * 100
        else:
            self.defective_probability = 0
        
        # 레이블 업데이트
        self.labels['총 개수'].setText(f"총 개수: {self.total_count}")
        self.labels['정품 개수'].setText(f"정품 개수: {self.good_count}")
        self.labels['불량품 개수'].setText(f"불량품 개수: {self.defective_count}")
        self.labels['불량품 확률'].setText(f"불량품 확률: {self.defective_probability:.2f}%")


    #종료시 출력할 문구
    def closeEvent(self, event):
        print("Closing the application.")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BadProductDetectionSystem(90, 0)
    time.sleep(2)
    window.updateUI(180, 20)  # 예시 업데이트 호출
    sys.exit(app.exec_())