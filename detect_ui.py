import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore

class BadProductDetectionSystem(QWidget):
    def __init__(self, total_count, good_count, defective_count, defective_probability):
        super().__init__()
        self.total_count = total_count
        self.good_count = good_count
        self.defective_count = defective_count
        self.defective_probability = defective_probability
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

        labels = {
            '총 개수': self.total_count,
            '정품 개수': self.good_count,
            '불량품 개수': self.defective_count,
            '불량품 확률': self.defective_probability
        }

        for i, (label, value) in enumerate(labels.items()):
            label_widget = QLabel(f"{label}: {value}")
            label_widget.setAlignment(QtCore.Qt.AlignCenter)  # 텍스트 가운데 정렬
            grid.addWidget(label_widget, i, 0)  # 레이블을 열이 아닌 행에 추가

        # 아이콘 추가
        icon_label = QLabel()
        pixmap = QPixmap('icon.png')  # 아이콘 이미지 파일 경로
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(QtCore.Qt.AlignCenter)  # 아이콘 가운데 정렬
        grid.addWidget(icon_label, 0, 1, len(labels), 1)  # 아이콘을 표시할 열 범위 지정

        # 아이콘 레이블 삭제
        icon_label.deleteLater()

        self.setGeometry(100, 100, 400, 200)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    total_count = 100
    good_count = 90
    defective_count = 10
    defective_probability = 0.1
    window = BadProductDetectionSystem(total_count, good_count, defective_count, defective_probability)
    sys.exit(app.exec_())
