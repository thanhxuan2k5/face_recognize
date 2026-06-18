
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow
from ui.styles import LIGHT_STYLE
from utils.logger import get_logger

log = get_logger("main")


def main() -> None:

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)



    app = QApplication(sys.argv)
    app.setApplicationName("Face Recognition AI")
    app.setStyleSheet(LIGHT_STYLE)

    window = MainWindow()
    window.show()

    log.info("Application started.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
