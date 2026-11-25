import sys
from PyQt6.QtWidgets import QApplication
from modelo import ModeloReconocimiento
from modelo_svm import ModeloSVM
from vista import VistaInteractiva
from controlador import ControladorReconocimiento

if __name__ == "__main__":
    app_qt = QApplication(sys.argv)

    modelo_hmm = ModeloReconocimiento()
    modelo_svm = ModeloSVM()
    vista = VistaInteractiva()
    controlador = ControladorReconocimiento(modelo_hmm, modelo_svm, vista)

    vista.show()
    sys.exit(app_qt.exec())