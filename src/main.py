from modelo import ModeloReconocimiento
from modelo_svm import ModeloSVM
from vista import VistaInteractiva
from controlador import ControladorReconocimiento

if __name__ == "__main__":
    modelo_hmm = ModeloReconocimiento()
    modelo_svm = ModeloSVM()
    vista = VistaInteractiva()
    
    app = ControladorReconocimiento(modelo_hmm, modelo_svm, vista)
    
    try:
        app.ejecutar()
    except KeyboardInterrupt:
        print("\n\n[!] Ejecución interrumpida por el usuario. Saliendo...")