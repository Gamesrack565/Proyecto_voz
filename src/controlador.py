import os

class ControladorReconocimiento:
    def __init__(self, modelo_hmm, modelo_svm, vista):
        self.modelo_hmm = modelo_hmm
        self.modelo_svm = modelo_svm
        self.vista = vista

    def ejecutar(self):
        while True:
            opcion = self.vista.mostrar_menu()
            
            if opcion == "1":
                self.vista.mostrar_mensaje("[*] Iniciando entrenamiento y comparativa (HMM vs SVM)...")
                if not os.path.exists("../corpus"):
                    self.vista.mostrar_mensaje("[!] Error: No se encontró el directorio '../corpus'.")
                    continue
                
                # 1. Entrenar y evaluar HMM
                datos_prueba = self.modelo_hmm.entrenar(ruta_corpus="../corpus", test_size=0.2)
                self.modelo_hmm.evaluar(datos_prueba)
                
                # 2. Entrenar y evaluar SVM
                datos_prueba_svm = self.modelo_svm.entrenar(ruta_corpus="../corpus", test_size=0.2)
                acc_svm, wer_svm = self.modelo_svm.evaluar(datos_prueba_svm)
                
                # 3. Mostrar tabla comparativa final
                if datos_prueba:
                    acc_hmm, wer_hmm = self._calcular_metricas_hmm(datos_prueba)
                    self.vista.mostrar_comparativa(acc_hmm, wer_hmm, acc_svm, wer_svm)
                
            elif opcion == "2":
                ruta = self.vista.solicitar_ruta_audio()
                if ruta:
                    self._procesar_y_predecir(ruta)
                    
            elif opcion == "3":
                ruta_temporal = self.vista.grabar_audio()
                self._procesar_y_predecir(ruta_temporal)
                
            elif opcion == "4":
                self.vista.mostrar_mensaje("[*] Saliendo del sistema...")
                break
                
            else:
                self.vista.mostrar_mensaje("[!] Opción no válida. Intenta de nuevo.")

    def _calcular_metricas_hmm(self, datos_prueba):
        aciertos = 0
        total = 0
        for etiqueta_real, audios_test in datos_prueba:
            for caracteristicas in audios_test:
                if self.modelo_hmm.predecir(caracteristicas) == etiqueta_real:
                    aciertos += 1
                total += 1
        acc = (aciertos / total) * 100 if total > 0 else 0.0
        return acc, 100.0 - acc

    def _procesar_y_predecir(self, ruta_audio):
        try:
            caracteristicas = self.modelo_hmm.preprocesador.procesar(ruta_audio)
            pred_hmm = self.modelo_hmm.predecir(caracteristicas)
            pred_svm = self.modelo_svm.predecir(caracteristicas)
            self.vista.mostrar_resultado(pred_hmm, pred_svm)
        except Exception as e:
            self.vista.mostrar_mensaje(f"[!] Aviso al procesar el audio: {e}")