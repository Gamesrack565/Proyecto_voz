import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os

class VistaInteractiva:
    def __init__(self, frecuencia_muestreo=16000):
        self.fs = frecuencia_muestreo

    def mostrar_menu(self):
        print("\n" + "="*55)
        print(" SISTEMA DE RECONOCIMIENTO DE VOZ (HMM vs SVM) ".center(55))
        print("="*55)
        print("1. Entrenar y Evaluar Modelos (HMM y SVM)")
        print("2. Predecir desde archivo de audio existente (.wav)")
        print("3. Predecir grabando desde el micrófono")
        print("4. Salir")
        return input("\nSelecciona una opción (1-4): ")

    def solicitar_ruta_audio(self):
        ruta = input("Ingresa la ruta del archivo .wav: ").strip()
        if os.path.exists(ruta):
            return ruta
        else:
            print("Error: El archivo especificado no existe.")
            return None

    def grabar_audio(self, duracion=2.0, archivo_salida="temp_audio.wav"):
        print("\n[!] Preparando micrófono...")
        duracion_total = duracion + 0.15
        print(f"[*] Grabando por {duracion} segundos... ¡Habla ahora!")
        
        grabacion = sd.rec(int(duracion_total * self.fs), samplerate=self.fs, channels=1, dtype=np.float32)
        sd.wait()
        
        recorte_inicio = int(0.15 * self.fs)
        grabacion_limpia = grabacion[recorte_inicio:]
        
        print("[*] Grabación finalizada.")
        write(archivo_salida, self.fs, grabacion_limpia)
        return archivo_salida

    def mostrar_resultado(self, pred_hmm, pred_svm):
        print("\n" + "-"*50)
        print(f" >> Predicción HMM : {pred_hmm.upper()}")
        print(f" >> Predicción SVM : {pred_svm.upper()}")
        print("-"*50 + "\n")

    def mostrar_comparativa(self, acc_hmm, wer_hmm, acc_svm, wer_svm):
        print("\n" + "="*55)
        print(" RESUMEN COMPARATIVO DE MODELOS (TEST SET) ".center(55))
        print("="*55)
        print(f" {'Modelo':<15} | {'Accuracy (%)':<15} | {'WER (%)':<15}")
        print("-" * 55)
        print(f" {'HMM (Bakis)':<15} | {acc_hmm:>12.2f}%   | {wer_hmm:>10.2f}%")
        print(f" {'SVM (RBF)':<15} | {acc_svm:>12.2f}%   | {wer_svm:>10.2f}%")
        print("="*55)
        ganador = "HMM (Modelos Ocultos de Markov)" if acc_hmm >= acc_svm else "SVM (Máquinas de Soporte Vectorial)"
        print(f" [*] Mejor desempeño general: {ganador}")
        print("="*55 + "\n")

    def mostrar_mensaje(self, mensaje):
        print(f"\n{mensaje}")