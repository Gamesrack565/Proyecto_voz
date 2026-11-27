import os
import sys
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from PyQt6.QtCore import QObject, QThread, pyqtSignal


class RedireccionConsola(QObject):
    """Captura los print() de los modelos y los envía a la interfaz."""
    texto_emitido = pyqtSignal(str)

    def write(self, texto):
        if texto:
            self.texto_emitido.emit(str(texto))

    def flush(self):
        pass


class HiloEntrenamiento(QThread):
    progreso = pyqtSignal(int, str)
    consola = pyqtSignal(str)
    comparativa_lista = pyqtSignal(float, float, float, float)
    finalizado = pyqtSignal()

    def __init__(self, modelo_hmm, modelo_svm, ruta_corpus="../corpus"):
        super().__init__()
        self.modelo_hmm = modelo_hmm
        self.modelo_svm = modelo_svm
        self.ruta_corpus = ruta_corpus

    def run(self):
        stdout_original = sys.stdout
        redirector = RedireccionConsola()
        redirector.texto_emitido.connect(self.consola.emit)
        sys.stdout = redirector

        try:
            if not os.path.exists(self.ruta_corpus):
                print(f"[!] Error: No se encontró el directorio '{self.ruta_corpus}'.")
                self.progreso.emit(0, "Error: carpeta corpus no encontrada")
                return

            self.progreso.emit(10, "Entrenando modelo HMM (Bakis)...")
            datos_prueba = self.modelo_hmm.entrenar(ruta_corpus=self.ruta_corpus, test_size=0.2)

            self.progreso.emit(45, "Evaluando modelo HMM...")
            acc_hmm, wer_hmm = self.modelo_hmm.evaluar(datos_prueba)

            self.progreso.emit(60, "Entrenando modelo SVM (Kernel RBF)...")
            datos_prueba_svm = self.modelo_svm.entrenar(ruta_corpus=self.ruta_corpus, test_size=0.2)

            self.progreso.emit(85, "Evaluando modelo SVM...")
            acc_svm, wer_svm = self.modelo_svm.evaluar(datos_prueba_svm)

            if datos_prueba:
                self.comparativa_lista.emit(acc_hmm, wer_hmm, acc_svm, wer_svm)

            self.progreso.emit(100, "Análisis y entrenamiento completados")
        except Exception as e:
            print(f"\n[!] Error durante el entrenamiento: {e}")
            self.progreso.emit(0, "Error en el proceso")
        finally:
            sys.stdout = stdout_original
            self.finalizado.emit()


class HiloGrabacion(QThread):
    """Graba audio con ganancia configurable, anima el espectro y emite el tiempo restante."""
    espectro_actualizado = pyqtSignal(object)
    tiempo_restante = pyqtSignal(float)
    grabacion_terminada = pyqtSignal(str)
    error_grabacion = pyqtSignal(str)

    def __init__(self, fs=16000, duracion=2.0, ganancia=1.0, archivo_salida="temp_audio.wav", num_barras=32):
        super().__init__()
        self.fs = fs
        self.duracion = duracion
        self.ganancia = ganancia
        self.archivo_salida = archivo_salida
        self.num_barras = num_barras

    def run(self):
        try:
            margen_inicio = 0.15
            duracion_total = self.duracion + margen_inicio
            muestras_totales = int(duracion_total * self.fs)
            tam_bloque = 512

            buffer_audio = []
            muestras_capturadas = 0

            def callback(indata, frames, time_info, status):
                nonlocal muestras_capturadas
                # Aplicar el control de volumen/ganancia evitando saturación (clipping)
                bloque = np.clip(indata[:, 0].copy() * self.ganancia, -1.0, 1.0)
                buffer_audio.append(bloque)
                muestras_capturadas += len(bloque)

                # Calcular espectro FFT para las barras visuales
                ventana = bloque * np.hanning(len(bloque))
                fft_mag = np.abs(np.fft.rfft(ventana))
                limite = min(len(fft_mag), 160)
                bandas = np.array_split(fft_mag[1:limite], self.num_barras)
                niveles = np.array([np.mean(b) for b in bandas], dtype=np.float32)

                niveles = np.log1p(niveles * 12.0) / 3.8
                niveles = np.clip(niveles, 0.0, 1.0)
                self.espectro_actualizado.emit(niveles)

            with sd.InputStream(
                samplerate=self.fs,
                channels=1,
                dtype=np.float32,
                blocksize=tam_bloque,
                callback=callback
            ):
                while muestras_capturadas < muestras_totales:
                    restante = max(0.0, min(self.duracion, (muestras_totales - muestras_capturadas) / self.fs))
                    self.tiempo_restante.emit(restante)
                    self.msleep(30)

            self.tiempo_restante.emit(0.0)
            audio_completo = np.concatenate(buffer_audio)[:muestras_totales]
            recorte_inicio = int(margen_inicio * self.fs)
            audio_limpio = audio_completo[recorte_inicio:]

            write(self.archivo_salida, self.fs, audio_limpio)
            self.grabacion_terminada.emit(self.archivo_salida)

        except Exception as e:
            self.error_grabacion.emit(f"[!] Error al acceder al micrófono: {e}")


class ControladorReconocimiento:
    def __init__(self, modelo_hmm, modelo_svm, vista):
        self.modelo_hmm = modelo_hmm
        self.modelo_svm = modelo_svm
        self.vista = vista
        self.hilo_entrenamiento = None
        self.hilo_grabacion = None

        self.vista.solicitar_entrenamiento.connect(self.iniciar_entrenamiento)
        self.vista.solicitar_prediccion_archivo.connect(
            lambda ruta: self._procesar_y_predecir(ruta, origen="archivo")
        )
        self.vista.solicitar_grabacion_microfono.connect(self.iniciar_grabacion)

    def iniciar_entrenamiento(self):
        self.vista.limpiar_consola()
        self.vista.bloquear_entrenamiento(True)
        self.vista.actualizar_progreso(5, "Preparando entorno de entrenamiento...")

        self.hilo_entrenamiento = HiloEntrenamiento(self.modelo_hmm, self.modelo_svm)
        self.hilo_entrenamiento.progreso.connect(self.vista.actualizar_progreso)
        self.hilo_entrenamiento.consola.connect(self.vista.agregar_texto_consola)
        self.hilo_entrenamiento.comparativa_lista.connect(self.vista.mostrar_comparativa)
        self.hilo_entrenamiento.finalizado.connect(lambda: self.vista.bloquear_entrenamiento(False))
        self.hilo_entrenamiento.start()

    def iniciar_grabacion(self, ganancia=1.0):
        self.vista.bloquear_microfono(True)
        self.vista.mostrar_aviso("micro", "")

        self.hilo_grabacion = HiloGrabacion(fs=self.vista.fs, duracion=2.0, ganancia=ganancia)
        self.hilo_grabacion.espectro_actualizado.connect(self.vista.actualizar_espectro)
        self.hilo_grabacion.tiempo_restante.connect(self.vista.actualizar_tiempo_restante)
        self.hilo_grabacion.grabacion_terminada.connect(self._al_terminar_grabacion)
        self.hilo_grabacion.error_grabacion.connect(self._al_error_grabacion)
        self.hilo_grabacion.start()

    def _al_terminar_grabacion(self, ruta_temporal):
        self.vista.bloquear_microfono(False)
        self._procesar_y_predecir(ruta_temporal, origen="micro")

    def _al_error_grabacion(self, mensaje):
        self.vista.bloquear_microfono(False)
        self.vista.mostrar_aviso("micro", mensaje)

    def _procesar_y_predecir(self, ruta_audio, origen="archivo"):
        try:
            caracteristicas = self.modelo_hmm.preprocesador.procesar(ruta_audio)
            pred_hmm = self.modelo_hmm.predecir(caracteristicas)
            pred_svm = self.modelo_svm.predecir(caracteristicas)
            self.vista.mostrar_resultado(origen, pred_hmm, pred_svm)
        except Exception as e:
            self.vista.mostrar_aviso(origen, f"[!] Aviso al procesar el audio: {e}")