import os
import random
import numpy as np
import librosa
from scipy.signal import wiener, butter, sosfilt
import joblib
from hmmlearn import hmm
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

class PreprocesadorAudio:
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
        # Filtro pasa-altas a 80 Hz para eliminar ruido eléctrico, pop del micrófono y ventiladores
        self.sos_hp = butter(4, 80, btype='highpass', fs=self.sr, output='sos')

    def _suprimir_clics_borde(self, senal, max_iter=3):
        """
        Silencia pops del micrófono: ráfagas cortas (<=150 ms) que tocan el inicio o el
        final de la grabación y son al menos el doble de fuertes que cualquier otro sonido.
        Si no se eliminan, el VAD se queda con el clic y descarta la palabra.
        """
        senal = senal.copy()
        toque_borde = int(0.03 * self.sr)
        duracion_max_clic = int(0.15 * self.sr)
        margen = int(0.01 * self.sr)
        n_clics = 0

        for _ in range(max_iter):
            intervalos = librosa.effects.split(senal, top_db=18, frame_length=400, hop_length=160)
            if len(intervalos) == 0:
                break
            picos = [np.max(np.abs(senal[a:b])) for a, b in intervalos]
            k = int(np.argmax(picos))
            a, b = intervalos[k]
            otros_picos = [p for i, p in enumerate(picos) if i != k]

            es_corto = (b - a) <= duracion_max_clic
            toca_borde = a <= toque_borde or b >= len(senal) - toque_borde
            domina = not otros_picos or picos[k] >= 2 * max(otros_picos)

            if not (es_corto and toca_borde and domina):
                break
            senal[max(0, a - margen):b + margen] = 0.0
            n_clics += 1

        return senal, n_clics

    def procesar(self, ruta_archivo):
        # 1. Conversión a mono y 16 kHz
        senal, sr = librosa.load(ruta_archivo, sr=self.sr, mono=True)
        
        # 2. Eliminar offset DC y aplicar filtro pasa-altas (limpieza de bajas frecuencias)
        senal = senal - np.mean(senal)
        senal = sosfilt(self.sos_hp, senal)
        
        # 3. Detección de silencio real post-filtrado
        rms = librosa.feature.rms(y=senal, frame_length=400, hop_length=160)
        if np.max(rms) < 0.008:
            raise ValueError("Silencio detectado. Habla más cerca del micrófono.")

        # 4. Eliminar clics/pops del micrófono pegados a los extremos de la grabación
        senal, n_clics = self._suprimir_clics_borde(senal)
        if n_clics:
            rms = librosa.feature.rms(y=senal, frame_length=400, hop_length=160)
            if np.max(rms) < 0.008:
                raise ValueError("Solo se detectó un clic del micrófono, sin voz.")

        # 5. Normalización base y recorte inteligente de voz activa (VAD)
        senal = librosa.util.normalize(senal)
        intervalos = librosa.effects.split(senal, top_db=18, frame_length=400, hop_length=160)

        if len(intervalos) > 0:
            # Conservar segmentos con al menos 5% de la energía del segmento más fuerte
            # (la energía acumulada distingue la palabra de ruidos breves mejor que el pico)
            energias = [np.sum(senal[a:b] ** 2) for a, b in intervalos]
            umbral = 0.05 * max(energias)
            intervalos_validos = [
                inter for inter, e in zip(intervalos, energias) if e >= umbral
            ]
                
            # Dejar un margen de 40 ms en los extremos para no cortar consonantes como /s/ o /t/
            margen = int(0.04 * self.sr)
            inicio = max(0, intervalos_validos[0][0] - margen)
            fin = min(len(senal), intervalos_validos[-1][1] + margen)
            senal_recortada = senal[inicio:fin]
        else:
            senal_recortada = senal

        if len(senal_recortada) < int(self.sr * 0.15):
            senal_recortada = senal

        # 6. Filtro de Wiener estabilizado + Pre-énfasis (realza consonantes agudas)
        senal_filtrada = wiener(senal_recortada + 1e-8, mysize=29)
        senal_filtrada = np.nan_to_num(senal_filtrada)
        senal_pre = librosa.effects.preemphasis(senal_filtrada, coef=0.97)
        senal_normalizada = librosa.util.normalize(senal_pre)
        
        # 7. Extracción de MFCCs con ventaneo estándar de voz (25 ms ventana / 10 ms salto)
        mfcc = librosa.feature.mfcc(
            y=senal_normalizada, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            n_fft=512,
            win_length=400,
            hop_length=160,
            fmin=80,
            fmax=7600,
            lifter=22
        )
        
        if mfcc.shape[1] < 9:
            faltantes = 9 - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, faltantes)), mode='edge')
            
        # 8. Normalización Cepstral de la Media (CMN): hace al sistema invariante al micrófono
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
            
        # 9. Derivadas dinámicas (Velocidad y Aceleración del tracto vocal)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        caracteristicas = np.vstack([mfcc, delta, delta2])
        return caracteristicas.T


class ModeloReconocimiento:
    def __init__(self, ruta_modelo="../modelos_entrenados/modelos_hmm.pkl"):
        self.ruta_modelo = ruta_modelo
        self.modelos = {} 
        self.preprocesador = PreprocesadorAudio()
        
        if os.path.exists(self.ruta_modelo):
            self.modelos = joblib.load(self.ruta_modelo)

    def _inicializar_hmm_izquierda_derecha(self, train_data, n_estados=5):
        """Inicializa el HMM alineando cronológicamente los segmentos de voz (Bakis)."""
        n_features = train_data[0].shape[1]
        
        # Dividir cada audio en 'n_estados' segmentos cronológicos iguales
        segmentos_por_estado = [[] for _ in range(n_estados)]
        for seq in train_data:
            partes = np.array_split(seq, n_estados, axis=0)
            for i in range(n_estados):
                segmentos_por_estado[i].append(partes[i])
                
        means = np.array([np.mean(np.vstack(seg), axis=0) for seg in segmentos_por_estado])
        covars = np.array([np.var(np.vstack(seg), axis=0) + 1e-2 for seg in segmentos_por_estado])
        
        startprob = np.zeros(n_estados)
        startprob[0] = 1.0
        
        transmat = np.zeros((n_estados, n_estados))
        for i in range(n_estados):
            if i == n_estados - 1:
                transmat[i, i] = 1.0
            else:
                transmat[i, i] = 0.6
                transmat[i, i + 1] = 0.4
                
        # init_params="" impide que K-Means desordene temporalmente los estados
        modelo_hmm = hmm.GaussianHMM(
            n_components=n_estados,
            covariance_type="diag",
            n_iter=60,
            tol=1e-2,
            min_covar=1e-2,
            init_params="",
            params="stmc"
        )
        modelo_hmm.startprob_ = startprob
        modelo_hmm.transmat_ = transmat
        modelo_hmm.means_ = means
        modelo_hmm.covars_ = covars
        return modelo_hmm

    def entrenar(self, ruta_corpus="../corpus", test_size=0.2, max_audios_por_clase=1000):
        print("\n[*] Procesando características y alineando estados temporales HMM...")
        self.modelos = {}
        datos_prueba = []
        
        for etiqueta in sorted(os.listdir(ruta_corpus)):
            ruta_clase = os.path.join(ruta_corpus, etiqueta)
            if not os.path.isdir(ruta_clase):
                continue
                
            archivos = [f for f in os.listdir(ruta_clase) if f.endswith('.wav')]
            if not archivos:
                continue
                
            # Limitar muestra aleatoria si el dataset es masivo para agilizar el entrenamiento
            random.seed(42)
            if len(archivos) > max_audios_por_clase:
                archivos = random.sample(archivos, max_audios_por_clase)
                
            X_clase = []
            descartados = []
            for archivo in archivos:
                ruta_archivo = os.path.join(ruta_clase, archivo)
                try:
                    caracteristicas = self.preprocesador.procesar(ruta_archivo)
                    if caracteristicas.shape[0] >= 9:
                        X_clase.append(caracteristicas)
                    else:
                        descartados.append((archivo, "muy pocos frames"))
                except Exception as e:
                    descartados.append((archivo, str(e)))

            if descartados:
                print(f" [!] '{etiqueta}': {len(descartados)}/{len(archivos)} audios descartados")
                for archivo, motivo in descartados:
                    print(f"     - {archivo}: {motivo}")

            if len(X_clase) < 3:
                print(f" [!] Muy pocos audios válidos en '{etiqueta}'. Omitiendo partición.")
                train_data, test_data = X_clase, []
            else:
                train_data, test_data = train_test_split(X_clase, test_size=test_size, random_state=42)
            
            if test_data:
                datos_prueba.append((etiqueta, test_data))
                
            if train_data:
                X_concat = np.concatenate(train_data)
                longitudes = [x.shape[0] for x in train_data]
                
                n_estados = 5
                modelo_hmm = self._inicializar_hmm_izquierda_derecha(train_data, n_estados=n_estados)
                modelo_hmm.fit(X_concat, longitudes)
                
                self.modelos[etiqueta] = modelo_hmm
                print(f" >> HMM entrenado: '{etiqueta}' | Audios Train: {len(train_data)} | Matriz: {X_concat.shape}")
        
        os.makedirs(os.path.dirname(self.ruta_modelo), exist_ok=True)
        joblib.dump(self.modelos, self.ruta_modelo)
        print(f"[*] Modelos guardados exitosamente en: {self.ruta_modelo}")
        
        return datos_prueba

    def evaluar(self, datos_prueba):
        if not datos_prueba:
            print("\n[!] No hay suficientes datos para evaluar.")
            return 0.0, 100.0

        print("\n" + "="*55)
        print(" EVALUACIÓN DEL MODELO (TEST SET) ".center(55))
        print("="*55)
        
        aciertos_totales = 0
        total_audios = 0
        
        for etiqueta_real, audios_test in datos_prueba:
            aciertos_clase = 0
            for caracteristicas in audios_test:
                prediccion = self.predecir(caracteristicas)
                if prediccion == etiqueta_real:
                    aciertos_clase += 1
                    aciertos_totales += 1
                total_audios += 1
                
            acc_clase = (aciertos_clase / len(audios_test)) * 100 if audios_test else 0
            print(f" -> Palabra '{etiqueta_real:<10}': {aciertos_clase}/{len(audios_test)} aciertos ({acc_clase:.2f}%)")
                
        accuracy = (aciertos_totales / total_audios) * 100 if total_audios > 0 else 0
        wer = 100.0 - accuracy
        
        print("-" * 55)
        print(f"[*] Total de audios evaluados : {total_audios}")
        print(f"[*] Tasa de Aciertos (Accuracy): {accuracy:.2f}%")
        print(f"[*] Tasa de Error (WER)        : {wer:.2f}%")
        print("="*55 + "\n")

        return accuracy, wer

    def predecir(self, caracteristicas):
        if not self.modelos:
            return "Error: Modelos no entrenados."
            
        puntajes = {}
        n_frames = len(caracteristicas)
        
        for etiqueta, modelo_hmm in self.modelos.items():
            try:
                # Normalizar el log-likelihood por la cantidad de frames
                puntajes[etiqueta] = modelo_hmm.score(caracteristicas) / n_frames
            except Exception:
                puntajes[etiqueta] = float('-inf')
                
        palabra_predicha = max(puntajes, key=puntajes.get)
        return palabra_predicha