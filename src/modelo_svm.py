import os
import random
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from modelo import PreprocesadorAudio

class ModeloSVM:
    def __init__(self, ruta_modelo="../modelos_entrenados/modelo_svm.pkl", n_segmentos=5):
        self.ruta_modelo = ruta_modelo
        self.n_segmentos = n_segmentos
        self.preprocesador = PreprocesadorAudio()
        self.pipeline = None
        
        if os.path.exists(self.ruta_modelo):
            self.pipeline = joblib.load(self.ruta_modelo)

    def _vectorizar_secuencia(self, caracteristicas):
        """
        Convierte una matriz temporal (frames x 39) en un vector 1D de longitud fija (390,)
        dividiendo el audio en segmentos cronológicos para preservar el orden fonético.
        """
        segmentos = np.array_split(caracteristicas, self.n_segmentos, axis=0)
        Estadisticos = []
        for seg in segmentos:
            media = np.mean(seg, axis=0)
            std = np.std(seg, axis=0)
            Estadisticos.extend([media, std])
            
        return np.concatenate(Estadisticos)

    def entrenar(self, ruta_corpus="../corpus", test_size=0.2, max_audios_por_clase=1000):
        print("\n[*] Procesando características y entrenando clasificador SVM (Kernel RBF)...")
        X_train, y_train = [], []
        datos_prueba = []
        
        for etiqueta in sorted(os.listdir(ruta_corpus)):
            ruta_clase = os.path.join(ruta_corpus, etiqueta)
            if not os.path.isdir(ruta_clase):
                continue
                
            archivos = [f for f in os.listdir(ruta_clase) if f.endswith('.wav')]
            if not archivos:
                continue
                
            # Usar la misma semilla (42) que el HMM para garantizar la misma partición de audios
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
                
            for seq in train_data:
                X_train.append(self._vectorizar_secuencia(seq))
                y_train.append(etiqueta)
                
        if len(set(y_train)) < 2:
            print("[!] Error SVM: Se necesitan al menos 2 clases (carpetas con audios) para entrenar un SVM.")
            return []
            
        X_train = np.array(X_train)
        
        # Pipeline con estandarización Z-score y SVM con Kernel RBF
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42))
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        os.makedirs(os.path.dirname(self.ruta_modelo), exist_ok=True)
        joblib.dump(self.pipeline, self.ruta_modelo)
        print(f" >> SVM entrenado con éxito | Muestras Train: {len(X_train)} | Dimensión Vector: {X_train.shape[1]}")
        print(f"[*] Modelo SVM guardado exitosamente en: {self.ruta_modelo}")
        
        return datos_prueba

    def evaluar(self, datos_prueba):
        if not datos_prueba or self.pipeline is None:
            print("\n[!] No hay datos de prueba o el modelo SVM no está entrenado.")
            return 0.0, 100.0
            
        print("\n" + "="*55)
        print(" EVALUACIÓN DEL MODELO SVM (TEST SET) ".center(55))
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
        if self.pipeline is None:
            return "Error: Modelo SVM no entrenado."
            
        vector = self._vectorizar_secuencia(caracteristicas).reshape(1, -1)
        prediccion = self.pipeline.predict(vector)[0]
        return prediccion