import os
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# Las 15 palabras definidas para el corpus organizadas por categoría
CATEGORIAS_VOCABULARIO = {
    "Sustantivos cotidianos": ["casa", "agua", "fuego", "libro", "reloj"],
    "Comandos de control": ["encender", "apagar", "abrir", "cerrar", "guardar"],
    "Navegación e interfaz": ["arriba", "abajo", "inicio", "salir", "ayuda"]
}

CANTIDAD_POR_PALABRA = 30

class RecolectorCorpus:
    def __init__(self, ruta_corpus="../corpus", fs=16000, duracion=2.0):
        self.ruta_corpus = ruta_corpus
        self.fs = fs              # Estándar del proyecto: 16 kHz
        self.duracion = duracion  # 2.0 segundos por muestra
        os.makedirs(self.ruta_corpus, exist_ok=True)

    def obtener_siguiente_indice(self, ruta_palabra):
        """Busca el último número de archivo guardado para no sobrescribir muestras."""
        archivos = [f for f in os.listdir(ruta_palabra) if f.endswith('.wav')]
        if not archivos:
            return 1
        indices = []
        for arch in archivos:
            try:
                num = int(arch.replace("muestra_", "").replace(".wav", ""))
                indices.append(num)
            except ValueError:
                continue
        return max(indices) + 1 if indices else 1

    def grabar_muestra(self):
        """Captura audio mono a 16 kHz eliminando el transitorio inicial del hardware."""
        margen_inicio = 0.15
        duracion_total = self.duracion + margen_inicio
        
        grabacion = sd.rec(
            int(duracion_total * self.fs),
            samplerate=self.fs,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        # Recortar los primeros 0.15s para evitar el 'pop' al abrir el micrófono
        muestras_recorte = int(margen_inicio * self.fs)
        grabacion_limpia = grabacion[muestras_recorte:]
        
        # Calcular energía RMS para verificar que realmente se habló
        rms = np.sqrt(np.mean(grabacion_limpia**2))
        
        # Convertir a PCM de 16 bits
        audio_int16 = np.int16(np.clip(grabacion_limpia, -1.0, 1.0) * 32767)
        return audio_int16, rms

    def grabar_lote_palabra(self, palabra, cantidad=CANTIDAD_POR_PALABRA):
        palabra = palabra.lower().strip()
        ruta_palabra = os.path.join(self.ruta_corpus, palabra)
        os.makedirs(ruta_palabra, exist_ok=True)
        
        indice_actual = self.obtener_siguiente_indice(ruta_palabra)
        completadas = 0
        
        print("\n" + "="*55)
        print(f" GRABANDO PALABRA: '{palabra.upper()}' ({cantidad} muestras) ".center(55))
        print("="*55)
        print("[*] Instrucciones: Habla claro cuando aparezca '>>> ¡HABLA AHORA! <<<'")
        print("[*] Presiona Ctrl+C en cualquier momento para pausar o volver al menú.\n")
        
        try:
            while completadas < cantidad:
                input(f"Presiona [ENTER] para grabar muestra {completadas + 1}/{cantidad} (archivo #{indice_actual:03d})...")
                
                for i in range(2, 0, -1):
                    print(f"  Grabando en {i}...", end="\r")
                    time.sleep(0.4)
                    
                print(f"  >>> ¡HABLA AHORA! ({self.duracion}s) <<<      ")
                audio, rms = self.grabar_muestra()
                
                # Validar que no sea silencio puro
                if rms < 0.005:
                    print("  [!] Audio demasiado bajo o silencio detectado. Repitiendo toma...\n")
                    continue
                    
                nombre_archivo = f"muestra_{indice_actual:03d}.wav"
                ruta_completa = os.path.join(ruta_palabra, nombre_archivo)
                write(ruta_completa, self.fs, audio)
                
                print(f"  [OK] Guardado: {ruta_completa} (Energía RMS: {rms:.4f})\n")
                indice_actual += 1
                completadas += 1
                
            print(f"[*] ¡Lote de {cantidad} muestras finalizado para '{palabra}'!")
            
        except KeyboardInterrupt:
            print("\n\n[!] Grabación de lote pausada por el usuario. Regresando...")

    def mostrar_estado_corpus(self):
        print("\n" + "="*60)
        print(" ESTADO ACTUAL DEL CORPUS (../corpus) ".center(60))
        print("="*60)
        
        total_audios = 0
        total_clases_listas = 0
        
        print(f" {'Categoría / Palabra':<25} | {'Audios (.wav)':<15} | {'Meta (30 c/u)'}")
        print("-" * 60)
        
        for categoria, palabras in CATEGORIAS_VOCABULARIO.items():
            print(f" [{categoria}]")
            for palabra in palabras:
                ruta_c = os.path.join(self.ruta_corpus, palabra)
                if os.path.exists(ruta_c):
                    n_wavs = len([f for f in os.listdir(ruta_c) if f.endswith('.wav')])
                else:
                    n_wavs = 0
                total_audios += n_wavs
                if n_wavs >= CANTIDAD_POR_PALABRA:
                    estado = "Completado"
                    total_clases_listas += 1
                else:
                    estado = f"Faltan {CANTIDAD_POR_PALABRA - n_wavs}"
                print(f"   - {palabra:<21} | {n_wavs:<15} | {estado}")
                
        print("-" * 60)
        print(f" Palabras completadas : {total_clases_listas}/15")
        print(f" Total de audios (.wav): {total_audios}")
        print("="*60)

    def iniciar_recorrido_guiado(self):
        total_palabras = sum(len(p) for p in CATEGORIAS_VOCABULARIO.values())
        contador_global = 1
        
        for categoria, palabras in CATEGORIAS_VOCABULARIO.items():
            print("\n" + "#"*60)
            print(f" CATEGORÍA: {categoria.upper()} ".center(60))
            print("#"*60)
            
            for palabra in palabras:
                print(f"\n---> Palabra {contador_global}/{total_palabras}: '{palabra.upper()}' ({CANTIDAD_POR_PALABRA} muestras) <---")
                accion = input(f"Presiona [ENTER] para grabar '{palabra}', 's' para saltar o 'q' para salir al menú: ").strip().lower()
                
                if accion == 'q':
                    return
                elif accion == 's':
                    contador_global += 1
                    continue
                    
                self.grabar_lote_palabra(palabra, CANTIDAD_POR_PALABRA)
                contador_global += 1

    def menu(self):
        while True:
            print("\n" + "="*55)
            print(" RECOLECTOR DE CORPUS DE VOZ (16 kHz - MONO) ".center(55))
            print("="*55)
            print("1. Ver estado actual del corpus")
            print("2. Iniciar recorrido guiado (15 palabras x 30 muestras)")
            print("3. Salir")
            
            opcion = input("\nSelecciona una opción (1-3): ").strip()
            
            if opcion == "1":
                self.mostrar_estado_corpus()
            elif opcion == "2":
                self.iniciar_recorrido_guiado()
            elif opcion == "3":
                print("\n[*] Cerrando recolector de corpus...")
                break
            else:
                print("\n[!] Opción no válida.")

if __name__ == "__main__":
    recolector = RecolectorCorpus()
    recolector.menu()