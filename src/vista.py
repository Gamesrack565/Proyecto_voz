import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QStackedWidget, QProgressBar, QPlainTextEdit, QFileDialog,
    QFrame, QGridLayout, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QTextCursor, QFont

# Edita aquí los nombres de los integrantes para la barra superior
DESARROLLADO_POR = "Desarrollado por: Angel Higuera, Edwin Perez"
ANCHO_SIDEBAR = 255


class EspectroWidget(QWidget):
    """Visualizador de espectro en tiempo real en tono claro."""
    def __init__(self, num_barras=32, parent=None):
        super().__init__(parent)
        self.num_barras = num_barras
        self.niveles = np.zeros(num_barras, dtype=np.float32)
        self.setMinimumHeight(175)

    def actualizar_niveles(self, nuevos_niveles):
        nuevos = np.asarray(nuevos_niveles, dtype=np.float32)
        if len(nuevos) == self.num_barras:
            self.niveles = 0.65 * nuevos + 0.35 * self.niveles
            self.update()

    def reiniciar(self):
        self.niveles = np.zeros(self.num_barras, dtype=np.float32)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        painter.setPen(QColor("#ddd6fe"))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 12, 12)

        painter.setPen(QColor("#ede9fe"))
        painter.drawLine(18, h // 2, w - 18, h // 2)

        margen_x = 22
        espacio_disponible = w - 2 * margen_x
        paso = espacio_disponible / self.num_barras
        ancho_barra = max(4.0, paso * 0.62)

        gradiente = QLinearGradient(0, 12, 0, h - 12)
        gradiente.setColorAt(0.0, QColor("#a855f7"))
        gradiente.setColorAt(0.5, QColor("#7c3aed"))
        gradiente.setColorAt(1.0, QColor("#6366f1"))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradiente)

        for i, nivel in enumerate(self.niveles):
            altura = max(6.0, float(nivel) * (h - 32))
            x = margen_x + i * paso + (paso - ancho_barra) / 2
            y = (h - altura) / 2
            painter.drawRoundedRect(QRectF(x, y, ancho_barra, altura), 4, 4)


class VistaInteractiva(QMainWindow):
    solicitar_entrenamiento = pyqtSignal()
    solicitar_prediccion_archivo = pyqtSignal(str)
    solicitar_grabacion_microfono = pyqtSignal(float)  # Emite el factor de ganancia

    def __init__(self, frecuencia_muestreo=16000):
        super().__init__()
        self.fs = frecuencia_muestreo
        self.ruta_archivo_actual = None

        self.setWindowTitle("Proyecto de Reconocimiento de Voz — HMM vs SVM")
        self.resize(1060, 700)
        self._construir_ui()
        self._aplicar_estilos()

    def _construir_ui(self):
        # Marco perimetral morado
        marco_morado = QWidget()
        marco_morado.setObjectName("MarcoMorado")
        layout_marco = QVBoxLayout(marco_morado)
        layout_marco.setContentsMargins(6, 6, 6, 6)

        contenedor_interior = QWidget()
        contenedor_interior.setObjectName("ContenedorInterior")
        layout_vertical_raiz = QVBoxLayout(contenedor_interior)
        layout_vertical_raiz.setContentsMargins(0, 0, 0, 0)
        layout_vertical_raiz.setSpacing(0)

        # ==================== 1. BARRA SUPERIOR ====================
        barra_superior = QFrame()
        barra_superior.setObjectName("BarraSuperior")
        barra_superior.setFixedHeight(48)
        layout_top = QHBoxLayout(barra_superior)
        layout_top.setContentsMargins(0, 0, 22, 0)
        layout_top.setSpacing(0)

        # Recuadro ASR: inicia en el borde izquierdo y termina en el borde derecho del sidebar
        lbl_badge = QLabel("ASR · HMM (Bakis) vs SVM (RBF)")
        lbl_badge.setObjectName("BadgeSuperior")
        lbl_badge.setFixedWidth(ANCHO_SIDEBAR)
        lbl_badge.setFixedHeight(48)
        lbl_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_autores = QLabel(DESARROLLADO_POR)
        lbl_autores.setObjectName("TextoAutores")
        lbl_autores.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout_top.addWidget(lbl_badge)
        layout_top.addStretch()
        layout_top.addWidget(lbl_autores)

        # ==================== CUERPO (SIDEBAR + CONTENIDO) ====================
        cuerpo = QWidget()
        layout_cuerpo = QHBoxLayout(cuerpo)
        layout_cuerpo.setContentsMargins(0, 0, 0, 0)
        layout_cuerpo.setSpacing(0)

        # ==================== 2. BARRA LATERAL IZQUIERDA ====================
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(ANCHO_SIDEBAR)
        layout_sidebar = QVBoxLayout(sidebar)
        layout_sidebar.setContentsMargins(18, 24, 18, 20)
        layout_sidebar.setSpacing(10)

        lbl_titulo_proyecto = QLabel("Proyecto de\nReconocimiento\nde Voz")
        lbl_titulo_proyecto.setObjectName("LogoTitulo")
        lbl_titulo_proyecto.setWordWrap(True)

        separador = QFrame()
        separador.setObjectName("SeparadorSidebar")
        separador.setFixedHeight(1)

        layout_sidebar.addWidget(lbl_titulo_proyecto)
        layout_sidebar.addSpacing(8)
        layout_sidebar.addWidget(separador)
        layout_sidebar.addSpacing(12)

        self.btn_nav_entrenar = self._crear_boton_nav("  Entrenar y Evaluar", 0)
        self.btn_nav_archivo = self._crear_boton_nav("  Predecir Archivo (.wav)", 1)
        self.btn_nav_micro = self._crear_boton_nav("  Grabar con Micrófono", 2)
        self.botones_nav = [self.btn_nav_entrenar, self.btn_nav_archivo, self.btn_nav_micro]

        for btn in self.botones_nav:
            layout_sidebar.addWidget(btn)

        layout_sidebar.addStretch()

        lbl_estado_pie = QLabel("16 kHz · Mono · Procesamiento Local")
        lbl_estado_pie.setObjectName("PieSidebar")
        layout_sidebar.addWidget(lbl_estado_pie)

        # ==================== ÁREA DE CONTENIDO ====================
        self.paginas = QStackedWidget()
        self.paginas.setObjectName("AreaContenido")

        self.paginas.addWidget(self._crear_pagina_entrenamiento())
        self.paginas.addWidget(self._crear_pagina_archivo())
        self.paginas.addWidget(self._crear_pagina_microfono())

        layout_cuerpo.addWidget(sidebar)
        layout_cuerpo.addWidget(self.paginas, 1)

        layout_vertical_raiz.addWidget(barra_superior)
        layout_vertical_raiz.addWidget(cuerpo, 1)

        layout_marco.addWidget(contenedor_interior)
        self.setCentralWidget(marco_morado)

        self._cambiar_pagina(0)

    def _crear_boton_nav(self, texto, indice):
        btn = QPushButton(texto)
        btn.setObjectName("BotonNav")
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self._cambiar_pagina(indice))
        return btn

    def _cambiar_pagina(self, indice):
        self.paginas.setCurrentIndex(indice)
        for i, btn in enumerate(self.botones_nav):
            btn.setChecked(i == indice)

    # ---------- PÁGINA 1: ENTRENAR Y EVALUAR ----------
    def _crear_pagina_entrenamiento(self):
        pagina = QWidget()
        layout = QVBoxLayout(pagina)
        layout.setContentsMargins(32, 26, 32, 26)
        layout.setSpacing(15)

        titulo = QLabel("Entrenamiento y Evaluación Comparativa")
        titulo.setObjectName("TituloPagina")
        subtitulo = QLabel("Entrena HMM y SVM con el corpus local (80% Train / 20% Test) y compara Accuracy y WER.")
        subtitulo.setObjectName("SubtituloPagina")

        fila_accion = QHBoxLayout()
        self.btn_iniciar_entrenamiento = QPushButton("Iniciar Análisis y Entrenamiento")
        self.btn_iniciar_entrenamiento.setObjectName("BotonPrimario")
        self.btn_iniciar_entrenamiento.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_iniciar_entrenamiento.clicked.connect(self.solicitar_entrenamiento.emit)

        self.lbl_estado_entrenamiento = QLabel("Listo para entrenar")
        self.lbl_estado_entrenamiento.setObjectName("EtiquetaEstado")

        fila_accion.addWidget(self.btn_iniciar_entrenamiento)
        fila_accion.addSpacing(12)
        fila_accion.addWidget(self.lbl_estado_entrenamiento)
        fila_accion.addStretch()

        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, 100)
        self.barra_progreso.setValue(0)
        self.barra_progreso.setTextVisible(True)

        grid_metricas = QGridLayout()
        grid_metricas.setSpacing(14)

        self.lbl_metrica_hmm = QLabel("—")
        self.lbl_metrica_svm = QLabel("—")
        self.lbl_ganador = QLabel("Pendiente")

        grid_metricas.addWidget(self._crear_tarjeta("HMM (Bakis)", self.lbl_metrica_hmm), 0, 0)
        grid_metricas.addWidget(self._crear_tarjeta("SVM (RBF)", self.lbl_metrica_svm), 0, 1)
        grid_metricas.addWidget(self._crear_tarjeta("Mejor Modelo", self.lbl_ganador), 0, 2)

        lbl_consola = QLabel("Salida de Consola")
        lbl_consola.setObjectName("EtiquetaSeccion")

        self.consola = QPlainTextEdit()
        self.consola.setObjectName("ConsolaSalida")
        self.consola.setReadOnly(True)
        self.consola.setFont(QFont("Monospace", 9))

        layout.addWidget(titulo)
        layout.addWidget(subtitulo)
        layout.addLayout(fila_accion)
        layout.addWidget(self.barra_progreso)
        layout.addLayout(grid_metricas)
        layout.addWidget(lbl_consola)
        layout.addWidget(self.consola, 1)
        return pagina

    # ---------- PÁGINA 2: PREDECIR DESDE ARCHIVO ----------
    def _crear_pagina_archivo(self):
        pagina = QWidget()
        layout = QVBoxLayout(pagina)
        layout.setContentsMargins(32, 26, 32, 26)
        layout.setSpacing(18)

        titulo = QLabel("Predecir desde Archivo de Audio (.wav)")
        titulo.setObjectName("TituloPagina")
        subtitulo = QLabel("Selecciona un archivo de audio local para clasificarlo con ambos modelos.")
        subtitulo.setObjectName("SubtituloPagina")

        tarjeta_selector = QFrame()
        tarjeta_selector.setObjectName("Tarjeta")
        lay_sel = QHBoxLayout(tarjeta_selector)
        lay_sel.setContentsMargins(18, 16, 18, 16)
        lay_sel.setSpacing(10)

        self.lbl_ruta_archivo = QLabel("Ningún archivo seleccionado")
        self.lbl_ruta_archivo.setObjectName("RutaArchivo")

        btn_examinar = QPushButton("Seleccionar .wav")
        btn_examinar.setObjectName("BotonSecundario")
        btn_examinar.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_examinar.clicked.connect(self._abrir_dialogo_archivo)

        self.btn_predecir_archivo = QPushButton("Analizar Audio")
        self.btn_predecir_archivo.setObjectName("BotonPrimario")
        self.btn_predecir_archivo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_predecir_archivo.setEnabled(False)
        self.btn_predecir_archivo.clicked.connect(self._emitir_prediccion_archivo)

        lay_sel.addWidget(self.lbl_ruta_archivo, 1)
        lay_sel.addWidget(btn_examinar)
        lay_sel.addWidget(self.btn_predecir_archivo)

        grid_res = QGridLayout()
        grid_res.setSpacing(16)
        self.lbl_res_archivo_hmm = QLabel("—")
        self.lbl_res_archivo_svm = QLabel("—")
        grid_res.addWidget(self._crear_tarjeta("Predicción HMM", self.lbl_res_archivo_hmm, grande=True), 0, 0)
        grid_res.addWidget(self._crear_tarjeta("Predicción SVM", self.lbl_res_archivo_svm, grande=True), 0, 1)

        self.lbl_aviso_archivo = QLabel("")
        self.lbl_aviso_archivo.setObjectName("EtiquetaAviso")

        layout.addWidget(titulo)
        layout.addWidget(subtitulo)
        layout.addWidget(tarjeta_selector)
        layout.addLayout(grid_res)
        layout.addWidget(self.lbl_aviso_archivo)
        layout.addStretch()
        return pagina

    # ---------- PÁGINA 3: GRABAR DESDE MICRÓFONO ----------
    def _crear_pagina_microfono(self):
        pagina = QWidget()
        layout = QVBoxLayout(pagina)
        layout.setContentsMargins(32, 26, 32, 26)
        layout.setSpacing(16)

        # Cabecera con título a la izquierda y contador regresivo pegado a la derecha
        fila_encabezado = QHBoxLayout()
        fila_encabezado.setSpacing(12)

        col_titulos = QVBoxLayout()
        col_titulos.setSpacing(4)
        titulo = QLabel("Reconocimiento en Vivo por Micrófono")
        titulo.setObjectName("TituloPagina")
        subtitulo = QLabel("Presiona grabar y pronuncia una palabra del vocabulario durante 2 segundos.")
        subtitulo.setObjectName("SubtituloPagina")
        col_titulos.addWidget(titulo)
        col_titulos.addWidget(subtitulo)

        # Cuadro contador de tiempo restante (oculto hasta que se presiona grabar)
        self.cuadro_contador = QFrame()
        self.cuadro_contador.setObjectName("CuadroContador")
        lay_contador = QVBoxLayout(self.cuadro_contador)
        lay_contador.setContentsMargins(16, 8, 16, 8)
        lay_contador.setSpacing(2)

        lbl_tit_contador = QLabel("CAPTURANDO VOZ")
        lbl_tit_contador.setObjectName("TituloContador")
        lbl_tit_contador.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_valor_contador = QLabel("2.0 s")
        self.lbl_valor_contador.setObjectName("ValorContador")
        self.lbl_valor_contador.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay_contador.addWidget(lbl_tit_contador)
        lay_contador.addWidget(self.lbl_valor_contador)
        self.cuadro_contador.setVisible(False)

        fila_encabezado.addLayout(col_titulos)
        fila_encabezado.addStretch()
        fila_encabezado.addWidget(self.cuadro_contador, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Espectro reactivo
        self.espectro = EspectroWidget(num_barras=32)

        # Control de volumen / ganancia del micrófono
        tarjeta_volumen = QFrame()
        tarjeta_volumen.setObjectName("Tarjeta")
        lay_vol = QHBoxLayout(tarjeta_volumen)
        lay_vol.setContentsMargins(18, 12, 18, 12)
        lay_vol.setSpacing(14)

        lbl_vol_titulo = QLabel("Volumen del micrófono:")
        lbl_vol_titulo.setObjectName("EtiquetaVolumen")

        self.slider_volumen = QSlider(Qt.Orientation.Horizontal)
        self.slider_volumen.setRange(10, 250)  # 10% a 250% de ganancia
        self.slider_volumen.setValue(100)
        self.slider_volumen.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider_volumen.valueChanged.connect(self._actualizar_etiqueta_volumen)

        self.lbl_valor_volumen = QLabel("100%")
        self.lbl_valor_volumen.setObjectName("ValorVolumen")
        self.lbl_valor_volumen.setFixedWidth(48)
        self.lbl_valor_volumen.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        lay_vol.addWidget(lbl_vol_titulo)
        lay_vol.addWidget(self.slider_volumen, 1)
        lay_vol.addWidget(self.lbl_valor_volumen)

        # Botón grabar y estado
        fila_btn = QHBoxLayout()
        self.btn_grabar = QPushButton("Grabar desde Micrófono (2.0 s)")
        self.btn_grabar.setObjectName("BotonPrimario")
        self.btn_grabar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_grabar.clicked.connect(self._emitir_grabacion)

        self.lbl_estado_micro = QLabel("Micrófono en espera")
        self.lbl_estado_micro.setObjectName("EtiquetaEstado")

        fila_btn.addWidget(self.btn_grabar)
        fila_btn.addSpacing(12)
        fila_btn.addWidget(self.lbl_estado_micro)
        fila_btn.addStretch()

        grid_res = QGridLayout()
        grid_res.setSpacing(16)
        self.lbl_res_micro_hmm = QLabel("—")
        self.lbl_res_micro_svm = QLabel("—")
        grid_res.addWidget(self._crear_tarjeta("Predicción HMM", self.lbl_res_micro_hmm, grande=True), 0, 0)
        grid_res.addWidget(self._crear_tarjeta("Predicción SVM", self.lbl_res_micro_svm, grande=True), 0, 1)

        self.lbl_aviso_micro = QLabel("")
        self.lbl_aviso_micro.setObjectName("EtiquetaAviso")

        layout.addLayout(fila_encabezado)
        layout.addWidget(self.espectro)
        layout.addWidget(tarjeta_volumen)
        layout.addLayout(fila_btn)
        layout.addLayout(grid_res)
        layout.addWidget(self.lbl_aviso_micro)
        layout.addStretch()
        return pagina

    def _crear_tarjeta(self, titulo, label_valor, grande=False):
        tarjeta = QFrame()
        tarjeta.setObjectName("Tarjeta")
        lay = QVBoxLayout(tarjeta)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(6)

        lbl_tit = QLabel(titulo)
        lbl_tit.setObjectName("TituloTarjeta")
        label_valor.setObjectName("ValorTarjetaGrande" if grande else "ValorTarjeta")
        label_valor.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        lay.addWidget(lbl_tit)
        lay.addWidget(label_valor)
        return tarjeta

    # ==================== MÉTODOS PÚBLICOS PARA EL CONTROLADOR ====================
    def _actualizar_etiqueta_volumen(self, valor):
        self.lbl_valor_volumen.setText(f"{valor}%")

    def _emitir_grabacion(self):
        ganancia = self.slider_volumen.value() / 100.0
        self.solicitar_grabacion_microfono.emit(ganancia)

    def _abrir_dialogo_archivo(self):
        dir_corpus = os.path.abspath("../corpus")
        dir_inicial = dir_corpus if os.path.isdir(dir_corpus) else os.getcwd()

        dialogo = QFileDialog(self, "Seleccionar audio WAV", dir_inicial, "Archivos de audio (*.wav)")
        dialogo.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialogo.setStyleSheet("""
            QFileDialog, QDialog {
                background-color: #ffffff;
                color: #1e1b2e;
            }
            QLabel, QLineEdit, QComboBox, QTreeView, QListView, QHeaderView::section {
                color: #1e1b2e;
                background-color: #ffffff;
            }
            QTreeView::item:selected, QListView::item:selected {
                background-color: #ede9fe;
                color: #5b21b6;
            }
            QPushButton {
                background-color: #f3effc;
                color: #1e1b2e;
                border: 1px solid #d8b4fe;
                border-radius: 6px;
                padding: 5px 14px;
            }
            QPushButton:hover {
                background-color: #e9d5ff;
            }
        """)

        if dialogo.exec():
            archivos = dialogo.selectedFiles()
            if archivos:
                ruta = archivos[0]
                self.ruta_archivo_actual = ruta
                self.lbl_ruta_archivo.setText(os.path.basename(ruta))
                self.lbl_ruta_archivo.setToolTip(ruta)
                self.btn_predecir_archivo.setEnabled(True)
                self.lbl_aviso_archivo.setText("")

    def _emitir_prediccion_archivo(self):
        if self.ruta_archivo_actual:
            self.solicitar_prediccion_archivo.emit(self.ruta_archivo_actual)

    def limpiar_consola(self):
        self.consola.clear()

    def agregar_texto_consola(self, texto):
        self.consola.moveCursor(QTextCursor.MoveOperation.End)
        self.consola.insertPlainText(texto)
        self.consola.moveCursor(QTextCursor.MoveOperation.End)

    def actualizar_progreso(self, valor, mensaje=None):
        self.barra_progreso.setValue(valor)
        if mensaje:
            self.lbl_estado_entrenamiento.setText(mensaje)

    def bloquear_entrenamiento(self, bloqueado):
        self.btn_iniciar_entrenamiento.setEnabled(not bloqueado)
        self.btn_iniciar_entrenamiento.setText(
            "Entrenando modelos..." if bloqueado else "Iniciar Análisis y Entrenamiento"
        )

    def mostrar_comparativa(self, acc_hmm, wer_hmm, acc_svm, wer_svm):
        self.lbl_metrica_hmm.setText(f"Acc: {acc_hmm:.2f}%  |  WER: {wer_hmm:.2f}%")
        self.lbl_metrica_svm.setText(f"Acc: {acc_svm:.2f}%  |  WER: {wer_svm:.2f}%")
        ganador = "HMM (Bakis)" if acc_hmm >= acc_svm else "SVM (RBF)"
        self.lbl_ganador.setText(ganador)

    def bloquear_microfono(self, grabando):
        self.btn_grabar.setEnabled(not grabando)
        self.slider_volumen.setEnabled(not grabando)
        self.btn_grabar.setText("Grabando... ¡Habla ahora!" if grabando else "Grabar desde Micrófono (2.0 s)")
        self.lbl_estado_micro.setText("Capturando audio a 16 kHz..." if grabando else "Micrófono en espera")
        self.cuadro_contador.setVisible(grabando)
        if grabando:
            self.lbl_valor_contador.setText("2.0 s")
        else:
            self.espectro.reiniciar()

    def actualizar_tiempo_restante(self, segundos):
        self.lbl_valor_contador.setText(f"{segundos:.1f} s")

    def actualizar_espectro(self, niveles):
        self.espectro.actualizar_niveles(niveles)

    def mostrar_resultado(self, origen, pred_hmm, pred_svm):
        if origen == "archivo":
            self.lbl_res_archivo_hmm.setText(pred_hmm.upper())
            self.lbl_res_archivo_svm.setText(pred_svm.upper())
            self.lbl_aviso_archivo.setText("")
        else:
            self.lbl_res_micro_hmm.setText(pred_hmm.upper())
            self.lbl_res_micro_svm.setText(pred_svm.upper())
            self.lbl_aviso_micro.setText("")

    def mostrar_aviso(self, origen, mensaje):
        if origen == "archivo":
            self.lbl_aviso_archivo.setText(mensaje)
        elif origen == "micro":
            self.lbl_aviso_micro.setText(mensaje)
        else:
            self.agregar_texto_consola(f"\n{mensaje}\n")

    # ==================== ESTILOS QSS (TONO CLARO) ====================
    def _aplicar_estilos(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Inter', 'Ubuntu', sans-serif;
                color: #1e1b2e;
            }
            #MarcoMorado {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7c3aed, stop:0.5 #9333ea, stop:1 #6d28d9
                );
            }
            #ContenedorInterior {
                background-color: #f8f7fc;
                border-radius: 10px;
            }
            #BarraSuperior {
                background-color: #ffffff;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                border-bottom: 1px solid #e5dff5;
            }
            /* Recuadro ASR alineado de borde izquierdo hasta el borde derecho del sidebar */
            #BadgeSuperior {
                font-size: 12px;
                font-weight: 700;
                color: #5b21b6;
                background-color: #ede9fe;
                border-top-left-radius: 10px;
                border-right: 1px solid #e2daf5;
                border-bottom: 1px solid #e5dff5;
            }
            #TextoAutores {
                font-size: 12px;
                font-weight: 600;
                color: #475569;
            }
            #Sidebar {
                background-color: #f2effa;
                border-bottom-left-radius: 10px;
                border-right: 1px solid #e2daf5;
            }
            #LogoTitulo {
                font-size: 19px;
                font-weight: 800;
                color: #4c1d95;
                line-height: 1.2;
            }
            #SeparadorSidebar {
                background-color: #ddd6fe;
            }
            #PieSidebar {
                font-size: 11px;
                color: #64748b;
            }
            #BotonNav {
                text-align: left;
                padding: 13px 16px;
                border-radius: 8px;
                background-color: transparent;
                color: #334155;
                font-size: 13px;
                font-weight: 600;
                border: 1px solid transparent;
            }
            #BotonNav:hover {
                background-color: #e9e3f8;
                color: #4c1d95;
            }
            #BotonNav:checked {
                background-color: #ffffff;
                color: #6d28d9;
                border: 1px solid #a78bfa;
            }
            #AreaContenido {
                background-color: #f8f7fc;
                border-bottom-right-radius: 10px;
            }
            #TituloPagina {
                font-size: 22px;
                font-weight: 700;
                color: #1e1b2e;
            }
            #SubtituloPagina {
                font-size: 13px;
                color: #64748b;
            }
            /* Cuadro contador de tiempo restante */
            #CuadroContador {
                background-color: #f3e8ff;
                border: 2px solid #8b5cf6;
                border-radius: 10px;
                min-width: 125px;
            }
            #TituloContador {
                font-size: 10px;
                font-weight: 800;
                color: #6d28d9;
                letter-spacing: 0.8px;
            }
            #ValorContador {
                font-size: 20px;
                font-weight: 800;
                color: #4c1d95;
            }
            /* Control de volumen */
            #EtiquetaVolumen {
                font-size: 13px;
                font-weight: 600;
                color: #334155;
            }
            #ValorVolumen {
                font-size: 13px;
                font-weight: 700;
                color: #6d28d9;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #ede9fe;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #7c3aed;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6d28d9;
                border: 2px solid #ffffff;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #8b5cf6;
            }
            #EtiquetaSeccion {
                font-size: 13px;
                font-weight: 700;
                color: #5b21b6;
            }
            #EtiquetaEstado {
                font-size: 13px;
                font-weight: 600;
                color: #7c3aed;
            }
            #EtiquetaAviso {
                font-size: 13px;
                color: #dc2626;
                font-weight: 600;
            }
            #Tarjeta {
                background-color: #ffffff;
                border: 1px solid #e2daf5;
                border-radius: 10px;
            }
            #TituloTarjeta {
                font-size: 11px;
                color: #64748b;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }
            #ValorTarjeta {
                font-size: 14px;
                font-weight: 700;
                color: #4c1d95;
            }
            #ValorTarjetaGrande {
                font-size: 26px;
                font-weight: 800;
                color: #6d28d9;
                letter-spacing: 1px;
            }
            #RutaArchivo {
                color: #334155;
                font-size: 13px;
                font-weight: 500;
            }
            #BotonPrimario {
                background-color: #7c3aed;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 11px 22px;
                font-size: 13px;
                font-weight: 700;
            }
            #BotonPrimario:hover {
                background-color: #6d28d9;
            }
            #BotonPrimario:disabled {
                background-color: #c4b5fd;
                color: #ffffff;
            }
            #BotonSecundario {
                background-color: #ede9fe;
                color: #5b21b6;
                border: 1px solid #c4b5fd;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 600;
            }
            #BotonSecundario:hover {
                background-color: #ddd6fe;
            }
            QProgressBar {
                background-color: #ede9fe;
                border: 1px solid #d8b4fe;
                border-radius: 7px;
                height: 20px;
                text-align: center;
                color: #2e1065;
                font-weight: 700;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7c3aed, stop:1 #a855f7
                );
                border-radius: 6px;
            }
            #ConsolaSalida {
                background-color: #ffffff;
                border: 1px solid #ddd6fe;
                border-radius: 8px;
                color: #2e1065;
                padding: 10px;
            }
            /* Barra deslizadora vertical y horizontal (ScrollBar) */
            QScrollBar:vertical {
                background: #f3effc;
                width: 10px;
                margin: 4px 2px 4px 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #c4b5fd;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #8b5cf6;
            }
            QScrollBar::handle:vertical:pressed {
                background: #6d28d9;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
                height: 0px;
            }

            QScrollBar:horizontal {
                background: #f3effc;
                height: 10px;
                margin: 2px 4px 2px 4px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #c4b5fd;
                min-width: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #8b5cf6;
            }
            QScrollBar::handle:horizontal:pressed {
                background: #6d28d9;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: none;
                width: 0px;
            }
        """)