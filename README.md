Sistema Inteligente de Detección de Ronquidos
<div align="center">
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/Arduino-Leonardo-orange
https://img.shields.io/badge/TensorFlow-Lite-FF6F00
https://img.shields.io/badge/License-MIT-green

Detección en tiempo real de ronquidos usando Machine Learning embebido

Sistema completo: desde la captura de audio hasta la intervención automática

</div>
🌟 Características Principales
🎤 Adquisición de Audio de Alta Calidad: Muestreo a 16 kHz con Arduino Leonardo

🧠 Modelo TinyML Optimizado: CNN 1D para clasificación eficiente en Raspberry Pi

⚡ Procesamiento en Tiempo Real: Latencia total < 2 segundos

🔊 Intervención No Invasiva: Activación de tonos suaves para mitigar ronquidos

📊 Dataset Especializado: +1,200 muestras de audio etiquetadas

🔧 Pipeline Completo: Entrenamiento, optimización y despliegue

🏗 Arquitectura del Sistema
text
Arduino Leonardo (16 kHz)
        ↓
Captura Audio → ADC 10-bit
        ↓
Serial (115200 baud) → Protocolo Binario
        ↓
Raspberry Pi 4/Zero
        ↓
Extracción MFCC → Normalización
        ↓
Modelo TFLite INT8 → Clasificación
        ↓
Decisión → Activación Buzzer
🚀 Comenzando Rápido
Hardware Requerido
Arduino Leonardo

Sensor KY-037 (Micrófono)

Raspberry Pi 4/Zero

Buzzer activo/pasivo

Protoboard y cables

Instalación Express
bash
# Clonar repositorio
git clone https://github.com/tuusuario/snore-detection-ai.git
cd snore-detection-ai

# Instalar dependencias Python
cd software/raspberry
pip install -r requirements.txt

# Cargar firmware Arduino (Abrir en Arduino IDE)
# firmware/ronquidos.ino
Configuración Hardware
Conexiones Arduino:

Sensor KY-037 OUT → Pin A0 (ADC7)

Buzzer → Pin 9

Alimentación: 5V y GND

Conexión Serial:

Arduino TX → Raspberry RX

Arduino RX → Raspberry TX

GND compartido

Uso Básico
bash
# Ejecutar sistema de detección
python snore_detector.py --model ../models/snore_model_int8.tflite --threshold 0.7

# Entrenar modelo personalizado
python ../ml/train_snore_end2end_optimized.py
Parámetros Principales
bash
# Ejemplo de uso completo
python snore_detector.py \
  --model ../models/snore_model_int8.tflite \
  --serial /dev/ttyACM0 \
  --threshold 0.65 \
  --win 1.5 \
  --hop 0.5 \
  --beep-ms 600 \
  --cooldown 2.0
📊 Rendimiento del Modelo
Métrica	Valor	Descripción
Precisión	94.2%	Clasificación correcta
Recall	92.8%	Detección de ronquidos reales
F1-Score	93.5%	Balance precisión-recall
Latencia	< 2s	Tiempo total de procesamiento
Tamaño Modelo	45 KB	Optimizado para edge
🗂 Estructura del Proyecto
text
snore-detection-ai/
├── firmware/
│   └── ronquidos.ino              # Código Arduino (muestreo audio)
├── software/
│   ├── raspberry/
│   │   ├── snore_detector.py      # Script principal de detección
│   │   └── requirements.txt       # Dependencias Python
│   └── ml/
│       └── train_snore_end2end_optimized.py  # Entrenamiento modelo
├── models/
│   ├── snore_model_int8.tflite    # Modelo optimizado INT8
│   └── snore_model_fp32.tflite    # Modelo precisión completa
├── datasets/                      # Estructura para datos de audio
└── results/                       # Métricas y evaluaciones
🔧 Componentes Técnicos
Arduino (firmware/ronquidos.ino)
Muestreo: 16 kHz estable con Timer1

ADC: 10-bit, centered en 512

Protocolo: Binario optimizado (0xAA 0x55 + datos)

Comandos: 'B' + 2 bytes LE para activar buzzer

Raspberry Pi (snore_detector.py)
Procesamiento: Ventanas de 1.5s con solapamiento 0.5s

Características: 20 MFCCs, 40 bandas mel (80-6000 Hz)

Modelo: TFLite INT8 para máxima eficiencia

Lógica: Histéresis y período de enfriamiento integrados

Entrenamiento ML (train_snore_end2end_optimized.py)
Arquitectura: Tiny DS-CNN optimizado

Aumentación: Ganancia aleatoria, ruido, time-shift

Exportación: TFLite FP32 e INT8

Balanceo: Aumentación específica para clase minoritaria

⚙️ Configuración Avanzada
Parámetros de Detección
python
--threshold 0.5      # Umbral de clasificación (0-1)
--hyst 0.1          # Histéresis para evitar flickering
--cooldown 2.0      # Segundos entre activaciones
--avg-k 5           # Promedio móvil de frames
--beep-ms 600       # Duración del tono en milisegundos
Optimización para Raspberry Pi Zero
bash
# Usar modelo INT8 para mejor rendimiento
python snore_detector.py --model ../models/snore_model_int8.tflite

# Reducir carga de CPU ajustando ventana
python snore_detector.py --win 1.0 --hop 1.0
🐛 Solución de Problemas
Problemas Comunes
Arduino no detectado:

bash
# Verificar puerto serial
ls /dev/ttyACM*
# Cambiar puerto en comando
python snore_detector.py --serial /dev/ttyACM1
Error de dependencias:

bash
# Actualizar pip e instalar
pip install --upgrade pip
pip install -r requirements.txt
Buzzer no suena:

Verificar conexiones (pin 9 y GND)

Confirmar si es buzzer activo o pasivo

Revisar código en ronquidos.ino (sección loop)

Logs y Debug
bash
# Ver datos en tiempo real
python snore_detector.py --threshold 0.5 --beep-ms 300

# Los archivos se guardan en:
# data/raw/ - Archivos de audio WAV
# data/events.csv - Registro de detecciones
