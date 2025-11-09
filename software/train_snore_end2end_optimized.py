#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenamiento TinyML optimizado para detección de ronquidos
Especialmente adaptado para Raspberry Pi Zero
- Incluye aumentación de datos
- Entrenamiento consciente de cuantización
- Exporta a TFLite INT8
"""

import os, glob, json
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# CONFIGURACION
SR = 16000
WIN_SEC = 1.5
N_MFCC = 20
HOP = 160
N_MELS = 40
FMIN, FMAX = 80, 6000
LABELS = ["background", "snore"]
T_FRAMES = int(WIN_SEC * SR / HOP)
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)

# AUMENTACION DE DATOS 
def random_gain(y, min_gain=0.7, max_gain=1.3):
    return y * np.random.uniform(min_gain, max_gain)

def add_noise(y, noise_factor=0.002):
    noise = np.random.normal(0, noise_factor, len(y))
    return y + noise

def time_shift(y, sr=SR, max_shift_seconds=0.3):
    shift = int(np.random.uniform(-max_shift_seconds * sr, max_shift_seconds * sr))
    return np.roll(y, shift)

def augment_audio(y):
    y = random_gain(y)
    if np.random.random() > 0.5:
        y = add_noise(y)
    if np.random.random() > 0.5:
        y = time_shift(y)
    return y

# UTILIDADES 
def ensure_len(y, sr=SR, sec=WIN_SEC):
    need = int(sr * sec)
    if len(y) > need:
        start = np.random.randint(0, len(y) - need)
        return y[start:start + need]
    return np.pad(y, (0, max(0, need - len(y))))[:need]

def wav_to_mfcc(y):
    # Normalize audio
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    # Extract MFCCs
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, 
                            n_fft=512, hop_length=HOP,
                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    
    # Normalize MFCCs
    m = (m - np.mean(m, axis=1, keepdims=True)) / (np.std(m, axis=1, keepdims=True) + 1e-8)
    
    m = m.T.astype(np.float32)
    return np.pad(m, ((0, max(0, T_FRAMES - len(m))), (0, 0)))[:T_FRAMES]

# CARGA DE DATOS 
def load_dataset(base, augment=False):
    X, y = [], []
    for li, lbl in enumerate(LABELS):
        paths = glob.glob(os.path.join(base, lbl, "*.wav"))
        print(f"Cargando {lbl}: {len(paths)} archivos")
        for p in paths:
            sig, _ = librosa.load(p, sr=SR, mono=True)
            sig = ensure_len(sig)
            
            # Original sample
            X.append(wav_to_mfcc(sig)[..., None])
            y.append(li)
            
            # Augmented samples
            if augment and lbl == "snore":  # Aumentar solo los ronquidos para balance
                for _ in range(2):  # 2 augmented versions per snore
                    aug_sig = augment_audio(sig.copy())
                    X.append(wav_to_mfcc(aug_sig)[..., None])
                    y.append(li)
                    
    return np.stack(X), np.array(y)

# MODELO
def build_model():
    # Tiny DS-CNN optimizado
    inp = tf.keras.Input(shape=(T_FRAMES, N_MFCC, 1))
    
    # First conv block
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # DS Conv blocks
    for filters in [32, 48]:
        x = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 1, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    # Global pooling and output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(len(LABELS), activation='softmax')(x)
    
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# EXPORTAR TFLITE
def export_tflite(model, X_test, output_dir):
    # Representative dataset for quantization
    def representative_dataset():
        for i in np.random.choice(len(X_test), min(100, len(X_test)), False):
            yield [X_test[i:i+1].astype(np.float32)]
    
    # FP32 version
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    fp32_path = os.path.join(output_dir, "snore_model_fp32.tflite")
    with open(fp32_path, 'wb') as f:
        f.write(tflite_fp32)
    print(f"✓ Modelo FP32: {fp32_path} ({len(tflite_fp32)/1024:.1f} KB)")
    
    # INT8 version
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    
    tflite_int8 = converter.convert()
    int8_path = os.path.join(output_dir, "snore_model_int8.tflite")
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8)
    print(f"✓ Modelo INT8: {int8_path} ({len(tflite_int8)/1024:.1f} KB)")
    
    return fp32_path, int8_path

# MAIN 
def main():
    base = "data"
    artifacts = "artifacts"
    os.makedirs(artifacts, exist_ok=True)
    
    # Cargar y aumentar dataset
    print("\n1. Cargando y aumentando dataset...")
    X, y = load_dataset(base, augment=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train: {X_train.shape} Test: {X_test.shape}")
    
    # Entrenar modelo
    print("\n2. Entrenando modelo...")
    model = build_model()
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar
    print("\n3. Evaluando modelo...")
    y_pred = model.predict(X_test).argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    
    # Guardar reporte
    with open(os.path.join(artifacts, "report.txt"), "w") as f:
        f.write(report + "\n\n")
        f.write("Matriz de confusión:\n")
        f.write(str(cm))
    
    # Guardar modelo y labels
    print("\n4. Guardando artefactos...")
    model.save(os.path.join(artifacts, "snore_model.keras"))
    with open(os.path.join(artifacts, "labels.txt"), "w") as f:
        f.write("\n".join(LABELS))
    
    # Exportar TFLite
    print("\n5. Exportando versiones TFLite...")
    fp32_path, int8_path = export_tflite(model, X_test, artifacts)
    
    print("\nListo! Revisa la carpeta artifacts para los resultados.")

if __name__ == "__main__":
    main()