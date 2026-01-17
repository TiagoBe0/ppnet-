#!/bin/bash
# Script para ejecutar el entrenamiento con las bibliotecas correctas

# Encontrar todas las ubicaciones posibles de bibliotecas de TensorFlow
TF_PATHS=$(python -c '
import tensorflow as tf
import os
import glob

paths = []
tf_dir = os.path.dirname(tf.__file__)
paths.append(tf_dir)

# Buscar en subdirectorios comunes
for subdir in ["", "python", "core", "lib"]:
    path = os.path.join(tf_dir, subdir)
    if os.path.exists(path):
        paths.append(path)

# Buscar donde está libtensorflow_framework.so
for root, dirs, files in os.walk(tf_dir):
    for file in files:
        if "libtensorflow_framework" in file and file.endswith(".so"):
            paths.append(root)
            break

# Eliminar duplicados
paths = list(set(paths))
print(":".join(paths))
')

# Agregar también rutas del sistema
SYSTEM_LIBS="/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib"

# Configurar LD_LIBRARY_PATH completo
export LD_LIBRARY_PATH="$TF_PATHS:$SYSTEM_LIBS:$LD_LIBRARY_PATH"

echo "LD_LIBRARY_PATH configurado con:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -10

# Ejecutar el entrenamiento
python train_crystal_classifier.py \
  --labels_csv data/labels.csv \
  --data_dir data/datos_convertidos \
  --num_points 32 \
  --batch_size 16 \
  --epochs 150 \
  --learning_rate 0.001 \
  --gpu 0
