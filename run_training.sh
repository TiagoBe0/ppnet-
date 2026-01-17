#!/bin/bash
# Script para ejecutar el entrenamiento con las bibliotecas correctas

# Encontrar la ubicación de TensorFlow
TF_LIB=$(python -c 'import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))')

# Configurar LD_LIBRARY_PATH para incluir las bibliotecas de TensorFlow
export LD_LIBRARY_PATH="$TF_LIB:$LD_LIBRARY_PATH"

# Ejecutar el entrenamiento
python train_crystal_classifier.py \
  --labels_csv data/labels.csv \
  --data_dir data/datos_convertidos \
  --num_points 32 \
  --batch_size 16 \
  --epochs 150 \
  --learning_rate 0.001 \
  --gpu 0
