#!/bin/bash
# Script de compilación de operadores customizados de TensorFlow
# Actualizado para TensorFlow 2.x y Python 3.x

set -e

echo "========================================"
echo "Compilando operadores customizados de TensorFlow"
echo "========================================"

# Detectar rutas de TensorFlow
echo "Detectando configuración de TensorFlow..."
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

echo "TF_CFLAGS: ${TF_CFLAGS[@]}"
echo "TF_LFLAGS: ${TF_LFLAGS[@]}"

# Detectar CUDA
if command -v nvcc &> /dev/null; then
    CUDA_PATH=$(which nvcc | sed 's|/bin/nvcc||')
    echo "CUDA encontrado en: $CUDA_PATH"
    HAS_CUDA=true
else
    echo "ADVERTENCIA: nvcc no encontrado. Buscando CUDA en ubicaciones comunes..."
    HAS_CUDA=false

    # Buscar en ubicaciones comunes
    for cuda_dir in /usr/local/cuda /usr/local/cuda-* /opt/cuda; do
        if [ -d "$cuda_dir" ] && [ -f "$cuda_dir/bin/nvcc" ]; then
            CUDA_PATH="$cuda_dir"
            export PATH="$CUDA_PATH/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
            echo "CUDA encontrado en: $CUDA_PATH"
            HAS_CUDA=true
            break
        fi
    done

    if [ "$HAS_CUDA" = false ]; then
        echo "ERROR: No se encontró CUDA instalado."
        echo "Para instalar CUDA, visita: https://developer.nvidia.com/cuda-downloads"
        echo ""
        echo "Alternativa: Si no tienes GPU, necesitarás usar una versión CPU-only del modelo,"
        echo "pero esto será significativamente más lento."
        exit 1
    fi
fi

# Compilar operador de sampling
echo ""
echo "========================================"
echo "1/3: Compilando tf_sampling..."
echo "========================================"
cd tf_ops/sampling

# Compilar con nvcc
nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# Compilar con g++
g++ -std=c++14 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
    -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -lcudart -O2

echo "✓ tf_sampling compilado exitosamente"

# Compilar operador de grouping
echo ""
echo "========================================"
echo "2/3: Compilando tf_grouping..."
echo "========================================"
cd ../grouping

nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++14 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
    -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -lcudart -O2

echo "✓ tf_grouping compilado exitosamente"

# Compilar operador de interpolación
echo ""
echo "========================================"
echo "3/3: Compilando tf_interpolate..."
echo "========================================"
cd ../3d_interpolation

nvcc tf_interpolate_g.cu -o tf_interpolate_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++14 tf_interpolate.cpp tf_interpolate_g.cu.o -o tf_interpolate_so.so -shared -fPIC \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
    -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -lcudart -O2

echo "✓ tf_interpolate compilado exitosamente"

cd ../..

echo ""
echo "========================================"
echo "✓ Todos los operadores compilados exitosamente!"
echo "========================================"
echo ""
echo "Ahora puedes ejecutar el entrenamiento con:"
echo "python train_crystal_classifier.py --labels_csv data/labels.csv --data_dir data/datos_convertidos --num_points 32 --batch_size 16 --epochs 150 --learning_rate 0.001 --gpu 0"
