#!/bin/bash
# Script de compilación manual simplificado
# Usa este script si compile_tf_ops.sh no funciona

set -e

echo "========================================"
echo "Compilación Manual de Operadores TF"
echo "========================================"

# Verificar que estamos en el directorio correcto
if [ ! -d "tf_ops" ]; then
    echo "ERROR: Debes ejecutar este script desde el directorio raíz del proyecto"
    echo "Ejecuta: cd ~/Escritorio/pnet++enero/ppnet-"
    exit 1
fi

# Verificar nvcc
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc no encontrado. CUDA no está instalado o no está en el PATH."
    echo ""
    echo "Opciones:"
    echo "1. Si CUDA está instalado pero no en PATH:"
    echo "   export PATH=/usr/local/cuda/bin:\$PATH"
    echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo ""
    echo "2. Si CUDA no está instalado, ve INSTRUCCIONES_COMPILACION.md"
    exit 1
fi

# Buscar CUDA
CUDA_PATH=$(which nvcc | sed 's|/bin/nvcc||')
echo "CUDA encontrado en: $CUDA_PATH"

# Obtener configuración de TensorFlow
echo "Obteniendo configuración de TensorFlow..."
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

echo "TF_CFLAGS: ${TF_CFLAGS[@]}"
echo "TF_LFLAGS: ${TF_LFLAGS[@]}"
echo ""

# Función para compilar un operador
compile_op() {
    local op_name=$1
    local op_dir=$2

    echo "========================================"
    echo "Compilando $op_name..."
    echo "========================================"

    cd "$op_dir"

    # Compilar CUDA
    echo "Compilando CUDA kernel..."
    nvcc tf_${op_name}_g.cu -o tf_${op_name}_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

    # Compilar C++
    echo "Compilando operador TF..."
    g++ -std=c++14 tf_${op_name}.cpp tf_${op_name}_g.cu.o -o tf_${op_name}_so.so -shared -fPIC \
        ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
        -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -lcudart -O2

    if [ -f "tf_${op_name}_so.so" ]; then
        echo "✓ $op_name compilado exitosamente"
    else
        echo "✗ Error compilando $op_name"
        exit 1
    fi

    cd - > /dev/null
    echo ""
}

# Compilar cada operador
compile_op "sampling" "tf_ops/sampling"
compile_op "grouping" "tf_ops/grouping"
compile_op "interpolate" "tf_ops/3d_interpolation"

echo "========================================"
echo "✓ Compilación completada!"
echo "========================================"
echo ""
echo "Archivos generados:"
ls -lh tf_ops/sampling/tf_sampling_so.so
ls -lh tf_ops/grouping/tf_grouping_so.so
ls -lh tf_ops/3d_interpolation/tf_interpolate_so.so
echo ""
echo "Ahora puedes ejecutar el entrenamiento:"
echo "python train_crystal_classifier.py --labels_csv data/labels.csv --data_dir data/datos_convertidos --num_points 32 --batch_size 16 --epochs 150 --learning_rate 0.001 --gpu 0"
