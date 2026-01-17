#!/bin/bash
# Script para encontrar la biblioteca de TensorFlow

echo "Buscando libtensorflow_framework.so..."
python -c "
import tensorflow as tf
import os
import glob

# Buscar en el directorio de TensorFlow
tf_dir = os.path.dirname(tf.__file__)
print(f'TensorFlow instalado en: {tf_dir}')

# Buscar la biblioteca
for pattern in ['**/libtensorflow_framework*.so*', '**/libtensorflow_framework*.so.*']:
    files = glob.glob(os.path.join(tf_dir, pattern), recursive=True)
    if files:
        print(f'\\nEncontrado:')
        for f in files:
            print(f'  {f}')
            print(f'  Directorio: {os.path.dirname(f)}')
"
