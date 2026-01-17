# Guía de Compilación de Operadores TensorFlow

## Problema

El error que estás viendo ocurre porque los operadores customizados de TensorFlow necesitan ser compilados antes de poder entrenar el modelo:

```
tensorflow.python.framework.errors_impl.NotFoundError: /home/santi/Escritorio/pnet++enero/ppnet-/tf_ops/sampling/tf_sampling_so.so: cannot open shared object file: No such file or directory
```

## Solución

### Paso 1: Verificar requisitos

Antes de compilar, asegúrate de tener instalado:

1. **NVIDIA CUDA Toolkit** (para operaciones GPU)
   - Descarga desde: https://developer.nvidia.com/cuda-downloads
   - Verifica con: `nvcc --version`

2. **g++ compilador** (usualmente ya está instalado)
   - Verifica con: `g++ --version`

3. **TensorFlow con soporte GPU**
   - Verifica con: `python -c "import tensorflow as tf; print(tf.__version__)"`

### Paso 2: Compilar los operadores

Ejecuta el script de compilación:

```bash
cd /Escritorio/pnet++enero/ppnet-
./compile_tf_ops.sh
```

Si el script no es ejecutable, primero hazlo ejecutable:

```bash
chmod +x compile_tf_ops.sh
```

### Paso 3: Verificar la compilación

Si la compilación es exitosa, deberías ver estos archivos creados:

```bash
ls tf_ops/sampling/tf_sampling_so.so
ls tf_ops/grouping/tf_grouping_so.so
ls tf_ops/3d_interpolation/tf_interpolate_so.so
```

### Paso 4: Ejecutar el entrenamiento

Ahora puedes ejecutar el entrenamiento:

```bash
python train_crystal_classifier.py \
  --labels_csv data/labels.csv \
  --data_dir data/datos_convertidos \
  --num_points 32 \
  --batch_size 16 \
  --epochs 150 \
  --learning_rate 0.001 \
  --gpu 0
```

## Errores Comunes

### Error: nvcc no encontrado

Si obtienes este error, CUDA no está instalado o no está en el PATH. Soluciones:

1. **Instalar CUDA**: Descarga e instala desde nvidia.com
2. **Agregar al PATH**:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

### Error: TensorFlow no encontrado

Asegúrate de estar en el entorno conda correcto:

```bash
conda activate base  # o el nombre de tu entorno
```

### Error de compilación con g++

Si obtienes errores de compilación, es posible que necesites ajustar la versión de C++. El script usa `-std=c++14`, pero puedes probar con `-std=c++11` si hay problemas.

## Cambios realizados

Los siguientes archivos fueron corregidos para Python 3:

1. **tf_ops/sampling/tf_sampling.py**:
   - Corregidos print statements (líneas 82, 84, 87)
   - Cambiado `import cPickle` a `import pickle` (línea 88)

2. **utils/pointnet_util.py**:
   - Corregida secuencia de escape `\sum_k` a `\\sum_k` (línea 169)

Estos cambios eran necesarios porque el código original fue escrito para Python 2.7 y ahora estás usando Python 3.x.
