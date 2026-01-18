# Instalación Completa de TensorFlow en Ubuntu 22.04 para PointNet++

## 🎯 Objetivo
Instalar TensorFlow 2.x con soporte GPU en Ubuntu 22.04 para ejecutar PointNet++, resolviendo problemas de compatibilidad con el código original (TensorFlow 1.2 / Python 2.7).

---

## 📋 Requisitos del Sistema

**Tu sistema actual:**
- ✅ Ubuntu 22.04
- ✅ Python 3.11.14
- ⚠️ No hay TensorFlow instalado
- ⚠️ No hay drivers NVIDIA instalados

---

## 🚀 PARTE 1: Instalación de Drivers NVIDIA y CUDA

### Paso 1.1: Verificar si tienes GPU NVIDIA

```bash
lspci | grep -i nvidia
```

**Si ves algo como** `NVIDIA Corporation ...` → Tienes GPU, continúa.
**Si no ves nada** → No tienes GPU NVIDIA, salta a PARTE 2 (CPU only).

---

### Paso 1.2: Instalar Drivers NVIDIA

```bash
# Actualizar repositorios
sudo apt update
sudo apt upgrade -y

# Ver drivers recomendados
ubuntu-drivers devices

# Instalar driver recomendado (generalmente 535 o superior)
sudo apt install -y nvidia-driver-535

# Reiniciar el sistema (OBLIGATORIO)
sudo reboot
```

**Después del reinicio, verificar:**
```bash
nvidia-smi
```

Deberías ver algo como:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2    |
+-----------------------------------------------------------------------------+
```

---

### Paso 1.3: Instalar CUDA Toolkit

**IMPORTANTE:** TensorFlow 2.15+ requiere CUDA 11.8 o 12.x

```bash
# Instalar CUDA 12.x desde repositorios de Ubuntu
sudo apt install -y nvidia-cuda-toolkit

# Verificar versión
nvcc --version
```

**Si nvcc no se encuentra**, añade al PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Hacer permanente
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### Paso 1.4: Instalar cuDNN

```bash
# cuDNN viene incluido en nvidia-cudnn
sudo apt install -y libcudnn8 libcudnn8-dev
```

---

## 🐍 PARTE 2: Configuración de Entorno Python

### Paso 2.1: Crear entorno virtual (RECOMENDADO)

```bash
# Instalar venv si no lo tienes
sudo apt install -y python3.11-venv python3-pip

# Ir al directorio del proyecto
cd ~/Escritorio/pnet++enero/ppnet-

# Crear entorno virtual
python3 -m venv venv_pointnet

# Activar entorno
source venv_pointnet/bin/activate

# Actualizar pip
pip install --upgrade pip setuptools wheel
```

**Nota:** A partir de ahora, SIEMPRE activa el entorno antes de trabajar:
```bash
source venv_pointnet/bin/activate
```

---

## 💻 PARTE 3: Instalación de TensorFlow

### Paso 3.1: Instalar TensorFlow con GPU

```bash
# Asegúrate de que el entorno virtual está activado
source venv_pointnet/bin/activate

# Instalar TensorFlow 2.15 (última versión compatible con CUDA 12.x)
pip install tensorflow[and-cuda]==2.15.0

# Instalar dependencias adicionales para PointNet++
pip install numpy==1.24.3
pip install h5py
pip install matplotlib
pip install scipy
```

**¿Por qué TensorFlow 2.15?**
- Compatible con Python 3.11
- Soporte nativo para CUDA 12.x
- Incluye paquetes de CUDA/cuDNN automáticamente

---

### Paso 3.2: Verificar instalación de TensorFlow

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

Deberías ver: `TensorFlow version: 2.15.0`

---

### Paso 3.3: Verificar detección de GPU

```bash
python -c "import tensorflow as tf; print('GPUs disponibles:', tf.config.list_physical_devices('GPU'))"
```

**Resultado esperado (con GPU):**
```
GPUs disponibles: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Resultado sin GPU:**
```
GPUs disponibles: []
```

---

## 🔧 PARTE 4: Compilar Operadores TensorFlow de PointNet++

### Paso 4.1: Instalar herramientas de compilación

```bash
sudo apt install -y build-essential g++ gcc

# Verificar
gcc --version
g++ --version
```

---

### Paso 4.2: Compilar los operadores

El proyecto ya tiene scripts de compilación actualizados para TensorFlow 2.x:

```bash
# Asegúrate de estar en el directorio del proyecto
cd ~/Escritorio/pnet++enero/ppnet-

# Asegúrate de que el entorno virtual está activado
source venv_pointnet/bin/activate

# Ejecutar script de compilación
./compile_tf_ops.sh
```

**Si falla**, intenta el script manual:
```bash
./compile_manual.sh
```

---

### Paso 4.3: Verificar compilación exitosa

```bash
ls -lh tf_ops/sampling/tf_sampling_so.so
ls -lh tf_ops/grouping/tf_grouping_so.so
ls -lh tf_ops/3d_interpolation/tf_interpolate_so.so
```

Deberías ver 3 archivos `.so` (bibliotecas compartidas).

---

## 🧪 PARTE 5: Probar la Instalación

### Paso 5.1: Test básico de importación

```bash
python -c "
import tensorflow as tf
import numpy as np

print('✓ TensorFlow version:', tf.__version__)
print('✓ NumPy version:', np.__version__)
print('✓ GPUs disponibles:', len(tf.config.list_physical_devices('GPU')))
print('✓ Eager execution:', tf.executing_eagerly())
"
```

---

### Paso 5.2: Test de operadores personalizados

```bash
python -c "
import tensorflow as tf
import sys
sys.path.insert(0, 'tf_ops/sampling')
from tf_sampling import farthest_point_sample, gather_point

print('✓ Operadores de sampling cargados correctamente')
"
```

---

### Paso 5.3: Test de entrenamiento (datos sintéticos)

```bash
# Probar con datos sintéticos pequeños
python train_crystal_classifier.py \
    --epochs 2 \
    --n_samples 50 \
    --batch_size 8 \
    --num_points 64
```

Si ves el progreso de las épocas sin errores → **¡Todo funciona!** ✅

---

## 🔥 PARTE 6: Compatibilidad TensorFlow 1.x → 2.x

El código original usa TensorFlow 1.x. Los archivos del proyecto ya han sido actualizados para TF 2.x, pero si encuentras errores, aquí están los cambios principales:

### Cambios realizados automáticamente:

1. **Sesiones eliminadas:**
   ```python
   # TF 1.x (VIEJO)
   with tf.Session() as sess:
       sess.run(...)

   # TF 2.x (NUEVO)
   # Eager execution por defecto, no necesitas sesiones
   ```

2. **Placeholders eliminados:**
   ```python
   # TF 1.x (VIEJO)
   x = tf.placeholder(tf.float32, [None, 1024, 3])

   # TF 2.x (NUEVO)
   # Usa @tf.function y tf.TensorSpec
   ```

3. **Variables:**
   ```python
   # TF 1.x (VIEJO)
   tf.get_variable('weights', ...)

   # TF 2.x (NUEVO)
   tf.Variable(name='weights', ...)
   ```

---

## 📝 PARTE 7: Flujo de Trabajo Completo

Una vez instalado todo, este es el flujo típico:

```bash
# 1. Activar entorno virtual (SIEMPRE al empezar)
cd ~/Escritorio/pnet++enero/ppnet-
source venv_pointnet/bin/activate

# 2. Convertir tus datos LAMMPS
python batch_convert_dumps.py \
    --input_dir ./mis_datos \
    --output_dir ./datos_procesados \
    --format off \
    --normalize \
    --keep_structure \
    --recursive

# 3. Entrenar el modelo
python train_crystal_classifier.py \
    --data_dir ./datos_procesados \
    --num_points 128 \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.001 \
    --gpu 0

# 4. Monitorear con TensorBoard (en otra terminal)
source venv_pointnet/bin/activate
tensorboard --logdir=log
# Abre http://localhost:6006 en tu navegador
```

---

## ❌ SOLUCIÓN DE PROBLEMAS COMUNES

### Problema 1: "ModuleNotFoundError: No module named 'tensorflow'"

**Causa:** Entorno virtual no activado.

**Solución:**
```bash
source venv_pointnet/bin/activate
```

---

### Problema 2: TensorFlow no detecta GPU

**Verificar:**
```bash
nvidia-smi  # ¿Funciona?
nvcc --version  # ¿Funciona?
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Soluciones:**

A) **Reinstalar TensorFlow:**
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0
```

B) **Variables de entorno:**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

C) **Verificar compatibilidad CUDA:**
```bash
python -c "from tensorflow.python.platform import build_info as tf_build_info; print('CUDA version:', tf_build_info.build_info['cuda_version']); print('cuDNN version:', tf_build_info.build_info['cudnn_version'])"
```

---

### Problema 3: "AttributeError: module 'tensorflow' has no attribute 'placeholder'"

**Causa:** Código TensorFlow 1.x en entorno TF 2.x.

**Solución:** El código del proyecto ya está actualizado. Si ves este error en archivos específicos, avísame para actualizarlos.

---

### Problema 4: Compilación de operadores falla

**Error común:** `fatal error: third_party/gpus/cuda/include/cuda.h: No such file or directory`

**Solución:**
```bash
# Buscar CUDA include
find /usr -name "cuda.h" 2>/dev/null

# Si está en /usr/include/cuda.h (no en /usr/local/cuda)
export CUDA_HOME=/usr

# Recompilar
./compile_manual.sh
```

---

### Problema 5: Python 3.11 no compatible con paquetes antiguos

**Solución:** Usa Python 3.10 en su lugar:

```bash
# Instalar Python 3.10
sudo apt install -y python3.10 python3.10-venv

# Recrear entorno con Python 3.10
python3.10 -m venv venv_pointnet
source venv_pointnet/bin/activate
pip install tensorflow[and-cuda]==2.15.0
```

---

## ✅ CHECKLIST FINAL

Antes de empezar a entrenar, verifica:

- [ ] `nvidia-smi` funciona y muestra tu GPU
- [ ] `nvcc --version` muestra CUDA instalado
- [ ] Entorno virtual creado y activado
- [ ] `python -c "import tensorflow as tf; print(tf.__version__)"` muestra 2.15.0
- [ ] `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` muestra tu GPU
- [ ] Existen 3 archivos `.so` en `tf_ops/*/`
- [ ] Test de entrenamiento sintético funciona sin errores

---

## 🎓 RESUMEN: Comandos Completos desde Cero

Aquí está todo en un solo bloque que puedes copiar y pegar (ajusta según necesites):

```bash
# === PARTE 1: DRIVERS Y CUDA ===
sudo apt update && sudo apt upgrade -y
sudo apt install -y nvidia-driver-535
# REINICIAR AQUÍ: sudo reboot

# Después del reinicio:
nvidia-smi  # Verificar
sudo apt install -y nvidia-cuda-toolkit libcudnn8 libcudnn8-dev
nvcc --version  # Verificar

# === PARTE 2: ENTORNO PYTHON ===
sudo apt install -y python3.11-venv python3-pip build-essential
cd ~/Escritorio/pnet++enero/ppnet-
python3 -m venv venv_pointnet
source venv_pointnet/bin/activate

# === PARTE 3: TENSORFLOW ===
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.15.0
pip install numpy==1.24.3 h5py matplotlib scipy

# === PARTE 4: COMPILAR OPERADORES ===
./compile_tf_ops.sh

# === PARTE 5: VERIFICAR ===
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

# === PARTE 6: PROBAR ===
python train_crystal_classifier.py --epochs 2 --n_samples 50 --batch_size 8
```

---

## 🔗 Archivos Relacionados

- `verificar_sistema.sh` - Diagnóstico completo del sistema
- `compile_tf_ops.sh` - Compilación automática de operadores
- `compile_manual.sh` - Compilación manual si la automática falla
- `INSTRUCCIONES_COMPILACION.md` - Detalles técnicos de compilación

---

## 📞 Soporte Adicional

Si después de seguir esta guía sigues teniendo problemas:

1. Ejecuta el script de verificación:
   ```bash
   ./verificar_sistema.sh
   ```

2. Copia TODO el output de:
   ```bash
   nvidia-smi
   nvcc --version
   python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
   ./compile_tf_ops.sh
   ```

3. Comparte los outputs para ayuda específica.

---

**¡Éxito! 🚀**
