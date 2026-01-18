# 🚀 Instalación Completa Desde Cero - PointNet++ en Ubuntu 22.04

**Guía definitiva para instalar TensorFlow y ejecutar PointNet++ correctamente**

---

## 📋 Resumen Ejecutivo

Esta guía te llevará desde un sistema Ubuntu 22.04 limpio hasta tener PointNet++ funcionando correctamente con TensorFlow 2.15.

**Tiempo estimado:** 30-60 minutos (dependiendo de velocidad de internet y si tienes GPU)

**Requisitos previos:**
- Ubuntu 22.04 (funciona en 20.04 y 22.04)
- Python 3.10 o 3.11
- (Opcional) GPU NVIDIA para entrenamiento acelerado

---

## 🎯 OPCIÓN A: Instalación Automática (Recomendada)

Si quieres la instalación más rápida y fácil, sigue estos pasos:

### Paso 1: Ubicarte en el directorio del proyecto

```bash
cd ~/Escritorio/pnet++enero/ppnet-
```

### Paso 2: Ejecutar el instalador automático

```bash
./instalar_tensorflow.sh
```

Selecciona opción **1** (Instalación completa).

El script hará:
- ✅ Verificar tu sistema
- ✅ Instalar drivers NVIDIA (si tienes GPU)
- ✅ Instalar CUDA Toolkit
- ✅ Crear entorno virtual Python
- ✅ Instalar TensorFlow 2.15
- ✅ Compilar operadores personalizados
- ✅ Ejecutar pruebas

**Si el script pide reiniciar**: Reinicia y vuelve a ejecutar el script.

### Paso 3: Parchar el código para compatibilidad TF 2.x

```bash
./parchar_para_tf2.sh
```

Di **s** (sí) cuando pregunte.

### Paso 4: Probar que todo funciona

```bash
# Activar entorno virtual
source venv_pointnet/bin/activate

# Test rápido de entrenamiento (2-3 minutos)
python train_crystal_classifier.py \
    --epochs 2 \
    --n_samples 50 \
    --batch_size 8 \
    --num_points 64
```

Si ves progreso de las épocas sin errores → **¡ÉXITO!** ✅

Ahora salta a la sección [Usar el Sistema](#usar-el-sistema).

---

## 🔧 OPCIÓN B: Instalación Manual Paso a Paso

Si prefieres entender cada paso o el instalador automático falla:

### PARTE 1: Verificar Sistema y GPU

```bash
# Ver si tienes GPU NVIDIA
lspci | grep -i nvidia

# Ver versión de Python
python3 --version

# Ver versión de Ubuntu
lsb_release -a
```

---

### PARTE 2: Instalar Drivers NVIDIA y CUDA (Solo si tienes GPU)

#### 2.1 Instalar Drivers NVIDIA

```bash
# Ver drivers disponibles
ubuntu-drivers devices

# Instalar driver recomendado (usualmente 535 o superior)
sudo apt update
sudo apt install -y nvidia-driver-535

# REINICIAR (obligatorio)
sudo reboot
```

#### 2.2 Verificar drivers después del reinicio

```bash
nvidia-smi
```

Deberías ver información de tu GPU.

#### 2.3 Instalar CUDA Toolkit

```bash
# Instalar CUDA desde repos de Ubuntu
sudo apt install -y nvidia-cuda-toolkit libcudnn8 libcudnn8-dev

# Añadir CUDA al PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verificar
nvcc --version
```

---

### PARTE 3: Configurar Entorno Python

```bash
# Instalar dependencias del sistema
sudo apt install -y python3-pip python3-venv build-essential g++ gcc

# Ir al directorio del proyecto
cd ~/Escritorio/pnet++enero/ppnet-

# Crear entorno virtual
python3 -m venv venv_pointnet

# Activar entorno
source venv_pointnet/bin/activate

# Deberías ver (venv_pointnet) en tu prompt
```

---

### PARTE 4: Instalar TensorFlow

```bash
# Asegúrate de que el entorno está activado
source venv_pointnet/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar TensorFlow 2.15 con GPU
pip install tensorflow[and-cuda]==2.15.0

# Si NO tienes GPU, instala solo:
# pip install tensorflow==2.15.0

# Instalar dependencias adicionales
pip install numpy==1.24.3 h5py matplotlib scipy

# Verificar instalación
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Resultado esperado:**
```
TensorFlow: 2.15.0
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

(Lista vacía `[]` si no tienes GPU)

---

### PARTE 5: Compilar Operadores TensorFlow Personalizados

```bash
# Asegúrate de estar en el directorio del proyecto
cd ~/Escritorio/pnet++enero/ppnet-

# Activar entorno
source venv_pointnet/bin/activate

# Intentar compilación automática
./compile_tf_ops.sh

# Si falla, intentar compilación manual
./compile_manual.sh
```

#### Verificar compilación exitosa:

```bash
ls -lh tf_ops/sampling/tf_sampling_so.so
ls -lh tf_ops/grouping/tf_grouping_so.so
ls -lh tf_ops/3d_interpolation/tf_interpolate_so.so
```

Deberías ver 3 archivos `.so`.

---

### PARTE 6: Parchar Código para TensorFlow 2.x

El código original usa TensorFlow 1.x. Necesitamos parchearlo:

```bash
# Ejecutar script de parcheado
./parchar_para_tf2.sh
```

Esto actualiza todos los archivos Python para usar `tensorflow.compat.v1`.

#### ¿Qué hace el parcheado?

- Cambia `import tensorflow as tf` → `import tensorflow.compat.v1 as tf`
- Añade `tf.disable_eager_execution()` (necesario para código TF 1.x)
- Reduce warnings de deprecación
- Crea backups (`.bak`) de archivos originales

---

### PARTE 7: Probar la Instalación

```bash
# Activar entorno
source venv_pointnet/bin/activate

# Test 1: Importar TensorFlow
python -c "import tensorflow as tf; print('✓ TensorFlow OK:', tf.__version__)"

# Test 2: Importar modelo
python -c "
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
sys.path.append('models')
import pointnet2_cls_ssg
print('✓ Modelo importa correctamente')
"

# Test 3: Mini-entrenamiento (2-3 minutos)
python train_crystal_classifier.py \
    --epochs 2 \
    --n_samples 50 \
    --batch_size 8 \
    --num_points 64
```

**Si todo funciona sin errores** → **¡INSTALACIÓN EXITOSA!** 🎉

---

## 📊 Usar el Sistema

Una vez instalado y probado, aquí está el flujo de trabajo típico:

### 1. Activar Entorno (SIEMPRE al empezar)

```bash
cd ~/Escritorio/pnet++enero/ppnet-
source venv_pointnet/bin/activate
```

### 2. Convertir Tus Datos LAMMPS

Si tienes archivos `.dump` de LAMMPS:

```bash
# Opción A: Archivos organizados por clase
python batch_convert_dumps.py \
    --input_dir ./mis_datos \
    --output_dir ./datos_procesados \
    --format off \
    --normalize \
    --keep_structure \
    --recursive

# Opción B: Crear CSV con etiquetas
python create_labels_csv.py \
    --data_dir ./mis_datos \
    --output labels.csv
```

Ver `GUIA_RAPIDA.md` para más detalles sobre conversión de datos.

### 3. Entrenar el Modelo

#### Con datos sintéticos (para aprender/probar):

```bash
python train_crystal_classifier.py \
    --epochs 50 \
    --n_samples 500 \
    --crystal_types fcc bcc hcp \
    --batch_size 32 \
    --num_points 512
```

#### Con tus datos reales:

```bash
python train_crystal_classifier.py \
    --data_dir ./datos_procesados \
    --num_points 128 \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.001 \
    --gpu 0
```

#### Parámetros importantes:

- `--num_points`: Número de átomos por muestra (ajusta según tu sistema)
- `--batch_size`: Reduce si tienes errores de memoria
- `--epochs`: Más epochs = mejor accuracy (100-150 recomendado)
- `--gpu`: GPU a usar (0 para primera GPU)

### 4. Monitorear Entrenamiento

En otra terminal:

```bash
cd ~/Escritorio/pnet++enero/ppnet-
source venv_pointnet/bin/activate
tensorboard --logdir=log
```

Abre en navegador: http://localhost:6006

### 5. Ver Uso de GPU Durante Entrenamiento

```bash
# En otra terminal
watch -n 1 nvidia-smi
```

Deberías ver uso de GPU en la columna "GPU-Util".

---

## ❌ Solución de Problemas Comunes

### Problema 1: "ModuleNotFoundError: No module named 'tensorflow'"

**Causa:** Entorno virtual no activado

**Solución:**
```bash
source venv_pointnet/bin/activate
```

---

### Problema 2: "nvidia-smi: command not found"

**Causa:** Drivers NVIDIA no instalados o instalación fallida

**Solución:**
```bash
# Ver si tienes GPU
lspci | grep -i nvidia

# Si muestra GPU, instalar drivers
sudo apt install -y nvidia-driver-535
sudo reboot

# Después del reinicio
nvidia-smi
```

---

### Problema 3: TensorFlow no detecta GPU

**Diagnóstico:**
```bash
nvidia-smi  # ¿Funciona?
nvcc --version  # ¿Funciona?
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Si lista vacía:**
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0

# Verificar de nuevo
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

### Problema 4: Errores de compilación de operadores

**Error común:** `fatal error: cuda.h: No such file or directory`

**Solución:**
```bash
# Buscar CUDA
find /usr -name "cuda.h" 2>/dev/null

# Si está en /usr/include/cuda.h
export CUDA_HOME=/usr

# Si está en /usr/local/cuda-XX/include/cuda.h
export CUDA_HOME=/usr/local/cuda-XX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Recompilar
./compile_manual.sh
```

---

### Problema 5: "AttributeError: module 'tensorflow' has no attribute 'Session'"

**Causa:** Código no parcheado o parcheado incorrectamente

**Solución:**
```bash
# Re-ejecutar parcheado
./parchar_para_tf2.sh

# Verificar que archivos tienen tensorflow.compat.v1
grep -n "tensorflow.compat.v1" train_crystal_classifier.py models/pointnet2_cls_ssg.py
```

---

### Problema 6: Entrenamiento muy lento (sin GPU)

**Diagnóstico:**
```bash
# Durante entrenamiento, ejecutar:
nvidia-smi

# ¿Muestra uso de GPU?
```

**Si no usa GPU:**
1. Verifica `nvidia-smi` funciona
2. Verifica TensorFlow detecta GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
3. Usa batch size más pequeño: `--batch_size 4` o `8`

---

### Problema 7: "Out of Memory" (OOM)

**Solución:**
```bash
# Reducir batch size
python train_crystal_classifier.py --batch_size 8  # o 4

# Reducir num_points
python train_crystal_classifier.py --num_points 256  # en vez de 512
```

---

## 📚 Documentación Adicional

| Archivo | Descripción |
|---------|-------------|
| `INSTALACION_TENSORFLOW_UBUNTU_22.04.md` | Guía detallada de instalación |
| `COMPATIBILIDAD_TF2.md` | Explicación de cambios TF 1.x → 2.x |
| `GUIA_RAPIDA.md` | Workflow de conversión de datos |
| `LAMMPS_CRYSTAL_CLASSIFICATION.md` | Tutorial completo de clasificación |
| `INICIO_RAPIDO.md` | Comandos rápidos para empezar |
| `verificar_sistema.sh` | Script de diagnóstico del sistema |

---

## ✅ Checklist Final

Antes de declarar la instalación exitosa, verifica:

- [ ] `nvidia-smi` funciona (si tienes GPU)
- [ ] `nvcc --version` funciona (si tienes GPU)
- [ ] Entorno virtual creado: `venv_pointnet/`
- [ ] TensorFlow instalado: `python -c "import tensorflow as tf; print(tf.__version__)"`
- [ ] GPU detectada por TF: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- [ ] Operadores compilados: `ls tf_ops/*/tf_*_so.so` muestra 3 archivos
- [ ] Código parcheado: `grep "tensorflow.compat.v1" train_crystal_classifier.py`
- [ ] Test de entrenamiento funciona sin errores
- [ ] `nvidia-smi` muestra uso de GPU durante entrenamiento

---

## 🎓 Resumen de Comandos Completos

Para copiar y pegar (ajusta según necesites):

```bash
# ===== INSTALACIÓN COMPLETA =====

# 1. Instalar drivers (requiere reinicio)
sudo apt update && sudo apt install -y nvidia-driver-535
sudo reboot

# 2. Después del reinicio: Instalar CUDA
sudo apt install -y nvidia-cuda-toolkit libcudnn8 libcudnn8-dev build-essential
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3. Crear entorno Python
cd ~/Escritorio/pnet++enero/ppnet-
python3 -m venv venv_pointnet
source venv_pointnet/bin/activate

# 4. Instalar TensorFlow
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.15.0
pip install numpy==1.24.3 h5py matplotlib scipy

# 5. Compilar operadores
./compile_tf_ops.sh

# 6. Parchar código
./parchar_para_tf2.sh

# 7. Probar
python train_crystal_classifier.py --epochs 2 --n_samples 50 --batch_size 8

# ===== USO DIARIO =====

# Activar entorno (siempre al empezar)
cd ~/Escritorio/pnet++enero/ppnet-
source venv_pointnet/bin/activate

# Convertir datos
python batch_convert_dumps.py --input_dir ./datos --output_dir ./procesados --format off

# Entrenar
python train_crystal_classifier.py --data_dir ./procesados --epochs 100 --batch_size 16
```

---

## 🆘 Si Nada Funciona

1. **Ejecuta el diagnóstico completo:**
   ```bash
   ./verificar_sistema.sh > diagnostico.txt
   ```

2. **Captura el error completo:**
   ```bash
   python train_crystal_classifier.py --epochs 1 --n_samples 10 2>&1 | tee error.log
   ```

3. **Comparte los archivos:**
   - `diagnostico.txt`
   - `error.log`
   - Output de `nvidia-smi`
   - Output de `nvcc --version`

---

**¡Éxito con tu proyecto! 🚀**

Si esta guía te fue útil, compártela con otros que tengan problemas similares de compatibilidad TensorFlow 1.x → 2.x.
