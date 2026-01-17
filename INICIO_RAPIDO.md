# INICIO RÁPIDO - Resolver Error de Compilación

## ⚡ TL;DR - Comandos para ejecutar AHORA

Copia y pega estos comandos en tu terminal **uno por uno**:

```bash
# 1. IR AL DIRECTORIO CORRECTO
cd ~/Escritorio/pnet++enero/ppnet-

# 2. GUARDAR TUS CAMBIOS LOCALES Y OBTENER ACTUALIZACIONES
git stash
git pull origin claude/fix-batch-convert-error-vKkS3

# 3. VERIFICAR QUE TU SISTEMA ESTÁ LISTO
./verificar_sistema.sh
```

**Después del paso 3:**
- Si todo está ✓ verde → Continúa al paso 4
- Si hay errores ✗ rojos → Ve a la sección "PROBLEMAS COMUNES" abajo

```bash
# 4. COMPILAR OPERADORES TENSORFLOW (solo si paso 3 fue exitoso)
./compile_tf_ops.sh

# Si compile_tf_ops.sh falla, intenta con:
./compile_manual.sh
```

```bash
# 5. VERIFICAR QUE SE COMPILARON CORRECTAMENTE
ls -l tf_ops/sampling/tf_sampling_so.so
ls -l tf_ops/grouping/tf_grouping_so.so
ls -l tf_ops/3d_interpolation/tf_interpolate_so.so

# Deberías ver 3 archivos .so
```

```bash
# 6. EJECUTAR ENTRENAMIENTO
python train_crystal_classifier.py \
  --labels_csv data/labels.csv \
  --data_dir data/datos_convertidos \
  --num_points 32 \
  --batch_size 16 \
  --epochs 150 \
  --learning_rate 0.001 \
  --gpu 0
```

---

## ❌ PROBLEMAS COMUNES

### Problema 1: "nvcc: command not found"

**Diagnóstico:**
```bash
nvcc --version
```

Si da error, CUDA no está instalado o no está en el PATH.

**Solución A:** CUDA instalado pero no en PATH
```bash
# Buscar CUDA
ls -d /usr/local/cuda* 2>/dev/null

# Si encuentra por ejemplo /usr/local/cuda-11.8, ejecuta:
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Verificar
nvcc --version

# Si ahora funciona, hacerlo permanente:
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Solución B:** CUDA no instalado

Necesitas instalar CUDA. **Opciones:**

**Opción 1 - Fácil pero puede ser versión antigua:**
```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
nvcc --version
```

**Opción 2 - Recomendado (CUDA 11.8):**
```bash
# Descargar
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Instalar
sudo sh cuda_11.8.0_520.61.05_linux.run
# Durante instalación: Di NO a drivers si nvidia-smi ya funciona, YES a toolkit

# Configurar PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Verificar
nvcc --version
```

### Problema 2: "nvidia-smi: command not found"

Necesitas instalar drivers NVIDIA primero.

```bash
# Ver si tienes GPU NVIDIA
lspci | grep -i nvidia

# Si muestra una GPU NVIDIA, instala drivers:
sudo apt update
sudo apt install nvidia-driver-535  # o la versión más reciente

# Reinicia el sistema
sudo reboot

# Después del reinicio, verifica:
nvidia-smi
```

### Problema 3: "ModuleNotFoundError: No module named 'tensorflow'"

```bash
# Instalar TensorFlow con soporte GPU
pip install tensorflow[and-cuda]

# Verificar
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Problema 4: Compilación falla con errores de g++

**Error común:** `-Wno-return-local-addr` o problemas de C++ standard

**Solución:** Edita `compile_manual.sh` y cambia `-std=c++14` a `-std=c++11`:

```bash
nano compile_manual.sh
# Busca la línea con "g++ -std=c++14" y cámbiala a "g++ -std=c++11"
# Guarda con Ctrl+O, sale con Ctrl+X

# Ejecuta de nuevo
./compile_manual.sh
```

### Problema 5: "cannot find -lcudart"

CUDA está instalado pero las librerías no se encuentran.

```bash
# Buscar libcudart.so
find /usr/local -name "libcudart.so*" 2>/dev/null

# Si lo encuentra en /usr/local/cuda-XX/lib64/, añade al PATH:
export LD_LIBRARY_PATH=/usr/local/cuda-XX/lib64:$LD_LIBRARY_PATH

# Hacerlo permanente
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-XX/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Problema 6: TensorFlow no detecta GPU

```bash
# Verificar que nvidia-smi funciona
nvidia-smi

# Verificar TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Si retorna lista vacía [], reinstala TensorFlow:
pip uninstall tensorflow
pip install tensorflow[and-cuda]

# Verificar de nuevo
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 📚 Documentación Adicional

Si los pasos rápidos no funcionan, consulta:

1. **INSTRUCCIONES_COMPILACION.md** - Guía detallada paso a paso
2. **COMPILAR_OPERADORES.md** - Explicación técnica de los cambios realizados
3. **verificar_sistema.sh** - Diagnóstico completo del sistema

---

## 🎯 Resumen de qué hace cada archivo

| Archivo | Propósito |
|---------|-----------|
| `verificar_sistema.sh` | Diagnostica si tu sistema tiene todo lo necesario |
| `compile_tf_ops.sh` | Compilación automática (detecta configuración) |
| `compile_manual.sh` | Compilación manual simplificada |
| `INSTRUCCIONES_COMPILACION.md` | Guía detallada completa |
| `INICIO_RAPIDO.md` | Este archivo - inicio rápido |

---

## ⚡ Una vez que compile exitosamente

Deberías ver algo como esto al ejecutar el entrenamiento:

```
Epoch 1/150
[======>...] 10% - loss: 2.3456 - accuracy: 0.1234
```

Si ves esto, **¡felicidades!** El entrenamiento está funcionando.

Si el entrenamiento es muy lento (< 1 epoch por minuto), verifica que esté usando GPU:
```bash
# En otra terminal mientras entrena:
nvidia-smi
# Deberías ver uso de GPU en la columna "GPU-Util"
```

---

## 🆘 Si nada funciona

1. Ejecuta `./verificar_sistema.sh` y copia todo el output
2. Ejecuta `./compile_manual.sh` (o el que hayas intentado) y copia el error completo
3. Comparte los outputs para ayuda adicional

---

## ✅ Checklist de verificación

- [ ] `nvidia-smi` funciona
- [ ] `nvcc --version` funciona
- [ ] `python -c "import tensorflow as tf; print(tf.__version__)"` funciona
- [ ] `./verificar_sistema.sh` muestra ✓ en todos los pasos principales
- [ ] `./compile_tf_ops.sh` o `./compile_manual.sh` completó sin errores
- [ ] Existen los 3 archivos `.so` en tf_ops/*/
- [ ] El entrenamiento inicia sin errores de "No such file"
- [ ] `nvidia-smi` muestra uso de GPU durante el entrenamiento

¡Buena suerte! 🚀
