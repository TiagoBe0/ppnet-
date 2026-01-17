# Instrucciones para Resolver Errores y Compilar Operadores

## Tu Situación Actual

Tienes dos problemas:
1. **Conflicto de git**: Cambios locales en `tf_sampling.py` que impiden hacer pull
2. **Operadores faltantes**: Solo tienes `tf_interpolate_so.so`, faltan `tf_sampling_so.so` y `tf_grouping_so.so`
3. **Confusión de directorios**: Estás en `ppnet-` pero el error muestra `ppnet-old`

## PASO 1: Verificar si CUDA está instalado

Ejecuta estos comandos para verificar:

```bash
# Ver si tienes GPU NVIDIA
nvidia-smi

# Ver si CUDA está instalado
nvcc --version

# Buscar instalaciones de CUDA
ls -d /usr/local/cuda* 2>/dev/null
```

### Interpretación de resultados:

**A) Si `nvidia-smi` funciona:**
- ✓ Tienes GPU NVIDIA instalada y drivers funcionando

**B) Si `nvcc --version` funciona:**
- ✓ CUDA ya está instalado, puedes seguir al PASO 2

**C) Si `nvcc: command not found` PERO existe `/usr/local/cuda/`:**
- CUDA está instalado pero no en el PATH
- Ejecuta estos comandos y luego repite `nvcc --version`:
  ```bash
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  nvcc --version
  ```
- Si ahora funciona, añade estas líneas a tu `~/.bashrc` para que sea permanente:
  ```bash
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  ```

**D) Si NADA funciona:**
- Necesitas instalar CUDA. Ve al ANEXO A al final de este documento.

## PASO 2: Resolver el conflicto de git

```bash
cd ~/Escritorio/pnet++enero/ppnet-

# Ver qué archivos tienen cambios
git status

# Guardar los cambios locales temporalmente
git stash

# Obtener los cambios del repositorio
git pull origin claude/fix-batch-convert-error-vKkS3

# Verificar que ahora tienes el script de compilación
ls -l compile_tf_ops.sh
```

## PASO 3: Compilar los operadores TensorFlow

```bash
cd ~/Escritorio/pnet++enero/ppnet-

# Dar permisos de ejecución al script
chmod +x compile_tf_ops.sh

# Ejecutar compilación
./compile_tf_ops.sh
```

### Si compile_tf_ops.sh falla:

**Opción A: Compilación manual simple**

Si el script falla pero `nvcc --version` funciona, compila manualmente:

```bash
cd ~/Escritorio/pnet++enero/ppnet-

# Obtener rutas de TensorFlow
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Compilar sampling
cd tf_ops/sampling
nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -O2

# Compilar grouping
cd ../grouping
nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -O2

# Compilar interpolation (si no existe ya)
cd ../3d_interpolation
nvcc tf_interpolate_g.cu -o tf_interpolate_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 tf_interpolate.cpp tf_interpolate_g.cu.o -o tf_interpolate_so.so -shared -fPIC \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -O2

cd ../..
```

**Opción B: Si tienes errores de compilación con CUDA path**

Encuentra donde está CUDA instalado y úsalo:

```bash
# Encontrar CUDA
find /usr/local -name "nvcc" 2>/dev/null

# Si lo encuentra en por ejemplo /usr/local/cuda-11.8/bin/nvcc
# usa esa ruta reemplazando /usr/local/cuda con /usr/local/cuda-11.8
```

## PASO 4: Verificar que todo está compilado

```bash
cd ~/Escritorio/pnet++enero/ppnet-

# Deberías ver estos 3 archivos .so
ls -l tf_ops/sampling/tf_sampling_so.so
ls -l tf_ops/grouping/tf_grouping_so.so
ls -l tf_ops/3d_interpolation/tf_interpolate_so.so
```

Si todos existen, ¡perfecto! Continúa al PASO 5.

## PASO 5: Probar el entrenamiento

```bash
cd ~/Escritorio/pnet++enero/ppnet-

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

## ANEXO A: Instalación de CUDA (si no está instalado)

### 1. Verificar que tienes GPU NVIDIA

```bash
lspci | grep -i nvidia
```

Si no muestra nada, no tienes GPU NVIDIA y no podrás usar CUDA.

### 2. Determinar tu versión de GPU y Ubuntu

```bash
# Ver GPU
nvidia-smi

# Ver versión de Ubuntu
lsb_release -a
```

### 3. Descargar e instalar CUDA

#### Opción Recomendada: CUDA 11.8 (compatible con TensorFlow 2.x)

```bash
# Descargar instalador de CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Instalar
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**Durante la instalación:**
- Si te pregunta sobre drivers NVIDIA: Selecciona "No" si `nvidia-smi` ya funciona
- Selecciona "Yes" para instalar CUDA Toolkit
- Selecciona "Yes" para instalar samples (opcional)

### 4. Configurar variables de entorno

```bash
# Añadir a ~/.bashrc
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Recargar
source ~/.bashrc

# Verificar
nvcc --version
```

#### Opción Alternativa: Instalar vía package manager (Ubuntu)

```bash
# Instalar CUDA desde repositorios de Ubuntu (más fácil pero puede ser versión antigua)
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verificar
nvcc --version
```

### 5. Después de instalar CUDA

Regresa al **PASO 3** de este documento para compilar los operadores.

---

## Solución de Problemas Comunes

### Error: "cannot find -lcudart"

```bash
# Verificar que libcudart.so existe
find /usr/local/cuda* -name "libcudart.so*" 2>/dev/null

# Si no existe, necesitas reinstalar CUDA
```

### Error: "tensorflow/core/framework/op.h: No such file"

```bash
# Verificar que TensorFlow está instalado
python -c "import tensorflow as tf; print(tf.__version__)"

# Si falla, instalar TensorFlow con GPU
pip install tensorflow[and-cuda]
```

### Error: "cc1plus: error: -Wno-return-local-addr"

Añade `-Wno-return-local-addr` a los flags de g++, o intenta con `-std=c++11` en lugar de `-std=c++14`.

### El entrenamiento es muy lento

Si el entrenamiento va muy lento, verifica:

```bash
# Durante el entrenamiento, en otra terminal ejecuta:
nvidia-smi

# Deberías ver uso de GPU. Si no, puede ser que TensorFlow no detecte la GPU.
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Resumen Rápido

```bash
# 1. Verificar CUDA
nvcc --version

# 2. Si no existe, instalar (ver ANEXO A)

# 3. Resolver git y compilar
cd ~/Escritorio/pnet++enero/ppnet-
git stash
git pull origin claude/fix-batch-convert-error-vKkS3
chmod +x compile_tf_ops.sh
./compile_tf_ops.sh

# 4. Verificar
ls -l tf_ops/*/tf_*_so.so

# 5. Entrenar
python train_crystal_classifier.py --labels_csv data/labels.csv --data_dir data/datos_convertidos --num_points 32 --batch_size 16 --epochs 150 --learning_rate 0.001 --gpu 0
```

¡Buena suerte!
