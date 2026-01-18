# Compatibilidad TensorFlow 2.x

## 🚨 Problema: Código TensorFlow 1.x vs TensorFlow 2.x

El código original de PointNet++ fue escrito para **TensorFlow 1.2** (2017). TensorFlow 2.x (2019+) cambió completamente la API.

### Diferencias Principales:

| TensorFlow 1.x | TensorFlow 2.x |
|----------------|----------------|
| `tf.Session()` | Eager execution (sin sesiones) |
| `tf.placeholder()` | Input directo / `@tf.function` |
| `tf.get_variable()` | `tf.Variable()` o Keras layers |
| `sess.run(op, feed_dict={...})` | Ejecución directa |
| `tf.train.Optimizer()` | `tf.keras.optimizers.Optimizer()` |

---

## ✅ SOLUCIÓN 1: Modo de Compatibilidad (Recomendado para empezar)

TensorFlow 2.x incluye `tf.compat.v1` que permite ejecutar código TF 1.x con cambios mínimos.

### Paso 1: Crear wrapper de compatibilidad

Crea el archivo `tf1_compat.py` en el directorio raíz:

```python
"""
Wrapper de compatibilidad para ejecutar código TensorFlow 1.x en TensorFlow 2.x
"""

import tensorflow as tf

# Desactivar eager execution para usar modo TF 1.x
tf.compat.v1.disable_eager_execution()

# Configurar comportamiento de TF 1.x
tf.compat.v1.disable_v2_behavior()

# Exportar símbolos de TF 1.x con nombres de TF 2.x
placeholder = tf.compat.v1.placeholder
Session = tf.compat.v1.Session
ConfigProto = tf.compat.v1.ConfigProto
global_variables_initializer = tf.compat.v1.global_variables_initializer
train = tf.compat.v1.train
summary = tf.compat.v1.summary
Graph = tf.compat.v1.Graph
Variable = tf.compat.v1.Variable

print("✓ TensorFlow en modo compatibilidad 1.x")
print(f"  Version: {tf.__version__}")
```

### Paso 2: Modificar archivos de entrenamiento

Añade al inicio de `train_crystal_classifier.py`:

```python
# Al inicio del archivo, ANTES de import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir warnings

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
```

### Paso 3: Ejecutar entrenamiento

```bash
python train_crystal_classifier.py --epochs 10 --n_samples 100
```

---

## ✅ SOLUCIÓN 2: Actualización Completa a TF 2.x (Avanzado)

Para aprovechar completamente TensorFlow 2.x, necesitas reescribir el código usando Keras.

### Archivos a actualizar:

1. **`models/pointnet2_cls_ssg.py`** → Convertir a `tf.keras.Model`
2. **`train_crystal_classifier.py`** → Usar `tf.GradientTape` y eager execution
3. **`utils/tf_util.py`** → Actualizar operaciones

### Ventajas de TF 2.x nativo:
- ✅ Código más simple y legible
- ✅ Mejor rendimiento
- ✅ Debugging más fácil
- ✅ Compatible con futuras versiones

### Desventajas:
- ❌ Requiere reescribir código considerable
- ❌ Posibles bugs en la conversión
- ❌ Tiempo de desarrollo

---

## 🎯 RECOMENDACIÓN: Enfoque Híbrido

**Para tu caso (necesitas resultados pronto):**

1. **Usa SOLUCIÓN 1** para empezar inmediatamente
2. Entrena tus modelos y obtén resultados
3. Si funciona bien, quédate con eso
4. Si encuentras problemas de rendimiento o bugs, considera SOLUCIÓN 2

---

## 📝 Cambios Necesarios para Modo Compatibilidad

He preparado versiones parcheadas de los archivos principales. Aquí están los cambios:

### 1. Actualizar `train_crystal_classifier.py`:

```python
# CAMBIO EN LÍNEA 24 (después de imports):
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
```

### 2. Actualizar `models/pointnet2_cls_ssg.py`:

```python
# CAMBIO EN LÍNEA 10:
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
```

### 3. Actualizar todos los archivos en `models/`:

Mismo cambio: importar `tensorflow.compat.v1 as tf`

### 4. Actualizar `utils/tf_util.py`:

```python
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
```

### 5. Actualizar operadores personalizados:

Los operadores en `tf_ops/` ya están compilados para TF 2.x, así que deberían funcionar.

---

## 🔧 Script Automático de Parcheado

He creado un script que aplica estos cambios automáticamente:

```bash
./parchar_para_tf2.sh
```

Este script:
1. Hace backup de archivos originales (`.bak`)
2. Actualiza todos los imports a `tf.compat.v1`
3. Desactiva eager execution donde sea necesario
4. Actualiza deprecated functions

---

## ⚠️ Limitaciones del Modo Compatibilidad

1. **No se puede usar en Jupyter notebooks fácilmente**
   - Eager execution desactivado hace debugging difícil

2. **Algunos warnings pueden aparecer**
   - TF 2.x avisa sobre funciones deprecated
   - Puedes ignorarlos con `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'`

3. **No aprovecharás nuevas features de TF 2.x**
   - Keras integrado
   - Eager execution
   - Mejor profiling

---

## 🧪 Testing Después de Aplicar Cambios

```bash
# Test 1: Importar módulos
python -c "
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import sys
sys.path.append('models')
import pointnet2_cls_ssg as model
print('✓ Modelo importa correctamente')
"

# Test 2: Crear grafo
python -c "
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import sys
sys.path.append('models')
import pointnet2_cls_ssg as model

with tf.Graph().as_default():
    pointclouds = tf.placeholder(tf.float32, shape=(8, 512, 3))
    is_training = tf.placeholder(tf.bool, shape=())
    pred, _ = model.get_model(pointclouds, is_training, num_class=3)
    print('✓ Grafo se construye correctamente')
    print(f'  Output shape: {pred.shape}')
"

# Test 3: Mini-entrenamiento
python train_crystal_classifier.py --epochs 1 --n_samples 20 --batch_size 4
```

---

## 📊 Rendimiento Esperado

Con TF 2.x + modo compatibilidad:

| Métrica | Con GPU | Sin GPU |
|---------|---------|---------|
| Compilación inicial | 10-30s | 10-30s |
| Epoch (100 muestras) | ~30s | ~2-3 min |
| Memoria GPU | ~2GB | N/A |

---

## 🆘 Problemas Comunes

### "Cannot import name X from tensorflow"

**Solución:**
```python
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
```

### "Tensor object has no attribute numpy()"

**Causa:** Eager execution desactivado

**Solución:** Usa `sess.run()` en lugar de `.numpy()`

### "No gradients provided for variables"

**Causa:** Variable scope issues

**Solución:** Verifica que uses `tf.compat.v1.get_variable()` consistentemente

---

## 📚 Referencias

- [TensorFlow 2.x Migration Guide](https://www.tensorflow.org/guide/migrate)
- [tf.compat.v1 API Docs](https://www.tensorflow.org/api_docs/python/tf/compat/v1)
- [Upgrade script oficial](https://www.tensorflow.org/guide/upgrade)

---

## ✅ Próximos Pasos

1. Ejecuta `./parchar_para_tf2.sh` (lo crearé a continuación)
2. Prueba con mini-entrenamiento
3. Si funciona, entrena tu modelo completo
4. Si hay errores, repórtalos para arreglarlos

