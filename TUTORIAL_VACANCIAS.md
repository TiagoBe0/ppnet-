# Tutorial: Detección de Divacancias con PointNet++

## 🎯 Objetivo

Entrenar PointNet++ para **detectar el patrón topológico** que dejan las divacancias en una red cristalina, basándose en el entorno atómico (nube de puntos 3D).

## 🧠 Concepto

**Idea clave:** Igual que PointNet++ puede aprender a reconocer una silla a partir de su geometría 3D, puede aprender a reconocer el patrón geométrico específico que crean los átomos alrededor de una divacancia.

### ¿Qué es una Divacancia?

Una divacancia (2 átomos faltantes) crea un patrón característico:
- **18 átomos vecinos** rodeando el hueco
- Geometría local **distorsionada** comparada con cristal perfecto
- **Número de coordinación reducido** para átomos cercanos
- **Patrón angular específico** en la disposición de vecinos

---

## 📁 Estructura de Datos Recomendada

### Opción 1: Organización Simple (Todos los archivos juntos)

```
mis_datos/
├── sim_vac_2_001.dump    # Divacancia, muestra 1
├── sim_vac_2_002.dump    # Divacancia, muestra 2
├── sim_vac_2_003.dump    # ...
├── sim_vac_2_004.dump
...
├── sim_vac_2_100.dump
└── labels.csv            # Se genera automáticamente
```

### Opción 2: Organización por Carpetas (Si tienes múltiples tipos)

```
mis_datos/
├── divacancias/
│   ├── sim_001.dump
│   ├── sim_002.dump
│   └── ...
├── trivacancias/
│   ├── sim_001.dump
│   └── ...
└── sin_vacancias/
    └── sim_001.dump
```

---

## 🚀 Workflow Completo

### **Paso 1: Preparar tus Archivos .dump**

Cada archivo .dump debe contener **solo los átomos del entorno** de la vacancia:
- ✅ 18-20 átomos alrededor de la divacancia
- ✅ Nombrados con patrón: `*vac_2*` (donde 2 = número de vacancias)
- ✅ En formato LAMMPS con `x y z` o `id type x y z`

**Ejemplo de tu formato (perfecto):**
```
ITEM: TIMESTEP
4000
ITEM: NUMBER OF ATOMS
18
ITEM: BOX BOUNDS abc origin pp pp pp
56.5754413011 0.0 0.0 -0.0317206506
0.0 56.5754413011 0.0 -0.0317206506
0.0 0.0 56.5754413011 -0.0317206506
ITEM: ATOMS id type x y z c_peatom ...
8734 1 26.4539 30.0398 28.3112 ...
...
```

---

### **Paso 2: Convertir .dump a .off**

Convierte todos tus archivos a formato OFF:

```bash
python batch_convert_dumps.py \
    --input_dir ./mis_divacancias \
    --output_dir ./datos_convertidos \
    --format off \
    --normalize \
    --recursive
```

Resultado:
```
datos_convertidos/
├── sim_vac_2_001.off
├── sim_vac_2_002.off
├── sim_vac_2_003.off
...
```

---

### **Paso 3: Generar labels.csv**

Crea automáticamente el archivo CSV de etiquetas:

```bash
python create_labels_csv.py \
    --data_dir ./datos_convertidos \
    --output labels.csv \
    --extension .off
```

**Output: `labels.csv`**
```csv
filename,label,label_name
sim_vac_2_001.off,0,vac_2
sim_vac_2_002.off,0,vac_2
sim_vac_2_003.off,0,vac_2
...
```

**Contenido del CSV:**
- `filename`: Nombre del archivo (relativo a `data_dir`)
- `label`: Índice numérico de la clase (0, 1, 2, ...)
- `label_name`: Nombre descriptivo de la clase (vac_2, vac_10, etc.)

---

### **Paso 4: Entrenar el Modelo**

Entrena PointNet++ usando el CSV:

```bash
python train_crystal_classifier.py \
    --labels_csv ./labels.csv \
    --data_dir ./datos_convertidos \
    --num_points 32 \
    --batch_size 16 \
    --epochs 150 \
    --learning_rate 0.001 \
    --gpu 0
```

**Parámetros importantes:**
- `--num_points 32`: Usa 32 puntos (tus muestras tienen 18 átomos)
- `--batch_size 16`: Ajustar según tu GPU
- `--epochs 150`: Más epochs para datasets pequeños

**¿Por qué `num_points=32` y no 18?**
- El modelo hace **resampling** para tener un número fijo
- Si tienes <32 átomos, repite algunos aleatoriamente
- Si tienes >32 átomos, muestrea 32
- Usar potencias de 2 (16, 32, 64) es más eficiente

---

### **Paso 5: Monitorear el Entrenamiento**

En otra terminal, abre TensorBoard:

```bash
tensorboard --logdir=log
```

Abre http://localhost:6006 en tu navegador para ver:
- 📊 Loss (train y val)
- 📈 Accuracy en tiempo real
- 🎯 Learning rate decay

---

## 📊 Caso de Uso: Solo Divacancias vs Cristal Perfecto

Si quieres detectar si hay una divacancia o no (clasificación binaria):

### Preparación de Datos

```
datos/
├── con_divacancia/
│   ├── vac_2_001.dump  (18 átomos alrededor del defecto)
│   ├── vac_2_002.dump
│   └── ...
└── sin_vacancia/
    ├── perfecto_001.dump  (18 átomos de cristal perfecto)
    ├── perfecto_002.dump
    └── ...
```

### Comandos

```bash
# 1. Convertir
python batch_convert_dumps.py \
    --input_dir ./datos \
    --output_dir ./convertidos \
    --format off \
    --normalize \
    --keep_structure \
    --recursive

# 2. Generar CSV
python create_labels_csv.py \
    --data_dir ./convertidos \
    --output labels.csv \
    --recursive

# 3. Entrenar
python train_crystal_classifier.py \
    --labels_csv labels.csv \
    --data_dir ./convertidos \
    --num_points 32 \
    --batch_size 32 \
    --epochs 100
```

---

## 📊 Caso de Uso: Múltiples Tipos de Vacancias

Si quieres clasificar diferentes números de vacancias:

### Nombrar tus archivos

```
sim_vac_1_001.dump   → Monovacancia
sim_vac_2_001.dump   → Divacancia
sim_vac_3_001.dump   → Trivacancia
sim_vac_4_001.dump   → Cuatrivacancia
```

El script `create_labels_csv.py` **automáticamente** detectará el patrón `vac_N` y creará las clases:

```csv
filename,label,label_name
sim_vac_1_001.off,0,vac_1
sim_vac_2_001.off,1,vac_2
sim_vac_3_001.off,2,vac_3
sim_vac_4_001.off,3,vac_4
```

---

## 🔧 Ajuste de Parámetros

### **num_points**: Número de Puntos a Muestrear

Tu caso: 18 átomos por muestra

| num_points | Recomendado Si... | Pros | Contras |
|------------|-------------------|------|---------|
| **16** | Siempre tienes <16 átomos | Más rápido | Puede perder información |
| **32** | Tienes 10-30 átomos | Balance óptimo | ✅ **RECOMENDADO** |
| **64** | Tienes 30-100 átomos | Más información | Más lento |
| **128+** | Tienes >100 átomos | Máxima info | Lento, más memoria |

### **batch_size**: Tamaño del Batch

| batch_size | GPU VRAM | Velocidad | Estabilidad |
|------------|----------|-----------|-------------|
| **8** | 2-4 GB | Lento | Menos estable |
| **16** | 4-8 GB | Medio | ✅ **RECOMENDADO** |
| **32** | 8-12 GB | Rápido | Muy estable |
| **64** | 12+ GB | Muy rápido | Máxima estabilidad |

### **learning_rate**: Tasa de Aprendizaje

| LR | Caso de Uso | Comportamiento |
|----|-------------|----------------|
| **0.0001** | Dataset muy pequeño (<100 muestras) | Aprende lento pero seguro |
| **0.001** | Dataset normal (100-1000 muestras) | ✅ **RECOMENDADO** |
| **0.01** | Dataset grande (>1000 muestras) | Aprende rápido, puede ser inestable |

---

## 🎯 Resultados Esperados

### Con Dataset Balanceado (100+ muestras por clase)

| Métrica | Valor Esperado | Interpretación |
|---------|----------------|----------------|
| **Train Accuracy** | >95% | Modelo aprende los patrones |
| **Val Accuracy** | >85% | Generaliza bien |
| **Train Loss** | <0.3 | Convergencia buena |
| **Val Loss** | <0.5 | No hay overfitting severo |

### Señales de Problemas

| Señal | Causa Probable | Solución |
|-------|----------------|----------|
| Train Acc <70% | Modelo muy simple o LR muy alto | Reducir LR a 0.0001 |
| Val Acc <<< Train Acc | Overfitting | Más datos o más augmentación |
| Loss no baja | LR muy bajo | Aumentar LR a 0.001 |
| Accuracy ~50% (2 clases) | Modelo no aprende | Revisar datos/features |

---

## 💡 Tips para Mejorar Resultados

### 1. **Data Augmentation** (Ya incluida)

El modelo automáticamente aplica:
- ✅ Rotación aleatoria 3D
- ✅ Escalado (±10%)
- ✅ Jitter (ruido pequeño)

### 2. **Aumentar Datos con Variaciones**

Para cada divacancia, genera múltiples orientaciones:

```python
from utils.lammps_reader import read_lammps_dump
from utils.lammps_to_off import write_off, normalize_point_cloud
import numpy as np

# Leer original
positions = read_lammps_dump('divacancia_001.dump')

# Crear 10 rotaciones diferentes
for i in range(10):
    # Rotación aleatoria
    angle = np.random.uniform(0, 2*np.pi, 3)
    # ... aplicar rotación ...
    # Guardar como archivo nuevo
    write_off(f'divacancia_001_rot_{i}.off', positions_rotated)
```

### 3. **Analizar Features antes de Entrenar**

Verifica que tus clases sean distinguibles:

```python
from utils.crystal_features import CrystalFeatureCalculator
from utils.lammps_reader import read_lammps_dump

# Analizar una divacancia
pos_vac2 = read_lammps_dump('divacancia.dump')
calc_vac2 = CrystalFeatureCalculator(pos_vac2, cutoff=5.0)
cn_vac2 = calc_vac2.compute_coordination_numbers()

# Analizar cristal perfecto
pos_perfect = read_lammps_dump('perfecto.dump')
calc_perfect = CrystalFeatureCalculator(pos_perfect, cutoff=5.0)
cn_perfect = calc_perfect.compute_coordination_numbers()

print(f"CN divacancia: {cn_vac2.mean():.2f} ± {cn_vac2.std():.2f}")
print(f"CN perfecto:   {cn_perfect.mean():.2f} ± {cn_perfect.std():.2f}")
```

**Esperado:** CN de divacancia debe ser ~10-15% menor que perfecto

### 4. **Ajustar Cutoff Apropiadamente**

El `cutoff` define qué tan lejos buscar vecinos:

```python
# Probar diferentes cutoffs
for cutoff in [4.0, 5.0, 6.0, 7.0]:
    calc = CrystalFeatureCalculator(positions, cutoff=cutoff)
    cn = calc.compute_coordination_numbers()
    print(f"Cutoff {cutoff}: CN medio = {cn.mean():.2f}")
```

**Regla:** Usar cutoff que capture primeros y segundos vecinos

---

## 🐛 Troubleshooting

### Error: "filename not found"
```
❌ FileNotFoundError: ./datos_convertidos/sim_vac_2_001.off
```

**Causa:** Rutas en CSV no coinciden con ubicación real

**Solución:**
```bash
# Opción 1: Usar rutas relativas correctas
python create_labels_csv.py --data_dir ./datos_convertidos --output labels.csv

# Opción 2: Entrenar especificando data_dir
python train_crystal_classifier.py --labels_csv labels.csv --data_dir ./datos_convertidos
```

### Accuracy muy baja (<60%)

**Posibles causas:**
1. **Clases muy similares**: Si vac_2 y vac_3 se ven muy parecidos
   - ✅ Solución: Aumenta `num_points` a 64
   - ✅ Solución: Aumenta `cutoff` para capturar más geometría

2. **Dataset muy pequeño**: <50 muestras por clase
   - ✅ Solución: Genera más muestras con augmentación
   - ✅ Solución: Reduce número de clases

3. **Parámetros incorrectos**:
   - ✅ Solución: Usa `--num_points 32` (no 512)
   - ✅ Solución: Reduce `--learning_rate` a 0.0001

### "Out of Memory" en GPU

```bash
# Reducir batch_size
python train_crystal_classifier.py ... --batch_size 8

# O usar CPU (muy lento)
python train_crystal_classifier.py ... --gpu -1
```

---

## 📚 Archivos de Referencia

| Archivo | Para qué sirve |
|---------|----------------|
| `create_labels_csv.py` | Genera CSV de etiquetas automáticamente |
| `batch_convert_dumps.py` | Convierte múltiples .dump a .off |
| `train_crystal_classifier.py` | Entrena PointNet++ con tu dataset |
| `test_user_format.py` | Verifica que tu formato funciona |
| `crystal_dataset.py` | Clase para cargar datos (con CSV support) |

---

## ✅ Checklist Final

Antes de entrenar, verifica:

- [ ] Archivos .dump tienen patrón `vac_N` en el nombre
- [ ] Todos los archivos están convertidos a .off
- [ ] `labels.csv` generado correctamente
- [ ] CSV tiene 3 columnas: filename, label, label_name
- [ ] Rutas en CSV coinciden con ubicación de archivos
- [ ] Tienes al menos 50 muestras por clase (recomendado 100+)
- [ ] `num_points` ajustado al tamaño de tus muestras
- [ ] TensorFlow instalado con soporte para GPU (opcional pero recomendado)

---

## 🚀 Comando Completo (Copy-Paste)

```bash
# 1. Convertir todos los dumps
python batch_convert_dumps.py \
    --input_dir ./mis_divacancias \
    --output_dir ./convertidos \
    --format off \
    --normalize

# 2. Generar labels.csv
python create_labels_csv.py \
    --data_dir ./convertidos \
    --output labels.csv

# 3. Entrenar (GPU)
python train_crystal_classifier.py \
    --labels_csv labels.csv \
    --data_dir ./convertidos \
    --num_points 32 \
    --batch_size 16 \
    --epochs 150 \
    --learning_rate 0.001 \
    --log_dir ./log_divacancias

# 4. Monitorear
tensorboard --logdir=./log_divacancias
```

---

**¡Listo para entrenar!** 🎉

Si tienes preguntas o problemas, revisa:
- `GUIA_RAPIDA.md` - Para conceptos básicos
- `LAMMPS_CRYSTAL_CLASSIFICATION.md` - Para documentación técnica detallada
