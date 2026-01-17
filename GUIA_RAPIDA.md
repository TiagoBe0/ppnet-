# Guía Rápida: De Archivos LAMMPS .dump a Clasificación Cristalina

## ✅ Tu Formato de Archivo Está Soportado

Tu formato de LAMMPS dump:
```
ITEM: TIMESTEP
1000
ITEM: NUMBER OF ATOMS
12
ITEM: BOX BOUNDS abc origin pp pp pp
56.586619445 0.0 0.0 -0.0373097225
0.0 56.586619445 0.0 -0.0373097225
0.0 0.0 56.586619445 -0.0373097225
ITEM: ATOMS x y z
22.9296 26.4484 24.7898
...
```

**¡Funciona perfectamente!** ✓

El sistema detecta automáticamente:
- ✅ Formato `abc origin` con origen no-cero
- ✅ Columnas `x y z` (sin id ni type)
- ✅ Cajas triclinicas o ortogonales

---

## 🚀 Workflow Recomendado

### **Paso 1: Probar con un archivo**

Primero, verifica que tu formato específico funciona:

```bash
python test_user_format.py
```

Este script:
1. Crea un archivo con tu formato exacto
2. Lo lee y procesa
3. Calcula features cristalográficas
4. Lo convierte a OFF

Deberías ver: `✓ TODAS LAS PRUEBAS PASARON EXITOSAMENTE`

---

### **Paso 2: Convertir tus archivos en lote**

Tienes dos opciones según tu organización de datos:

#### **Opción A: Archivos sueltos (sin organizar por clase)**

Si tus archivos están todos juntos:

```bash
python batch_convert_dumps.py \
    --input_dir ./mis_dumps \
    --output_dir ./convertidos \
    --format off \
    --normalize
```

#### **Opción B: Archivos organizados por clase (RECOMENDADO)**

Si ya tienes tus archivos organizados por tipo cristalino:

```
mis_datos/
├── fcc/
│   ├── sim_001.dump
│   ├── sim_002.dump
│   └── ...
├── bcc/
│   ├── sim_001.dump
│   └── ...
└── hcp/
    └── sim_001.dump
```

Convierte manteniendo la estructura:

```bash
python batch_convert_dumps.py \
    --input_dir ./mis_datos \
    --output_dir ./procesados \
    --format off \
    --normalize \
    --keep_structure \
    --recursive
```

Resultado:
```
procesados/
├── fcc/
│   ├── sim_001.off
│   ├── sim_002.off
│   └── ...
├── bcc/
│   └── sim_001.off
└── hcp/
    └── sim_001.off
```

---

### **Paso 3: Entrenar el Modelo**

Una vez convertidos, entrena PointNet++:

#### **Con datos sintéticos (para empezar/probar)**

```bash
python train_crystal_classifier.py \
    --epochs 50 \
    --n_samples 500 \
    --crystal_types fcc bcc hcp \
    --batch_size 32
```

Esto genera datos automáticamente para aprender el sistema.

#### **Con TUS datos reales**

```bash
python train_crystal_classifier.py \
    --data_dir ./procesados \
    --num_points 512 \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.001
```

**Nota importante sobre `num_points`:**
- Tus archivos tienen 12 átomos (muy pequeño para cristales típicos)
- Si todos tus archivos tienen ~10-50 átomos, usa `--num_points 128`
- Para sistemas más grandes (100-1000 átomos), usa `--num_points 512` o `1024`

---

## 📊 Análisis de tus Estructuras

### Verificar número de coordinación

Antes de entrenar, es útil verificar que tus estructuras tienen sentido:

```python
from utils.lammps_reader import read_lammps_dump
from utils.crystal_features import CrystalFeatureCalculator

# Leer un archivo de ejemplo
positions = read_lammps_dump('mis_datos/fcc/ejemplo.dump')

# Analizar
calc = CrystalFeatureCalculator(positions, cutoff=5.0)
cn = calc.compute_coordination_numbers()

print(f"Número de átomos: {len(positions)}")
print(f"CN promedio: {cn.mean():.2f}")
print(f"CN por átomo: {cn}")
```

**Valores esperados de CN:**
- **FCC**: ~12 (primeros vecinos)
- **BCC**: ~8 (primeros vecinos)
- **HCP**: ~12 (como FCC)
- **Superficie/cluster**: Menor (6-10)

⚠️ **Si tu CN promedio es ~11-12**: Probablemente sean clusters pequeños, no cristales bulk periódicos. Esto está bien, pero ajusta el `cutoff` apropiadamente.

---

## 🔧 Ajuste del Parámetro `cutoff`

El `cutoff` define qué tan lejos buscar vecinos. Es **crítico** para calcular features correctamente.

### Cómo elegir el cutoff correcto:

1. **Mira tus datos**:
```python
from utils.lammps_reader import read_lammps_dump
import numpy as np

positions = read_lammps_dump('tu_archivo.dump')

# Calcular todas las distancias
from scipy.spatial.distance import pdist
distances = pdist(positions)

print(f"Distancia mínima: {distances.min():.3f} Å")
print(f"Percentil 10%: {np.percentile(distances, 10):.3f} Å")
print(f"Mediana: {np.median(distances):.3f} Å")
```

2. **Regla general**:
   - `cutoff = 1.5 × distancia_primer_vecino`
   - Para metales típicos: 3.5 - 5.0 Å
   - Para tu caso (12 átomos): Prueba con 4.0 - 6.0 Å

3. **Verificar**:
```python
calc = CrystalFeatureCalculator(positions, cutoff=5.0)
cn = calc.compute_coordination_numbers()
print(f"Con cutoff=5.0: CN promedio = {cn.mean():.2f}")

calc = CrystalFeatureCalculator(positions, cutoff=6.0)
cn = calc.compute_coordination_numbers()
print(f"Con cutoff=6.0: CN promedio = {cn.mean():.2f}")
```

---

## 💡 Tips Específicos para tu Caso

### 1. **Sistemas Pequeños (12 átomos)**

Tus estructuras son clusters pequeños, no cristales bulk:
- ✅ Usa `--num_points 64` o `128` (no 1024)
- ✅ Aumenta `cutoff` a 5-6 Å para capturar más vecinos
- ✅ El modelo puede necesitar más epochs (~100-200)

### 2. **Formato con Origen**

Tu formato tiene `origin` no-cero (-0.037):
- ✅ Automáticamente manejado por el código
- ✅ Las coordenadas se centran automáticamente
- ✅ No necesitas hacer nada especial

### 3. **Sin ID ni Type**

Tu formato solo tiene `x y z`:
- ✅ Suficiente para clasificación de estructura
- ✅ Si necesitas tipos atómicos, modifica tu script LAMMPS dump
- ✅ Para mezclas multicomponente, añade columna `type`

---

## 🎯 Caso de Uso Completo: Ejemplo Práctico

Imagina que tienes 100 archivos por clase:

```bash
# 1. Organizar archivos
mis_simulaciones/
├── fcc/  # 100 archivos .dump
├── bcc/  # 100 archivos .dump
└── hcp/  # 100 archivos .dump

# 2. Convertir todo (toma ~1 min)
python batch_convert_dumps.py \
    --input_dir ./mis_simulaciones \
    --output_dir ./datos_entrenamiento \
    --format off \
    --normalize \
    --keep_structure \
    --recursive

# 3. Verificar una muestra
python -c "
from utils.lammps_to_off import read_off
vertices, _ = read_off('datos_entrenamiento/fcc/sim_001.off')
print(f'Leídos {len(vertices)} vértices')
print(f'Rango: [{vertices.min():.3f}, {vertices.max():.3f}]')
"

# 4. Entrenar (GPU recomendada)
python train_crystal_classifier.py \
    --data_dir ./datos_entrenamiento \
    --num_points 128 \
    --batch_size 32 \
    --epochs 150 \
    --learning_rate 0.001 \
    --gpu 0

# 5. Ver progreso con TensorBoard (en otra terminal)
tensorboard --logdir=log
# Abre http://localhost:6006
```

Resultados esperados:
- **Train accuracy**: >95% después de 100 epochs
- **Val accuracy**: >85-90%
- **Mejor modelo**: guardado en `log/<timestamp>/best_model.ckpt`

---

## 🐛 Troubleshooting

### Error: "ValueError: No se encontraron columnas de posición"
- **Causa**: Tu formato es diferente al esperado
- **Solución**: Revisa que tu dump tenga `ITEM: ATOMS x y z`

### Accuracy muy baja (<50%)
- **Causa**: Estructuras muy pequeñas o cutoff incorrecto
- **Solución**:
  1. Ajusta `--num_points` al tamaño de tus sistemas
  2. Verifica CN con diferentes cutoffs
  3. Aumenta número de epochs

### "Out of memory"
- **Causa**: Batch size o num_points muy grande
- **Solución**: Reduce `--batch_size` a 8 o 16

### Las 3 clases tienen accuracy similar
- **Causa**: El modelo no puede distinguir las estructuras
- **Solución**:
  1. Verifica que las clases sean realmente diferentes (calcula CN)
  2. Aumenta `cutoff` para capturar más geometría local
  3. Usa más muestras de entrenamiento

---

## 📚 Archivos Útiles

| Archivo | Descripción |
|---------|-------------|
| `test_user_format.py` | Prueba tu formato específico |
| `batch_convert_dumps.py` | Convierte múltiples archivos |
| `train_crystal_classifier.py` | Entrena el modelo |
| `example_workflow.py` | Ejemplo completo de todo el sistema |
| `LAMMPS_CRYSTAL_CLASSIFICATION.md` | Tutorial detallado |

---

## ✉️ Próximos Pasos

1. ✅ **Ejecuta**: `python test_user_format.py` para verificar compatibilidad
2. ✅ **Convierte**: Usa `batch_convert_dumps.py` con tus archivos
3. ✅ **Analiza**: Calcula CN para entender tus estructuras
4. ✅ **Entrena**: Empieza con 50 epochs y datos sintéticos
5. ✅ **Ajusta**: Usa tus datos reales y optimiza hiperparámetros

---

**¿Preguntas?** Revisa `LAMMPS_CRYSTAL_CLASSIFICATION.md` para más detalles técnicos.
