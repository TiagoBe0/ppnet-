# Clasificación de Estructuras Cristalinas con PointNet++

## 📚 Tutorial Completo: De LAMMPS a Machine Learning

Este tutorial te enseña cómo usar **PointNet++** para detectar automáticamente patrones cristalinos (FCC, BCC, HCP, etc.) en simulaciones de dinámica molecular de LAMMPS.

---

## 🎯 Objetivo

**¿Es posible que una red neuronal detecte estructuras cristalinas?**

**¡Sí!** PointNet++ puede aprender a reconocer patrones geométricos locales y globales en nubes de puntos 3D. Las redes cristalinas tienen geometrías características:

| Estructura | Número Coordinación | Ángulos Característicos | Aplicaciones |
|------------|---------------------|------------------------|--------------|
| **FCC** | 12 | 60°, 90°, 120° | Cu, Au, Ag, Al, Ni |
| **BCC** | 8 + 6 | ~70.5°, 109.5° | Fe, Cr, W, Mo |
| **HCP** | 12 | Similar a FCC | Mg, Zn, Ti, Co |
| **Diamond** | 4 | 109.5° (tetraédrico) | C, Si, Ge |

---

## 🏗️ Arquitectura del Sistema

```
Simulación LAMMPS (.dump)
         ↓
  [Lectura y Conversión]
         ↓
   Nube de Puntos 3D (N × 3)
         ↓
  [Normalización]
         ↓
   PointNet++ (Deep Learning)
         ↓
  Clasificación: FCC, BCC, HCP, etc.
```

---

## 📁 Estructura del Código

```
ppnet-/
├── utils/
│   ├── lammps_reader.py          # Lector de archivos .dump de LAMMPS
│   ├── lammps_to_off.py          # Conversor LAMMPS → OFF/PLY
│   ├── crystal_generator.py      # Generador de estructuras sintéticas
│   └── crystal_features.py       # Cálculo de features cristalográficas
├── crystal_dataset.py             # Dataset class para PyTorch-style
├── train_crystal_classifier.py   # Script de entrenamiento
└── LAMMPS_CRYSTAL_CLASSIFICATION.md  # Este archivo
```

---

## 🚀 Guía de Uso Paso a Paso

### **Paso 1: Convertir archivo LAMMPS dump a formato OFF**

```python
from utils.lammps_to_off import convert_lammps_to_off

# Convertir archivo .dump a .off
stats = convert_lammps_to_off(
    dump_file='simulation.dump',
    off_file='output.off',
    normalize=True,
    norm_method='sphere'
)

print(f"Convertidos {stats['n_atoms']} átomos")
```

#### Ejemplo de archivo LAMMPS dump:

```
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
256
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 1.8075 1.8075 0.0
...
```

---

### **Paso 2: Generar Dataset Sintético (para entrenamiento)**

Si no tienes muchos datos reales, puedes generar estructuras cristalinas perfectas:

```python
from utils.crystal_generator import generate_crystal_dataset

# Generar 1000 muestras de FCC, BCC, HCP
point_clouds, labels = generate_crystal_dataset(
    n_samples_per_class=1000,
    crystal_types=['fcc', 'bcc', 'hcp'],
    add_noise=True,
    noise_level=0.1
)

print(f"Generadas {len(point_clouds)} muestras")
```

**¿Por qué funciona el dataset sintético?**

Las redes cristalinas tienen geometrías bien definidas. Al generar estructuras perfectas con:
- Variación en tamaño (2×2×2 hasta 5×5×5 celdas)
- Ruido térmico realista
- Defectos ocasionales (vacantes, intersticiales)

El modelo aprende las características geométricas esenciales de cada estructura.

---

### **Paso 3: Entrenar PointNet++**

#### Opción A: Con datos sintéticos (recomendado para empezar)

```bash
python train_crystal_classifier.py \
    --num_points 512 \
    --n_samples 1000 \
    --crystal_types fcc bcc hcp \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.001
```

#### Opción B: Con tus propios datos LAMMPS

1. Organiza tus archivos en esta estructura:

```
data/
├── fcc/
│   ├── sample1.off
│   ├── sample2.off
│   └── ...
├── bcc/
│   ├── sample1.off
│   └── ...
└── hcp/
    ├── sample1.off
    └── ...
```

2. Entrena:

```bash
python train_crystal_classifier.py \
    --data_dir ./data \
    --num_points 512 \
    --batch_size 32 \
    --epochs 100
```

---

### **Paso 4: Analizar Features Cristalográficas**

Antes o después del entrenamiento, puedes analizar las características cristalográficas:

```python
from utils.crystal_features import CrystalFeatureCalculator
from utils.lammps_reader import read_lammps_dump

# Leer posiciones
positions = read_lammps_dump('simulation.dump', center=True)

# Calcular features
calc = CrystalFeatureCalculator(positions, cutoff=4.0)

# Número de coordinación
cn = calc.compute_coordination_numbers()
print(f"CN promedio: {cn.mean():.2f}")

# Parámetro de orden Q4
q4 = calc.compute_steinhardt_q4()
print(f"Q4 promedio: {q4.mean():.4f}")

# Función de distribución radial
r, g_r = calc.compute_rdf(n_bins=100)

# Clasificación automática (basada en reglas)
structure = calc.classify_structure()
print(f"Estructura detectada: {structure}")
```

**Interpretación de Features:**

| Feature | FCC | BCC | HCP | Significado |
|---------|-----|-----|-----|-------------|
| CN (cutoff=4Å) | ~12 | ~8 | ~12 | Vecinos cercanos |
| Q4 | ~0.19 | ~0.04 | ~0.10 | Orden local |
| Ángulo principal | 60°, 90° | 70.5° | 60°, 90° | Geometría |

---

## 🧠 Cómo Funciona PointNet++

### Arquitectura para Clasificación

```
Input Point Cloud (B × N × 3)
      ↓
Set Abstraction Layer 1
  - Farthest Point Sampling: N → N/2
  - K-NN Grouping: radio = 0.2
  - PointNet: extraer features locales
      ↓ (B × N/2 × 128)
Set Abstraction Layer 2
  - FPS: N/2 → N/4
  - K-NN: radio = 0.4
  - PointNet: features más abstractas
      ↓ (B × N/4 × 256)
Global Aggregation
  - Max pooling: N/4 → 1
      ↓ (B × 1 × 1024)
Fully Connected Layers
  - FC(512) + ReLU + Dropout
  - FC(256) + ReLU + Dropout
  - FC(num_classes)
      ↓
Softmax → Predicción (B × num_classes)
```

### ¿Por qué funciona para cristales?

1. **Invariancia a permutaciones**: El orden de los átomos no importa
2. **Invariancia a transformaciones**: Aprende features geométricas robustas
3. **Features jerárquicas**:
   - Nivel bajo: ángulos de enlace, distancias
   - Nivel medio: motivos de coordinación local
   - Nivel alto: estructura global

---

## 📊 Resultados Esperados

Con el dataset sintético, deberías obtener:

| Métrica | Valor Esperado |
|---------|----------------|
| **Train Accuracy** | > 95% |
| **Val Accuracy** | > 90% |
| **Convergencia** | ~30-50 epochs |

### Matriz de Confusión Típica (después de 50 epochs):

```
           Predicted
         FCC  BCC  HCP
Actual
FCC      95%   2%   3%
BCC       1%  97%   2%
HCP       4%   1%  95%
```

**Nota:** FCC y HCP son más difíciles de distinguir porque ambos tienen CN=12.

---

## 🔬 Experimentos Avanzados

### 1. Detección de Defectos

```python
from utils.crystal_generator import CrystalGenerator

gen = CrystalGenerator(lattice_constant=3.615)

# FCC perfecto
fcc_perfect = gen.generate_fcc(5, 5, 5)

# FCC con vacantes (5%)
fcc_vacancies = gen.add_vacancies(fcc_perfect, vacancy_fraction=0.05)

# FCC con intersticiales
fcc_interstitials = gen.add_interstitials(fcc_perfect, interstitial_fraction=0.02)

# Entrenar modelo para clasificar: perfecto vs. defectuoso
```

### 2. Detección de Bordes de Grano

```python
# Crear estructura con borde de grano
grain1 = gen.generate_fcc(4, 4, 4)
grain2 = gen.generate_fcc(4, 4, 4)
gb_structure = gen.create_grain_boundary(grain1, grain2, angle=30)

# Analizar
calc = CrystalFeatureCalculator(gb_structure, cutoff=4.0)
cn = calc.compute_coordination_numbers()

# Los átomos en el borde de grano tendrán CN reducido
print(f"CN mínimo (borde): {cn.min()}")
print(f"CN máximo (bulk): {cn.max()}")
```

### 3. Clasificación Multinivel

Entrena un clasificador jerárquico:

1. **Nivel 1**: Estructura general (FCC, BCC, HCP)
2. **Nivel 2**: Calidad (perfecto, defectos, amorfo)
3. **Nivel 3**: Tipo de defecto (vacante, intersticial, dislocación)

---

## 🛠️ API Completa

### Módulo: `lammps_reader.py`

```python
from utils.lammps_reader import LAMMPSDumpReader, read_lammps_dump

# Opción 1: Lectura rápida
positions = read_lammps_dump('file.dump', center=True)

# Opción 2: Lectura completa
reader = LAMMPSDumpReader('file.dump')
data = reader.read()
positions = reader.get_positions()
types = reader.get_types()
box_size = reader.get_box_size()
```

### Módulo: `lammps_to_off.py`

```python
from utils.lammps_to_off import (
    convert_lammps_to_off,
    convert_lammps_to_ply_with_features,
    write_off,
    read_off,
    normalize_point_cloud
)

# Conversión básica
convert_lammps_to_off('in.dump', 'out.off')

# Con features (colorea por CN)
convert_lammps_to_ply_with_features('in.dump', 'out.ply', cutoff=4.0)

# Normalización manual
positions_norm = normalize_point_cloud(positions, method='sphere')
```

### Módulo: `crystal_generator.py`

```python
from utils.crystal_generator import CrystalGenerator

gen = CrystalGenerator(lattice_constant=3.615)

# Generar estructuras
fcc = gen.generate_fcc(nx=3, ny=3, nz=3)
bcc = gen.generate_bcc(nx=3, ny=3, nz=3)
hcp = gen.generate_hcp(nx=3, ny=3, nz=2)
diamond = gen.generate_diamond(nx=2, ny=2, nz=2)

# Añadir efectos realistas
fcc_noisy = gen.add_thermal_noise(fcc, temperature=300.0)
fcc_defects = gen.add_vacancies(fcc, vacancy_fraction=0.05)
```

### Módulo: `crystal_features.py`

```python
from utils.crystal_features import CrystalFeatureCalculator, compute_all_features

calc = CrystalFeatureCalculator(positions, cutoff=4.0)

# Features individuales
cn = calc.compute_coordination_numbers()
q4 = calc.compute_steinhardt_q4()
r, g_r = calc.compute_rdf(n_bins=100)
angles = calc.compute_bond_angles(atom_idx=0)

# Vector de features completo
features = calc.compute_feature_vector(n_angle_bins=36)

# Clasificación automática
structure_type = calc.classify_structure()
```

### Módulo: `crystal_dataset.py`

```python
from crystal_dataset import CrystalDataset

# Crear dataset
dataset = CrystalDataset(
    num_points=1024,
    split='train',
    data_augmentation=True
)

# Opción 1: Generar sintético
dataset.generate_synthetic(
    n_samples_per_class=1000,
    crystal_types=['fcc', 'bcc', 'hcp']
)

# Opción 2: Cargar desde archivos
dataset.load_from_directory('./data', file_extension='.off')

# Uso
batch_data, batch_labels = dataset.get_batch(batch_size=32)
train_ds, val_ds = dataset.split_train_val(val_fraction=0.2)
```

---

## 💡 Tips y Mejores Prácticas

### 1. **Tamaño de Sistema**

- **Mínimo**: 100-200 átomos (3×3×3 celdas)
- **Óptimo**: 500-1000 átomos (5×5×5 celdas)
- **Máximo**: Limitado por memoria (~10,000 átomos)

### 2. **Número de Puntos (num_points)**

- Para cristales: 256-512 puntos suelen ser suficientes
- Más puntos = más precisión pero más lento
- Usa muestreo FPS (Farthest Point Sampling) para representatividad

### 3. **Cutoff para Vecindad**

| Material | a (Å) | Cutoff Recomendado |
|----------|-------|-------------------|
| Cu (FCC) | 3.615 | 3.5 - 4.5 Å |
| Fe (BCC) | 2.866 | 3.0 - 4.0 Å |
| Mg (HCP) | 3.209 | 3.5 - 4.5 Å |

**Regla general**: cutoff ≈ 1.2 × (primer mínimo de g(r))

### 4. **Augmentación de Datos**

```python
# En train_crystal_classifier.py, ya está implementado:
- Rotación aleatoria (todas las direcciones)
- Escalado (±10%)
- Jitter (ruido pequeño σ=0.02)
```

### 5. **Combinar con Features Tradicionales**

Para mejor rendimiento, concatena:
- Features aprendidas (PointNet++)
- Features cristalográficas (CN, Q4, Q6)

```python
# Ejemplo conceptual
pointnet_features = model.extract_features(point_cloud)  # (1024,)
crystal_features = calc.compute_feature_vector()  # (40,)
combined = np.concatenate([pointnet_features, crystal_features])
```

---

## 🐛 Troubleshooting

### Error: "No module named 'plyfile'"

```bash
pip install plyfile
```

### Error: "GPU out of memory"

Reduce `batch_size` o `num_points`:

```bash
python train_crystal_classifier.py --batch_size 8 --num_points 256
```

### Warning: "Low training accuracy"

1. Verifica que las estructuras estén normalizadas
2. Aumenta el número de epochs
3. Reduce learning rate
4. Aumenta tamaño del dataset

### Accuracy alta en train, baja en val

Overfitting. Soluciones:
- Más augmentación de datos
- Más dropout
- Más datos de entrenamiento
- Regularización L2

---

## 📚 Referencias

### Papers

1. **PointNet++** (Qi et al., 2017)
   - "PointNet++: Deep Hierarchical Feature Learning on Point Sets"
   - [arXiv:1706.02413](https://arxiv.org/abs/1706.02413)

2. **Steinhardt Parameters** (Steinhardt et al., 1983)
   - "Bond-orientational order in liquids and glasses"
   - Physical Review B, 28(2), 784

3. **Common Neighbor Analysis** (Honeycutt & Andersen, 1987)
   - "Molecular dynamics study of melting and freezing"
   - J. Phys. Chem., 91(19), 4950-4963

### Software

- **LAMMPS**: https://www.lammps.org
- **OVITO**: https://www.ovito.org (visualización y análisis)
- **freud**: https://freud.readthedocs.io (análisis cristalográfico en Python)

---

## 🎓 Ejercicios Propuestos

### Básico

1. Genera un dataset de 100 muestras por clase y entrena por 20 epochs
2. Visualiza la matriz de confusión
3. Calcula CN para una estructura FCC y verifica que sea ~12

### Intermedio

4. Implementa detección de estructuras amorfas vs. cristalinas
5. Añade clasificación de estructuras SC (Simple Cubic)
6. Experimenta con diferentes valores de `cutoff` y observa el efecto en CN

### Avanzado

7. Implementa clasificación de fases sólido/líquido basada en Q6
8. Detecta automáticamente bordes de grano en policristales
9. Clasifica el tipo de dislocación (edge vs. screw) basándote en CN local
10. Combina PointNet++ con features de CNA para mejorar accuracy

---

## 👨‍💻 Contribuciones

Este código es educativo y está diseñado para aprendizaje. Siéntete libre de:

- Experimentar con diferentes arquitecturas
- Añadir nuevas estructuras cristalinas
- Mejorar los calculadores de features
- Compartir tus resultados

---

## 📄 Licencia

Este tutorial es de código abierto y gratuito para uso educativo y de investigación.

---

## 🙏 Agradecimientos

- Implementación basada en PointNet++ de Qi et al.
- Agradecimientos a la comunidad de LAMMPS
- Inspirado en trabajos de machine learning para ciencia de materiales

---

**¡Feliz clasificación cristalina! 🔬✨**

*¿Preguntas? Revisa los ejemplos en cada módulo o consulta los docstrings.*
