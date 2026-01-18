# 👋 LEE ESTO PRIMERO - Instalación de PointNet++ en Ubuntu 22.04

## 🚨 IMPORTANTE: Problemas de Compatibilidad TensorFlow

Este proyecto fue originalmente escrito para **TensorFlow 1.2** (2017) y **Python 2.7**.
Si estás en **Ubuntu 22.04** con **Python 3.11**, necesitas seguir esta guía para que funcione.

---

## ⚡ INICIO RÁPIDO (5 minutos)

### Opción 1: Instalación Automática Completa ⭐ RECOMENDADA

```bash
# 1. Ve al directorio del proyecto
cd ~/Escritorio/pnet++enero/ppnet-

# 2. Ejecuta el instalador
./instalar_tensorflow.sh
# Selecciona opción 1 (Instalación completa)

# 3. Parcha el código para TensorFlow 2.x
./parchar_para_tf2.sh

# 4. Activa el entorno virtual
source venv_pointnet/bin/activate

# 5. Prueba que funciona
python train_crystal_classifier.py --epochs 2 --n_samples 50 --batch_size 8
```

**Si ves progreso de entrenamiento → ¡Ya está!** 🎉

---

### Opción 2: Ya tengo TensorFlow, solo necesito parchar el código

```bash
./parchar_para_tf2.sh
```

---

## 📖 GUÍAS DETALLADAS

Tenemos documentación completa para cada escenario:

### 🔧 **Instalación y Configuración:**

| Archivo | Para qué es |
|---------|-------------|
| **[INSTALACION_COMPLETA_DESDE_CERO.md](INSTALACION_COMPLETA_DESDE_CERO.md)** | 📌 **EMPIEZA AQUÍ** - Guía completa paso a paso desde Ubuntu limpio |
| [INSTALACION_TENSORFLOW_UBUNTU_22.04.md](INSTALACION_TENSORFLOW_UBUNTU_22.04.md) | Guía detallada solo de instalación de TensorFlow |
| [COMPATIBILIDAD_TF2.md](COMPATIBILIDAD_TF2.md) | Explicación técnica de problemas TF 1.x vs 2.x |
| [INICIO_RAPIDO.md](INICIO_RAPIDO.md) | Comandos rápidos si ya tienes todo instalado |

### 📊 **Uso del Sistema:**

| Archivo | Para qué es |
|---------|-------------|
| [GUIA_RAPIDA.md](GUIA_RAPIDA.md) | Workflow completo: datos → entrenamiento → resultados |
| [LAMMPS_CRYSTAL_CLASSIFICATION.md](LAMMPS_CRYSTAL_CLASSIFICATION.md) | Tutorial detallado de clasificación de cristales |
| [TUTORIAL_VACANCIAS.md](TUTORIAL_VACANCIAS.md) | Tutorial específico para detección de vacancias |

### 🛠️ **Scripts y Herramientas:**

| Script | Descripción |
|--------|-------------|
| `instalar_tensorflow.sh` | Instalador automático completo |
| `parchar_para_tf2.sh` | Actualiza código a TensorFlow 2.x |
| `verificar_sistema.sh` | Diagnóstico completo del sistema |
| `compile_tf_ops.sh` | Compila operadores TensorFlow |

---

## 🎯 ¿Qué Tengo que Hacer?

### Si eres NUEVO en el proyecto:

1. Lee: [INSTALACION_COMPLETA_DESDE_CERO.md](INSTALACION_COMPLETA_DESDE_CERO.md)
2. Ejecuta: `./instalar_tensorflow.sh` (opción 1)
3. Ejecuta: `./parchar_para_tf2.sh`
4. Lee: [GUIA_RAPIDA.md](GUIA_RAPIDA.md) para aprender a usar el sistema

### Si ya instalaste TensorFlow pero tienes ERRORES:

1. Ejecuta: `./parchar_para_tf2.sh`
2. Prueba: `python train_crystal_classifier.py --epochs 1 --n_samples 10`
3. Si falla, lee: [COMPATIBILIDAD_TF2.md](COMPATIBILIDAD_TF2.md)

### Si tienes GPU pero no la detecta:

1. Ejecuta: `./verificar_sistema.sh`
2. Sigue las recomendaciones del script
3. Reinstala TensorFlow: `pip install tensorflow[and-cuda]==2.15.0`

### Si solo quieres ENTRENAR con tus datos:

1. Asegúrate de tener el entorno activado: `source venv_pointnet/bin/activate`
2. Convierte tus datos: `python batch_convert_dumps.py --input_dir ./datos --output_dir ./procesados`
3. Entrena: `python train_crystal_classifier.py --data_dir ./procesados --epochs 100`

---

## ❓ Preguntas Frecuentes

### ¿Por qué tantos scripts de instalación?

El código original es de 2017 y usa TensorFlow 1.2. TensorFlow cambió completamente en la versión 2.0 (2019). Los scripts solucionan estos problemas de compatibilidad.

### ¿Puedo usar TensorFlow 1.x en su lugar?

No. TensorFlow 1.x ya no es soportado y no se puede instalar en Python 3.11.

### ¿Necesito GPU?

No es obligatorio, pero ALTAMENTE recomendado. Sin GPU, el entrenamiento puede tomar 10-20x más tiempo.

### ¿Funciona en Windows?

Este proyecto está diseñado para Linux. En Windows necesitarás WSL2 (Windows Subsystem for Linux).

### ¿Funciona en Mac?

Puede funcionar en Mac Intel con algunas modificaciones. Mac M1/M2 requiere configuración especial para TensorFlow.

---

## 🐛 Solución de Problemas

### "nvidia-smi: command not found"

→ Lee la sección de instalación de drivers en [INSTALACION_COMPLETA_DESDE_CERO.md](INSTALACION_COMPLETA_DESDE_CERO.md)

### "ModuleNotFoundError: No module named 'tensorflow'"

→ Activa el entorno: `source venv_pointnet/bin/activate`

### "AttributeError: module 'tensorflow' has no attribute 'Session'"

→ Ejecuta: `./parchar_para_tf2.sh`

### TensorFlow no detecta mi GPU

→ Ejecuta: `./verificar_sistema.sh` y sigue las recomendaciones

### Compilación de operadores falla

→ Lee: [INSTRUCCIONES_COMPILACION.md](INSTRUCCIONES_COMPILACION.md)

### Para cualquier otro error:

1. Ejecuta: `./verificar_sistema.sh > diagnostico.txt`
2. Comparte el archivo `diagnostico.txt`

---

## 📁 Estructura del Proyecto

```
ppnet-/
├── README.md                              # Documentación original de PointNet++
├── LEEME_PRIMERO.md                       # 👈 ESTE ARCHIVO
├── INSTALACION_COMPLETA_DESDE_CERO.md    # 📌 Guía principal de instalación
├── GUIA_RAPIDA.md                         # Workflow de uso
│
├── instalar_tensorflow.sh                 # 🚀 Instalador automático
├── parchar_para_tf2.sh                    # 🔧 Parchear código para TF 2.x
├── verificar_sistema.sh                   # 🔍 Diagnóstico del sistema
│
├── train_crystal_classifier.py            # Script principal de entrenamiento
├── batch_convert_dumps.py                 # Convertir datos LAMMPS
├── create_labels_csv.py                   # Crear CSV con etiquetas
│
├── models/                                # Modelos PointNet++
├── tf_ops/                                # Operadores TensorFlow personalizados
├── utils/                                 # Utilidades
└── data/                                  # Tus datos
```

---

## ✅ Checklist de Verificación

Antes de empezar a usar el sistema, verifica:

- [ ] Leí [INSTALACION_COMPLETA_DESDE_CERO.md](INSTALACION_COMPLETA_DESDE_CERO.md)
- [ ] Ejecuté `./instalar_tensorflow.sh` exitosamente
- [ ] Ejecuté `./parchar_para_tf2.sh` exitosamente
- [ ] `source venv_pointnet/bin/activate` activa el entorno
- [ ] `nvidia-smi` muestra mi GPU (si tengo GPU)
- [ ] El test de entrenamiento funciona sin errores
- [ ] Leí [GUIA_RAPIDA.md](GUIA_RAPIDA.md) para entender el workflow

---

## 🎓 Próximos Pasos

Una vez que tengas todo instalado y funcionando:

1. **Lee el tutorial completo:** [LAMMPS_CRYSTAL_CLASSIFICATION.md](LAMMPS_CRYSTAL_CLASSIFICATION.md)
2. **Convierte tus datos:** Usa `batch_convert_dumps.py`
3. **Entrena tu primer modelo:** Con datos sintéticos para aprender
4. **Entrena con tus datos reales:** Ajusta hiperparámetros según necesites
5. **Monitorea resultados:** Usa TensorBoard para ver progreso

---

## 🆘 ¿Necesitas Ayuda?

1. **Primero**: Lee la documentación correspondiente arriba
2. **Segundo**: Ejecuta `./verificar_sistema.sh` para diagnóstico
3. **Tercero**: Revisa la sección de solución de problemas en las guías
4. **Último recurso**: Comparte los archivos de diagnóstico

---

**¡Buena suerte con tu proyecto de clasificación de cristales! 🚀💎**

_Documentación actualizada para Ubuntu 22.04 + Python 3.11 + TensorFlow 2.15_
