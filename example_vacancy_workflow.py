"""
Ejemplo Completo: Workflow para Detección de Divacancias
==========================================================

Este script demuestra el workflow completo desde archivos .dump
hasta el entrenamiento con CSV de etiquetas.

Ejecuta: python example_vacancy_workflow.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from lammps_to_off import convert_lammps_to_off, write_off
from crystal_features import CrystalFeatureCalculator


def create_example_divacancy_dump():
    """Crea un archivo dump de ejemplo con 18 átomos (divacancia)."""

    dump_content = """ITEM: TIMESTEP
4000
ITEM: NUMBER OF ATOMS
18
ITEM: BOX BOUNDS abc origin pp pp pp
56.5754413011 0.0 0.0 -0.0317206506
0.0 56.5754413011 0.0 -0.0317206506
0.0 0.0 56.5754413011 -0.0317206506
ITEM: ATOMS id type x y z c_peatom c_satom1 c_satom2 c_satom3 c_satom4 c_satom5 c_satom6
8734 1 26.4539 30.0398 28.3112 -4.29746 4.66717e05 -7.64882e05 -1.20518e05 -2.22975e05 -1.88624e05 25791.5
8735 1 26.4575 28.2783 29.985 -4.31738 1.27496e05 2.84744e05 -1.34742e05 2446.0 2.37186e05 5.24317e05
8737 1 28.257 28.3255 28.3357 -4.2958 -2.24009e05 74839.6 -2.11207e05 -4.90752e05 2.48662e05 6.97141e05
8738 1 30.0088 29.908 28.2472 -4.12351 -1.03961e06 -96706.6 -3.91977e05 1.45179e05 -3.67949e05 -1.02025e05
8739 1 30.1414 28.3376 30.1313 -4.26182 4.33398e05 5.18871e05 1.31743e05 3.24876e05 2.32556e05 96007.0
8744 1 31.8415 30.1599 30.0155 -4.27352 5.70352e05 5.30761e05 1.10147e06 -4.37524e05 -1.3585e05 20247.8
8799 1 26.3959 31.7619 29.9951 -4.28429 1.87637e05 2.26868e05 -1.13377e05 81330.7 1.59206e05 23613.7
8801 1 28.2999 31.616 28.3686 -4.17023 5.66784e05 1.13843e05 9.67314e05 -1.27114e05 1.49557e05 -1.03605e06
8802 1 29.963 33.491 28.1893 -4.2927 -1.83849e05 40333.3 -78660.4 4.31748e05 3.7045e05 -31337.4
8804 1 28.2466 33.5239 29.9018 -4.27486 2.02145e05 -1.29565e05 98104.8 1.31476e05 -53367.2 2.45968e05
8805 1 31.7612 31.8548 28.1985 -4.29247 6.76469e05 2.19301e05 -15988.2 3.94726e05 -1.27376e05 5.76692e05
8808 1 31.8698 33.5918 29.9427 -4.28937 -2.98689e05 -3.55542e05 6.5022e05 6.32836e05 1.4521e05 -1.29177e05
9758 1 26.5774 30.0579 31.8025 -4.30823 8.94521e05 8.3368e05 5.23589e05 4.96977e05 11699.0 4.09632e05
9761 1 28.3209 28.3801 31.6993 -4.28421 -2.32372e05 1.03116e06 1.02227e06 65800.9 1.1735e05 -1.71939e05
9762 1 30.0547 30.1146 31.8501 -4.16491 -2.74444e05 -7.03549e05 5.93754e05 -22408.0 47881.0 -2.20158e05
9825 1 28.395 31.8538 31.7272 -4.16167 2.09584e05 -3.90892e05 1.11076e06 3475.35 -1.91862e05 2.63592e05
9826 1 30.023 33.535 31.6353 -4.25642 1.13132e05 -2.99357e05 8.24523e05 -2.24998e05 -1.09861e05 7.31004e05
9829 1 31.7753 31.7796 31.8398 -4.29924 7.44149e05 2.08708e05 5.76253e05 -66059.5 2.07943e05 -7.23215e05
"""

    return dump_content


def print_section(title):
    """Imprime un encabezado de sección."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def main():
    print_section("EJEMPLO: WORKFLOW DE DETECCIÓN DE DIVACANCIAS")

    # =========================================================================
    # PASO 1: Crear archivos de ejemplo
    # =========================================================================
    print_section("PASO 1: Crear Archivos de Ejemplo")

    os.makedirs('/tmp/vacancy_example', exist_ok=True)

    # Crear múltiples archivos de divacancia
    print("Creando archivos .dump de ejemplo...")

    for i in range(1, 6):
        filename = f'/tmp/vacancy_example/sim_vac_2_{i:03d}.dump'
        with open(filename, 'w') as f:
            f.write(create_example_divacancy_dump())
        print(f"  ✓ Creado: {filename}")

    # Crear archivos de monovacancia (simulados con menos átomos)
    print("\nCreando archivos de monovacancia (simulados)...")
    # Para el ejemplo, usamos el mismo pero con diferente nombre
    for i in range(1, 4):
        filename = f'/tmp/vacancy_example/sim_vac_1_{i:03d}.dump'
        with open(filename, 'w') as f:
            # Simular monovacancia (en realidad sería diferente)
            f.write(create_example_divacancy_dump())
        print(f"  ✓ Creado: {filename}")

    print(f"\n✓ Total: 8 archivos .dump creados")

    # =========================================================================
    # PASO 2: Convertir a OFF
    # =========================================================================
    print_section("PASO 2: Convertir .dump a .off")

    from batch_convert_dumps import batch_convert

    batch_convert(
        input_dir='/tmp/vacancy_example',
        output_dir='/tmp/vacancy_example/off',
        normalize=True,
        norm_method='sphere',
        output_format='off',
        cutoff=5.0,
        keep_structure=False,
        recursive=False
    )

    # =========================================================================
    # PASO 3: Analizar Features Cristalográficas
    # =========================================================================
    print_section("PASO 3: Analizar Features Cristalográficas")

    from lammps_reader import read_lammps_dump

    # Analizar un archivo
    print("Analizando una divacancia de ejemplo...")
    positions = read_lammps_dump('/tmp/vacancy_example/sim_vac_2_001.dump')

    print(f"  Número de átomos: {len(positions)}")

    # Calcular features
    calc = CrystalFeatureCalculator(positions, cutoff=5.0)
    cn = calc.compute_coordination_numbers()

    print(f"\n  Número de Coordinación:")
    print(f"    Promedio: {cn.mean():.2f}")
    print(f"    Std Dev:  {cn.std():.2f}")
    print(f"    Mínimo:   {cn.min():.0f}")
    print(f"    Máximo:   {cn.max():.0f}")

    print(f"\n  Distribución de CN:")
    unique_cn, counts = np.unique(cn, return_counts=True)
    for cn_val, count in zip(unique_cn, counts):
        bar = '█' * int(count / counts.max() * 20)
        print(f"    CN={int(cn_val)}: {bar} ({count} átomos)")

    # =========================================================================
    # PASO 4: Generar labels.csv
    # =========================================================================
    print_section("PASO 4: Generar labels.csv")

    from create_labels_csv import create_labels_csv

    create_labels_csv(
        data_dir='/tmp/vacancy_example/off',
        output_file='/tmp/vacancy_example/labels.csv',
        file_extension='.off',
        recursive=False,
        relative_paths=True
    )

    # Mostrar contenido del CSV
    print("\nContenido de labels.csv:")
    with open('/tmp/vacancy_example/labels.csv', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(f"  {line.rstrip()}")
            if i >= 10:  # Mostrar solo primeras 10 líneas
                print(f"  ... y {len(lines) - 11} líneas más")
                break

    # =========================================================================
    # PASO 5: Cargar Dataset
    # =========================================================================
    print_section("PASO 5: Cargar Dataset con CSV")

    sys.path.insert(0, os.path.dirname(__file__))
    from crystal_dataset import CrystalDataset

    dataset = CrystalDataset(
        num_points=32,
        split='train',
        data_augmentation=True,
        normalize=True
    )

    dataset.load_from_csv(
        labels_csv='/tmp/vacancy_example/labels.csv',
        data_dir='/tmp/vacancy_example/off',
        file_extension='.off'
    )

    # Verificar dataset
    print(f"\nVerificación del dataset:")
    print(f"  Total de muestras: {len(dataset)}")
    print(f"  Clases: {dataset.class_names}")

    # Obtener una muestra
    points, label = dataset[0]
    print(f"\n  Muestra de ejemplo:")
    print(f"    Shape: {points.shape}")
    print(f"    Label: {label} ({dataset.class_names[label]})")
    print(f"    Rango: [{points.min():.3f}, {points.max():.3f}]")

    # =========================================================================
    # PASO 6: Preparar para Entrenamiento
    # =========================================================================
    print_section("PASO 6: Preparar para Entrenamiento")

    train_ds, val_ds = dataset.split_train_val(val_fraction=0.2)

    print(f"Dataset dividido:")
    print(f"  Train: {len(train_ds)} muestras")
    print(f"  Val:   {len(val_ds)} muestras")

    # Obtener un batch
    batch_data, batch_labels = train_ds.get_batch(batch_size=4)
    print(f"\nBatch de ejemplo:")
    print(f"  Shape: {batch_data.shape}")
    print(f"  Labels: {batch_labels}")

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print_section("RESUMEN Y PRÓXIMOS PASOS")

    print("✓ Workflow completado exitosamente!\n")

    print("Archivos generados:")
    print("  📁 /tmp/vacancy_example/")
    print("     ├── sim_vac_1_*.dump  (3 archivos)")
    print("     ├── sim_vac_2_*.dump  (5 archivos)")
    print("     ├── off/")
    print("     │   └── sim_vac_*.off (8 archivos convertidos)")
    print("     └── labels.csv")

    print("\nPróximos pasos:\n")

    print("1. Para entrenar con estos datos de ejemplo:")
    print("   python train_crystal_classifier.py \\")
    print("       --labels_csv /tmp/vacancy_example/labels.csv \\")
    print("       --data_dir /tmp/vacancy_example/off \\")
    print("       --num_points 32 \\")
    print("       --batch_size 4 \\")
    print("       --epochs 20\n")

    print("2. Para usar tus propios datos:")
    print("   a. Coloca tus archivos .dump con nombres: sim_vac_2_*.dump")
    print("   b. Ejecuta: python batch_convert_dumps.py --input_dir ./tus_datos --output_dir ./off")
    print("   c. Ejecuta: python create_labels_csv.py --data_dir ./off --output labels.csv")
    print("   d. Entrena: python train_crystal_classifier.py --labels_csv labels.csv\n")

    print("3. Monitorear entrenamiento:")
    print("   tensorboard --logdir=log\n")

    print("📚 Documentación:")
    print("   - TUTORIAL_VACANCIAS.md: Tutorial completo")
    print("   - GUIA_RAPIDA.md: Guía rápida de referencia")


if __name__ == '__main__':
    main()
