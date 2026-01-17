"""
Ejemplo Completo: Pipeline de Clasificación de Estructuras Cristalinas
=======================================================================

Este script demuestra todo el workflow desde la generación de datos
hasta el análisis de resultados.

Ejecuta este script para ver todo el sistema en acción:
    python example_workflow.py
"""

import numpy as np
import os
import sys

# Añadir paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from crystal_generator import CrystalGenerator, generate_crystal_dataset
from lammps_to_off import (
    convert_lammps_to_off,
    convert_lammps_to_ply_with_features,
    write_off,
    normalize_point_cloud
)
from crystal_features import CrystalFeatureCalculator, compute_all_features


def print_section(title):
    """Imprime un encabezado de sección."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def example_1_generate_structures():
    """Ejemplo 1: Generar estructuras cristalinas."""
    print_section("EJEMPLO 1: Generar Estructuras Cristalinas")

    gen = CrystalGenerator(lattice_constant=3.615)

    print("Generando diferentes estructuras cristalinas...\n")

    # FCC
    print("1. FCC (Face-Centered Cubic) - como Cu, Au, Ag")
    fcc = gen.generate_fcc(3, 3, 3)
    print(f"   ✓ Generados {len(fcc)} átomos")

    # BCC
    print("\n2. BCC (Body-Centered Cubic) - como Fe, Cr, W")
    bcc = gen.generate_bcc(3, 3, 3)
    print(f"   ✓ Generados {len(bcc)} átomos")

    # HCP
    print("\n3. HCP (Hexagonal Close-Packed) - como Mg, Zn, Ti")
    hcp = gen.generate_hcp(3, 3, 2)
    print(f"   ✓ Generados {len(hcp)} átomos")

    # Diamond
    print("\n4. Diamond - como C, Si, Ge")
    diamond = gen.generate_diamond(2, 2, 2)
    print(f"   ✓ Generados {len(diamond)} átomos")

    # Guardar como OFF
    os.makedirs('/tmp/crystal_examples', exist_ok=True)

    structures = {
        'fcc': fcc,
        'bcc': bcc,
        'hcp': hcp,
        'diamond': diamond
    }

    print("\n5. Guardando estructuras en formato OFF...")
    for name, positions in structures.items():
        # Normalizar
        positions_norm = normalize_point_cloud(positions, method='sphere')

        # Guardar
        filename = f'/tmp/crystal_examples/{name}.off'
        write_off(filename, positions_norm)
        print(f"   ✓ {filename}")

    return structures


def example_2_add_realism():
    """Ejemplo 2: Añadir efectos realistas."""
    print_section("EJEMPLO 2: Añadir Efectos Realistas")

    gen = CrystalGenerator(lattice_constant=3.615)
    fcc = gen.generate_fcc(4, 4, 4)

    print("Partiendo de una estructura FCC perfecta...\n")

    # Ruido térmico
    print("1. Añadiendo ruido térmico (T=300K)")
    fcc_300K = gen.add_thermal_noise(fcc.copy(), temperature=300.0)
    displacement = np.linalg.norm(fcc_300K - fcc, axis=1)
    print(f"   ✓ Desplazamiento RMS: {np.sqrt(np.mean(displacement**2)):.4f} Å")

    print("\n2. Añadiendo ruido térmico (T=900K)")
    fcc_900K = gen.add_thermal_noise(fcc.copy(), temperature=900.0)
    displacement = np.linalg.norm(fcc_900K - fcc, axis=1)
    print(f"   ✓ Desplazamiento RMS: {np.sqrt(np.mean(displacement**2)):.4f} Å")

    # Vacantes
    print("\n3. Creando vacantes (5% de átomos)")
    fcc_vacancies = gen.add_vacancies(fcc.copy(), vacancy_fraction=0.05)
    n_removed = len(fcc) - len(fcc_vacancies)
    print(f"   ✓ Átomos removidos: {n_removed}")
    print(f"   ✓ Átomos restantes: {len(fcc_vacancies)}")

    # Intersticiales
    print("\n4. Añadiendo átomos intersticiales (2%)")
    fcc_interstitials = gen.add_interstitials(fcc.copy(), interstitial_fraction=0.02)
    n_added = len(fcc_interstitials) - len(fcc)
    print(f"   ✓ Átomos añadidos: {n_added}")
    print(f"   ✓ Átomos totales: {len(fcc_interstitials)}")

    # Borde de grano
    print("\n5. Creando borde de grano (rotación 30°)")
    grain1 = gen.generate_fcc(3, 3, 3)
    grain2 = gen.generate_fcc(3, 3, 3)
    polycrystal = gen.create_grain_boundary(grain1, grain2, angle=30.0)
    print(f"   ✓ Átomos totales: {len(polycrystal)}")
    print(f"   ✓ Grain 1: {len(grain1)} átomos")
    print(f"   ✓ Grain 2: {len(grain2)} átomos")


def example_3_calculate_features():
    """Ejemplo 3: Calcular features cristalográficas."""
    print_section("EJEMPLO 3: Calcular Features Cristalográficas")

    gen = CrystalGenerator(lattice_constant=3.615)

    structures = {
        'FCC': gen.generate_fcc(4, 4, 4),
        'BCC': gen.generate_bcc(4, 4, 4),
        'HCP': gen.generate_hcp(4, 4, 3)
    }

    print("Analizando diferentes estructuras cristalinas...\n")
    print(f"{'Estructura':<12} {'CN medio':<12} {'CN std':<12} {'Q4 medio':<12} {'Detectado':<12}")
    print("-" * 70)

    for name, positions in structures.items():
        # Centrar
        positions = positions - positions.mean(axis=0)

        # Calcular features
        calc = CrystalFeatureCalculator(positions, cutoff=4.0)

        cn = calc.compute_coordination_numbers()
        q4 = calc.compute_steinhardt_q4()
        detected = calc.classify_structure()

        print(f"{name:<12} {cn.mean():<12.2f} {cn.std():<12.2f} "
              f"{q4.mean():<12.4f} {detected:<12}")

    # RDF para FCC
    print("\n" + "-" * 70)
    print("Función de Distribución Radial (RDF) para FCC:")
    print("-" * 70)

    calc_fcc = CrystalFeatureCalculator(structures['FCC'], cutoff=10.0)
    r, g_r = calc_fcc.compute_rdf(n_bins=50, r_max=10.0)

    # Encontrar picos
    peaks_idx = np.where(g_r > 1.5)[0]
    if len(peaks_idx) > 0:
        peaks_r = r[peaks_idx]
        print(f"\nPicos significativos en g(r):")
        for i, peak_r in enumerate(peaks_r[:5]):  # Mostrar primeros 5
            print(f"  Pico {i+1}: r = {peak_r:.3f} Å")

    # Ángulos de enlace
    print("\n" + "-" * 70)
    print("Ángulos de enlace para un átomo central en FCC:")
    print("-" * 70)

    angles = calc_fcc.compute_bond_angles(atom_idx=50)  # Átomo cerca del centro
    if len(angles) > 0:
        # Histograma de ángulos
        hist, bin_edges = np.histogram(angles, bins=18, range=(0, 180))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        print(f"\nDistribución de ángulos:")
        for i, (angle, count) in enumerate(zip(bin_centers, hist)):
            if count > 0:
                bar = '█' * int(count / hist.max() * 30)
                print(f"  {angle:5.1f}°: {bar} ({count})")


def example_4_generate_dataset():
    """Ejemplo 4: Generar dataset para entrenamiento."""
    print_section("EJEMPLO 4: Generar Dataset de Entrenamiento")

    print("Generando dataset sintético para machine learning...\n")

    # Generar dataset pequeño para demostración
    n_samples = 50  # 50 muestras por clase
    crystal_types = ['fcc', 'bcc', 'hcp']

    print(f"Configuración:")
    print(f"  - Muestras por clase: {n_samples}")
    print(f"  - Clases: {crystal_types}")
    print(f"  - Ruido térmico: Sí (T=300±150K)")
    print(f"  - Defectos: 10% de probabilidad")
    print()

    point_clouds, labels = generate_crystal_dataset(
        n_samples_per_class=n_samples,
        crystal_types=crystal_types,
        add_noise=True,
        noise_level=0.1
    )

    print(f"\n✓ Dataset generado:")
    print(f"  - Total de muestras: {len(point_clouds)}")
    print(f"  - Distribución de clases: {dict(zip(crystal_types, np.bincount(labels)))}")

    # Estadísticas del dataset
    sizes = [len(pc) for pc in point_clouds]
    print(f"\n  Estadísticas de tamaño:")
    print(f"  - Mínimo: {min(sizes)} átomos")
    print(f"  - Máximo: {max(sizes)} átomos")
    print(f"  - Promedio: {np.mean(sizes):.1f} átomos")

    # Analizar algunas muestras
    print(f"\n  Análisis de features (primeras 5 muestras):")
    print(f"  {'Muestra':<10} {'Clase':<10} {'N átomos':<12} {'CN medio':<12} {'Q4 medio':<12}")
    print("  " + "-" * 60)

    for i in range(min(5, len(point_clouds))):
        calc = CrystalFeatureCalculator(point_clouds[i], cutoff=4.0)
        cn = calc.compute_coordination_numbers()
        q4 = calc.compute_steinhardt_q4()
        class_name = crystal_types[labels[i]]

        print(f"  {i:<10} {class_name:<10} {len(point_clouds[i]):<12} "
              f"{cn.mean():<12.2f} {q4.mean():<12.4f}")


def example_5_lammps_conversion():
    """Ejemplo 5: Conversión de archivos LAMMPS."""
    print_section("EJEMPLO 5: Conversión de Archivos LAMMPS")

    # Crear archivo LAMMPS de ejemplo
    print("Creando archivo LAMMPS dump de ejemplo (FCC)...\n")

    gen = CrystalGenerator(lattice_constant=3.615)
    fcc = gen.generate_fcc(3, 3, 3)

    # Crear archivo dump
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
108
ITEM: BOX BOUNDS pp pp pp
0.0 10.845
0.0 10.845
0.0 10.845
ITEM: ATOMS id type x y z
"""

    for i, pos in enumerate(fcc):
        dump_content += f"{i+1} 1 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"

    dump_file = '/tmp/crystal_examples/example_fcc.dump'
    os.makedirs('/tmp/crystal_examples', exist_ok=True)

    with open(dump_file, 'w') as f:
        f.write(dump_content)

    print(f"✓ Archivo dump creado: {dump_file}")

    # Convertir a OFF
    print("\n1. Convirtiendo a formato OFF...")
    off_file = '/tmp/crystal_examples/example_fcc.off'
    stats = convert_lammps_to_off(
        dump_file=dump_file,
        off_file=off_file,
        normalize=True,
        norm_method='sphere'
    )

    print(f"   ✓ Archivo OFF: {off_file}")
    print(f"   ✓ Átomos convertidos: {stats['n_atoms']}")
    print(f"   ✓ Timestep: {stats['timestep']}")
    print(f"   ✓ Tamaño de caja: {stats['box_size']}")

    # Convertir a PLY con features
    print("\n2. Convirtiendo a PLY con features cristalográficas...")
    ply_file = '/tmp/crystal_examples/example_fcc.ply'
    stats_ply = convert_lammps_to_ply_with_features(
        dump_file=dump_file,
        ply_file=ply_file,
        cutoff=4.0,
        normalize=True
    )

    print(f"   ✓ Archivo PLY: {ply_file}")
    print(f"   ✓ CN promedio: {stats_ply['cn_mean']:.2f} ± {stats_ply['cn_std']:.2f}")
    print(f"   ✓ CN rango: [{stats_ply['cn_min']}, {stats_ply['cn_max']}]")

    print("\n✓ Los archivos pueden visualizarse con:")
    print("  - MeshLab (OFF, PLY)")
    print("  - CloudCompare (PLY)")
    print("  - OVITO (dump)")


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("\n" + "="*70)
    print("  EJEMPLOS DE CLASIFICACIÓN DE ESTRUCTURAS CRISTALINAS")
    print("  Tutorial Completo con PointNet++")
    print("="*70)

    try:
        # Ejemplo 1: Generar estructuras
        example_1_generate_structures()

        # Ejemplo 2: Añadir realismo
        example_2_add_realism()

        # Ejemplo 3: Calcular features
        example_3_calculate_features()

        # Ejemplo 4: Generar dataset
        example_4_generate_dataset()

        # Ejemplo 5: Conversión LAMMPS
        example_5_lammps_conversion()

        # Resumen final
        print_section("RESUMEN Y PRÓXIMOS PASOS")

        print("✓ Todos los ejemplos completados exitosamente!\n")

        print("Archivos generados en /tmp/crystal_examples/:")
        print("  - fcc.off, bcc.off, hcp.off, diamond.off")
        print("  - example_fcc.dump")
        print("  - example_fcc.off")
        print("  - example_fcc.ply (con colores por CN)")

        print("\nPróximos pasos:\n")
        print("1. Entrenar el modelo:")
        print("   python train_crystal_classifier.py --epochs 50 --n_samples 500\n")

        print("2. Usar tus propios datos LAMMPS:")
        print("   - Convierte tus archivos .dump a .off")
        print("   - Organízalos por carpetas (fcc/, bcc/, etc.)")
        print("   - Entrena: python train_crystal_classifier.py --data_dir ./mi_data\n")

        print("3. Analizar estructuras desconocidas:")
        print("   - Usa crystal_features.py para análisis preliminar")
        print("   - Usa el modelo entrenado para clasificación automática\n")

        print("4. Experimentos avanzados:")
        print("   - Detección de defectos cristalinos")
        print("   - Identificación de bordes de grano")
        print("   - Clasificación de fases (sólido/líquido)\n")

        print("Para más información, consulta:")
        print("  → LAMMPS_CRYSTAL_CLASSIFICATION.md")

    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
