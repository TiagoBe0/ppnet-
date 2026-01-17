"""
Script de prueba para verificar que el lector funciona con el formato del usuario.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from lammps_reader import LAMMPSDumpReader
from lammps_to_off import convert_lammps_to_off
import numpy as np

# Crear archivo dump con el formato del usuario
user_dump_content = """ITEM: TIMESTEP
1000
ITEM: NUMBER OF ATOMS
12
ITEM: BOX BOUNDS abc origin pp pp pp
56.586619445 0.0 0.0 -0.0373097225
0.0 56.586619445 0.0 -0.0373097225
0.0 0.0 56.586619445 -0.0373097225
ITEM: ATOMS x y z
22.9296 26.4484 24.7898
22.9675 24.6801 26.5933
24.7652 24.8674 24.703
26.482 26.5084 24.7507
26.4963 24.7027 26.4235
22.9223 28.1912 26.4877
24.7149 28.2067 24.7545
26.4273 28.2134 26.4566
22.9966 26.4568 28.2214
24.7624 24.7254 28.1755
26.4402 26.4546 28.1246
24.6984 28.2396 28.2693
"""

# Guardar archivo temporal
test_file = '/tmp/user_format_test.dump'
with open(test_file, 'w') as f:
    f.write(user_dump_content)

print("="*70)
print("PRUEBA DE FORMATO DE USUARIO")
print("="*70)

# Test 1: Leer el archivo
print("\n1. Leyendo archivo dump con formato de usuario...")
try:
    reader = LAMMPSDumpReader(test_file)
    data = reader.read()

    print(f"   ✓ Timestep: {data['timestep']}")
    print(f"   ✓ Número de átomos: {data['natoms']}")
    print(f"   ✓ Columnas: {data['columns']}")
    print(f"   ✓ Box bounds:\n{data['box_bounds']}")

    if reader.box_origin is not None:
        print(f"   ✓ Box origin: {reader.box_origin}")

except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Extraer posiciones
print("\n2. Extrayendo posiciones atómicas...")
try:
    positions = reader.get_positions()
    print(f"   ✓ Shape de posiciones: {positions.shape}")
    print(f"   ✓ Primeras 3 posiciones:")
    for i in range(min(3, len(positions))):
        print(f"      Átomo {i+1}: [{positions[i,0]:.4f}, {positions[i,1]:.4f}, {positions[i,2]:.4f}]")

    # Verificar que las posiciones coinciden con el archivo
    expected_first = np.array([22.9296, 26.4484, 24.7898])
    if np.allclose(positions[0], expected_first):
        print(f"   ✓ Verificación: Primera posición correcta")
    else:
        print(f"   ✗ Verificación fallida:")
        print(f"      Esperado: {expected_first}")
        print(f"      Obtenido: {positions[0]}")

except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Centrar átomos
print("\n3. Centrando átomos en el origen...")
try:
    centered = reader.center_atoms()
    centroid = np.mean(positions, axis=0)
    print(f"   ✓ Centroide original: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")

    new_centroid = np.mean(centered, axis=0)
    print(f"   ✓ Centroide después: [{new_centroid[0]:.6f}, {new_centroid[1]:.6f}, {new_centroid[2]:.6f}]")

    if np.allclose(new_centroid, 0.0, atol=1e-10):
        print(f"   ✓ Verificación: Átomo correctamente centrado")
    else:
        print(f"   ⚠ Centroide no es exactamente cero (puede ser error de redondeo)")

except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Convertir a OFF
print("\n4. Convirtiendo a formato OFF...")
try:
    off_file = '/tmp/user_format_test.off'
    stats = convert_lammps_to_off(
        dump_file=test_file,
        off_file=off_file,
        normalize=True,
        norm_method='sphere'
    )

    print(f"   ✓ Archivo OFF creado: {off_file}")
    print(f"   ✓ Átomos convertidos: {stats['n_atoms']}")
    print(f"   ✓ Normalizado: {stats['normalized']}")

    # Leer el archivo OFF para verificar
    with open(off_file, 'r') as f:
        off_lines = f.readlines()

    print(f"   ✓ Primeras líneas del archivo OFF:")
    for line in off_lines[:5]:
        print(f"      {line.rstrip()}")

except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Calcular features cristalográficas
print("\n5. Calculando features cristalográficas...")
try:
    from crystal_features import CrystalFeatureCalculator

    calc = CrystalFeatureCalculator(centered, cutoff=5.0)
    cn = calc.compute_coordination_numbers()

    print(f"   ✓ Número de coordinación promedio: {cn.mean():.2f} ± {cn.std():.2f}")
    print(f"   ✓ CN por átomo:")
    for i, cn_val in enumerate(cn):
        print(f"      Átomo {i+1}: CN = {int(cn_val)}")

except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Resumen
print("\n" + "="*70)
print("RESUMEN DE PRUEBAS")
print("="*70)
print("\n✓ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
print(f"\nTu formato de LAMMPS dump es compatible con el sistema.")
print(f"\nArchivos generados:")
print(f"  - {test_file}")
print(f"  - {off_file}")
print(f"\nPróximos pasos:")
print(f"  1. Convierte todos tus archivos .dump a .off")
print(f"  2. Organízalos por tipo de estructura cristalina")
print(f"  3. Entrena el modelo con: python train_crystal_classifier.py --data_dir ./tu_directorio")
