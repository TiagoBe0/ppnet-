"""
Conversor de LAMMPS Dump a formatos OFF/PLY
============================================

Este módulo convierte archivos .dump de LAMMPS a formatos utilizables
para redes neuronales de punto (PointNet++).

Formatos soportados:
- .off: Object File Format (simple, solo geometría)
- .ply: Polygon File Format (más flexible, puede incluir colores/normales)
- .txt: Formato de texto simple (x y z por línea)

Para clasificación de estructuras cristalinas, es importante:
1. Normalizar las posiciones
2. Opcionalmente incluir información de vecindad local
3. Calcular features cristalográficas (número de coordinación, parámetros de orden)
"""

import numpy as np
import os
from typing import Optional, Tuple

# Import handling for both module and direct execution
try:
    from .lammps_reader import LAMMPSDumpReader, read_lammps_dump
except ImportError:
    from lammps_reader import LAMMPSDumpReader, read_lammps_dump


def write_off(filename: str, points: np.ndarray, faces: Optional[np.ndarray] = None):
    """
    Escribe puntos en formato OFF (Object File Format).

    Formato OFF:
        OFF
        n_vertices n_faces n_edges
        x1 y1 z1
        x2 y2 z2
        ...
        [faces si existen]

    Args:
        filename: Ruta del archivo de salida
        points: Array (N, 3) de coordenadas XYZ
        faces: Array (M, 3) de índices de caras (opcional, para mallas)

    Example:
        >>> points = np.random.rand(100, 3)
        >>> write_off('output.off', points)
    """
    n_vertices = points.shape[0]
    n_faces = 0 if faces is None else faces.shape[0]
    n_edges = 0  # No se usan normalmente

    with open(filename, 'w') as f:
        # Header
        f.write('OFF\n')
        f.write(f'{n_vertices} {n_faces} {n_edges}\n')

        # Vertices
        for point in points:
            f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n')

        # Faces (si existen)
        if faces is not None:
            for face in faces:
                f.write(f'3 {face[0]} {face[1]} {face[2]}\n')


def read_off(filename: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Lee un archivo OFF.

    Args:
        filename: Ruta del archivo OFF

    Returns:
        Tuple de (vertices, faces) donde vertices es (N, 3) y faces es (M, 3) o None
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Saltar línea OFF
    if lines[0].strip() == 'OFF':
        start_idx = 1
    else:
        start_idx = 0

    # Leer header
    header = lines[start_idx].strip().split()
    n_vertices = int(header[0])
    n_faces = int(header[1])

    # Leer vertices
    vertices = []
    for i in range(n_vertices):
        coords = lines[start_idx + 1 + i].strip().split()
        vertices.append([float(coords[0]), float(coords[1]), float(coords[2])])

    vertices = np.array(vertices)

    # Leer faces si existen
    faces = None
    if n_faces > 0:
        faces = []
        for i in range(n_faces):
            face_line = lines[start_idx + 1 + n_vertices + i].strip().split()
            # face_line[0] es el número de vértices (usualmente 3)
            face_indices = [int(face_line[j]) for j in range(1, 4)]
            faces.append(face_indices)
        faces = np.array(faces)

    return vertices, faces


def write_ply_simple(filename: str, points: np.ndarray):
    """
    Escribe puntos en formato PLY simple (sin colores).

    Args:
        filename: Ruta del archivo de salida
        points: Array (N, 3) de coordenadas XYZ
    """
    n_points = points.shape[0]

    with open(filename, 'w') as f:
        # Header PLY
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n_points}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')

        # Data
        for point in points:
            f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n')


def normalize_point_cloud(points: np.ndarray,
                          method: str = 'sphere') -> np.ndarray:
    """
    Normaliza una nube de puntos.

    Args:
        points: Array (N, 3) de coordenadas
        method: Método de normalización
            - 'sphere': Escala para que quepa en esfera unitaria
            - 'box': Escala para que quepa en cubo unitario [-1, 1]^3
            - 'center': Solo centra en origen (sin escalar)

    Returns:
        Array (N, 3) normalizado
    """
    # Centrar en origen
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    if method == 'center':
        return points_centered

    elif method == 'sphere':
        # Escalar para que el punto más lejano esté a distancia 1
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        return points_centered / max_dist

    elif method == 'box':
        # Escalar para que quepa en [-1, 1]^3
        max_range = np.max(np.abs(points_centered))
        return points_centered / max_range

    else:
        raise ValueError(f"Método de normalización '{method}' no reconocido")


def compute_coordination_number(points: np.ndarray,
                                cutoff: float = 3.5,
                                return_features: bool = False) -> np.ndarray:
    """
    Calcula el número de coordinación de cada átomo.

    El número de coordinación es cuántos vecinos tiene un átomo dentro
    de una distancia de corte. Útil para identificar estructuras:
    - FCC: CN = 12 (primeros vecinos)
    - BCC: CN = 8 (primeros vecinos) + 6 (segundos)
    - HCP: CN = 12 (primeros vecinos)

    Args:
        points: Array (N, 3) de posiciones atómicas
        cutoff: Distancia de corte para vecinos (en mismas unidades que points)
        return_features: Si True, devuelve features adicionales por átomo

    Returns:
        Si return_features=False: Array (N,) con números de coordinación
        Si return_features=True: Array (N, 4) con [CN, mean_dist, std_dist, radial_density]
    """
    n_atoms = points.shape[0]
    coordination_numbers = np.zeros(n_atoms)

    if return_features:
        mean_distances = np.zeros(n_atoms)
        std_distances = np.zeros(n_atoms)
        radial_densities = np.zeros(n_atoms)

    for i in range(n_atoms):
        # Calcular distancias a todos los demás átomos
        distances = np.linalg.norm(points - points[i], axis=1)

        # Excluir el átomo mismo (distancia 0) y contar vecinos dentro del cutoff
        neighbors_mask = (distances > 0) & (distances < cutoff)
        coordination_numbers[i] = np.sum(neighbors_mask)

        if return_features:
            neighbor_distances = distances[neighbors_mask]
            if len(neighbor_distances) > 0:
                mean_distances[i] = np.mean(neighbor_distances)
                std_distances[i] = np.std(neighbor_distances)
                # Densidad radial: número de vecinos / volumen de esfera
                volume = (4/3) * np.pi * cutoff**3
                radial_densities[i] = coordination_numbers[i] / volume
            else:
                mean_distances[i] = 0
                std_distances[i] = 0
                radial_densities[i] = 0

    if return_features:
        return np.column_stack([coordination_numbers,
                               mean_distances,
                               std_distances,
                               radial_densities])
    else:
        return coordination_numbers


def convert_lammps_to_off(dump_file: str,
                          off_file: str,
                          normalize: bool = True,
                          norm_method: str = 'sphere') -> dict:
    """
    Convierte un archivo LAMMPS dump a formato OFF.

    Args:
        dump_file: Ruta del archivo .dump de entrada
        off_file: Ruta del archivo .off de salida
        normalize: Si True, normaliza las coordenadas
        norm_method: Método de normalización ('sphere', 'box', 'center')

    Returns:
        Dict con estadísticas de la conversión

    Example:
        >>> stats = convert_lammps_to_off('input.dump', 'output.off')
        >>> print(f"Convertidos {stats['n_atoms']} átomos")
    """
    # Leer dump file
    reader = LAMMPSDumpReader(dump_file)
    reader.read()
    positions = reader.center_atoms()  # Ya centra automáticamente

    # Normalizar si se solicita
    if normalize:
        positions = normalize_point_cloud(positions, method=norm_method)

    # Escribir OFF
    write_off(off_file, positions)

    # Calcular estadísticas
    stats = {
        'n_atoms': len(positions),
        'timestep': reader.timestep,
        'box_size': reader.get_box_size(),
        'centroid': np.mean(reader.get_positions(), axis=0),
        'normalized': normalize
    }

    return stats


def convert_lammps_to_ply_with_features(dump_file: str,
                                        ply_file: str,
                                        cutoff: float = 3.5,
                                        normalize: bool = True) -> dict:
    """
    Convierte LAMMPS dump a PLY con features cristalográficas como colores.

    Los números de coordinación se mapean a colores para visualización:
    - Azul: CN bajo (superficie, defectos)
    - Verde: CN medio
    - Rojo: CN alto (bulk cristalino)

    Args:
        dump_file: Ruta del archivo .dump
        ply_file: Ruta del archivo .ply de salida
        cutoff: Distancia de corte para cálculo de CN
        normalize: Si True, normaliza coordenadas

    Returns:
        Dict con estadísticas
    """
    # Leer posiciones
    positions = read_lammps_dump(dump_file, center=True)

    if normalize:
        positions = normalize_point_cloud(positions, method='sphere')

    # Calcular número de coordinación
    cn = compute_coordination_number(positions, cutoff=cutoff)

    # Mapear CN a colores (normalizar a rango [0, 255])
    cn_min, cn_max = cn.min(), cn.max()
    if cn_max > cn_min:
        cn_normalized = (cn - cn_min) / (cn_max - cn_min)
    else:
        cn_normalized = np.zeros_like(cn)

    # Colormap: azul (bajo CN) -> verde -> rojo (alto CN)
    colors = np.zeros((len(positions), 3), dtype=np.uint8)
    for i, val in enumerate(cn_normalized):
        if val < 0.5:
            # Azul -> Verde
            colors[i] = [0, int(255 * val * 2), int(255 * (1 - val * 2))]
        else:
            # Verde -> Rojo
            colors[i] = [int(255 * (val - 0.5) * 2), int(255 * (1 - (val - 0.5) * 2)), 0]

    # Escribir PLY con colores
    with open(ply_file, 'w') as f:
        n_points = len(positions)
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n_points}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for i in range(n_points):
            f.write(f'{positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f} '
                   f'{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n')

    stats = {
        'n_atoms': len(positions),
        'cn_min': cn_min,
        'cn_max': cn_max,
        'cn_mean': np.mean(cn),
        'cn_std': np.std(cn)
    }

    return stats


if __name__ == '__main__':
    print("=== LAMMPS to OFF/PLY Converter - Ejemplo ===\n")

    # Crear archivo dump de ejemplo (FCC)
    example_dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
32
ITEM: BOX BOUNDS pp pp pp
0.0 7.23
0.0 7.23
0.0 7.23
ITEM: ATOMS id type x y z
"""

    # Generar red FCC simple (a = 3.615 Å para Cu)
    a = 3.615
    positions = []
    atom_id = 1

    # 2x2x2 celdas unitarias FCC
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Átomos base FCC
                base = np.array([
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5]
                ]) * a

                offset = np.array([i, j, k]) * a
                for atom in base:
                    pos = atom + offset
                    example_dump += f"{atom_id} 1 {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n"
                    atom_id += 1

    # Guardar dump temporal
    dump_path = '/tmp/example_fcc.dump'
    with open(dump_path, 'w') as f:
        f.write(example_dump)

    # Convertir a OFF
    print("1. Convirtiendo a OFF...")
    stats_off = convert_lammps_to_off(dump_path, '/tmp/example.off')
    print(f"   ✓ Convertidos {stats_off['n_atoms']} átomos")
    print(f"   ✓ Timestep: {stats_off['timestep']}")

    # Convertir a PLY con features
    print("\n2. Convirtiendo a PLY con features cristalográficas...")
    stats_ply = convert_lammps_to_ply_with_features(dump_path, '/tmp/example.ply', cutoff=4.0)
    print(f"   ✓ Número de coordinación promedio: {stats_ply['cn_mean']:.2f} ± {stats_ply['cn_std']:.2f}")
    print(f"   ✓ CN rango: [{stats_ply['cn_min']}, {stats_ply['cn_max']}]")

    # Leer OFF de vuelta
    print("\n3. Verificando lectura de OFF...")
    vertices, faces = read_off('/tmp/example.off')
    print(f"   ✓ Leídos {len(vertices)} vértices")

    print("\n✓ Conversión exitosa!")
    print(f"\nArchivos generados:")
    print(f"  - {dump_path}")
    print(f"  - /tmp/example.off")
    print(f"  - /tmp/example.ply")
