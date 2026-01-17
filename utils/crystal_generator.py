"""
Generador de Estructuras Cristalinas Sintéticas
================================================

Este módulo genera estructuras cristalinas perfectas y con defectos
para entrenar redes neuronales en detección de patrones cristalinos.

Estructuras soportadas:
- FCC (Face-Centered Cubic): Cu, Au, Ag, Al, Ni
- BCC (Body-Centered Cubic): Fe, Cr, W, Mo
- HCP (Hexagonal Close-Packed): Mg, Zn, Ti, Co
- SC (Simple Cubic): Po (raro en la naturaleza)
- Diamond: C, Si, Ge

Útil para:
1. Generar datasets de entrenamiento etiquetados
2. Probar algoritmos de detección de estructura
3. Estudiar el efecto de defectos en la clasificación
"""

import numpy as np
from typing import Tuple, Optional, List
import random


class CrystalGenerator:
    """
    Generador de estructuras cristalinas.

    Attributes:
        lattice_constant (float): Parámetro de red (Angstroms)
        crystal_type (str): Tipo de cristal ('fcc', 'bcc', 'hcp', etc.)
    """

    def __init__(self, lattice_constant: float = 3.615):
        """
        Inicializa el generador.

        Args:
            lattice_constant: Parámetro de red en Angstroms
                             (default: 3.615 Å para Cu)
        """
        self.lattice_constant = lattice_constant

    def generate_fcc(self, nx: int = 3, ny: int = 3, nz: int = 3) -> np.ndarray:
        """
        Genera una estructura FCC (Face-Centered Cubic).

        La celda unitaria FCC tiene 4 átomos:
        - (0, 0, 0)
        - (1/2, 1/2, 0)
        - (1/2, 0, 1/2)
        - (0, 1/2, 1/2)

        Número de coordinación: 12 (primeros vecinos)

        Args:
            nx, ny, nz: Número de celdas unitarias en cada dirección

        Returns:
            np.ndarray de forma (N, 3) con posiciones atómicas
        """
        a = self.lattice_constant
        positions = []

        # Base de la celda FCC
        base = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = np.array([i, j, k])
                    for atom in base:
                        pos = (atom + offset) * a
                        positions.append(pos)

        return np.array(positions)

    def generate_bcc(self, nx: int = 3, ny: int = 3, nz: int = 3) -> np.ndarray:
        """
        Genera una estructura BCC (Body-Centered Cubic).

        La celda unitaria BCC tiene 2 átomos:
        - (0, 0, 0)
        - (1/2, 1/2, 1/2)

        Número de coordinación: 8 (primeros vecinos) + 6 (segundos)

        Args:
            nx, ny, nz: Número de celdas unitarias en cada dirección

        Returns:
            np.ndarray de forma (N, 3) con posiciones atómicas
        """
        a = self.lattice_constant
        positions = []

        # Base de la celda BCC
        base = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = np.array([i, j, k])
                    for atom in base:
                        pos = (atom + offset) * a
                        positions.append(pos)

        return np.array(positions)

    def generate_hcp(self, nx: int = 3, ny: int = 3, nz: int = 2) -> np.ndarray:
        """
        Genera una estructura HCP (Hexagonal Close-Packed).

        La celda unitaria HCP es más compleja. Relación ideal: c/a = √(8/3) ≈ 1.633

        Número de coordinación: 12 (como FCC, pero diferente arreglo)

        Args:
            nx, ny, nz: Número de celdas unitarias en cada dirección

        Returns:
            np.ndarray de forma (N, 3) con posiciones atómicas
        """
        a = self.lattice_constant
        c = a * np.sqrt(8/3)  # Relación ideal c/a

        positions = []

        # Base de la celda HCP (2 átomos)
        base = np.array([
            [0.0, 0.0, 0.0],
            [1/3, 2/3, 0.5]
        ])

        # Vectores de la red hexagonal
        a1 = np.array([1.0, 0.0, 0.0]) * a
        a2 = np.array([0.5, np.sqrt(3)/2, 0.0]) * a
        a3 = np.array([0.0, 0.0, c/a])  # Normalizado respecto a 'a'

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for atom in base:
                        pos = (atom[0] * a1 + atom[1] * a2 +
                              (k + atom[2]) * a3)
                        pos += i * a1 + j * a2
                        positions.append(pos)

        return np.array(positions)

    def generate_sc(self, nx: int = 3, ny: int = 3, nz: int = 3) -> np.ndarray:
        """
        Genera una estructura SC (Simple Cubic).

        La celda unitaria SC tiene 1 átomo en (0, 0, 0)

        Número de coordinación: 6

        Args:
            nx, ny, nz: Número de celdas unitarias en cada dirección

        Returns:
            np.ndarray de forma (N, 3) con posiciones atómicas
        """
        a = self.lattice_constant
        positions = []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pos = np.array([i, j, k]) * a
                    positions.append(pos)

        return np.array(positions)

    def generate_diamond(self, nx: int = 2, ny: int = 2, nz: int = 2) -> np.ndarray:
        """
        Genera una estructura Diamond (como Si, Ge).

        La estructura diamond es como dos redes FCC interpenetradas.

        Número de coordinación: 4 (tetraédrico)

        Args:
            nx, ny, nz: Número de celdas unitarias en cada dirección

        Returns:
            np.ndarray de forma (N, 3) con posiciones atómicas
        """
        a = self.lattice_constant
        positions = []

        # Base de la celda diamond (8 átomos)
        base = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75]
        ])

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = np.array([i, j, k])
                    for atom in base:
                        pos = (atom + offset) * a
                        positions.append(pos)

        return np.array(positions)

    def add_thermal_noise(self, positions: np.ndarray,
                         temperature: float = 300.0,
                         mass: float = 63.546) -> np.ndarray:
        """
        Añade desplazamientos térmicos a las posiciones atómicas.

        Simula vibraciones térmicas usando distribución de Maxwell-Boltzmann.

        Args:
            positions: Array (N, 3) de posiciones
            temperature: Temperatura en Kelvin
            mass: Masa atómica en u.m.a. (default: Cu = 63.546)

        Returns:
            Posiciones con desplazamientos térmicos añadidos
        """
        # Constantes
        kB = 1.380649e-23  # J/K (Boltzmann)
        u = 1.66053906660e-27  # kg (unidad de masa atómica)

        # Desviación estándar de los desplazamientos (en Å)
        # Basado en teorema de equipartición: <x^2> = kB*T / (m*omega^2)
        # Aproximación simple: sigma ≈ sqrt(kB*T / (m * omega^2))
        # Para metales a temperatura ambiente, típicamente ~0.1 Å

        sigma = 0.1 * np.sqrt(temperature / 300.0)  # Escalado simple

        # Añadir ruido gaussiano
        noise = np.random.normal(0, sigma, positions.shape)
        return positions + noise

    def add_vacancies(self, positions: np.ndarray,
                     vacancy_fraction: float = 0.01) -> np.ndarray:
        """
        Elimina átomos aleatoriamente para crear vacantes.

        Args:
            positions: Array (N, 3) de posiciones
            vacancy_fraction: Fracción de átomos a eliminar (0-1)

        Returns:
            Posiciones con vacantes
        """
        n_atoms = len(positions)
        n_remove = int(n_atoms * vacancy_fraction)

        # Seleccionar índices aleatorios para eliminar
        remove_indices = random.sample(range(n_atoms), n_remove)

        # Crear máscara para mantener átomos
        keep_mask = np.ones(n_atoms, dtype=bool)
        keep_mask[remove_indices] = False

        return positions[keep_mask]

    def add_interstitials(self, positions: np.ndarray,
                         interstitial_fraction: float = 0.01) -> np.ndarray:
        """
        Añade átomos intersticiales en posiciones aleatorias.

        Args:
            positions: Array (N, 3) de posiciones
            interstitial_fraction: Fracción de átomos intersticiales a añadir

        Returns:
            Posiciones con intersticiales
        """
        n_atoms = len(positions)
        n_add = int(n_atoms * interstitial_fraction)

        # Calcular rango de coordenadas
        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)

        # Generar posiciones aleatorias para intersticiales
        interstitials = np.random.uniform(min_coords, max_coords, (n_add, 3))

        return np.vstack([positions, interstitials])

    def create_grain_boundary(self, positions1: np.ndarray,
                             positions2: np.ndarray,
                             angle: float = 30.0) -> np.ndarray:
        """
        Crea una estructura con borde de grano simple.

        Rota la segunda estructura y las une.

        Args:
            positions1: Primera estructura
            positions2: Segunda estructura
            angle: Ángulo de rotación en grados

        Returns:
            Estructura combinada con borde de grano
        """
        # Matriz de rotación alrededor del eje Z
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Rotar segunda estructura
        positions2_rotated = positions2 @ rotation_matrix.T

        # Desplazar la segunda estructura en Z
        z_offset = positions1[:, 2].max() + 0.5 * self.lattice_constant
        positions2_rotated[:, 2] += z_offset

        # Combinar
        return np.vstack([positions1, positions2_rotated])


def generate_crystal_dataset(n_samples_per_class: int = 100,
                            crystal_types: List[str] = ['fcc', 'bcc', 'hcp'],
                            add_noise: bool = True,
                            noise_level: float = 0.1,
                            size_range: Tuple[int, int] = (2, 5)) -> Tuple[List[np.ndarray], List[int]]:
    """
    Genera un dataset de estructuras cristalinas para entrenamiento.

    Args:
        n_samples_per_class: Número de muestras por tipo de cristal
        crystal_types: Lista de tipos de cristal a generar
        add_noise: Si True, añade ruido térmico
        noise_level: Nivel de ruido (sigma en Å)
        size_range: Rango de tamaños (min, max) de celdas unitarias

    Returns:
        Tuple de (point_clouds, labels) donde:
        - point_clouds es lista de arrays (N_i, 3)
        - labels es lista de enteros (0=fcc, 1=bcc, 2=hcp, etc.)

    Example:
        >>> clouds, labels = generate_crystal_dataset(n_samples_per_class=50)
        >>> print(f"Generadas {len(clouds)} muestras")
    """
    point_clouds = []
    labels = []

    # Mapeo de tipos a índices
    type_to_idx = {ctype: idx for idx, ctype in enumerate(crystal_types)}

    for ctype in crystal_types:
        print(f"Generando {n_samples_per_class} muestras de {ctype.upper()}...")

        for i in range(n_samples_per_class):
            # Tamaño aleatorio
            nx = random.randint(*size_range)
            ny = random.randint(*size_range)
            nz = random.randint(*size_range)

            # Parámetro de red aleatorio (variación ±10%)
            a = 3.615 * random.uniform(0.9, 1.1)

            # Generar estructura
            generator = CrystalGenerator(lattice_constant=a)

            if ctype == 'fcc':
                positions = generator.generate_fcc(nx, ny, nz)
            elif ctype == 'bcc':
                positions = generator.generate_bcc(nx, ny, nz)
            elif ctype == 'hcp':
                positions = generator.generate_hcp(nx, ny, nz)
            elif ctype == 'sc':
                positions = generator.generate_sc(nx, ny, nz)
            elif ctype == 'diamond':
                positions = generator.generate_diamond(nx, ny, nz)
            else:
                raise ValueError(f"Tipo de cristal '{ctype}' no soportado")

            # Añadir ruido térmico
            if add_noise:
                positions = generator.add_thermal_noise(
                    positions,
                    temperature=300.0 * random.uniform(0.5, 1.5)
                )

            # Ocasionalmente añadir defectos
            if random.random() < 0.1:  # 10% de probabilidad
                if random.random() < 0.5:
                    positions = generator.add_vacancies(positions, 0.02)
                else:
                    positions = generator.add_interstitials(positions, 0.02)

            # Centrar en origen
            positions -= positions.mean(axis=0)

            point_clouds.append(positions)
            labels.append(type_to_idx[ctype])

    return point_clouds, labels


if __name__ == '__main__':
    print("=== Generador de Estructuras Cristalinas - Ejemplo ===\n")

    # Crear generador
    gen = CrystalGenerator(lattice_constant=3.615)

    # Generar diferentes estructuras
    print("1. Generando FCC (3x3x3)...")
    fcc = gen.generate_fcc(3, 3, 3)
    print(f"   ✓ {len(fcc)} átomos generados")

    print("\n2. Generando BCC (3x3x3)...")
    bcc = gen.generate_bcc(3, 3, 3)
    print(f"   ✓ {len(bcc)} átomos generados")

    print("\n3. Generando HCP (3x3x2)...")
    hcp = gen.generate_hcp(3, 3, 2)
    print(f"   ✓ {len(hcp)} átomos generados")

    # Añadir ruido
    print("\n4. Añadiendo ruido térmico a FCC...")
    fcc_noisy = gen.add_thermal_noise(fcc, temperature=300.0)
    displacement = np.linalg.norm(fcc_noisy - fcc, axis=1)
    print(f"   ✓ Desplazamiento promedio: {displacement.mean():.4f} Å")

    # Crear defectos
    print("\n5. Creando FCC con 5% de vacantes...")
    fcc_vacancies = gen.add_vacancies(fcc, vacancy_fraction=0.05)
    print(f"   ✓ Átomos: {len(fcc)} → {len(fcc_vacancies)} ({len(fcc)-len(fcc_vacancies)} vacantes)")

    # Generar dataset
    print("\n6. Generando dataset de entrenamiento...")
    clouds, labels = generate_crystal_dataset(n_samples_per_class=10,
                                             crystal_types=['fcc', 'bcc', 'hcp'])
    print(f"   ✓ Total de muestras: {len(clouds)}")
    print(f"   ✓ Distribución de clases: {np.bincount(labels)}")
    print(f"   ✓ Tamaños de muestra (min/max): {min(len(c) for c in clouds)}/{max(len(c) for c in clouds)}")

    print("\n✓ Generación exitosa!")
