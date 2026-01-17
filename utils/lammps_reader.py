"""
LAMMPS Dump File Reader
=======================

Este módulo lee archivos .dump de LAMMPS y extrae información atómica.

LAMMPS dump format (ejemplo):
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
    2 1 1.0 0.0 0.0
    ...

Autor: Tutorial educativo para detección de patrones cristalinos
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional


class LAMMPSDumpReader:
    """
    Clase para leer archivos .dump de LAMMPS.

    Attributes:
        filename (str): Ruta al archivo .dump
        timestep (int): Paso temporal de la simulación
        natoms (int): Número de átomos
        box_bounds (np.ndarray): Límites de la caja de simulación (3x2)
        atoms_data (np.ndarray): Datos atómicos (N x C)
        atom_columns (list): Nombres de las columnas de datos atómicos
    """

    def __init__(self, filename: str):
        """
        Inicializa el lector.

        Args:
            filename: Ruta al archivo .dump de LAMMPS
        """
        self.filename = filename
        self.timestep = None
        self.natoms = None
        self.box_bounds = None
        self.atoms_data = None
        self.atom_columns = []

    def read(self) -> Dict:
        """
        Lee el archivo dump y extrae toda la información.

        Returns:
            Dict con keys: 'timestep', 'natoms', 'box_bounds', 'atoms_data', 'columns'

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato es incorrecto
        """
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Leer timestep
            if 'ITEM: TIMESTEP' in line:
                self.timestep = int(lines[i + 1].strip())
                i += 2

            # Leer número de átomos
            elif 'ITEM: NUMBER OF ATOMS' in line:
                self.natoms = int(lines[i + 1].strip())
                i += 2

            # Leer límites de la caja
            elif 'ITEM: BOX BOUNDS' in line:
                self.box_bounds = np.zeros((3, 2))
                for j in range(3):
                    bounds = lines[i + 1 + j].strip().split()
                    self.box_bounds[j, 0] = float(bounds[0])
                    self.box_bounds[j, 1] = float(bounds[1])
                i += 4

            # Leer datos atómicos
            elif 'ITEM: ATOMS' in line:
                # Extraer nombres de columnas
                self.atom_columns = line.split()[2:]

                # Leer datos
                atom_lines = []
                for j in range(self.natoms):
                    atom_lines.append(lines[i + 1 + j].strip().split())

                # Convertir a numpy array
                self.atoms_data = np.array(atom_lines, dtype=float)
                i += self.natoms + 1
            else:
                i += 1

        return {
            'timestep': self.timestep,
            'natoms': self.natoms,
            'box_bounds': self.box_bounds,
            'atoms_data': self.atoms_data,
            'columns': self.atom_columns
        }

    def get_positions(self) -> np.ndarray:
        """
        Extrae solo las posiciones atómicas (x, y, z).

        Returns:
            np.ndarray de forma (N, 3) con coordenadas XYZ

        Raises:
            ValueError: Si las columnas x, y, z no existen
        """
        if self.atoms_data is None:
            self.read()

        # Buscar índices de columnas x, y, z
        try:
            x_idx = self.atom_columns.index('x')
            y_idx = self.atom_columns.index('y')
            z_idx = self.atom_columns.index('z')
        except ValueError:
            # Intentar con coordenadas escaladas
            try:
                x_idx = self.atom_columns.index('xs')
                y_idx = self.atom_columns.index('ys')
                z_idx = self.atom_columns.index('zs')
                # Escalar a coordenadas reales
                positions = self.atoms_data[:, [x_idx, y_idx, z_idx]]
                for i in range(3):
                    positions[:, i] = (positions[:, i] *
                                      (self.box_bounds[i, 1] - self.box_bounds[i, 0]) +
                                      self.box_bounds[i, 0])
                return positions
            except ValueError:
                raise ValueError("No se encontraron columnas de posición (x,y,z o xs,ys,zs)")

        return self.atoms_data[:, [x_idx, y_idx, z_idx]]

    def get_types(self) -> Optional[np.ndarray]:
        """
        Extrae tipos atómicos si existen.

        Returns:
            np.ndarray de forma (N,) con tipos atómicos, o None si no existe
        """
        if self.atoms_data is None:
            self.read()

        if 'type' in self.atom_columns:
            type_idx = self.atom_columns.index('type')
            return self.atoms_data[:, type_idx].astype(int)
        return None

    def get_box_size(self) -> np.ndarray:
        """
        Calcula el tamaño de la caja de simulación.

        Returns:
            np.ndarray de forma (3,) con [Lx, Ly, Lz]
        """
        if self.box_bounds is None:
            self.read()

        return self.box_bounds[:, 1] - self.box_bounds[:, 0]

    def center_atoms(self) -> np.ndarray:
        """
        Centra los átomos en el origen.

        Returns:
            np.ndarray de forma (N, 3) con posiciones centradas
        """
        positions = self.get_positions()
        center = np.mean(positions, axis=0)
        return positions - center

    def apply_pbc(self, positions: np.ndarray) -> np.ndarray:
        """
        Aplica condiciones periódicas de frontera (PBC).

        Útil para calcular distancias mínimas entre átomos.

        Args:
            positions: Array (N, 3) de posiciones

        Returns:
            Posiciones envueltas dentro de la caja
        """
        box_size = self.get_box_size()
        box_min = self.box_bounds[:, 0]

        wrapped = positions.copy()
        for i in range(3):
            wrapped[:, i] = ((positions[:, i] - box_min[i]) % box_size[i]) + box_min[i]

        return wrapped


def read_lammps_dump(filename: str, center: bool = True) -> np.ndarray:
    """
    Función de conveniencia para leer rápidamente un archivo dump.

    Args:
        filename: Ruta al archivo .dump
        center: Si True, centra los átomos en el origen

    Returns:
        np.ndarray de forma (N, 3) con posiciones XYZ

    Example:
        >>> positions = read_lammps_dump('simulation.dump')
        >>> print(positions.shape)
        (256, 3)
    """
    reader = LAMMPSDumpReader(filename)
    reader.read()

    if center:
        return reader.center_atoms()
    else:
        return reader.get_positions()


if __name__ == '__main__':
    # Ejemplo de uso
    print("=== LAMMPS Dump Reader - Ejemplo de Uso ===\n")

    # Crear un archivo dump de ejemplo
    example_dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0.0 3.615
0.0 3.615
0.0 3.615
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 1.8075 1.8075 0.0
3 1 1.8075 0.0 1.8075
4 1 0.0 1.8075 1.8075
"""

    # Guardar ejemplo
    with open('/tmp/example_fcc.dump', 'w') as f:
        f.write(example_dump)

    # Leer y procesar
    reader = LAMMPSDumpReader('/tmp/example_fcc.dump')
    data = reader.read()

    print(f"Timestep: {data['timestep']}")
    print(f"Número de átomos: {data['natoms']}")
    print(f"Límites de caja:\n{data['box_bounds']}")
    print(f"Columnas: {data['columns']}")
    print(f"\nPosiciones atómicas:\n{reader.get_positions()}")
    print(f"\nPosiciones centradas:\n{reader.center_atoms()}")
