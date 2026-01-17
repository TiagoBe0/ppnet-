"""
Calculador de Features Cristalográficas
========================================

Este módulo calcula features descriptivas para caracterizar estructuras cristalinas.

Features implementadas:
1. Número de Coordinación (CN)
2. Radial Distribution Function (RDF/g(r))
3. Parámetros de orden de Steinhardt (Q4, Q6)
4. Common Neighbor Analysis (CNA)
5. Ángulos de enlace

Estas features son útiles para:
- Pre-procesamiento de datos para ML
- Análisis de estructuras cristalinas
- Validación de resultados de clasificación
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy.spatial import cKDTree
from collections import Counter


class CrystalFeatureCalculator:
    """
    Calculador de features cristalográficas.

    Attributes:
        positions (np.ndarray): Posiciones atómicas (N, 3)
        cutoff (float): Radio de corte para vecindad
        kdtree (cKDTree): Árbol KD para búsqueda eficiente de vecinos
    """

    def __init__(self, positions: np.ndarray, cutoff: float = 4.0):
        """
        Inicializa el calculador.

        Args:
            positions: Array (N, 3) de posiciones atómicas
            cutoff: Radio de corte para vecinos (Å)
        """
        self.positions = positions
        self.cutoff = cutoff
        self.kdtree = cKDTree(positions)
        self._neighbors_cache = None

    def get_neighbors(self, use_cache: bool = True) -> List[np.ndarray]:
        """
        Obtiene vecinos dentro del cutoff para cada átomo.

        Args:
            use_cache: Si True, usa caché de vecinos

        Returns:
            Lista de arrays con índices de vecinos para cada átomo
        """
        if use_cache and self._neighbors_cache is not None:
            return self._neighbors_cache

        neighbors = []
        for i in range(len(self.positions)):
            # Buscar vecinos dentro del cutoff
            indices = self.kdtree.query_ball_point(self.positions[i], self.cutoff)
            # Excluir el átomo mismo
            indices = [idx for idx in indices if idx != i]
            neighbors.append(np.array(indices))

        if use_cache:
            self._neighbors_cache = neighbors

        return neighbors

    def compute_coordination_numbers(self) -> np.ndarray:
        """
        Calcula el número de coordinación para cada átomo.

        Returns:
            Array (N,) con números de coordinación

        Example:
            >>> calc = CrystalFeatureCalculator(positions, cutoff=3.5)
            >>> cn = calc.compute_coordination_numbers()
            >>> print(f"CN promedio: {cn.mean():.2f}")
        """
        neighbors = self.get_neighbors()
        return np.array([len(neigh) for neigh in neighbors])

    def compute_rdf(self, n_bins: int = 100, r_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la función de distribución radial g(r).

        La RDF muestra cómo varía la densidad atómica con la distancia.
        Picos en g(r) indican capas de coordinación.

        Args:
            n_bins: Número de bins para el histograma
            r_max: Radio máximo (si None, usa cutoff)

        Returns:
            Tuple de (r, g_r) donde:
            - r: Array con distancias (centros de bins)
            - g_r: Array con valores de g(r)
        """
        if r_max is None:
            r_max = self.cutoff

        # Calcular todas las distancias por pares
        distances = []
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < r_max:
                    distances.append(dist)

        distances = np.array(distances)

        # Crear histograma
        counts, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
        r = (bin_edges[:-1] + bin_edges[1:]) / 2  # Centros de bins

        # Normalizar por volumen de cáscara esférica y densidad
        bin_width = r_max / n_bins
        n_atoms = len(self.positions)

        # Volumen de cada cáscara esférica
        shell_volumes = 4 * np.pi * r**2 * bin_width

        # Densidad de número (átomos por volumen)
        # Estimación simple: usar volumen de la caja
        box_volume = self._estimate_box_volume()
        number_density = n_atoms / box_volume

        # g(r) = (número de pares en cáscara) / (número esperado en cáscara)
        # Factores: 2 porque cada par se cuenta una vez, n_atoms porque es promedio por átomo
        g_r = counts / (shell_volumes * number_density * n_atoms)

        return r, g_r

    def _estimate_box_volume(self) -> float:
        """Estima el volumen de la caja de simulación."""
        ranges = self.positions.max(axis=0) - self.positions.min(axis=0)
        return np.prod(ranges)

    def compute_steinhardt_q4(self) -> np.ndarray:
        """
        Calcula el parámetro de orden de Steinhardt Q4.

        Q4 es útil para distinguir estructuras:
        - FCC: Q4 ≈ 0.191
        - HCP: Q4 ≈ 0.097
        - BCC: Q4 ≈ 0.036
        - Líquido: Q4 ≈ 0

        Returns:
            Array (N,) con valores de Q4 para cada átomo

        Note:
            Implementación simplificada. Para resultados precisos,
            usar librerías como freud o OVITO.
        """
        neighbors_list = self.get_neighbors()
        q4_values = []

        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) == 0:
                q4_values.append(0.0)
                continue

            # Vectores de enlace
            bonds = self.positions[neighbors] - self.positions[i]

            # Normalizar a vectores unitarios
            bond_lengths = np.linalg.norm(bonds, axis=1, keepdims=True)
            bonds_normalized = bonds / (bond_lengths + 1e-10)

            # Calcular ángulos polares (theta, phi)
            # Simplificación: usar solo phi (ángulo azimutal)
            # Implementación completa requeriría armónicos esféricos

            # Aproximación simple: varianza angular
            # Menor varianza = más ordenado
            phi = np.arctan2(bonds_normalized[:, 1], bonds_normalized[:, 0])
            theta = np.arccos(np.clip(bonds_normalized[:, 2], -1, 1))

            # Medida simple de orden basada en dispersión angular
            phi_std = np.std(phi)
            theta_std = np.std(theta)

            # Valor aproximado (no es el verdadero Q4, pero correlaciona con orden)
            q4_approx = 1.0 / (1.0 + phi_std + theta_std)

            q4_values.append(q4_approx)

        return np.array(q4_values)

    def compute_cna_signature(self, atom_idx: int) -> Tuple[int, int, int]:
        """
        Calcula la signatura de Common Neighbor Analysis (CNA) para un átomo.

        CNA identifica estructuras locales:
        - FCC: mayormente (421)
        - HCP: mezcla de (421) y (422)
        - BCC: mayormente (666)

        Args:
            atom_idx: Índice del átomo a analizar

        Returns:
            Tuple (n_common, n_bonds, longest_chain) representando la signatura

        Note:
            Implementación simplificada. Para análisis completo usar OVITO.
        """
        neighbors = self.get_neighbors()[atom_idx]

        if len(neighbors) == 0:
            return (0, 0, 0)

        # Contar vecinos comunes entre pares de vecinos
        neighbor_set = set(neighbors)
        common_counts = []

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                neigh_i_neighbors = set(self.get_neighbors()[neighbors[i]])
                neigh_j_neighbors = set(self.get_neighbors()[neighbors[j]])

                # Vecinos comunes (excluyendo el átomo central)
                common = (neigh_i_neighbors & neigh_j_neighbors & neighbor_set)
                common_counts.append(len(common))

        if len(common_counts) == 0:
            return (0, 0, 0)

        # Signatura simplificada
        avg_common = int(np.mean(common_counts))
        max_common = int(np.max(common_counts))
        n_neighbors = len(neighbors)

        return (avg_common, max_common, n_neighbors)

    def compute_bond_angles(self, atom_idx: int) -> np.ndarray:
        """
        Calcula ángulos entre enlaces para un átomo.

        Los ángulos de enlace son característicos de cada estructura:
        - FCC: 60°, 90°, 120° principalmente
        - BCC: ~70.5°, 109.5°
        - Diamond: 109.5° (tetraédrico)

        Args:
            atom_idx: Índice del átomo

        Returns:
            Array con ángulos de enlace en grados
        """
        neighbors = self.get_neighbors()[atom_idx]

        if len(neighbors) < 2:
            return np.array([])

        # Vectores de enlace
        bonds = self.positions[neighbors] - self.positions[atom_idx]
        bonds_normalized = bonds / np.linalg.norm(bonds, axis=1, keepdims=True)

        # Calcular todos los ángulos por pares
        angles = []
        for i in range(len(bonds_normalized)):
            for j in range(i + 1, len(bonds_normalized)):
                cos_angle = np.dot(bonds_normalized[i], bonds_normalized[j])
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(np.degrees(angle))

        return np.array(angles)

    def compute_feature_vector(self, n_angle_bins: int = 36) -> np.ndarray:
        """
        Calcula un vector de features completo para cada átomo.

        Features incluidas:
        - Número de coordinación
        - Q4 (parámetro de Steinhardt)
        - Histograma de ángulos de enlace
        - Distancia promedio a vecinos
        - Desviación estándar de distancias a vecinos

        Args:
            n_angle_bins: Número de bins para histograma de ángulos (0-180°)

        Returns:
            Array (N, D) donde D es la dimensionalidad del feature vector
        """
        n_atoms = len(self.positions)
        neighbors_list = self.get_neighbors()

        # Calcular features básicas
        cn = self.compute_coordination_numbers()
        q4 = self.compute_steinhardt_q4()

        # Features por átomo
        feature_vectors = []

        for i in range(n_atoms):
            features = []

            # 1. Número de coordinación (1 feature)
            features.append(cn[i])

            # 2. Q4 (1 feature)
            features.append(q4[i])

            # 3. Estadísticas de distancias a vecinos (2 features)
            if len(neighbors_list[i]) > 0:
                distances = np.linalg.norm(
                    self.positions[neighbors_list[i]] - self.positions[i],
                    axis=1
                )
                features.append(np.mean(distances))
                features.append(np.std(distances))
            else:
                features.append(0.0)
                features.append(0.0)

            # 4. Histograma de ángulos de enlace (n_angle_bins features)
            angles = self.compute_bond_angles(i)
            if len(angles) > 0:
                hist, _ = np.histogram(angles, bins=n_angle_bins, range=(0, 180))
                # Normalizar por número total de ángulos
                hist = hist.astype(float) / (len(angles) + 1e-10)
            else:
                hist = np.zeros(n_angle_bins)

            features.extend(hist)

            feature_vectors.append(features)

        return np.array(feature_vectors)

    def classify_structure(self) -> str:
        """
        Intenta clasificar la estructura cristalina basándose en features.

        Método simple basado en número de coordinación promedio.

        Returns:
            String con tipo predicho: 'FCC', 'BCC', 'HCP', 'SC', 'Diamond', 'Unknown'
        """
        cn = self.compute_coordination_numbers()
        cn_mean = np.mean(cn)
        cn_std = np.std(cn)

        # Clasificación simple basada en CN promedio
        if 11.5 < cn_mean < 12.5 and cn_std < 1.0:
            # Distinguir FCC de HCP requiere más análisis
            # Por ahora, asumir FCC
            return 'FCC'
        elif 7.5 < cn_mean < 8.5:
            return 'BCC'
        elif 5.5 < cn_mean < 6.5:
            return 'SC'
        elif 3.5 < cn_mean < 4.5:
            return 'Diamond'
        else:
            return 'Unknown'


def compute_all_features(positions: np.ndarray,
                        cutoff: float = 4.0) -> Dict[str, np.ndarray]:
    """
    Función de conveniencia para calcular todas las features.

    Args:
        positions: Array (N, 3) de posiciones atómicas
        cutoff: Radio de corte para vecinos

    Returns:
        Dict con todas las features calculadas

    Example:
        >>> features = compute_all_features(positions)
        >>> print(f"CN promedio: {features['cn'].mean():.2f}")
    """
    calc = CrystalFeatureCalculator(positions, cutoff=cutoff)

    features = {
        'cn': calc.compute_coordination_numbers(),
        'q4': calc.compute_steinhardt_q4(),
        'feature_vectors': calc.compute_feature_vector(),
        'structure_type': calc.classify_structure()
    }

    return features


if __name__ == '__main__':
    print("=== Crystal Feature Calculator - Ejemplo ===\n")

    # Generar estructura FCC de ejemplo
    from crystal_generator import CrystalGenerator

    gen = CrystalGenerator(lattice_constant=3.615)
    positions_fcc = gen.generate_fcc(3, 3, 3)

    print("1. Analizando estructura FCC...")
    calc_fcc = CrystalFeatureCalculator(positions_fcc, cutoff=4.0)

    cn_fcc = calc_fcc.compute_coordination_numbers()
    print(f"   ✓ CN promedio: {cn_fcc.mean():.2f} ± {cn_fcc.std():.2f}")

    q4_fcc = calc_fcc.compute_steinhardt_q4()
    print(f"   ✓ Q4 promedio: {q4_fcc.mean():.4f}")

    structure_fcc = calc_fcc.classify_structure()
    print(f"   ✓ Estructura detectada: {structure_fcc}")

    # Generar estructura BCC
    positions_bcc = gen.generate_bcc(3, 3, 3)

    print("\n2. Analizando estructura BCC...")
    calc_bcc = CrystalFeatureCalculator(positions_bcc, cutoff=4.0)

    cn_bcc = calc_bcc.compute_coordination_numbers()
    print(f"   ✓ CN promedio: {cn_bcc.mean():.2f} ± {cn_bcc.std():.2f}")

    q4_bcc = calc_bcc.compute_steinhardt_q4()
    print(f"   ✓ Q4 promedio: {q4_bcc.mean():.4f}")

    structure_bcc = calc_bcc.classify_structure()
    print(f"   ✓ Estructura detectada: {structure_bcc}")

    # Calcular RDF
    print("\n3. Calculando RDF...")
    r, g_r = calc_fcc.compute_rdf(n_bins=50, r_max=10.0)
    peaks = r[g_r > 1.5]  # Encontrar picos significativos
    print(f"   ✓ Picos en g(r) en: {peaks[:3]} Å")

    # Feature vectors
    print("\n4. Calculando feature vectors...")
    features_fcc = calc_fcc.compute_feature_vector(n_angle_bins=18)
    print(f"   ✓ Dimensionalidad: {features_fcc.shape}")
    print(f"   ✓ Features por átomo: {features_fcc.shape[1]}")

    print("\n✓ Análisis completado!")
