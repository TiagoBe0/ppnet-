"""
Dataset de Estructuras Cristalinas para PointNet++
===================================================

Este módulo proporciona una clase de dataset para entrenar PointNet++
en la tarea de clasificación de estructuras cristalinas.

Compatible con el pipeline de entrenamiento existente de PointNet++.

Soporta:
- Carga de archivos LAMMPS .dump
- Carga de archivos .off
- Generación sintética de estructuras
- Augmentación de datos (rotación, ruido, scaling)
"""

import os
import sys
import numpy as np
import pickle
from typing import List, Tuple, Optional

# Añadir directorio utils al path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from crystal_generator import CrystalGenerator, generate_crystal_dataset
from lammps_reader import read_lammps_dump
from lammps_to_off import read_off, normalize_point_cloud


class CrystalDataset:
    """
    Dataset de estructuras cristalinas.

    Attributes:
        data (List[np.ndarray]): Lista de nubes de puntos (N_i, 3)
        labels (List[int]): Lista de etiquetas
        class_names (List[str]): Nombres de las clases
        num_points (int): Número de puntos a muestrear por nube
    """

    def __init__(self,
                 num_points: int = 1024,
                 split: str = 'train',
                 data_augmentation: bool = True,
                 normalize: bool = True):
        """
        Inicializa el dataset.

        Args:
            num_points: Número de puntos a muestrear de cada nube
            split: 'train', 'val', o 'test'
            data_augmentation: Si True, aplica augmentación de datos
            normalize: Si True, normaliza las nubes de puntos
        """
        self.num_points = num_points
        self.split = split
        self.data_augmentation = data_augmentation
        self.normalize = normalize

        self.data = []
        self.labels = []
        self.class_names = []

    def load_from_directory(self,
                           data_dir: str,
                           file_extension: str = '.off') -> None:
        """
        Carga datos desde un directorio organizado por clases.

        Estructura esperada:
        data_dir/
            fcc/
                sample1.off
                sample2.off
            bcc/
                sample1.off
            hcp/
                sample1.off

        Args:
            data_dir: Directorio raíz de los datos
            file_extension: Extensión de archivos a cargar (.off, .dump, etc.)
        """
        # Listar subdirectorios (clases)
        class_dirs = [d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))]
        class_dirs.sort()

        self.class_names = class_dirs
        print(f"Clases encontradas: {self.class_names}")

        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(data_dir, class_name)

            # Listar archivos en la clase
            files = [f for f in os.listdir(class_dir)
                    if f.endswith(file_extension)]

            print(f"Cargando {class_name}: {len(files)} archivos...")

            for filename in files:
                filepath = os.path.join(class_dir, filename)

                try:
                    # Cargar según extensión
                    if file_extension == '.off':
                        positions, _ = read_off(filepath)
                    elif file_extension == '.dump':
                        positions = read_lammps_dump(filepath, center=True)
                    else:
                        raise ValueError(f"Extensión {file_extension} no soportada")

                    # Normalizar si se solicita
                    if self.normalize:
                        positions = normalize_point_cloud(positions, method='sphere')

                    self.data.append(positions)
                    self.labels.append(class_idx)

                except Exception as e:
                    print(f"  Error cargando {filename}: {e}")

        print(f"\nTotal cargado: {len(self.data)} muestras")

    def load_from_csv(self,
                     labels_csv: str,
                     data_dir: Optional[str] = None,
                     file_extension: str = '.off') -> None:
        """
        Carga datos usando un archivo CSV de etiquetas.

        El CSV debe tener el formato:
            filename,label,label_name
            sim_vac_2_001.off,0,vac_2
            sim_vac_2_002.off,0,vac_2
            sim_vac_10_001.off,1,vac_10

        Args:
            labels_csv: Ruta al archivo CSV con etiquetas
            data_dir: Directorio base para rutas relativas (si None, usa dir del CSV)
            file_extension: Extensión esperada de archivos

        Example:
            >>> dataset = CrystalDataset(num_points=64)
            >>> dataset.load_from_csv('labels.csv', data_dir='./datos')
        """
        import csv

        # Determinar directorio base
        if data_dir is None:
            data_dir = os.path.dirname(labels_csv)

        print(f"Cargando datos desde CSV: {labels_csv}")
        print(f"Directorio base: {data_dir}\n")

        # Leer CSV
        with open(labels_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            print("❌ Error: El archivo CSV está vacío")
            return

        print(f"Encontradas {len(rows)} entradas en el CSV")

        # Extraer nombres de clases únicos
        label_to_name = {}
        for row in rows:
            label_idx = int(row['label'])
            label_name = row['label_name']
            if label_idx not in label_to_name:
                label_to_name[label_idx] = label_name

        # Ordenar por índice de label
        self.class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]

        print(f"Clases detectadas: {self.class_names}\n")

        # Cargar archivos
        errors = []
        loaded_count = 0

        for row in rows:
            filename = row['filename']
            label = int(row['label'])

            # Construir ruta completa
            if os.path.isabs(filename):
                filepath = filename
            else:
                filepath = os.path.join(data_dir, filename)

            try:
                # Cargar según extensión
                if filepath.endswith('.off') or file_extension == '.off':
                    positions, _ = read_off(filepath)
                elif filepath.endswith('.dump') or file_extension == '.dump':
                    positions = read_lammps_dump(filepath, center=True)
                else:
                    raise ValueError(f"Extensión no soportada: {filepath}")

                # Normalizar si se solicita
                if self.normalize:
                    positions = normalize_point_cloud(positions, method='sphere')

                self.data.append(positions)
                self.labels.append(label)
                loaded_count += 1

            except Exception as e:
                errors.append((filename, str(e)))

        # Resumen
        print(f"\n✓ Total cargado: {loaded_count} muestras")

        if errors:
            print(f"⚠️  Errores: {len(errors)}")
            for filename, error in errors[:5]:  # Mostrar primeros 5
                print(f"   - {filename}: {error}")
            if len(errors) > 5:
                print(f"   ... y {len(errors) - 5} errores más")

        # Mostrar distribución de clases
        if loaded_count > 0:
            import numpy as np
            unique, counts = np.unique(self.labels, return_counts=True)
            print(f"\nDistribución de clases:")
            for label_idx, count in zip(unique, counts):
                print(f"  {self.class_names[label_idx]}: {count} muestras")

    def generate_synthetic(self,
                          n_samples_per_class: int = 1000,
                          crystal_types: List[str] = ['fcc', 'bcc', 'hcp'],
                          save_to_disk: bool = False,
                          save_dir: Optional[str] = None) -> None:
        """
        Genera datos sintéticos de estructuras cristalinas.

        Args:
            n_samples_per_class: Número de muestras por clase
            crystal_types: Tipos de cristal a generar
            save_to_disk: Si True, guarda los datos generados
            save_dir: Directorio donde guardar (si save_to_disk=True)
        """
        print(f"Generando dataset sintético...")
        print(f"  - {n_samples_per_class} muestras por clase")
        print(f"  - Clases: {crystal_types}")

        self.data, self.labels = generate_crystal_dataset(
            n_samples_per_class=n_samples_per_class,
            crystal_types=crystal_types,
            add_noise=True,
            noise_level=0.1
        )

        self.class_names = crystal_types

        # Normalizar
        if self.normalize:
            self.data = [normalize_point_cloud(pc, method='sphere')
                        for pc in self.data]

        print(f"✓ Generadas {len(self.data)} muestras")

        # Guardar a disco si se solicita
        if save_to_disk and save_dir is not None:
            self._save_to_disk(save_dir)

    def _save_to_disk(self, save_dir: str) -> None:
        """Guarda el dataset a disco en formato pickle."""
        os.makedirs(save_dir, exist_ok=True)

        data_dict = {
            'data': self.data,
            'labels': self.labels,
            'class_names': self.class_names
        }

        filepath = os.path.join(save_dir, f'crystal_dataset_{self.split}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"✓ Dataset guardado en: {filepath}")

    def load_from_pickle(self, filepath: str) -> None:
        """Carga el dataset desde un archivo pickle."""
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)

        self.data = data_dict['data']
        self.labels = data_dict['labels']
        self.class_names = data_dict['class_names']

        print(f"✓ Dataset cargado desde: {filepath}")
        print(f"  - {len(self.data)} muestras")
        print(f"  - {len(self.class_names)} clases")

    def __len__(self) -> int:
        """Retorna el número de muestras."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Obtiene una muestra del dataset.

        Args:
            idx: Índice de la muestra

        Returns:
            Tuple de (point_cloud, label) donde:
            - point_cloud: array (num_points, 3)
            - label: int (índice de clase)
        """
        positions = self.data[idx].copy()
        label = self.labels[idx]

        # Muestrear/rellenar a num_points
        positions = self._sample_points(positions)

        # Aplicar augmentación si está en modo train
        if self.data_augmentation and self.split == 'train':
            positions = self._augment(positions)

        return positions, label

    def _sample_points(self, positions: np.ndarray) -> np.ndarray:
        """
        Muestrea o rellena para obtener exactamente num_points.

        Args:
            positions: Array (N, 3) de posiciones

        Returns:
            Array (num_points, 3)
        """
        n_points = len(positions)

        if n_points == self.num_points:
            return positions
        elif n_points > self.num_points:
            # Muestreo aleatorio
            indices = np.random.choice(n_points, self.num_points, replace=False)
            return positions[indices]
        else:
            # Rellenar con muestreo aleatorio con reemplazo
            indices = np.random.choice(n_points, self.num_points, replace=True)
            return positions[indices]

    def _augment(self, positions: np.ndarray) -> np.ndarray:
        """
        Aplica augmentación de datos.

        Transformaciones:
        - Rotación aleatoria
        - Escalado aleatorio
        - Jitter (ruido pequeño)

        Args:
            positions: Array (N, 3) de posiciones

        Returns:
            Posiciones augmentadas
        """
        # Rotación aleatoria
        positions = self._rotate_point_cloud(positions)

        # Escalado aleatorio (±10%)
        scale = np.random.uniform(0.9, 1.1)
        positions *= scale

        # Jitter
        noise = np.random.normal(0, 0.02, positions.shape)
        positions += noise

        return positions

    def _rotate_point_cloud(self, positions: np.ndarray) -> np.ndarray:
        """
        Rota la nube de puntos aleatoriamente.

        Args:
            positions: Array (N, 3)

        Returns:
            Posiciones rotadas
        """
        # Ángulos aleatorios
        angles = np.random.uniform(0, 2 * np.pi, 3)

        # Matrices de rotación
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])

        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])

        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])

        # Aplicar rotaciones
        R = Rz @ Ry @ Rx
        return positions @ R.T

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera un batch de datos.

        Args:
            batch_size: Tamaño del batch

        Returns:
            Tuple de (batch_data, batch_labels) donde:
            - batch_data: array (batch_size, num_points, 3)
            - batch_labels: array (batch_size,)
        """
        indices = np.random.choice(len(self), batch_size, replace=False)

        batch_data = []
        batch_labels = []

        for idx in indices:
            data, label = self[idx]
            batch_data.append(data)
            batch_labels.append(label)

        return np.array(batch_data), np.array(batch_labels)

    def split_train_val(self, val_fraction: float = 0.2) -> Tuple['CrystalDataset', 'CrystalDataset']:
        """
        Divide el dataset en train y validación.

        Args:
            val_fraction: Fracción de datos para validación

        Returns:
            Tuple de (train_dataset, val_dataset)
        """
        n_samples = len(self)
        n_val = int(n_samples * val_fraction)

        # Shuffle indices
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        # Crear datasets
        train_dataset = CrystalDataset(
            num_points=self.num_points,
            split='train',
            data_augmentation=self.data_augmentation,
            normalize=self.normalize
        )
        train_dataset.data = [self.data[i] for i in train_indices]
        train_dataset.labels = [self.labels[i] for i in train_indices]
        train_dataset.class_names = self.class_names

        val_dataset = CrystalDataset(
            num_points=self.num_points,
            split='val',
            data_augmentation=False,  # No augmentation for validation
            normalize=self.normalize
        )
        val_dataset.data = [self.data[i] for i in val_indices]
        val_dataset.labels = [self.labels[i] for i in val_indices]
        val_dataset.class_names = self.class_names

        return train_dataset, val_dataset


if __name__ == '__main__':
    print("=== Crystal Dataset - Ejemplo de Uso ===\n")

    # 1. Generar dataset sintético
    print("1. Generando dataset sintético...")
    dataset = CrystalDataset(num_points=1024, split='train', data_augmentation=True)
    dataset.generate_synthetic(
        n_samples_per_class=100,
        crystal_types=['fcc', 'bcc', 'hcp']
    )

    # 2. Información del dataset
    print(f"\n2. Información del dataset:")
    print(f"   - Número de muestras: {len(dataset)}")
    print(f"   - Clases: {dataset.class_names}")
    print(f"   - Puntos por muestra: {dataset.num_points}")

    # 3. Obtener una muestra
    print(f"\n3. Obteniendo muestra...")
    points, label = dataset[0]
    print(f"   - Shape de puntos: {points.shape}")
    print(f"   - Etiqueta: {label} ({dataset.class_names[label]})")
    print(f"   - Rango de coordenadas: [{points.min():.3f}, {points.max():.3f}]")

    # 4. Obtener un batch
    print(f"\n4. Generando batch...")
    batch_data, batch_labels = dataset.get_batch(batch_size=8)
    print(f"   - Shape del batch: {batch_data.shape}")
    print(f"   - Etiquetas: {batch_labels}")

    # 5. División train/val
    print(f"\n5. Dividiendo en train/val...")
    train_ds, val_ds = dataset.split_train_val(val_fraction=0.2)
    print(f"   - Train: {len(train_ds)} muestras")
    print(f"   - Val: {len(val_ds)} muestras")

    # 6. Guardar dataset
    print(f"\n6. Guardando dataset...")
    dataset._save_to_disk('/tmp/crystal_data')

    # 7. Cargar dataset
    print(f"\n7. Cargando dataset guardado...")
    new_dataset = CrystalDataset(num_points=1024)
    new_dataset.load_from_pickle('/tmp/crystal_data/crystal_dataset_train.pkl')

    print("\n✓ Todas las operaciones completadas exitosamente!")
