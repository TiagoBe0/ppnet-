"""
Generador de labels.csv para Dataset de Vacancias
==================================================

Este script escanea un directorio con archivos .dump o .off y genera
un archivo labels.csv basándose en el nombre del archivo.

Formato esperado en nombres de archivo:
    - sim_vac_2_001.dump  -> clase: 2 (divacancia)
    - config_vac_10.dump  -> clase: 10 (10 vacancias)
    - estructura_vac_1_test.dump -> clase: 1 (monovacancia)

El script busca el patrón 'vac_N' donde N es el número de vacancias.

Output: labels.csv con formato:
    filename,label,label_name
    sim_vac_2_001.off,0,vac_2
    sim_vac_2_002.off,0,vac_2
    sim_vac_10_001.off,1,vac_10

Uso:
    python create_labels_csv.py --data_dir ./mis_datos --output labels.csv
"""

import argparse
import os
import re
import csv
from pathlib import Path
from collections import defaultdict


def extract_vacancy_number(filename: str) -> int:
    """
    Extrae el número de vacancias del nombre del archivo.

    Busca el patrón 'vac_N' donde N es un número.

    Args:
        filename: Nombre del archivo

    Returns:
        Número de vacancias, o None si no se encuentra el patrón

    Examples:
        >>> extract_vacancy_number('sim_vac_2_001.dump')
        2
        >>> extract_vacancy_number('config_vac_10.off')
        10
        >>> extract_vacancy_number('test_vac_1.dump')
        1
    """
    # Buscar patrón vac_N o vac_N_ o vac_N. o vac_N-
    match = re.search(r'vac[_-](\d+)', filename, re.IGNORECASE)

    if match:
        return int(match.group(1))

    return None


def scan_directory(data_dir: str, file_extension: str = '.off', recursive: bool = False):
    """
    Escanea un directorio y extrae información de los archivos.

    Args:
        data_dir: Directorio a escanear
        file_extension: Extensión de archivos a buscar (.off, .dump, etc.)
        recursive: Si True, busca recursivamente

    Returns:
        Lista de tuplas (filepath, num_vacancias)
    """
    path = Path(data_dir)

    # Buscar archivos
    if recursive:
        files = list(path.rglob(f'*{file_extension}'))
    else:
        files = list(path.glob(f'*{file_extension}'))

    # Extraer información
    file_info = []

    for filepath in files:
        num_vac = extract_vacancy_number(filepath.name)

        if num_vac is not None:
            file_info.append((filepath, num_vac))
        else:
            print(f"⚠️  Advertencia: No se pudo extraer número de vacancias de '{filepath.name}'")

    return file_info


def create_labels_csv(data_dir: str,
                     output_file: str,
                     file_extension: str = '.off',
                     recursive: bool = False,
                     relative_paths: bool = True):
    """
    Crea un archivo labels.csv basándose en los nombres de archivo.

    Args:
        data_dir: Directorio con los archivos
        output_file: Ruta del archivo CSV de salida
        file_extension: Extensión de archivos a procesar
        recursive: Si True, busca recursivamente
        relative_paths: Si True, usa rutas relativas en el CSV
    """
    print(f"\n{'='*70}")
    print(f"GENERADOR DE LABELS.CSV")
    print(f"{'='*70}\n")

    # Escanear directorio
    print(f"Escaneando: {data_dir}")
    print(f"Buscando archivos: *{file_extension}")
    print(f"Recursivo: {recursive}\n")

    file_info = scan_directory(data_dir, file_extension, recursive)

    if len(file_info) == 0:
        print(f"❌ No se encontraron archivos con patrón 'vac_N' en {data_dir}")
        return

    print(f"✓ Encontrados {len(file_info)} archivos con etiquetas\n")

    # Agrupar por número de vacancias
    vacancy_groups = defaultdict(list)
    for filepath, num_vac in file_info:
        vacancy_groups[num_vac].append(filepath)

    # Crear mapeo de num_vac -> label (índice)
    unique_vacancies = sorted(vacancy_groups.keys())
    vac_to_label = {vac: idx for idx, vac in enumerate(unique_vacancies)}

    print(f"Clases detectadas:")
    for vac, label_idx in vac_to_label.items():
        count = len(vacancy_groups[vac])
        print(f"  - vac_{vac}: {count} archivos (label={label_idx})")

    # Crear CSV
    print(f"\nCreando {output_file}...")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(['filename', 'label', 'label_name'])

        # Datos
        for filepath, num_vac in sorted(file_info, key=lambda x: (x[1], str(x[0]))):
            # Determinar ruta a escribir
            if relative_paths:
                try:
                    file_path_str = os.path.relpath(filepath, data_dir)
                except ValueError:
                    # En Windows, si están en drives diferentes
                    file_path_str = str(filepath)
            else:
                file_path_str = str(filepath)

            label_idx = vac_to_label[num_vac]
            label_name = f"vac_{num_vac}"

            writer.writerow([file_path_str, label_idx, label_name])

    print(f"✓ Archivo creado: {output_file}")

    # Resumen
    print(f"\n{'='*70}")
    print(f"RESUMEN")
    print(f"{'='*70}")
    print(f"Total de archivos: {len(file_info)}")
    print(f"Número de clases: {len(unique_vacancies)}")
    print(f"Clases: {', '.join([f'vac_{v}' for v in unique_vacancies])}")
    print(f"\nArchivo labels.csv listo para usar con:")
    print(f"  python train_crystal_classifier.py --labels_csv {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Genera labels.csv a partir de nombres de archivo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Generar labels.csv para archivos .off
  python create_labels_csv.py --data_dir ./datos_convertidos --output labels.csv

  # Buscar recursivamente en subdirectorios
  python create_labels_csv.py --data_dir ./datos --output labels.csv --recursive

  # Para archivos .dump (antes de convertir)
  python create_labels_csv.py --data_dir ./dumps --output labels.csv --extension .dump

  # Rutas absolutas en el CSV
  python create_labels_csv.py --data_dir ./datos --output labels.csv --absolute_paths

Formato de nombres esperado:
  - sim_vac_2_001.dump  → 2 vacancias (divacancia)
  - config_vac_10.dump  → 10 vacancias
  - test_vac_1.off      → 1 vacancia (monovacancia)
        """
    )

    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directorio con los archivos de datos')
    parser.add_argument('--output', type=str, default='labels.csv',
                       help='Ruta del archivo CSV de salida (default: labels.csv)')
    parser.add_argument('--extension', type=str, default='.off',
                       help='Extensión de archivos a procesar (default: .off)')
    parser.add_argument('--recursive', action='store_true',
                       help='Buscar archivos recursivamente en subdirectorios')
    parser.add_argument('--absolute_paths', action='store_true',
                       help='Usar rutas absolutas en lugar de relativas')

    args = parser.parse_args()

    # Verificar que el directorio existe
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: El directorio {args.data_dir} no existe")
        return

    # Generar CSV
    create_labels_csv(
        data_dir=args.data_dir,
        output_file=args.output,
        file_extension=args.extension,
        recursive=args.recursive,
        relative_paths=not args.absolute_paths
    )


if __name__ == '__main__':
    main()
