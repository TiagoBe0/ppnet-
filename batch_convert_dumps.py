"""
Script para convertir múltiples archivos LAMMPS .dump a formato .off

Uso:
    # Convertir todos los .dump en un directorio
    python batch_convert_dumps.py --input_dir ./dumps --output_dir ./off_files

    # Convertir con opciones específicas
    python batch_convert_dumps.py --input_dir ./dumps --output_dir ./off_files --normalize --cutoff 5.0

    # Mantener estructura de subdirectorios (útil para datasets organizados por clase)
    python batch_convert_dumps.py --input_dir ./datos/fcc --output_dir ./procesados/fcc --keep_structure
"""

import argparse
import os
import sys
from pathlib import Path

# Try to import tqdm, fallback to simple progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc="Progress"):
        """Simple fallback for tqdm"""
        total = len(list(iterable)) if hasattr(iterable, '__len__') else 0
        for i, item in enumerate(iterable):
            if total > 0:
                print(f"\r{desc}: {i+1}/{total} ({100*(i+1)/total:.1f}%)", end='', flush=True)
            yield item
        print()  # New line after completion

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from lammps_to_off import convert_lammps_to_off, convert_lammps_to_ply_with_features


def find_dump_files(directory: str, recursive: bool = False) -> list:
    """
    Encuentra todos los archivos .dump en un directorio.

    Args:
        directory: Ruta del directorio a buscar
        recursive: Si True, busca recursivamente en subdirectorios

    Returns:
        Lista de rutas de archivos .dump
    """
    dump_files = []
    path = Path(directory)

    if recursive:
        dump_files = list(path.rglob('*.dump'))
    else:
        dump_files = list(path.glob('*.dump'))

    return [str(f) for f in dump_files]


def batch_convert(input_dir: str,
                 output_dir: str,
                 normalize: bool = True,
                 norm_method: str = 'sphere',
                 output_format: str = 'off',
                 cutoff: float = 4.0,
                 keep_structure: bool = False,
                 recursive: bool = False):
    """
    Convierte múltiples archivos .dump a formato OFF/PLY.

    Args:
        input_dir: Directorio con archivos .dump
        output_dir: Directorio de salida
        normalize: Si True, normaliza las coordenadas
        norm_method: Método de normalización ('sphere', 'box', 'center')
        output_format: Formato de salida ('off' o 'ply')
        cutoff: Radio de corte para features cristalográficas (solo para PLY)
        keep_structure: Si True, mantiene estructura de subdirectorios
        recursive: Si True, busca archivos recursivamente
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Encontrar archivos
    print(f"\nBuscando archivos .dump en {input_dir}...")
    dump_files = find_dump_files(input_dir, recursive=recursive)

    if len(dump_files) == 0:
        print(f"❌ No se encontraron archivos .dump en {input_dir}")
        return

    print(f"✓ Encontrados {len(dump_files)} archivos\n")

    # Convertir archivos
    successful = 0
    failed = 0
    errors = []

    print(f"Convirtiendo a formato {output_format.upper()}...")

    for dump_file in tqdm(dump_files, desc="Progreso"):
        try:
            # Determinar ruta de salida
            if keep_structure:
                # Mantener estructura de directorios
                rel_path = os.path.relpath(dump_file, input_dir)
                output_path = os.path.join(output_dir, rel_path)
            else:
                # Colocar todos los archivos en output_dir
                output_path = os.path.join(output_dir, os.path.basename(dump_file))

            # Cambiar extensión
            output_path = output_path.replace('.dump', f'.{output_format}')

            # Crear subdirectorios si es necesario
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Convertir
            if output_format == 'off':
                stats = convert_lammps_to_off(
                    dump_file=dump_file,
                    off_file=output_path,
                    normalize=normalize,
                    norm_method=norm_method
                )
            elif output_format == 'ply':
                stats = convert_lammps_to_ply_with_features(
                    dump_file=dump_file,
                    ply_file=output_path,
                    cutoff=cutoff,
                    normalize=normalize
                )
            else:
                raise ValueError(f"Formato {output_format} no soportado")

            successful += 1

        except Exception as e:
            failed += 1
            errors.append((dump_file, str(e)))

    # Resumen
    print(f"\n{'='*70}")
    print(f"RESUMEN DE CONVERSIÓN")
    print(f"{'='*70}")
    print(f"✓ Exitosas: {successful}")
    print(f"✗ Fallidas: {failed}")

    if failed > 0:
        print(f"\nErrores encontrados:")
        for dump_file, error in errors[:10]:  # Mostrar solo primeros 10 errores
            print(f"  - {os.path.basename(dump_file)}: {error}")
        if len(errors) > 10:
            print(f"  ... y {len(errors) - 10} errores más")

    print(f"\nArchivos convertidos guardados en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convierte archivos LAMMPS .dump a formato OFF/PLY',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Convertir todos los .dump en un directorio
  python batch_convert_dumps.py --input_dir ./dumps --output_dir ./off_files

  # Convertir a PLY con features cristalográficas
  python batch_convert_dumps.py --input_dir ./dumps --output_dir ./ply_files --format ply

  # Mantener estructura de directorios (útil para datasets organizados)
  python batch_convert_dumps.py --input_dir ./datos --output_dir ./procesados --keep_structure --recursive

  # Conversión para dataset de entrenamiento
  python batch_convert_dumps.py --input_dir ./training_data --output_dir ./processed --normalize --format off
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directorio con archivos .dump')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directorio de salida')
    parser.add_argument('--format', type=str, default='off', choices=['off', 'ply'],
                       help='Formato de salida (default: off)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalizar coordenadas')
    parser.add_argument('--norm_method', type=str, default='sphere',
                       choices=['sphere', 'box', 'center'],
                       help='Método de normalización (default: sphere)')
    parser.add_argument('--cutoff', type=float, default=4.0,
                       help='Radio de corte para features (solo para PLY, default: 4.0)')
    parser.add_argument('--keep_structure', action='store_true',
                       help='Mantener estructura de subdirectorios')
    parser.add_argument('--recursive', action='store_true',
                       help='Buscar archivos recursivamente en subdirectorios')

    args = parser.parse_args()

    # Verificar que el directorio de entrada existe
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: El directorio {args.input_dir} no existe")
        sys.exit(1)

    # Mostrar configuración
    print("="*70)
    print("CONVERSIÓN POR LOTES DE ARCHIVOS LAMMPS")
    print("="*70)
    print(f"\nConfiguración:")
    print(f"  Input:      {args.input_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Formato:    {args.format.upper()}")
    print(f"  Normalizar: {args.normalize}")
    if args.normalize:
        print(f"  Método:     {args.norm_method}")
    if args.format == 'ply':
        print(f"  Cutoff:     {args.cutoff} Å")
    print(f"  Recursivo:  {args.recursive}")
    print(f"  Mantener estructura: {args.keep_structure}")

    # Convertir
    batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        normalize=args.normalize,
        norm_method=args.norm_method,
        output_format=args.format,
        cutoff=args.cutoff,
        keep_structure=args.keep_structure,
        recursive=args.recursive
    )

    print("\n✓ Conversión completada!")


if __name__ == '__main__':
    main()
