#!/bin/bash

# ============================================================================
# Script para Parchar Código TensorFlow 1.x → TensorFlow 2.x (Modo Compat)
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# BACKUP de archivos originales
# ============================================================================

backup_file() {
    local file=$1
    if [ -f "$file" ] && [ ! -f "${file}.bak" ]; then
        cp "$file" "${file}.bak"
        print_info "Backup creado: ${file}.bak"
    fi
}

# ============================================================================
# APLICAR PARCHES
# ============================================================================

patch_file() {
    local file=$1
    local description=$2

    if [ ! -f "$file" ]; then
        print_warning "Archivo no encontrado: $file (saltando)"
        return
    fi

    print_info "Parcheando: $file"
    backup_file "$file"

    # Verificar si ya está parcheado
    if grep -q "tensorflow.compat.v1" "$file"; then
        print_success "$file ya está parcheado"
        return
    fi

    # Crear archivo temporal
    local temp_file="${file}.tmp"

    # Leer el archivo línea por línea
    local in_imports=true
    local added_compat=false

    while IFS= read -r line; do
        # Si encontramos "import tensorflow as tf" sin compat
        if echo "$line" | grep -q "^import tensorflow as tf$"; then
            # Añadir versión de compatibilidad
            echo "import os"
            echo "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir warnings"
            echo "import tensorflow.compat.v1 as tf"
            echo "tf.disable_eager_execution()"
            added_compat=true
        elif echo "$line" | grep -q "^from tensorflow import"; then
            # Cambiar from tensorflow import X a usar compat
            echo "$line" | sed 's/from tensorflow import/from tensorflow.compat.v1 import/'
        else
            echo "$line"
        fi
    done < "$file" > "$temp_file"

    # Mover archivo temporal al original
    mv "$temp_file" "$file"

    if [ "$added_compat" = true ]; then
        print_success "$description"
    else
        print_warning "$description (no se encontró import tensorflow)"
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    clear
    print_header "Parchar Código para TensorFlow 2.x (Modo Compatibilidad)"

    echo "Este script actualiza el código de TensorFlow 1.x para que funcione"
    echo "en TensorFlow 2.x usando tf.compat.v1"
    echo ""
    echo "Se crearán backups (.bak) de todos los archivos modificados."
    echo ""
    read -p "¿Continuar? (s/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        print_error "Operación cancelada"
        exit 1
    fi

    print_header "PASO 1: Parcheando Archivos de Entrenamiento"

    patch_file "train.py" "train.py parcheado"
    patch_file "train_multi_gpu.py" "train_multi_gpu.py parcheado"
    patch_file "train_crystal_classifier.py" "train_crystal_classifier.py parcheado"
    patch_file "evaluate.py" "evaluate.py parcheado"

    print_header "PASO 2: Parcheando Modelos"

    for model_file in models/*.py; do
        if [ -f "$model_file" ]; then
            filename=$(basename "$model_file")
            patch_file "$model_file" "Modelo $filename parcheado"
        fi
    done

    print_header "PASO 3: Parcheando Utilities"

    patch_file "utils/tf_util.py" "tf_util.py parcheado"
    patch_file "utils/pointnet_util.py" "pointnet_util.py parcheado" 2>/dev/null || true

    print_header "PASO 4: Parcheando Datasets"

    patch_file "modelnet_dataset.py" "modelnet_dataset.py parcheado"
    patch_file "modelnet_h5_dataset.py" "modelnet_h5_dataset.py parcheado"
    patch_file "crystal_dataset.py" "crystal_dataset.py parcheado"

    print_header "PASO 5: Parcheando Part Segmentation"

    if [ -d "part_seg" ]; then
        for seg_file in part_seg/*.py; do
            if [ -f "$seg_file" ]; then
                filename=$(basename "$seg_file")
                patch_file "$seg_file" "Part seg: $filename parcheado"
            fi
        done
    fi

    print_header "PASO 6: Parcheando ScanNet"

    if [ -d "scannet" ]; then
        for scan_file in scannet/*.py; do
            if [ -f "$scan_file" ]; then
                filename=$(basename "$scan_file")
                patch_file "$scan_file" "ScanNet: $filename parcheado"
            fi
        done
    fi

    print_header "PASO 7: Parcheando Operadores TF"

    # Los operadores personalizados ya deberían funcionar con TF 2.x
    # pero vamos a parchar los archivos Python auxiliares

    for op_dir in tf_ops/*/; do
        if [ -d "$op_dir" ]; then
            for py_file in ${op_dir}*.py; do
                if [ -f "$py_file" ] && [[ $(basename "$py_file") != "test"* ]]; then
                    filename=$(basename "$py_file")
                    patch_file "$py_file" "TF op: $filename parcheado"
                fi
            done
        fi
    done

    print_header "PARCHEADO COMPLETADO"

    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}    ¡PARCHEADO EXITOSO! 🎉${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

    echo "📋 CAMBIOS REALIZADOS:"
    echo ""
    echo "  ✓ Imports actualizados a tensorflow.compat.v1"
    echo "  ✓ Eager execution desactivado"
    echo "  ✓ Warnings de TF reducidos"
    echo "  ✓ Backups creados (.bak)"
    echo ""

    echo "🧪 PRÓXIMOS PASOS:"
    echo ""
    echo "1. Probar imports:"
    echo -e "   ${YELLOW}python -c 'import tensorflow.compat.v1 as tf; import sys; sys.path.append(\"models\"); import pointnet2_cls_ssg; print(\"✓ OK\")'${NC}"
    echo ""
    echo "2. Test mini-entrenamiento:"
    echo -e "   ${YELLOW}python train_crystal_classifier.py --epochs 1 --n_samples 20 --batch_size 4${NC}"
    echo ""

    echo "⚠️  SI ALGO FALLA:"
    echo ""
    echo "Restaurar backups:"
    echo -e "   ${YELLOW}for f in **/*.bak; do mv \"\$f\" \"\${f%.bak}\"; done${NC}"
    echo ""

    print_success "Listo para usar TensorFlow 2.x en modo compatibilidad!"
}

main
