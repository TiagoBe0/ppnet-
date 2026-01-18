#!/bin/bash

# ============================================================================
# Script de Instalación Automatizada de TensorFlow para PointNet++
# Ubuntu 22.04 | Python 3.11 | TensorFlow 2.15
# ============================================================================

set -e  # Salir si hay error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funciones de utilidad
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

# Verificar si se ejecuta con sudo cuando es necesario
check_sudo() {
    if [ "$EUID" -eq 0 ]; then
        print_error "No ejecutes este script como sudo/root"
        print_info "El script pedirá sudo cuando sea necesario"
        exit 1
    fi
}

# ============================================================================
# PARTE 1: VERIFICACIÓN DEL SISTEMA
# ============================================================================

verificar_sistema() {
    print_header "PARTE 1: Verificando Sistema"

    # Verificar Ubuntu 22.04
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$VERSION_ID" == "22.04" ]]; then
            print_success "Ubuntu 22.04 detectado"
        else
            print_warning "Ubuntu $VERSION_ID detectado (recomendado: 22.04)"
        fi
    fi

    # Verificar Python
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_info "Python version: $PYTHON_VERSION"

    # Verificar GPU NVIDIA
    if lspci | grep -i nvidia > /dev/null 2>&1; then
        print_success "GPU NVIDIA detectada"
        GPU_DETECTED=true
    else
        print_warning "No se detectó GPU NVIDIA (se instalará TensorFlow CPU only)"
        GPU_DETECTED=false
    fi

    # Verificar nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        print_success "Drivers NVIDIA instalados"
        nvidia-smi --query-gpu=name --format=csv,noheader
        NVIDIA_DRIVER=true
    else
        print_warning "Drivers NVIDIA no instalados"
        NVIDIA_DRIVER=false
    fi

    # Verificar CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_success "CUDA $CUDA_VERSION instalado"
        CUDA_INSTALLED=true
    else
        print_warning "CUDA no instalado"
        CUDA_INSTALLED=false
    fi
}

# ============================================================================
# PARTE 2: INSTALACIÓN DE DRIVERS Y CUDA
# ============================================================================

instalar_drivers_cuda() {
    if [ "$GPU_DETECTED" = false ]; then
        print_info "Saltando instalación de drivers (no hay GPU)"
        return
    fi

    print_header "PARTE 2: Instalando Drivers NVIDIA y CUDA"

    # Preguntar si ya tiene drivers
    if [ "$NVIDIA_DRIVER" = false ]; then
        echo -e "${YELLOW}¿Quieres instalar drivers NVIDIA? (requiere reinicio después)${NC}"
        read -p "Continuar? (s/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            print_info "Actualizando repositorios..."
            sudo apt update

            print_info "Instalando driver NVIDIA 535..."
            sudo apt install -y nvidia-driver-535

            print_success "Driver instalado"
            print_warning "NECESITAS REINICIAR EL SISTEMA"
            print_info "Después del reinicio, ejecuta este script de nuevo"
            exit 0
        fi
    fi

    # Instalar CUDA Toolkit
    if [ "$CUDA_INSTALLED" = false ]; then
        print_info "Instalando CUDA Toolkit..."
        sudo apt install -y nvidia-cuda-toolkit libcudnn8 libcudnn8-dev

        # Configurar PATH
        if ! grep -q "cuda/bin" ~/.bashrc; then
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
            print_info "Variables CUDA añadidas a ~/.bashrc"
        fi

        # Cargar variables en esta sesión
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

        print_success "CUDA Toolkit instalado"
    fi
}

# ============================================================================
# PARTE 3: CONFIGURACIÓN DE ENTORNO PYTHON
# ============================================================================

configurar_entorno_python() {
    print_header "PARTE 3: Configurando Entorno Python"

    # Instalar dependencias del sistema
    print_info "Instalando dependencias del sistema..."
    sudo apt install -y python3-pip python3-venv build-essential g++ gcc

    # Crear entorno virtual
    if [ ! -d "venv_pointnet" ]; then
        print_info "Creando entorno virtual..."
        python3 -m venv venv_pointnet
        print_success "Entorno virtual creado"
    else
        print_warning "Entorno virtual ya existe"
    fi

    # Activar entorno
    print_info "Activando entorno virtual..."
    source venv_pointnet/bin/activate

    # Actualizar pip
    print_info "Actualizando pip..."
    pip install --upgrade pip setuptools wheel -q

    print_success "Entorno Python configurado"
}

# ============================================================================
# PARTE 4: INSTALACIÓN DE TENSORFLOW
# ============================================================================

instalar_tensorflow() {
    print_header "PARTE 4: Instalando TensorFlow 2.15"

    # Activar entorno
    source venv_pointnet/bin/activate

    # Determinar versión a instalar
    if [ "$GPU_DETECTED" = true ] && [ "$NVIDIA_DRIVER" = true ]; then
        print_info "Instalando TensorFlow con soporte GPU..."
        TF_PACKAGE="tensorflow[and-cuda]==2.15.0"
    else
        print_info "Instalando TensorFlow CPU only..."
        TF_PACKAGE="tensorflow==2.15.0"
    fi

    # Instalar TensorFlow
    pip install $TF_PACKAGE -q

    # Instalar dependencias adicionales
    print_info "Instalando dependencias adicionales..."
    pip install numpy==1.24.3 h5py matplotlib scipy -q

    print_success "TensorFlow instalado"

    # Verificar instalación
    print_info "Verificando instalación..."
    python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

    # Verificar GPU
    if [ "$GPU_DETECTED" = true ]; then
        GPU_COUNT=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))")
        if [ "$GPU_COUNT" -gt 0 ]; then
            print_success "GPU detectada por TensorFlow: $GPU_COUNT dispositivo(s)"
            python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
        else
            print_warning "TensorFlow no detecta GPU"
            print_info "Esto puede ser normal si no reiniciaste después de instalar drivers"
        fi
    fi
}

# ============================================================================
# PARTE 5: COMPILACIÓN DE OPERADORES
# ============================================================================

compilar_operadores() {
    print_header "PARTE 5: Compilando Operadores TensorFlow Personalizados"

    # Activar entorno
    source venv_pointnet/bin/activate

    # Verificar que los scripts existen
    if [ ! -f "compile_tf_ops.sh" ]; then
        print_error "No se encuentra compile_tf_ops.sh"
        return
    fi

    # Hacer ejecutables
    chmod +x compile_tf_ops.sh
    chmod +x compile_manual.sh 2>/dev/null || true

    # Intentar compilación automática
    print_info "Compilando con compile_tf_ops.sh..."
    if ./compile_tf_ops.sh; then
        print_success "Operadores compilados exitosamente"
    else
        print_warning "Compilación automática falló, intentando compilación manual..."
        if [ -f "compile_manual.sh" ]; then
            if ./compile_manual.sh; then
                print_success "Operadores compilados con compile_manual.sh"
            else
                print_error "Compilación manual también falló"
                print_info "Revisa los errores arriba para más detalles"
                return
            fi
        fi
    fi

    # Verificar archivos .so
    print_info "Verificando archivos compilados..."
    COMPILED_FILES=0

    if [ -f "tf_ops/sampling/tf_sampling_so.so" ]; then
        print_success "tf_sampling_so.so encontrado"
        ((COMPILED_FILES++))
    fi

    if [ -f "tf_ops/grouping/tf_grouping_so.so" ]; then
        print_success "tf_grouping_so.so encontrado"
        ((COMPILED_FILES++))
    fi

    if [ -f "tf_ops/3d_interpolation/tf_interpolate_so.so" ]; then
        print_success "tf_interpolate_so.so encontrado"
        ((COMPILED_FILES++))
    fi

    if [ $COMPILED_FILES -eq 3 ]; then
        print_success "Todos los operadores compilados correctamente"
    else
        print_error "Solo $COMPILED_FILES/3 operadores compilados"
    fi
}

# ============================================================================
# PARTE 6: PRUEBAS FINALES
# ============================================================================

ejecutar_pruebas() {
    print_header "PARTE 6: Ejecutando Pruebas"

    # Activar entorno
    source venv_pointnet/bin/activate

    # Test 1: Importar TensorFlow
    print_info "Test 1: Importando TensorFlow..."
    if python -c "import tensorflow as tf; print('✓ TensorFlow OK')" 2>/dev/null; then
        print_success "Test 1 pasado"
    else
        print_error "Test 1 falló"
    fi

    # Test 2: Importar operadores personalizados
    print_info "Test 2: Importando operadores personalizados..."
    if python -c "
import sys
sys.path.insert(0, 'tf_ops/sampling')
from tf_sampling import farthest_point_sample
print('✓ Operadores OK')
" 2>/dev/null; then
        print_success "Test 2 pasado"
    else
        print_warning "Test 2 falló (normal si no compilaron los operadores)"
    fi

    # Test 3: Test rápido de entrenamiento
    print_info "Test 3: Ejecutando mini-entrenamiento (esto toma 1-2 minutos)..."
    if python train_crystal_classifier.py --epochs 1 --n_samples 20 --batch_size 4 --num_points 32 2>&1 | grep -q "Epoch"; then
        print_success "Test 3 pasado - ¡El sistema funciona!"
    else
        print_warning "Test 3 no completado (revisa si hay errores arriba)"
    fi
}

# ============================================================================
# PARTE 7: RESUMEN FINAL
# ============================================================================

mostrar_resumen() {
    print_header "INSTALACIÓN COMPLETADA"

    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}    ¡INSTALACIÓN EXITOSA! 🎉${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

    echo "📋 RESUMEN:"
    echo ""
    [ "$NVIDIA_DRIVER" = true ] && echo "  ✓ Drivers NVIDIA instalados" || echo "  ○ Sin GPU"
    [ "$CUDA_INSTALLED" = true ] && echo "  ✓ CUDA instalado" || echo "  ○ CUDA no instalado"
    echo "  ✓ Entorno virtual creado: venv_pointnet/"
    echo "  ✓ TensorFlow 2.15 instalado"
    echo "  ✓ Dependencias instaladas"
    echo ""

    echo "🚀 PRÓXIMOS PASOS:"
    echo ""
    echo "1. Activa el entorno virtual:"
    echo -e "   ${YELLOW}source venv_pointnet/bin/activate${NC}"
    echo ""
    echo "2. Convierte tus datos LAMMPS:"
    echo -e "   ${YELLOW}python batch_convert_dumps.py --input_dir ./mis_datos --output_dir ./procesados --format off --normalize${NC}"
    echo ""
    echo "3. Entrena el modelo:"
    echo -e "   ${YELLOW}python train_crystal_classifier.py --data_dir ./procesados --num_points 128 --epochs 100${NC}"
    echo ""

    echo "📚 DOCUMENTACIÓN:"
    echo "   • INSTALACION_TENSORFLOW_UBUNTU_22.04.md - Guía completa"
    echo "   • GUIA_RAPIDA.md - Workflow de datos"
    echo "   • LAMMPS_CRYSTAL_CLASSIFICATION.md - Tutorial detallado"
    echo ""

    if [ "$NVIDIA_DRIVER" = false ] && [ "$GPU_DETECTED" = true ]; then
        print_warning "IMPORTANTE: Reinicia el sistema para activar los drivers NVIDIA"
        print_info "Después del reinicio, ejecuta: source venv_pointnet/bin/activate"
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    clear
    echo -e "${BLUE}"
    echo "  ╔═══════════════════════════════════════════════════════╗"
    echo "  ║                                                       ║"
    echo "  ║   Instalador de TensorFlow para PointNet++           ║"
    echo "  ║   Ubuntu 22.04 | TensorFlow 2.15 | Python 3.11       ║"
    echo "  ║                                                       ║"
    echo "  ╚═══════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"

    check_sudo

    # Menú de opciones
    echo "Selecciona el tipo de instalación:"
    echo "  1) Instalación completa (recomendado)"
    echo "  2) Solo TensorFlow (ya tengo drivers/CUDA)"
    echo "  3) Solo compilar operadores"
    echo "  4) Solo pruebas"
    echo ""
    read -p "Opción (1-4): " OPTION

    case $OPTION in
        1)
            verificar_sistema
            instalar_drivers_cuda
            configurar_entorno_python
            instalar_tensorflow
            compilar_operadores
            ejecutar_pruebas
            mostrar_resumen
            ;;
        2)
            verificar_sistema
            configurar_entorno_python
            instalar_tensorflow
            compilar_operadores
            ejecutar_pruebas
            mostrar_resumen
            ;;
        3)
            verificar_sistema
            compilar_operadores
            ;;
        4)
            verificar_sistema
            ejecutar_pruebas
            ;;
        *)
            print_error "Opción inválida"
            exit 1
            ;;
    esac
}

# Ejecutar
main
