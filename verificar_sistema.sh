#!/bin/bash
# Script para verificar que el sistema tiene todo lo necesario para compilar

echo "========================================================================"
echo "VERIFICACIÓN DEL SISTEMA PARA COMPILACIÓN DE OPERADORES TENSORFLOW"
echo "========================================================================"
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ok_count=0
warn_count=0
error_count=0

check_ok() {
    echo -e "${GREEN}✓${NC} $1"
    ((ok_count++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((warn_count++))
}

check_error() {
    echo -e "${RED}✗${NC} $1"
    ((error_count++))
}

echo "1. Verificando GPU NVIDIA..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        check_ok "GPU NVIDIA detectada"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
            echo "   └─ $line"
        done
    else
        check_error "nvidia-smi encontrado pero no funciona. Puede que los drivers no estén instalados correctamente"
    fi
else
    check_error "nvidia-smi no encontrado. No hay GPU NVIDIA o drivers no instalados"
    echo "   └─ Instala drivers NVIDIA desde: https://www.nvidia.com/Download/index.aspx"
fi
echo ""

echo "2. Verificando CUDA..."
if command -v nvcc &> /dev/null; then
    check_ok "nvcc encontrado"
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "   └─ Versión: $nvcc_version"
    CUDA_PATH=$(which nvcc | sed 's|/bin/nvcc||')
    echo "   └─ Ubicación: $CUDA_PATH"
else
    # Buscar CUDA en ubicaciones comunes
    found_cuda=false
    for cuda_dir in /usr/local/cuda /usr/local/cuda-* /opt/cuda; do
        if [ -d "$cuda_dir" ] && [ -f "$cuda_dir/bin/nvcc" ]; then
            check_warn "CUDA instalado pero nvcc no está en PATH"
            echo "   └─ Encontrado en: $cuda_dir"
            echo "   └─ Ejecuta estos comandos:"
            echo "      export PATH=$cuda_dir/bin:\$PATH"
            echo "      export LD_LIBRARY_PATH=$cuda_dir/lib64:\$LD_LIBRARY_PATH"
            found_cuda=true
            break
        fi
    done

    if [ "$found_cuda" = false ]; then
        check_error "CUDA no encontrado"
        echo "   └─ Necesitas instalar CUDA Toolkit"
        echo "   └─ Ver INSTRUCCIONES_COMPILACION.md (ANEXO A)"
    fi
fi
echo ""

echo "3. Verificando compilador g++..."
if command -v g++ &> /dev/null; then
    check_ok "g++ encontrado"
    gcc_version=$(g++ --version | head -n1)
    echo "   └─ $gcc_version"
else
    check_error "g++ no encontrado"
    echo "   └─ Instala con: sudo apt install build-essential"
fi
echo ""

echo "4. Verificando Python..."
if command -v python &> /dev/null; then
    check_ok "Python encontrado"
    python_version=$(python --version 2>&1)
    echo "   └─ $python_version"

    # Verificar que es Python 3
    if python -c "import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)" 2>/dev/null; then
        check_ok "Python 3.x confirmado"
    else
        check_warn "Estás usando Python 2.x, pero el código requiere Python 3.x"
    fi
else
    check_error "Python no encontrado"
fi
echo ""

echo "5. Verificando TensorFlow..."
if python -c "import tensorflow" 2>/dev/null; then
    check_ok "TensorFlow instalado"
    tf_version=$(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
    echo "   └─ Versión: $tf_version"

    # Verificar GPUs detectadas
    gpu_count=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null)
    if [ "$gpu_count" -gt 0 ]; then
        check_ok "TensorFlow detectó $gpu_count GPU(s)"
    else
        check_warn "TensorFlow no detecta GPUs"
        echo "   └─ El entrenamiento será extremadamente lento sin GPU"
        echo "   └─ Verifica que TensorFlow esté instalado con soporte GPU"
        echo "   └─ Reinstala con: pip install tensorflow[and-cuda]"
    fi

    # Verificar que puede obtener configuración de compilación
    if python -c "import tensorflow as tf; tf.sysconfig.get_compile_flags()" 2>/dev/null; then
        check_ok "Configuración de compilación de TensorFlow disponible"
    else
        check_error "No se puede obtener configuración de compilación de TensorFlow"
    fi
else
    check_error "TensorFlow no está instalado"
    echo "   └─ Instala con: pip install tensorflow[and-cuda]"
fi
echo ""

echo "6. Verificando operadores ya compilados..."
ops_found=0
if [ -f "tf_ops/sampling/tf_sampling_so.so" ]; then
    check_ok "tf_sampling_so.so encontrado"
    ((ops_found++))
else
    check_warn "tf_sampling_so.so no encontrado (necesita compilación)"
fi

if [ -f "tf_ops/grouping/tf_grouping_so.so" ]; then
    check_ok "tf_grouping_so.so encontrado"
    ((ops_found++))
else
    check_warn "tf_grouping_so.so no encontrado (necesita compilación)"
fi

if [ -f "tf_ops/3d_interpolation/tf_interpolate_so.so" ]; then
    check_ok "tf_interpolate_so.so encontrado"
    ((ops_found++))
else
    check_warn "tf_interpolate_so.so no encontrado (necesita compilación)"
fi

if [ $ops_found -eq 3 ]; then
    echo "   └─ Todos los operadores están compilados"
else
    echo "   └─ Faltan $(( 3 - ops_found )) operadores por compilar"
fi
echo ""

echo "========================================================================"
echo "RESUMEN"
echo "========================================================================"
echo -e "${GREEN}✓ Correcto: $ok_count${NC}"
echo -e "${YELLOW}⚠ Advertencias: $warn_count${NC}"
echo -e "${RED}✗ Errores: $error_count${NC}"
echo ""

if [ $error_count -eq 0 ] && [ $warn_count -eq 0 ]; then
    echo -e "${GREEN}¡Todo listo para compilar!${NC}"
    echo ""
    echo "Ejecuta:"
    echo "  ./compile_tf_ops.sh"
    echo "o"
    echo "  ./compile_manual.sh"
elif [ $error_count -eq 0 ]; then
    echo -e "${YELLOW}Sistema listo pero con advertencias${NC}"
    echo "Puedes intentar compilar, pero puede haber problemas."
    echo ""
    echo "Ejecuta:"
    echo "  ./compile_tf_ops.sh"
else
    echo -e "${RED}Hay errores que deben resolverse antes de compilar${NC}"
    echo ""
    echo "Pasos recomendados:"
    if ! command -v nvidia-smi &> /dev/null; then
        echo "1. Instala drivers NVIDIA"
    fi
    if ! command -v nvcc &> /dev/null; then
        echo "2. Instala CUDA Toolkit (ver INSTRUCCIONES_COMPILACION.md)"
    fi
    if ! python -c "import tensorflow" 2>/dev/null; then
        echo "3. Instala TensorFlow: pip install tensorflow[and-cuda]"
    fi
fi
echo ""
