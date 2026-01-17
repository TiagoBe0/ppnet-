"""
Script de Entrenamiento para Clasificación de Estructuras Cristalinas
======================================================================

Este script entrena PointNet++ para clasificar estructuras cristalinas
(FCC, BCC, HCP, etc.) a partir de simulaciones LAMMPS.

Tutorial completo que incluye:
1. Generación de dataset sintético
2. Configuración del modelo PointNet++
3. Loop de entrenamiento
4. Evaluación y visualización de resultados

Uso:
    python train_crystal_classifier.py --epochs 100 --batch_size 32

Autor: Tutorial educativo para detección de patrones cristalinos
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

# Añadir paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from crystal_dataset import CrystalDataset
import provider

# Importar modelo PointNet++
import pointnet2_cls_ssg as MODEL


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Train PointNet++ for Crystal Classification')

    # Parámetros de datos
    parser.add_argument('--num_points', type=int, default=512,
                       help='Número de puntos por muestra [default: 512]')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Muestras por clase para entrenamiento [default: 500]')
    parser.add_argument('--crystal_types', nargs='+', default=['fcc', 'bcc', 'hcp'],
                       help='Tipos de cristal a clasificar [default: fcc bcc hcp]')

    # Parámetros de entrenamiento
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size [default: 16]')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de epochs [default: 50]')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate inicial [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum para optimizer [default: 0.9]')
    parser.add_argument('--decay_step', type=int, default=200000,
                       help='Decay step para learning rate [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                       help='Decay rate para learning rate [default: 0.7]')

    # Parámetros del modelo
    parser.add_argument('--model', type=str, default='pointnet2_cls_ssg',
                       help='Modelo a usar [default: pointnet2_cls_ssg]')

    # Directorios
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directorio para logs [default: log/<timestamp>]')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directorio con datos (si no se genera sintético)')

    # Opciones
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU a usar [default: 0]')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Desactivar augmentación de datos')

    return parser.parse_args()


def get_learning_rate(batch, base_lr, batch_size, decay_step, decay_rate):
    """
    Calcula learning rate con decaimiento exponencial.

    Args:
        batch: Número de batch actual (global step)
        base_lr: Learning rate base
        batch_size: Tamaño del batch
        decay_step: Steps entre decaimientos
        decay_rate: Factor de decaimiento

    Returns:
        Learning rate actual
    """
    learning_rate = tf.train.exponential_decay(
        base_lr,  # Base learning rate
        batch * batch_size,  # Current index into dataset
        decay_step,  # Decay step
        decay_rate,  # Decay rate
        staircase=True
    )
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch, bn_init_decay, batch_size, bn_decay_step, bn_decay_rate, bn_decay_clip):
    """
    Calcula batch normalization decay.

    Args:
        batch: Batch actual
        bn_init_decay: Decay inicial para BN
        batch_size: Tamaño del batch
        bn_decay_step: Steps entre decaimientos
        bn_decay_rate: Factor de decaimiento
        bn_decay_clip: Valor máximo de decay

    Returns:
        BN decay actual
    """
    bn_momentum = tf.train.exponential_decay(
        bn_init_decay,
        batch * batch_size,
        bn_decay_step,
        bn_decay_rate,
        staircase=True
    )
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    return bn_decay


def train_one_epoch(sess, ops, train_dataset, epoch_num, batch_size, augment=True):
    """
    Entrena por un epoch.

    Args:
        sess: Sesión de TensorFlow
        ops: Dict con operaciones de TF
        train_dataset: Dataset de entrenamiento
        epoch_num: Número de epoch actual
        batch_size: Tamaño del batch
        augment: Si True, aplica augmentación

    Returns:
        Loss y accuracy promedio del epoch
    """
    is_training = True

    # Shuffle del dataset (mezclando índices)
    n_samples = len(train_dataset)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    n_batches = n_samples // batch_size

    for batch_idx in range(n_batches):
        # Obtener batch
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch_indices = indices[start_idx:end_idx]

        batch_data = []
        batch_labels = []

        for idx in batch_indices:
            data, label = train_dataset[idx]
            batch_data.append(data)
            batch_labels.append(label)

        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)

        # Augmentación adicional con provider
        if augment:
            batch_data = provider.random_scale_point_cloud(batch_data)
            batch_data = provider.shift_point_cloud(batch_data)

        # Feed dict
        feed_dict = {
            ops['pointclouds_pl']: batch_data,
            ops['labels_pl']: batch_labels,
            ops['is_training_pl']: is_training,
        }

        # Train step
        summary, step, _, loss_val, pred_val = sess.run(
            [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
            feed_dict=feed_dict
        )

        # Estadísticas
        pred_class = np.argmax(pred_val, 1)
        correct = np.sum(pred_class == batch_labels)

        total_correct += correct
        total_seen += batch_size
        total_loss += loss_val

        # Log cada 10 batches
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{n_batches} | '
                  f'Loss: {loss_val:.4f} | '
                  f'Acc: {correct/batch_size:.4f}')

    avg_loss = total_loss / n_batches
    avg_acc = total_correct / total_seen

    return avg_loss, avg_acc


def eval_one_epoch(sess, ops, val_dataset, batch_size):
    """
    Evalúa el modelo en el conjunto de validación.

    Args:
        sess: Sesión de TensorFlow
        ops: Dict con operaciones de TF
        val_dataset: Dataset de validación
        batch_size: Tamaño del batch

    Returns:
        Loss y accuracy promedio
    """
    is_training = False

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    n_samples = len(val_dataset)
    n_batches = n_samples // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch_data = []
        batch_labels = []

        for idx in range(start_idx, end_idx):
            data, label = val_dataset[idx]
            batch_data.append(data)
            batch_labels.append(label)

        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)

        feed_dict = {
            ops['pointclouds_pl']: batch_data,
            ops['labels_pl']: batch_labels,
            ops['is_training_pl']: is_training,
        }

        loss_val, pred_val = sess.run(
            [ops['loss'], ops['pred']],
            feed_dict=feed_dict
        )

        pred_class = np.argmax(pred_val, 1)
        correct = np.sum(pred_class == batch_labels)

        total_correct += correct
        total_seen += batch_size
        total_loss += loss_val

    avg_loss = total_loss / n_batches
    avg_acc = total_correct / total_seen

    return avg_loss, avg_acc


def train(args):
    """Función principal de entrenamiento."""

    print("\n" + "="*70)
    print("ENTRENAMIENTO DE CLASIFICADOR DE ESTRUCTURAS CRISTALINAS")
    print("="*70 + "\n")

    # =========================================================================
    # PASO 1: Preparar datos
    # =========================================================================
    print("PASO 1: Preparando datos...\n")

    if args.data_dir is not None:
        # Cargar datos desde directorio
        print(f"Cargando datos desde: {args.data_dir}")
        dataset = CrystalDataset(
            num_points=args.num_points,
            split='train',
            data_augmentation=not args.no_augmentation
        )
        dataset.load_from_directory(args.data_dir, file_extension='.off')

    else:
        # Generar datos sintéticos
        print(f"Generando datos sintéticos...")
        print(f"  - Clases: {args.crystal_types}")
        print(f"  - Muestras por clase: {args.n_samples}")

        dataset = CrystalDataset(
            num_points=args.num_points,
            split='train',
            data_augmentation=not args.no_augmentation
        )
        dataset.generate_synthetic(
            n_samples_per_class=args.n_samples,
            crystal_types=args.crystal_types
        )

    # Dividir en train/val
    train_dataset, val_dataset = dataset.split_train_val(val_fraction=0.2)

    print(f"\n✓ Datos preparados:")
    print(f"  - Train: {len(train_dataset)} muestras")
    print(f"  - Val: {len(val_dataset)} muestras")
    print(f"  - Clases: {dataset.class_names}")
    print(f"  - Puntos por muestra: {args.num_points}")

    NUM_CLASSES = len(dataset.class_names)

    # =========================================================================
    # PASO 2: Crear modelo
    # =========================================================================
    print(f"\nPASO 2: Creando modelo PointNet++...\n")

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(args.gpu)):
            # Placeholders
            pointclouds_pl = tf.placeholder(tf.float32, shape=(args.batch_size, args.num_points, 3))
            labels_pl = tf.placeholder(tf.int32, shape=(args.batch_size))
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)

            # Get learning rate and bn decay
            learning_rate = get_learning_rate(batch, args.learning_rate, args.batch_size,
                                             args.decay_step, args.decay_rate)
            bn_decay = get_bn_decay(batch, 0.5, args.batch_size, args.decay_step,
                                   args.decay_rate, 0.99)

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl,
                                              num_class=NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)

            # Add summary
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('bn_decay', bn_decay)
            tf.summary.scalar('loss', loss)

            # Get training operator
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        if args.log_dir is None:
            log_dir = os.path.join('log', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            log_dir = args.log_dir

        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(log_dir, 'val'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {
            'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'loss': loss,
            'train_op': train_op,
            'merged': tf.summary.merge_all(),
            'step': batch
        }

        print(f"✓ Modelo creado:")
        print(f"  - Arquitectura: {args.model}")
        print(f"  - Número de clases: {NUM_CLASSES}")
        print(f"  - Learning rate: {args.learning_rate}")
        print(f"  - Batch size: {args.batch_size}")

        # =====================================================================
        # PASO 3: Entrenar
        # =====================================================================
        print(f"\nPASO 3: Entrenando...\n")
        print("="*70)

        best_val_acc = 0.0

        for epoch in range(args.epochs):
            print(f'\n*** EPOCH {epoch+1:03d}/{args.epochs:03d} ***')

            # Train
            train_loss, train_acc = train_one_epoch(
                sess, ops, train_dataset, epoch,
                args.batch_size, augment=not args.no_augmentation
            )

            print(f'\n  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}')

            # Validate
            val_loss, val_acc = eval_one_epoch(sess, ops, val_dataset, args.batch_size)

            print(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = saver.save(sess, os.path.join(log_dir, 'best_model.ckpt'))
                print(f'  ✓ Mejor modelo guardado: {save_path}')

            # Save checkpoint cada 10 epochs
            if (epoch + 1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(log_dir, f'model_epoch_{epoch+1}.ckpt'))
                print(f'  ✓ Checkpoint guardado: {save_path}')

        # =====================================================================
        # PASO 4: Resultados finales
        # =====================================================================
        print("\n" + "="*70)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*70)
        print(f"\n✓ Mejor accuracy en validación: {best_val_acc:.4f}")
        print(f"✓ Modelo guardado en: {log_dir}")
        print(f"\nPara visualizar en TensorBoard:")
        print(f"  tensorboard --logdir={log_dir}")


if __name__ == '__main__':
    args = parse_args()

    print("\n" + "="*70)
    print("CLASIFICADOR DE ESTRUCTURAS CRISTALINAS CON POINTNET++")
    print("="*70)
    print("\nConfiguración:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    train(args)

    print("\n" + "="*70)
    print("TUTORIAL COMPLETADO")
    print("="*70)
    print("\nPróximos pasos:")
    print("  1. Usa tus propios archivos LAMMPS .dump")
    print("  2. Experimenta con diferentes estructuras cristalinas")
    print("  3. Ajusta hiperparámetros para mejorar accuracy")
    print("  4. Visualiza las predicciones del modelo")
    print("\n")
