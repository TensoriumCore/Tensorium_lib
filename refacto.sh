#!/bin/bash

# Définition de la racine des includes
BASE="includes/Tensorium"

echo "🔄 Début de la réorganisation de Tensorium..."

# 1. Création de la nouvelle structure
mkdir -p "$BASE/Backend/CPU_Kernels"
mkdir -p "$BASE/Backend/CUDA"
mkdir -p "$BASE/Physics"
mkdir -p "$BASE/Utils"

# 2. Déplacement des Kernels CPU (Le nommage propre)
echo "📦 Déplacement et renommage des Kernels..."
mv "$BASE/Core/MatrixKernels/GemmKernel_big.hpp" "$BASE/Backend/CPU_Kernels/GemmKernel_Ref.hpp"
mv "$BASE/Core/MatrixKernels/GemmKernel_bigger.hpp" "$BASE/Backend/CPU_Kernels/GemmKernel_Optimized.hpp"
mv "$BASE/Core/MatrixKernels/MatrixKernel.hpp" "$BASE/Backend/CPU_Kernels/MatrixKernel.hpp"
rmdir "$BASE/Core/MatrixKernels"

# 3. Déplacement de SIMD
mv "$BASE/SIMD" "$BASE/Backend/SIMD"

# 4. Intégration de CUDA (Fini la racine séparée !)
echo "📦 Intégration de CUDA..."
# On suppose que Tensorium_CUDA est dans includes/Tensorium_CUDA
if [ -d "includes/Tensorium_CUDA" ]; then
    mv "includes/Tensorium_CUDA/"* "$BASE/Backend/CUDA/"
    rmdir "includes/Tensorium_CUDA"
fi

# 5. Déplacement de la Physique (DiffGeometry)
echo "📦 Isolation de la Physique..."
mv "$BASE/DiffGeometry" "$BASE/Physics/DiffGeometry"

# 6. Déplacement des Utilitaires
echo "📦 Rangement des Utils..."
mv "$BASE/MathUtils" "$BASE/Utils/MathUtils"
mv "$BASE/IO" "$BASE/Utils/IO"

# 7. Correction de la typo "Functionnal"
if [ -d "$BASE/Functionnal" ]; then
    mv "$BASE/Functionnal" "$BASE/Functional"
    # Renommage du fichier interne s'il a la typo
    if [ -f "$BASE/Functional/Functionnal.hpp" ]; then
        mv "$BASE/Functional/Functionnal.hpp" "$BASE/Functional/Functional.hpp"
    fi
fi

# 8. Update automatique des includes dans les fichiers (Sed magic)
# Attention : C'est brutal mais ça fait 90% du boulot.
# On remplace les anciens chemins par les nouveaux.

echo "sed -i 's|Core/MatrixKernels/GemmKernel_bigger.hpp|Backend/CPU_Kernels/GemmKernel_Optimized.hpp|g' $BASE/Core/Matrix.hpp"
# (Je ne l'exécute pas automatiquement pour te laisser le contrôle, 
# mais c'est ce qu'il faudra faire dans ton IDE : Search & Replace).

echo "✅ Terminé ! Ta structure est maintenant :"
tree includes/Tensorium
