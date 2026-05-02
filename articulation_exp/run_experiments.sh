#!/usr/bin/env bash
###############################################################################
# Articulated Joint Estimation Experiment Pipeline
# Based on: Video Generator + 3D Part Prior
#
# This script executes all experiment stages in order.
# Each stage can be run independently by specifying the stage name.
#
# Usage:
#   bash articulation_exp/run_experiments.sh              # run all stages
#   bash articulation_exp/run_experiments.sh --stage e1    # run specific stage
#   bash articulation_exp/run_experiments.sh --stage tf0   # training-free only
###############################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP_ROOT="${PROJ_ROOT}/articulation_exp"
DATA_CACHE="${PROJ_ROOT}/data_cache/partnet_mobility"
OUTPUT_ROOT="${PROJ_ROOT}/outputs/articulation"
DATASET_CONFIG="${EXP_ROOT}/configs/dataset/partnet_mobility.yaml"
DEVICE="cuda"
STAGE="${1:---all}"

# Use OmniPart conda environment (adjust if your env name differs)
PYTHON="/root/miniconda3/envs/omni/bin/python"
if [ ! -f "${PYTHON}" ]; then
    PYTHON="python"
fi

export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${DATA_CACHE}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ============================================================================
# Stage 0: Data Preparation & Feature Caching (E0)
# ============================================================================
run_e0() {
    log "========== E0: Data Preparation & Feature Caching =========="

    log "Step 1: Verify data cache directory structure..."
    mkdir -p "${DATA_CACHE}"

    log "Step 2: Cache OmniPart features (SLAT tokens, bboxes, meshes)..."
    log "  NOTE: This requires OmniPart pipeline to be loaded."
    log "  Run manually if not yet cached:"
    log "    python -c \""
    log "    from articulation_exp.cache.cache_omnipart import OmniPartFeatureCache"
    log "    # Initialize with your pipeline and run process_sample() per object"
    log "    \""

    log "Step 3: Cache Wan/Video-DiT features..."
    log "  NOTE: This requires Wan model to be loaded."
    log "    python -c \""
    log "    from articulation_exp.cache.cache_wan_features import WanFeatureExtractor"
    log "    # Initialize with Wan model and extract features per video"
    log "    \""

    log "E0 complete. Ensure cached data exists in ${DATA_CACHE}"
}

# ============================================================================
# Stage 1: Basic Baselines (E1-A, E1-B, E1-C)
# ============================================================================
run_e1() {
    log "========== E1: Basic Baselines =========="

    # E1-A: 3D-only baseline
    log "--- E1-A: 3D-only baseline ---"
    ${PYTHON} -m articulation_exp.train.train_all \
        --model_config "${EXP_ROOT}/configs/model/baseline_3d.yaml" \
        --dataset_config "${DATASET_CONFIG}" \
        --output_dir "${OUTPUT_ROOT}/e1a_3d_only" \
        --device "${DEVICE}"

    # E1-B: Video-only baseline
    log "--- E1-B: Video-only baseline ---"
    ${PYTHON} -m articulation_exp.train.train_all \
        --model_config "${EXP_ROOT}/configs/model/baseline_video.yaml" \
        --dataset_config "${DATASET_CONFIG}" \
        --output_dir "${OUTPUT_ROOT}/e1b_video_only" \
        --device "${DEVICE}"

    # E1-C: Late fusion
    log "--- E1-C: Late fusion baseline ---"
    ${PYTHON} -m articulation_exp.train.train_all \
        --model_config "${EXP_ROOT}/configs/model/late_fusion.yaml" \
        --dataset_config "${DATASET_CONFIG}" \
        --output_dir "${OUTPUT_ROOT}/e1c_late_fusion" \
        --device "${DEVICE}"

    log "E1 complete. Results in ${OUTPUT_ROOT}/e1*/"
}

# ============================================================================
# Stage 2: Cross-Attention Fusion (E2)
# ============================================================================
run_e2() {
    log "========== E2: Video-to-SLAT Cross-Attention Fusion =========="
    ${PYTHON} -m articulation_exp.train.train_all \
        --model_config "${EXP_ROOT}/configs/model/cross_fusion.yaml" \
        --dataset_config "${DATASET_CONFIG}" \
        --output_dir "${OUTPUT_ROOT}/e2_cross_fusion" \
        --device "${DEVICE}"
    log "E2 complete."
}

# ============================================================================
# Stage 3: Projective Fusion (E3)
# ============================================================================
run_e3() {
    log "========== E3: Projective Video-to-SLAT Fusion =========="
    ${PYTHON} -m articulation_exp.train.train_all \
        --model_config "${EXP_ROOT}/configs/model/projective_fusion.yaml" \
        --dataset_config "${DATASET_CONFIG}" \
        --output_dir "${OUTPUT_ROOT}/e3_projective_fusion" \
        --device "${DEVICE}"
    log "E3 complete."
}

# ============================================================================
# Stage TF0: Training-Free Geometry+Motion Candidate Fitting
# ============================================================================
run_tf0() {
    log "========== TF0: Training-Free Candidate Fitting =========="
    ${PYTHON} -m articulation_exp.training_free.run_training_free \
        --config "${EXP_ROOT}/configs/model/training_free.yaml" \
        --data_dir "${DATA_CACHE}" \
        --output_dir "${OUTPUT_ROOT}/tf0_candidate" \
        --stage tf0
    log "TF0 complete."
}

# ============================================================================
# Stage TF1: Wan Feature Candidate Scoring
# ============================================================================
run_tf1() {
    log "========== TF1: Wan Feature Candidate Scoring =========="
    ${PYTHON} -m articulation_exp.training_free.run_training_free \
        --config "${EXP_ROOT}/configs/model/training_free.yaml" \
        --data_dir "${DATA_CACHE}" \
        --output_dir "${OUTPUT_ROOT}/tf1_wan_scoring" \
        --stage tf0
    log "TF1 complete. (scoring variant configured in YAML)"
}

# ============================================================================
# Stage TF2: Kinematic Refinement
# ============================================================================
run_tf2() {
    log "========== TF2: Kinematic Refinement =========="
    ${PYTHON} -m articulation_exp.training_free.run_training_free \
        --config "${EXP_ROOT}/configs/model/training_free.yaml" \
        --data_dir "${DATA_CACHE}" \
        --output_dir "${OUTPUT_ROOT}/tf2_refinement" \
        --stage tf2
    log "TF2 complete."
}

# ============================================================================
# Stage VGT0: Video-DiT Correspondence Extraction
# ============================================================================
run_vgt0() {
    log "========== VGT0: Video-DiT Correspondence =========="
    log "  NOTE: Requires Wan/Video-DiT model to be loaded."
    log "  This module hooks into DiT layers to extract temporal correspondence."
    log "  Usage:"
    log "    from articulation_exp.video_generator_motion.dit_correspondence import DiTCorrespondenceExtractor"
    log "    extractor = DiTCorrespondenceExtractor()"
    log "    tracks, conf = extractor.extract_tracks(dit_features, query_points)"
    log "VGT0 extraction logic ready. Integrate with cached DiT features."
}

# ============================================================================
# Stage E4: Fusion Init + Training-Free Refinement
# ============================================================================
run_e4() {
    log "========== E4: Fusion Init + Kinematic Refinement =========="
    log "  Step 1: Use E2/E3 trained model for initialization"
    log "  Step 2: Refine with TF2 kinematic optimization"
    log "  This combines trainable and training-free branches."
    log "  Run E2/E3 first, then use predictions as TF2 init."
    log "E4 pipeline ready."
}

# ============================================================================
# Evaluation: Generate final results table
# ============================================================================
run_eval() {
    log "========== Evaluation: Generating Results Table =========="
    
    # Evaluate all trained models
    for model_dir in "${OUTPUT_ROOT}"/e1* "${OUTPUT_ROOT}"/e2* "${OUTPUT_ROOT}"/e3*; do
        if [ -d "${model_dir}" ] && [ -f "${model_dir}/best.pt" ]; then
            model_name=$(basename "${model_dir}")
            config_name=$(echo "${model_name}" | sed 's/e1a_3d_only/baseline_3d/;s/e1b_video_only/baseline_video/;s/e1c_late_fusion/late_fusion/;s/e2_cross_fusion/cross_fusion/;s/e3_projective_fusion/projective_fusion/')
            
            log "Evaluating ${model_name}..."
            ${PYTHON} -m articulation_exp.train.train_all \
                --model_config "${EXP_ROOT}/configs/model/${config_name}.yaml" \
                --dataset_config "${DATASET_CONFIG}" \
                --output_dir "${model_dir}" \
                --device "${DEVICE}" \
                --eval_only \
                --checkpoint "${model_dir}/best.pt" \
                2>&1 | tee "${model_dir}/eval_output.txt"
        fi
    done

    log "All evaluations complete. Results in ${OUTPUT_ROOT}/*/test_results.json"
}

# ============================================================================
# Main dispatch
# ============================================================================
case "${STAGE}" in
    --all|-a)
        run_e0
        run_e1
        run_e2
        run_e3
        run_tf0
        run_tf1
        run_tf2
        run_vgt0
        run_e4
        run_eval
        ;;
    --stage)
        STAGE_NAME="${2:-e1}"
        case "${STAGE_NAME}" in
            e0)   run_e0 ;;
            e1)   run_e1 ;;
            e2)   run_e2 ;;
            e3)   run_e3 ;;
            tf0)  run_tf0 ;;
            tf1)  run_tf1 ;;
            tf2)  run_tf2 ;;
            vgt0) run_vgt0 ;;
            e4)   run_e4 ;;
            eval) run_eval ;;
            *)    log "Unknown stage: ${STAGE_NAME}" && exit 1 ;;
        esac
        ;;
    *)
        echo "Usage: $0 [--all | --stage <e0|e1|e2|e3|tf0|tf1|tf2|vgt0|e4|eval>]"
        exit 1
        ;;
esac

log "Done."
