#!/bin/bash

set -e

ROOT=$(pwd)
DATE=$(date +"%Y%m%d_%H%M%S")

ARCHIVE="${ROOT}/_cleanup_archive/final_cleanup_${DATE}"

mkdir -p "$ARCHIVE"


echo "================================="
echo "Balloon radar project cleanup"
echo "Archive:"
echo "$ARCHIVE"
echo "================================="


move_item(){

    if [ -e "$1" ]; then
        echo "[MOVE] $1"
        mkdir -p "$ARCHIVE/$(dirname "$1")"
        mv "$1" "$ARCHIVE/$1"
    fi

}



echo ""
echo "=== archive root packages ==="


for f in *.zip
do
    [ -e "$f" ] && move_item "$f"
done


move_item "INSTALL_MANIFEST.json"
move_item "PATCH_MANIFEST.json"


echo ""
echo "=== archive staging ==="

move_item "_staging"
move_item "files"
move_item "backups"



echo ""
echo "=== archive old datasets ==="


move_item "datasets/radar_dataset.py"
move_item "datasets/detection_dataset_v2.py"
move_item "datasets/polarimetric_detection_dataset_v1.py"



echo ""
echo "=== archive old training ==="


move_item "training/train_polarimetric_representation_fcn.py"

move_item "training/train_background_calibrator.py"

move_item "training/train_background_tail_calibrator.py"



echo ""
echo "=== archive old scripts ==="



OLD_SCRIPTS=(

scripts/run_polarimetric_representation_benchmark_v1.py

scripts/summarize_polarimetric_representation_benchmark_v1.py

scripts/run_bc_dpg_v2_tail.py

scripts/run_bc_dpg_v31_shift_reg_sweep.py

scripts/run_grouped_5fold_v2.py

scripts/run_dual_branch_gated.py

scripts/run_hv_ablation.py

scripts/test_detection_baseline_v2.py

scripts/test_detection_dataloader_v2.py

scripts/test_hv_ablation_pipeline.py

scripts/test_radar_dataset.py

)



for f in "${OLD_SCRIPTS[@]}"
do
    move_item "$f"
done



echo ""
echo "=== archive root helper scripts ==="


ROOT_FILES=(

apply_polarimetric_final_analysis_usage_tools_v1.py

apply_polarimetric_twofold_diagnostics_collection_v1.py

apply_roi_polarimetric_stage4_context_collection_v1.py

apply_roi_polarimetric_stage4_v1.py

apply_stage4_next_all_in_one_v1.py

cleanup_balloon_radar_project_v1.py

package_balloon_radar_project.sh

collect_bc_dpg_v3_context.sh

)


for f in "${ROOT_FILES[@]}"
do
    move_item "$f"
done



echo ""
echo "=== remove python cache ==="


find . -type d -name "__pycache__" -print0 \
| xargs -0 rm -rf



find . -name "*.pyc" -delete



echo ""
echo "=== cleanup empty logs ==="


mkdir -p "$ARCHIVE/logs"

for f in *.log
do
    [ -e "$f" ] && move_item "$f"
done



echo ""
echo "=== generate report ==="


cat > cleanup_report.txt <<EOF

Cleanup finished.

Time:
$(date)

Archive:
$ARCHIVE


Core pipeline kept:

BC-DPG-FCN v3
Polarimetric representation V2
ROI Stage4
Six-fold evidence
Paper assets


EOF


echo ""
echo "================================="
echo "Cleanup finished"
echo "Report: cleanup_report.txt"
echo "================================="
