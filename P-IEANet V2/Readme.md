# IEA Reasoning Code + Partition Manifest, No Weights

This lightweight package includes optimized code and **only the Data Partitioning split manifest**. It does not include model weights, source images, full labels, or mask PNG files.

**The model weights and all datasets will be released publicly after the paper is accepted.**

## Included

- `code/`: optimized project code.
- `data/partition_manifest/`: CSV/TXT/JSON files listing train/val/test sample IDs, analysis JSON paths, and mask paths.
- `scripts/rebuild_data_partitioning_from_manifest.py`: rebuilds `data/Data Partitioning/` from a local full Dataset Part2 directory by symlink or copy.
- `weights/`: placeholder folders and placement instructions.

## Not Included

No model weight files are included. The package contains no `.pt`, `.pth`, `.bin`, `.safetensors`, or `.ckpt` files.

The following weight directories are placeholders only:

```text
weights/best_checkpoint/
weights/llama8b_base/
weights/grounded_sam/
weights/bert-base-uncased/
```

## Where To Put Released Weights

After the paper is accepted and weights are released, place them here:

```text
weights/best_checkpoint/reasoning_grounding_model_best_epoch_010.pt
weights/llama8b_base/
weights/grounded_sam/sam_vit_h_4b8939.pth
weights/grounded_sam/groundingdino_swint_ogc.pth
weights/bert-base-uncased/
weights/wavelet_epoch_2_val_loss0.03022034629540784.pth
```

## Rebuild Data Partitioning

Provide the full Dataset Part2 directory containing `analysis_only/` and `segmentations/`:

```bash
python scripts/rebuild_data_partitioning_from_manifest.py \
  --dataset-part2-root "/path/to/Dataset Part2_IEA-Reasoning" \
  --output-root "data/Data Partitioning" \
  --mode symlink
```

Use `--mode copy` if you need real files instead of symlinks.

For evaluation, also provide source images:

```bash
export IMAGE_ROOT=/path/to/Dataset\ Part1_IEA40K/IEA_img
export SPLIT=test
bash scripts/evaluate_epoch10_with_rebuilt_partition.sh
```

## Expected Epoch 10 Metrics With Released Weights

```text
epoch          = 10
BLEU@4         = 0.5040559977
mIoU@0.5       = 0.3844857886
best_mIoU      = 0.3855560642
best_threshold = 0.4
mAcc           = 0.8098743888
pixel_accuracy = 0.7972168769
```

## Demo Presentation
https://github.com/user-attachments/assets/f0da7fcd-8fda-488d-8a15-14b750cff5dd

https://github.com/user-attachments/assets/1a309d65-50f2-4a47-9c33-04c81287a092







