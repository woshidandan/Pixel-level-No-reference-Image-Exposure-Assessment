import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == 'copy':
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)


def main() -> None:
    parser = argparse.ArgumentParser(description='Rebuild Data Partitioning folders from lightweight manifest files.')
    parser.add_argument('--manifest-dir', default='data/partition_manifest')
    parser.add_argument('--dataset-part2-root', required=True, help='Path to Dataset Part2_IEA-Reasoning containing analysis_only and segmentations')
    parser.add_argument('--output-root', default='data/Data Partitioning')
    parser.add_argument('--mode', choices=['symlink', 'copy'], default='symlink')
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    part2 = Path(args.dataset_part2_root)
    out = Path(args.output_root)
    seen_analysis = set()
    seen_masks = set()
    for csv_path in sorted(manifest_dir.glob('*_manifest.csv')):
        with csv_path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                split = row['split']
                sample_id = row['sample_id']
                analysis_src = part2 / 'analysis_only' / f'{sample_id}.json'
                analysis_dst = out / split / 'analysis_only' / f'{sample_id}.json'
                if analysis_dst not in seen_analysis:
                    if not analysis_src.exists():
                        raise FileNotFoundError(analysis_src)
                    link_or_copy(analysis_src, analysis_dst, args.mode)
                    seen_analysis.add(analysis_dst)
                mask_filename = row.get('mask_filename') or ''
                if mask_filename:
                    mask_src = part2 / 'segmentations' / sample_id / mask_filename
                    mask_dst = out / split / 'segmentations' / sample_id / mask_filename
                    if mask_dst not in seen_masks:
                        if not mask_src.exists():
                            raise FileNotFoundError(mask_src)
                        link_or_copy(mask_src, mask_dst, args.mode)
                        seen_masks.add(mask_dst)
    print(json.dumps({'analysis_files': len(seen_analysis), 'mask_files': len(seen_masks), 'output_root': str(out), 'mode': args.mode}, indent=2))


if __name__ == '__main__':
    main()
