#!/usr/bin/env python3
"""Full offline pipeline: ODT → processed CSV → FIP inference → T-pose calibration → video.

Usage:
    python run_pipeline.py                        # Full pipeline (all steps, all motions)
    python run_pipeline.py --step render          # Only render videos
    python run_pipeline.py --step infer           # Only inference + calibration
    python run_pipeline.py --motions m2 m3        # Process specific motions
    python run_pipeline.py --step render --motions m2  # Render only m2
"""
import argparse
import os
import sys
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'human_body_prior'))

from config import (
    MOTION_IDS, TPOSE_ID, DEVICE, BODY_PARAMS,
    MODEL_CHECKPOINT, SMPL_MODEL, RAW_DATA_DIR, OUTPUT_DIR,
    RENDER_FPS, RENDER_WIDTH, RENDER_HEIGHT,
)


def step_preprocess(motion_ids):
    """Step 1: ODT → processed CSV."""
    from pipeline.preprocess import preprocess_motion

    print("\n" + "=" * 60)
    print("Step 1: Preprocess  (ODT → CSV)")
    print("=" * 60)

    csv_dir = os.path.join(OUTPUT_DIR, 'csv')
    for mid in motion_ids:
        odt_path = os.path.join(RAW_DATA_DIR, f'{mid}.odt')
        csv_path = os.path.join(csv_dir, f'{mid}_processed.csv')
        if os.path.exists(odt_path):
            preprocess_motion(odt_path, csv_path)
        else:
            print(f"  Warning: {odt_path} not found")
    return csv_dir


def step_inference(motion_ids, csv_dir):
    """Step 2: CSV → calibrated poses."""
    from pipeline.inference import load_model, process_all_motions

    print("\n" + "=" * 60)
    print("Step 2: Inference + Calibration  (CSV → poses)")
    print("=" * 60)

    model = load_model(MODEL_CHECKPOINT, DEVICE)
    print(f"  Model loaded on {DEVICE}")

    pose_dir = os.path.join(OUTPUT_DIR, 'poses')
    return process_all_motions(
        csv_dir, pose_dir, model, DEVICE, BODY_PARAMS, motion_ids, TPOSE_ID,
    )


def step_render(motion_ids):
    """Step 3: Calibrated poses → video."""
    from pipeline.renderer import SMPLRenderer

    print("\n" + "=" * 60)
    print("Step 3: Render  (poses → video)")
    print("=" * 60)

    renderer = SMPLRenderer(
        SMPL_MODEL, device=DEVICE, width=RENDER_WIDTH, height=RENDER_HEIGHT,
    )

    pose_dir = os.path.join(OUTPUT_DIR, 'poses')
    video_dir = os.path.join(OUTPUT_DIR, 'video')

    for mid in motion_ids:
        pose_path = os.path.join(pose_dir, f'{mid}_calibrated.pt')
        video_path = os.path.join(video_dir, f'{mid}_animation.mp4')

        if not os.path.exists(pose_path):
            print(f"  Warning: {pose_path} not found, skipping")
            continue

        print(f"\n  [{mid}]")
        n = renderer.render_video(pose_path, video_path, fps=RENDER_FPS)
        print(f"  {mid}: {n} frames → {video_path}")

    renderer.cleanup()


def main():
    parser = argparse.ArgumentParser(description='FIP Motion Reconstruction Pipeline')
    parser.add_argument('--step', choices=['preprocess', 'infer', 'render', 'all'], default='all',
                        help='Which pipeline step to run (default: all)')
    parser.add_argument('--motions', nargs='+', default=MOTION_IDS,
                        help='Motion IDs to process (default: m1-m7)')
    args = parser.parse_args()

    print("=" * 60)
    print("FIP Motion Reconstruction Pipeline")
    print("=" * 60)
    print(f"  Device:  {DEVICE}")
    print(f"  Motions: {args.motions}")
    print(f"  Step:    {args.step}")

    t0 = time.time()

    csv_dir = os.path.join(OUTPUT_DIR, 'csv')

    if args.step in ('all', 'preprocess'):
        csv_dir = step_preprocess(args.motions)

    if args.step in ('all', 'infer'):
        step_inference(args.motions, csv_dir)

    if args.step in ('all', 'render'):
        step_render(args.motions)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
