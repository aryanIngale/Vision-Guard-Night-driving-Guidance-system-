"""
scripts/prepare_dataset.py  —  Night Guidance System
Sets up the expected dataset/ folder structure from a BDD100K download.

"""

import os
import sys
import shutil
import argparse
from tqdm import tqdm


def prepare(bdd_root: str, out_root: str):
    splits = ["train", "val"]

    for split in splits:
        # Locate source folders
        img_src  = os.path.join(bdd_root, "images", split)
        mask_src = os.path.join(bdd_root, "labels", "sem_seg", "masks", split)

        # Fallback: some Kaggle variants use a slightly different path
        if not os.path.exists(mask_src):
            mask_src = os.path.join(bdd_root, "bdd100k", "labels",
                                    "sem_seg", "masks", split)

        if not os.path.exists(img_src):
            print(f"  [WARN] Image dir not found: {img_src}  — skipping {split}")
            continue
        if not os.path.exists(mask_src):
            print(f"  [WARN] Mask dir not found:  {mask_src}  — skipping {split}")
            continue

        img_dst  = os.path.join(out_root, split, "images")
        mask_dst = os.path.join(out_root, split, "masks")
        os.makedirs(img_dst,  exist_ok=True)
        os.makedirs(mask_dst, exist_ok=True)

        images = [f for f in os.listdir(img_src)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"\n[{split}] Linking {len(images)} images …")

        linked_pairs = 0
        for img_file in tqdm(images):
            stem      = os.path.splitext(img_file)[0]
            mask_file = stem + "_train_id.png"
            mask_path = os.path.join(mask_src, mask_file)

            if not os.path.exists(mask_path):
                continue   # silently skip unpaired images

            src_img  = os.path.join(img_src,  img_file)
            dst_img  = os.path.join(img_dst,  img_file)
            src_mask = mask_path
            dst_mask = os.path.join(mask_dst, mask_file)

            # Prefer symlinks (saves disk space); fall back to copy
            for src, dst in [(src_img, dst_img), (src_mask, dst_mask)]:
                if os.path.exists(dst):
                    continue
                try:
                    os.symlink(os.path.abspath(src), dst)
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

            linked_pairs += 1

        print(f"  {linked_pairs} image-mask pairs ready in {os.path.join(out_root, split)}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BDD100K dataset folder for Night Guidance System"
    )
    parser.add_argument(
        "--bdd_root", required=True,
        help="Root folder of your BDD100K download"
    )
    parser.add_argument(
        "--out_root", default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"
        ),
        help="Where to create dataset/  (default: project/dataset/)"
    )
    args = parser.parse_args()

    print(f"\nBDD100K root : {args.bdd_root}")
    print(f"Output root  : {args.out_root}\n")
    prepare(args.bdd_root, args.out_root)
    print("\nDone.  Run scripts/datacheck.py to verify.\n")


if __name__ == "__main__":
    main()