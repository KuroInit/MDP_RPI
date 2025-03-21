
import os
import random
import glob
import argparse
from PIL import Image, ImageOps
import numpy as np
import shutil
import cv2
import concurrent.futures

os.system("taskset -p 0xffffffffffffffffffffffff %d" % os.getpid())

"""
python /p/home/jusers/xu17/juwels/code/MDP_CV/scripts/add_background.py \
    --images /p/data1/mmlaion/y4/MDP_CV-2/valid/images \
    --labels /p/data1/mmlaion/y4/MDP_CV-2/valid/labels \
    --backgrounds /p/home/jusers/xu17/juwels/code/MDP_CV/backgrounds \
    --output /p/data1/mmlaion/y4/MDP_CV-2/valid_edge \
    --threshold 120


python /p/home/jusers/xu17/juwels/code/MDP_CV/scripts/add_background.py \
    --images /p/data1/mmlaion/y4/MDP_CV-2/train/images \
    --labels /p/data1/mmlaion/y4/MDP_CV-2/train/labels \
    --backgrounds /p/home/jusers/xu17/juwels/code/MDP_CV/backgrounds \
    --output /p/data1/mmlaion/y4/MDP_CV-2/train_edge \
    --threshold 120
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Background Replacement for YOLO Dataset (only within bounding boxes)")
    parser.add_argument("--images", type=str, required=True,
                        help="Path to the folder containing original images (e.g. 'train/images')")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to the folder containing YOLO label files (e.g. 'train/labels')")
    parser.add_argument("--backgrounds", type=str, required=True,
                        help="Path to the folder with random background images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory where new images and labels will be saved")
    parser.add_argument("--threshold", type=int, default=30,
                        help="Threshold for masking black background (0-255). Increase if black background is not perfectly removed.")
    parser.add_argument("--workers", type=int, default=96,
                        help="Number of parallel worker processes.")
    return parser.parse_args()

def apply_edge_detection(pil_image, threshold1=30, threshold2=100):
    cv_img = np.array(pil_image)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def get_mask_remove_black(pil_image, threshold=30):
    """
    Creates a mask that removes near-black pixels from the image.
    A pixel is considered 'black' if R, G, and B are all below the threshold.
    Returns an RGBA image where non-black pixels retain their original color
    and black pixels become transparent.
    """
    img_array = np.array(pil_image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        black_mask = (img_array[:,:,0] < threshold) & \
                     (img_array[:,:,1] < threshold) & \
                     (img_array[:,:,2] < threshold)
    else:
        raise ValueError("Expected an RGB image. Got shape: {}".format(img_array.shape))
    foreground_mask = ~black_mask
    alpha = np.zeros_like(img_array[:,:,0], dtype=np.uint8)
    alpha[foreground_mask] = 255
    rgba_image = Image.fromarray(np.dstack([img_array, alpha]), mode="RGBA")
    return rgba_image

def process_image(img_path, args, bg_files, out_images_dir, out_labels_dir):
    try:
        filename = os.path.basename(img_path)
        name_no_ext, ext = os.path.splitext(filename)
        label_file = os.path.join(args.labels, f"{name_no_ext}.txt")
        if not os.path.exists(label_file):
            return f"Skipped {filename} (no label file)"
        
        # Load the original image
        original_img = Image.open(img_path).convert("RGB")
        img_width, img_height = original_img.size

        # Pick a random background image and resize it
        bg_path = random.choice(bg_files)
        bg_img_full = Image.open(bg_path).convert("RGB")
        bg_img_full = bg_img_full.resize((img_width, img_height))

        # Read the label file (YOLO format)
        with open(label_file, 'r') as f:
            boxes = f.readlines()

        for line in boxes:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
            except ValueError:
                continue

            # Calculate the original bounding box dimensions and center
            box_width = int(width_norm * img_width)
            box_height = int(height_norm * img_height)
            x_center_pixel = x_center_norm * img_width
            y_center_pixel = y_center_norm * img_height

            # Expand the bounding box by a factor of 2
            new_box_width = box_width * 2
            new_box_height = box_height * 2

            x1 = int(x_center_pixel - new_box_width / 2)
            y1 = int(y_center_pixel - new_box_height / 2)
            x2 = x1 + new_box_width
            y2 = y1 + new_box_height

            # Clamp coordinates to image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            # Crop regions from the original and background images
            region_orig = original_img.crop((x1, y1, x2, y2))
            region_bg = bg_img_full.crop((x1, y1, x2, y2))

            # Create an RGBA region with the mask applied
            region_fg_rgba = get_mask_remove_black(region_orig, threshold=args.threshold)
            region_composite = Image.alpha_composite(region_bg.convert("RGBA"), region_fg_rgba)
            region_composite_rgb = region_composite.convert("RGB")

            # Paste the composite back into the original image
            original_img.paste(region_composite_rgb, (x1, y1))

        # Edge detection and postprocessing
        edge_image = apply_edge_detection(original_img, threshold1=30, threshold2=100)
        cv_img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        edge_cv = cv2.cvtColor(np.array(edge_image), cv2.COLOR_RGB2BGR)
        fused_cv = cv2.addWeighted(cv_img, 0.7, edge_cv, 0.3, 0)
        fused_image = Image.fromarray(cv2.cvtColor(fused_cv, cv2.COLOR_BGR2RGB))
        # inverted_img = ImageOps.invert(fused_image.convert("RGB"))
        # grayscale_img = inverted_img.convert("L")
        final_image = fused_image

        # Save modified image and copy label file
        new_image_path = os.path.join(out_images_dir, filename)
        final_image.save(new_image_path)
        new_label_path = os.path.join(out_labels_dir, f"{name_no_ext}.txt")
        shutil.copy(label_file, new_label_path)
        return f"Processed {filename} -> {new_image_path}"
    except Exception as e:
        return f"Error processing {img_path}: {str(e)}"

def main():
    args = parse_args()

    # Create output subfolders
    out_images_dir = os.path.join(args.output, "images")
    out_labels_dir = os.path.join(args.output, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # Gather background images
    bg_files = glob.glob(os.path.join(args.backgrounds, "*.*"))
    if not bg_files:
        print("No background images found in:", args.backgrounds)
        return

    # Gather image paths from the input dataset
    image_paths = glob.glob(os.path.join(args.images, "*.*"))
    if not image_paths:
        print("No images found in:", args.images)
        return

    # Use ProcessPoolExecutor to parallelize processing across workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for img_path in image_paths:
            futures.append(
                executor.submit(process_image, img_path, args, bg_files, out_images_dir, out_labels_dir)
            )
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print("Done! New dataset is in:", args.output)

if __name__ == "__main__":
    main()

