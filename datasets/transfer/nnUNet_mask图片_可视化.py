from PIL import Image
import numpy as np
import os


def convert_mask_to_rgb_and_save_pair(mask_path, paired_dir, groundTruth_dir, output_dir):
    # Extract filename
    filename = os.path.basename(mask_path)
    file_root = os.path.splitext(filename)[0]

    # Create a directory for this file in the output_dir
    file_output_dir = os.path.join(output_dir, file_root)
    os.makedirs(file_output_dir, exist_ok=True)

    # Load and process the mask image
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    scaled_mask = mask_array * 255
    scaled_mask = scaled_mask.astype(np.uint8)
    scaled_mask_rgb = np.stack([scaled_mask] * 3, axis=-1)

    # Load the paired image
    filename = filename[:-4] + '_0000' + filename[-4:]
    paired_image_path = os.path.join(paired_dir, filename)
    if not os.path.exists(paired_image_path):
        print(f"Paired image not found for {filename}")
        return

    paired_image = Image.open(paired_image_path)

    # Save the paired image (before overlay)
    paired_image_save_path = os.path.join(file_output_dir, "original_" + filename)
    paired_image.save(paired_image_save_path)

    # Overlay and save the original mask on the paired image
    ouline_overlay = overlay_outline_mask_on_image(paired_image, scaled_mask_rgb)
    ouline_overlay = ouline_overlay.convert("RGB")
    ouline_overlay_path = os.path.join(file_output_dir, "paired_with_outline_mask_" + filename)
    ouline_overlay.save(ouline_overlay_path)

    overlay_original = overlay_mask_on_image(paired_image, scaled_mask_rgb)
    overlay_original = overlay_original.convert("RGB")
    overlay_original_save_path = os.path.join(file_output_dir, "paired_with_mask_" + filename)
    overlay_original.save(overlay_original_save_path)

    # # Load and process the ground truth mask
    ground_truth_path = os.path.join(groundTruth_dir, filename)
    if os.path.exists(ground_truth_path):
        ground_truth_mask = Image.open(ground_truth_path)
        ground_truth_mask_array = np.array(ground_truth_mask)
        scaled_ground_truth_mask = ground_truth_mask_array * 255
        scaled_ground_truth_mask = scaled_ground_truth_mask.astype(np.uint8)
        scaled_ground_truth_mask_rgb = np.stack([scaled_ground_truth_mask] * 3, axis=-1)

        # Overlay and save the ground truth mask on the paired image
        overlay_ground_truth = overlay_outline_mask_on_image(paired_image, scaled_ground_truth_mask_rgb)
        overlay_ground_truth.save(os.path.join(file_output_dir, "paired_with_ground_truth_" + filename))
    else:
        print(f"Ground truth mask not found for {filename}")


import cv2


def get_mask_outline(mask, thickness=1):
    # Use OpenCV to find edges
    edges = cv2.Canny(mask, 100, 200)
    # Dilate the edges to make them thicker
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    return dilated_edges


def overlay_outline_mask_on_image(paired_image, mask, color=(0, 255, 0), thickness=1):
    # Convert paired image to numpy array
    paired_image_np = np.array(paired_image)

    # Get the outline of the mask
    outline = get_mask_outline(mask, thickness)

    # Overlay the outline
    paired_image_np[outline > 0] = color  # Set the color for the outline

    # Convert back to PIL Image
    overlayed_image = Image.fromarray(paired_image_np)
    return overlayed_image


def overlay_mask_on_image(paired_image, mask, opacity=0.2):
    # Convert paired image to numpy array with RGBA format
    paired_image_np = np.array(paired_image.convert("RGBA"))

    # Ensure the mask is in RGBA format
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[..., 0:3] = mask  # Use the RGB channels of the mask
    mask_rgba[..., 3] = (mask[..., 0] / 255.0) * opacity * 255  # Set opacity based on the first channel of the mask

    # Overlay the mask
    overlayed_image = Image.alpha_composite(Image.fromarray(paired_image_np), Image.fromarray(mask_rgba))

    return overlayed_image


def process_all_masks(mask_dir, paired_dir, groundTruth_dir, output_dir):
    # Process all images in the mask directory
    for mask_filename in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_filename)
        if mask_path.endswith('.jpg') and os.path.isfile(mask_path):
            convert_mask_to_rgb_and_save_pair(mask_path, paired_dir, groundTruth_dir, output_dir)


# Example usage
process_all_masks(
    "/home/zzm/nnUNet/output_image",
    "/home/zzm/pic_new_resize",
    "/home/zzm/gt (2)",
    "/home/zzm/xihei_Clival_recess2"
)
