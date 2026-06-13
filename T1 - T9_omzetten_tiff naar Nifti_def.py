import os
import json
import numpy as np
import tifffile as tiff
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, binary_dilation
from tqdm import tqdm

# ============================================================
# LOAD TIFF STACK
# ============================================================
def load_volume(folder_path, z_max=460): # Use 460 for T1 - T9 and 620 for T0
    tif_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".tif") or f.endswith(".tiff")
    ])[:z_max]

    slices = []
    global_min = np.inf
    global_max = -np.inf

    for f in tif_files:
        img = tiff.imread(os.path.join(folder_path, f))
        global_min = min(global_min, img.min())
        global_max = max(global_max, img.max())
        slices.append(img)

    volume = np.stack(slices, axis=0)
    return volume, global_min, global_max


# ============================================================
# SEGMENT + CROP
# ============================================================
def segment_and_crop(volume):
    img_sitk = sitk.GetImageFromArray(volume)

    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)

    mask = sitk.GetArrayFromImage(otsu.Execute(img_sitk)).astype(bool)
    threshold = otsu.GetThreshold()

    # Morphology
    structure = np.ones((3, 3, 3), dtype=bool)
    mask = binary_erosion(mask, structure, iterations=1)
    mask = binary_dilation(mask, structure, iterations=1)

    # Bounding box
    coords = np.where(mask)

    z_min = max(coords[0].min() - 10, 0)
    z_max = min(coords[0].max() + 10, volume.shape[0] - 1)
    y_min = max(coords[1].min() - 10, 0)
    y_max = min(coords[1].max() + 10, volume.shape[1] - 1)
    x_min = max(coords[2].min() - 10, 0)
    x_max = min(coords[2].max() + 10, volume.shape[2] - 1)

    cropped = volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

    crop_info = {
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "z_min": int(z_min),
        "z_max": int(z_max),
        "original_shape": list(volume.shape),
        "otsu_threshold": int(threshold)
    }

    return cropped, crop_info, threshold

# ============================================================
# RESAMPLING FUNCTION
# ============================================================
def resample_image(image, new_spacing=(0.2, 0.2, 0.2)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)

# ============================================================
# CREATE AFFINE
# ============================================================
def create_affine_supine(spacing):
    z, y, x = spacing
    # Affine translation for supine orientation
    return np.array([
        [-x,  0, 0, 0],     # voxel x → Anterior
        [0, 0, -y, 0],      # voxel y → Superior
        [0, -z, 0, 0],     # voxel z → Left
        [0, 0, 0, 1]
    ])

def create_affine_decubitus(spacing):
    z, y, x = spacing
    # Affine translation for decubitus orientation
    return np.array([
        [0, 0, x, 0],
        [-y, 0, 0, 0],
        [0, -z, 0, 0],
        [0, 0, 0, 1]
    ])

def get_affine(filename, spacing):
    if any(fid in filename for fid in decubitus_ids):
        return create_affine_decubitus(spacing)
    else:
        return create_affine_supine(spacing)

# ============================================================
# PROCESS ONE SCAN
# ============================================================
def process_scan(folder_path, output_folder, name):

    # ---- LOAD ----
    volume, vmin, vmax = load_volume(folder_path)

    # ---- SEGMENT + CROP ----
    cropped, crop_info, threshold = segment_and_crop(volume)
    print(f"Otsu threshold: {threshold}")

    # ---- REORIENT ----
    volume_reoriented = np.transpose(cropped, (2, 1, 0))

    # ---- RESAMPLE ----
    original_spacing = (0.1, 0.1, 0.1)
    # new_spacing = (0.2, 0.2, 0.2)

    # img_sitk = sitk.GetImageFromArray(volume_reoriented)
    # img_sitk.SetSpacing(original_spacing)

    # resampled_img = resample_image(img_sitk, new_spacing)
    # resampled = sitk.GetArrayFromImage(resampled_img)

    print("Cropped shape:", volume_reoriented.shape)
    # print("Resampled shape:", resampled.shape)

    # ---- NORMALIZE ----
    normalized = (volume_reoriented - vmin) / (vmax - vmin)
    normalized = (normalized * 255).astype(np.uint8)

    # ---- AFFINES ----
    affine_orig = get_affine(folder, original_spacing)
    # affine_orig = create_affine_supine(original_spacing)
    # affine_resampled = create_affine(new_spacing)

    # ---- SAVE ----
    os.makedirs(output_folder, exist_ok=True)
    cropped_folder = os.path.join(output_folder, "cropped_scans_16bit")
    os.makedirs(cropped_folder, exist_ok=True)
    nib.save(nib.Nifti1Image(volume_reoriented, affine_orig),
             os.path.join(cropped_folder, name + "_cropped.nii.gz"))
    
    # resampled_folder = os.path.join(output_folder, "resampled")
    # os.makedirs(resampled_folder, exist_ok=True)
    # nib.save(nib.Nifti1Image(resampled.astype(np.float32), affine_resampled),
    #          os.path.join(resampled_folder, name + "_resampled.nii.gz"))

    normalized_folder = os.path.join(output_folder, "normalized_scans_8bit")
    os.makedirs(normalized_folder, exist_ok=True)
    nib.save(nib.Nifti1Image(normalized, affine_orig),
             os.path.join(normalized_folder, name + "_normalized.nii.gz"))

    # ---- SAVE JSON ----
    crop_info_folder = os.path.join(output_folder, "crop_info")
    os.makedirs(crop_info_folder, exist_ok=True)
    with open(os.path.join(crop_info_folder, name + "_crop_info.json"), "w") as f:
        json.dump(crop_info, f, indent=4)

    print("✅ Saved:", name)


# ============================================================
# MAIN LOOP
# ============================================================

tif_root = r"L:\Basic\dive\Micro-CT\Scans\Puck\Studie protocol comparison\T1_BHC0.4"
output_root = r"L:\Basic\divg\aef\3DAtlas-students\Dafne S Dijkman\07- Handmatige segmentaties\Nifti files\T1"
decubitus_ids = ["TOP156", "TOP369", "TOP409", "TOP463", "TOP510"]

folders = os.listdir(tif_root)

for folder in tqdm(folders, desc="Processing scans"):
    folder_path = os.path.join(tif_root, folder, "recon")

    if not os.path.isdir(folder_path):
        continue

    try:
        process_scan(folder_path, output_root, folder + "_T1_BHC0.4")
    except Exception as e:
        print(f"❌ Error in {folder}: {e}")