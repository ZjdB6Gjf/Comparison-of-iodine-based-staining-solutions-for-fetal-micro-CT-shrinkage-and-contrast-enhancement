import os
import numpy as np
import re
import pandas as pd
import nibabel as nib

input_folder = r"L:\Basic\divg\aef\3DAtlas-students\Dafne S Dijkman\07- Handmatige segmentaties\Automatic segmentations\Files Marcella_all timepoints_uCT_29-04-2026\Organs_uCT_T3-T9"

labels_organs = {
    1: "brain",
    2: "lung",
    3: "liver"
}

pattern = re.compile(
    r"(TOP\d+)_T(\d+)_(organs|wholefetus)(?:\([^)]*\))?_(?:\d+_)?(MRI|uCT)_(.+)\.nii"
)

# --- DEFINE GLOBAL AXES ---
tops = sorted(set([f.split("_")[0] for f in os.listdir(input_folder)]))
structures = ["wholefetus", "brain", "lung", "liver"]
timepoints = ["MRI_T0", "uCT_T0"] + [f"T{i}" for i in range(1, 10)]

# --- CREATE DF ONCE ---
index = pd.MultiIndex.from_product([tops, structures], names=["TOP", "structure"])
df = pd.DataFrame(index=index, columns=timepoints, dtype=float)


# --- FUNCTIONS ---
def voxel_volume_from_affine(nii):
    return abs(np.linalg.det(nii.affine[:3, :3]))


# --- LOOP ---
for file in os.listdir(input_folder):
    if not (file.endswith(".nii") or file.endswith(".nii.gz")):
        continue

    match = pattern.search(file)
    if not match:
        continue

    top, t, seg_type, modality, _ = match.groups()
    t = int(t)

    path = os.path.join(input_folder, file)
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.int32)

    voxel_vol = voxel_volume_from_affine(nii)

    # --- COLUMN NAME ---
    if modality == "MRI" and t == 0:
        col = "MRI_T0"
    elif modality == "uCT" and t == 0:
        col = "uCT_T0"
    else:
        col = f"T{t}"

    # --- WHOLE FETUS ---
    if seg_type == "wholefetus":
        vol = np.sum(data > 0) * voxel_vol
        df.loc[(top, "wholefetus"), col] = vol

    # --- ORGANS (PER LABEL!) ---
    elif seg_type == "organs":
        for label, organ in labels_organs.items():
            voxels = np.sum(data == label)
            vol = voxels * voxel_vol

            # optional correction factors
            if modality == "uCT":
                vol
            elif modality == "MRI":
                vol *= 5

            df.loc[(top, organ), col] = vol


# --- SAVE ---
df.to_csv(r"L:\Basic\divg\aef\3DAtlas-students\Dafne S Dijkman\07- Handmatige segmentaties\volumes.csv", float_format="%.2f")

print("Done.")
