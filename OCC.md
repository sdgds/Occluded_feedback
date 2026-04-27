# Low-Dimensional Frontal Feedback Resolves High-Dimensional Visual Ambiguity in Human Visual Cortex

This repository contains the code and bundled data used for the paper *Low-Dimensional Frontal Feedback Resolves High-Dimensional Visual Ambiguity in Human Visual Cortex*. The computational model implements hierarchical interactions between a ventral temporal cortex (VTC) module and a ventrolateral prefrontal cortex (vlPFC) module to study occluded face processing.

The release includes:

- model code: `BrainSOM.py`, `Hopfield_VTCSOM.py`
- stimulus folders for six conditions: intact face, no-eyes, upper-half, lower-half, eyes-only, and tools
- pretrained model weights and normalization statistics
- precomputed model outputs in `model_results/`
- analysis notebooks used to reproduce the main computational figures

The repository is notebook-centered. The main entry points are:

- `Stimuli_formal.ipynb`: stimulus preparation and Grad-CAM based face-information estimation
- `Occluded_face_formal.ipynb`: hierarchical VTC-vlPFC simulation
- `Model_results_formal.ipynb`: manifold, decoding, and time-series analysis

## Overview

The model follows the paper's computational pipeline:

1. Each image is resized/cropped to `224 x 224`, processed by AlexNet, and represented in a `1000`-dimensional feature space.
2. The feature vector is normalized and projected to the first `4` PCA components.
3. The `4`-dimensional feature drives a `200 x 200` VTC self-organizing map.
4. VTC activity evolves under stochastic Hopfield dynamics.
5. In the feedback model, the current VTC state is projected to a `20 x 20` vlPFC SOM, updated by vlPFC recurrent dynamics, and projected back to VTC as top-down feedback.
6. The code stores VTC and vlPFC state trajectories, which are then analyzed in the paper through decoding, manifold geometry, and energy-landscape visualization.

The provided stimuli implement the paper's Information-Graded Occluded Faces (IGOF) design:

- `face`: intact faces
- `noeye`: eyes occluded
- `top_face`: upper-half face
- `down_face`: lower-half face
- `eyes`: eyes-only
- `tools`: non-face comparison category

Each folder contains `20` images.

## System Requirements

### Hardware

The repository is large and memory-intensive.

- CPU: standard 64-bit desktop CPU
- RAM:
  - `32 GB` recommended for the precomputed-results demo in `Model_results_formal.ipynb`
  - `64 GB` recommended for rerunning the full VTC-vlPFC simulation in `Occluded_face_formal.ipynb`
- Storage:
  - the current `Formal/` folder occupies about `26.4 GB`
  - reserve at least `40 GB` free disk space to run the notebooks comfortably
- GPU: optional; helpful for AlexNet-based feature extraction and Grad-CAM, but not required for the bundled-results demo

### Software

The code was checked in the author's conda environment `occluded_face` with:

- OS: Windows 11 24H2, 64-bit (`build 26100`; reported by Python as `Windows-10-10.0.26100-SP0`)
- Python: `3.9.21`

Primary Python packages used by the notebooks and source code:

- `numpy==1.23.5`
- `scipy==1.9.3`
- `pandas==2.2.3`
- `torch==2.7.1`
- `torchvision==0.22.1`
- `scikit-learn==1.6.1`
- `matplotlib==3.9.4`
- `seaborn==0.13.2`
- `Pillow==11.2.1`
- `opencv-python==4.11.0.86`
- `imageio==2.37.0`
- `umap-learn==0.5.7`
- `statsmodels==0.14.4`
- `patsy==1.0.1`
- `tqdm==4.67.1`
- `h5py==3.13.0`
- `joblib==1.5.1`
- `minisom==2.3.5`
- `dhnn==0.1.12`
- `ipykernel==6.29.5`
- `ipywidgets==8.1.7`

Notes:

- `requirements.txt` is included, but the code also requires `minisom` and `dhnn`, which are imported by `BrainSOM.py` and `Hopfield_VTCSOM.py`.
- If you want to execute the notebooks from the command line, install `notebook` or `jupyter` as well.

## Installation Guide

### Recommended installation

```bash
conda create -n occluded_face python=3.9.21 -y
conda activate occluded_face
python -m pip install -r requirements.txt
python -m pip install minisom==2.3.5 dhnn==0.1.12 notebook ipywidgets
```

If you need a CUDA-enabled PyTorch build, install PyTorch and torchvision from the official PyTorch channel first, then install the remaining packages.

### Typical install time

Typical install time on an ordinary desktop computer with a stable internet connection: `~20-30 minutes`.

### Files already included in this release

No additional download is required for the bundled demo below. This release already contains:

- pretrained weights such as `model_VTC_weights.npy`, `model_vlPFC_weights.npy`, `som_sigma_6.2.npy`, `som_vlPFC_weights.npy`
- normalization statistics: `mean.npy`, `std.npy`
- PCA fitting data: `Data.npy`
- precomputed demo outputs in `model_results/`

## Demo

### Recommended demo for editors and reviewers

The fastest way to verify the release is to run the analysis notebook on the bundled precomputed results.

From the `Formal/` directory:

```bash
conda activate occluded_face
jupyter nbconvert --to notebook --execute --inplace Model_results_formal.ipynb
```

This demo uses the provided result files in `model_results/` and does not require rerunning the full stochastic simulation.

### Expected demo output

Running `Model_results_formal.ipynb` should produce:

- printed statistics for manifold dimensionality and manifold radius of vlPFC and VTC
- a bar plot of normalized face-vs-tool decoding scores across the five occlusion conditions
- a time-series plot showing how the VTC face-selective response evolves over model time under different occlusion levels

The bundled result files loaded by the notebook have the following structure:

- `Dynamic_states_VTC`: shape `(20, 162, 200, 200)` for face-like conditions
- `Dynamic_states_vlPFC`: shape `(20, 30, 20, 20)`
- `H_top_down`: shape `(20, 299, 200, 200)`

Qualitatively, the expected result is consistent with the paper:

- intact faces show the strongest and fastest face-like VTC response
- more severely occluded faces show delayed or weaker trajectories
- the feedback model still preserves face-related information across occlusion levels better than a purely feedforward account

### Expected demo run time

Expected run time on an ordinary desktop computer with `32 GB` RAM and an SSD: `~10-20 minutes`.

Most of this time is spent loading the bundled `model_results/` files, which occupy about `13 GB` in total.

### Optional full rerun from the bundled stimuli

If you want to rerun the hierarchical model itself instead of the precomputed demo:

```bash
conda activate occluded_face
jupyter nbconvert --to notebook --execute --inplace Occluded_face_formal.ipynb
```

This full rerun is substantially heavier than the recommended demo because it loads `model_VTC_weights.npy` (`~11.9 GB`) and performs long stochastic updates for each stimulus. On a CPU desktop, this should be treated as a long job rather than a quick verification step.

## Instructions for Use

### Using the provided stimuli

`Occluded_face_formal.ipynb` is the main notebook for running the model on image folders. The notebook:

1. loads AlexNet
2. fits PCA on `Data.npy`
3. loads the pretrained SOM and Hopfield weights
4. converts each image into a `4`-dimensional PCA feature
5. runs the VTC-only or VTC-vlPFC stochastic dynamics
6. saves the outputs as Python dictionaries containing dynamic states and feedback terms

The notebook currently points to:

- input root: `Stim_for_model/`
- default output root: `VTC_vlPFC_model/`

If you want the saved files to be read directly by `Model_results_formal.ipynb`, save them with the same naming convention used in `model_results/`, for example:

- `Face_feedback_results.npy`
- `Top_face_feedback_results.npy`
- `Noeye_feedback_results.npy`
- `Down_face_feedback_results.npy`
- `Eyes_feedback_results.npy`
- `Tool_feedback_results.npy`

### Using your own data

To run the model on your own images:

1. Create a new folder under `Stim_for_model/`, for example `Stim_for_model/my_condition/`.
2. Put your `.png`, `.jpg`, or `.bmp` images into that folder.
3. Open `Occluded_face_formal.ipynb`.
4. Keep the same preprocessing used by the paper:
   - resize to `256`
   - center crop to `224 x 224`
   - normalize with ImageNet mean/std inside the notebook
   - extract AlexNet features
   - z-score/normalize with the bundled `mean.npy` and `std.npy`
   - project to the first four PCA components fitted from `Data.npy`
5. Replace the input path in the notebook, or call the helper already used there:

```python
images_response, Dynamic_states_VTC, Dynamic_states_vlPFC, F_all, H_top_down_all = Feedback_results(
    'Stim_for_model/my_condition/', mean, std
)
```

6. Save the output dictionary with `pickle.dump(...)` using the same keys as the bundled files:
   - `Dynamic_states_VTC`
   - `Dynamic_states_vlPFC`
   - `F`
   - `H_top_down`

### If you want to compare your own condition against tools

The decoding analysis in `Model_results_formal.ipynb` is face-vs-tool based. Therefore, if you want to reproduce the same type of decoding on your own data, you should also prepare a non-face comparison folder analogous to `Stim_for_model/tools/` and generate a matching `Tool_feedback_results.npy`.

### Important note about `Stimuli_formal.ipynb`

`Stimuli_formal.ipynb` is used for the Grad-CAM based face-information analysis described in the manuscript. In the current release, the notebook is configured to train a simple local face-vs-tool classifier head from the bundled `Stim_for_model/face/` and `Stim_for_model/tools/` images before running Grad-CAM, so no extra external checkpoint is required.

## Repository Contents

```text
Formal/
|-- BrainSOM.py
|-- Hopfield_VTCSOM.py
|-- Stimuli_formal.ipynb
|-- Occluded_face_formal.ipynb
|-- Model_results_formal.ipynb
|-- Stim_for_model/
|   |-- face/
|   |-- noeye/
|   |-- top_face/
|   |-- down_face/
|   |-- eyes/
|   `-- tools/
|-- model_results/
|-- Data.npy
|-- mean.npy
|-- std.npy
|-- face_mask.npy
|-- object_mask.npy
|-- som_sigma_6.2.npy
|-- som_vlPFC_weights.npy
|-- model_VTC_weights.npy
|-- model_vlPFC_weights.npy
|-- X_VTC.npy
|-- X_vlPFC.npy
|-- H_VTC_recurrent.npy
|-- H_VTC_feedback.npy
|-- H_vlPFC_feedback.npy
|-- model_dimensions_VTC.npy
|-- model_dimensions_vlPFC.npy
|-- model_radii_VTC.npy
|-- model_radii_vlPFC.npy
|-- requirements.txt
`-- README.md
```

## License

This project is released under the MIT License. See `LICENSE`.
