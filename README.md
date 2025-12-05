# F25_CVE_Denver – Using Computer Vision to Screen Angiograms

This repository contains the code and experiments for our **F25 Computer Vision (CVE) project** on **Using Computer Vision to Screen Angiograms**.

The goal is to build a **clinical-grade pipeline** that:
- Segments coronary vessels from X-ray angiograms
- Extracts accurate vessel **centerlines and diameters**
- Quantifies **% Diameter Stenosis (%DS)** along the main vessel
- Produces rich **visual and tabular outputs** for clinical interpretation
- Scales to **batch processing of large datasets**

> Core work and full pipeline are currently developed in the team branches (`Sangram`, `Vrishabh`, `Rajiv`, `Shweta`). The `main` branch is a lightweight landing branch.

---

## 1. Project Overview

Coronary stenosis (narrowing of arteries) is usually quantified using the **% Diameter Stenosis** metric:

\[
\%DS = \left(1 - \frac{D_{\text{stenosis}}}{D_{\text{ref}}}\right) \times 100\%
\]

where:

- \( D_{\text{stenosis}} \) is the minimum lumen diameter at the lesion
- \( D_{\text{ref}} \) is a rolling “healthy” reference diameter upstream

This project implements an automated pipeline to:

1. Preprocess raw angiogram images (contrast enhancement, noise reduction, field-of-view masking)
2. Perform **advanced vessel segmentation** using Frangi vesselness + morphological ops
3. Extract the **vessel skeleton** and estimate **diameter** along the centerline
4. Compute **reference diameter profiles** and **%DS** along the vessel
5. Detect and rank significant stenoses
6. Generate **PNG overlays, diameter profile plots, and CSV summaries**, both per-image and for entire datasets

---

## 2. Key Features

- **Frangi Vesselness Filtering**
  - Multi-scale enhancement of tubular structures (vessels)
  - Parameters tuned for coronary angiograms (e.g., \( \beta \approx 0.5, \gamma \approx 20\text{–}25 \))

- **Robust Lumen Segmentation**
  - CLAHE-based contrast enhancement
  - Black-hat transform to remove low-frequency background
  - Otsu thresholding gated by vesselness ridges
  - Morphological cleanup (closing, opening, hole/fragment removal)
  - Field-of-view masking to ignore non-anatomical regions

- **Skeletonization & Diameter Measurement**
  - Medial axis transform for centerline and distance map
  - Diameter from distance transform: \( D = 2 \times \text{distance} \)
  - Alternative edge-based diameter using normals and edge hits

- **Graph-based Centerline Extraction**
  - Skeleton → pixel graph (nodes = skeleton pixels, weighted edges)
  - Longest-path search on largest connected component for main vessel

- **Stenosis Quantification**
  - Rolling reference diameter \( D_{\text{ref}}(i) = P_{90} \) over a sliding window
  - %DS profile:
    \[
    \%DS(i) = \left[1 - \frac{D(i)}{D_{\text{ref}}(i) + \varepsilon}\right] \times 100\%
    \]
  - Lesion region expansion where \(\%DS\) stays above a severity threshold (e.g. 50%)

- **Batch Processing**
  - Multi-threaded execution for hundreds of images
  - Per-image output folders + global summary tables
  - Non-interactive Matplotlib backend (`Agg`) for headless processing

---

## 3. Repository & Branch Structure

### Default Branch

- **`main`**
  - Minimal landing branch
  - Placeholder `README.md`
  - Meant to be updated with the consolidated project description (this file)

### Active Development Branches

- **`Sangram` branch**
  - `advanced_vessel_analysis.py`  
    Single-image advanced pipeline (preprocessing → segmentation → skeleton → diameter → %DS → visualizations).
  - `detailed_batch_analysis.py`  
    Batch pipeline that mirrors clinical workflow and generates rich outputs for each case.
  - `advanced_output/`  
    Example per-image output folders (vesselness, mask, skeleton, clinical overlay, diameter profile, CSVs).
  - `MIDTERM_REPORT.md`  
    Detailed midterm report documenting algorithm design, performance, and results.
  - `workdone.txt`  
    High-level changelog / notes.

- **`Vrishabh` branch**
  - `stenosis_centerline_batch.py`  
    Production-style **batch stenosis analysis** script:
    - CLAHE + Gaussian preprocessing
    - Frangi-based segmentation and FOV masking
    - Medial axis skeletonization and diameter estimation
    - Rolling reference diameter and %DS calculation
    - Overlay rendering (stenosis markers, calipers, %DS labels)
  - `Data/`  
    Input angiogram image samples.
  - `Out/`, `Out_all/`  
    Example output directories for batch runs.
  - `todo*.txt`  
    Implementation notes / next steps.

- **`Rajiv` branch**
  - `blockage_pipeline.py`  
    Prototype pipeline for stenosis (“blockage”) detection on sample angiograms.
  - `Angiogram_1.png`, `Angiogram_2.png`, `Angiogram_3.png`  
    Sample input images used for early experiments.

- **`Shweta` branch**
  - `binary.py`  
    Image binarization / segmentation experiments.
  - `Angiogram_*bin*.png`  
    Example binarized outputs for different thresholds / configurations.

> Each branch represents individual contributions and experiments. The final consolidated pipeline is mostly reflected in the `Sangram` and `Vrishabh` branches.

---

## 4. Installation

### 4.1. Clone the repository

```bash
git clone https://github.com/Sangramxd/F25_CVE_Denver.git
cd F25_CVE_Denver

