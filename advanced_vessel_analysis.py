#!/usr/bin/env python3
"""
Advanced Vessel Analysis Pipeline
Combines sophisticated vessel segmentation, skeletonization, diameter measurement,
and stenosis quantification for angiogram analysis.

Key Features:
- Frangi vesselness filtering for enhanced vessel detection
- Medial axis transform for accurate diameter measurement
- Graph-based skeleton analysis for vessel centerlines
- Quantitative stenosis detection with %DS calculation
- Comprehensive outputs: overlays, profiles, and CSV data
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import io, img_as_float
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import (closing, opening, remove_small_holes,
                               remove_small_objects, disk, medial_axis, dilation)
from skimage.segmentation import clear_border
import networkx as nx

# ---------------------------
# Configuration Parameters
# ---------------------------
MIN_OBJECT_SIZE = 200
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSS_SIGMA = 1.0
FRANGI_BETA = 0.5
FRANGI_GAMMA = 25
CLOSE_RADIUS = 3
OPEN_RADIUS = 1

ROLL_WINDOW = 80          # samples for proximal reference
ROLL_PERCENT = 0.90       # P90
EDGE_EXCLUDE = 15         # ignore ends when picking min diameter
MIN_REF_DIAM = 1.0        # px; ignore minima with tiny reference

# ---------------------------
# Core Analysis Functions
# ---------------------------

def enhance_gray(gray, clahe_clip=CLAHE_CLIP, clahe_tile=CLAHE_TILE, sigma=GAUSS_SIGMA):
    """Enhanced contrast and noise reduction for angiogram images"""
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    g8 = clahe.apply(g8)
    g = g8.astype(np.float32) / 255.0
    g = gaussian_filter(g, sigma=sigma)
    return g

def build_fov_mask(gray, right_strip_px=110, bottom_strip_px=15):
    """Create field-of-view mask to exclude imaging artifacts"""
    m = np.ones_like(gray, dtype=bool)
    m[:6, :] = False
    m[-6:, :] = False
    m[:, :6] = False
    m[:, -6:] = False
    if right_strip_px > 0:
        m[:, -right_strip_px:] = False
    if bottom_strip_px > 0:
        m[-bottom_strip_px:, :] = False
    return m

def segment_vessels_advanced(gray, fov_mask, min_obj=MIN_OBJECT_SIZE,
                           close_r=CLOSE_RADIUS, open_r=OPEN_RADIUS, ridge_dilate=3):
    """
    Advanced vessel segmentation using Frangi vesselness and morphological operations
    Returns (vessel_mask, vesselness_map)
    """
    # Enhanced preprocessing
    g = enhance_gray(gray)
    inv = 1.0 - g  # arteries darker → higher in inv

    # Black-hat transform to remove low-frequency background
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closing_g = cv2.morphologyEx((g * 255).astype(np.uint8), cv2.MORPH_CLOSE, se)
    blackhat = (closing_g.astype(np.float32) - (g * 255)).clip(0, 255) / 255.0

    # Frangi vesselness on inverted gray (dark vessels become bright)
    vesselness = frangi(inv, beta=FRANGI_BETA, gamma=FRANGI_GAMMA)
    vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)

    # Candidate vessels by Otsu thresholding on blackhat within FOV
    bh = blackhat.copy()
    bh[~fov_mask] = 0
    t_bh = threshold_otsu(bh[fov_mask])
    vessels_initial = bh > t_bh

    # Gate with vesselness ridges (dilated) to keep only vascular structures
    ridge = vesselness > np.percentile(vesselness[fov_mask], 70)
    ridge = dilation(ridge, disk(ridge_dilate))
    vessel_mask = vessels_initial & ridge

    # Morphological cleanup
    vessel_mask = binary_fill_holes(vessel_mask)
    vessel_mask = closing(vessel_mask, disk(close_r))
    vessel_mask = opening(vessel_mask, disk(open_r))
    vessel_mask = remove_small_objects(vessel_mask, min_size=min_obj)
    vessel_mask = remove_small_holes(vessel_mask, area_threshold=min_obj // 2)
    vessel_mask = clear_border(vessel_mask)

    # Keep largest connected component
    num_labels, labels = cv2.connectedComponents(vessel_mask.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        counts = np.bincount(labels.ravel())
        keep = np.argmax(counts[1:]) + 1 if counts.size > 1 else 0
        vessel_mask = (labels == keep)
    else:
        vessel_mask = (labels > 0)

    return vessel_mask, vesselness

def extract_skeleton_with_diameters(mask):
    """Extract skeleton using medial axis transform with diameter measurements"""
    skel, dist = medial_axis(mask, return_distance=True)
    y, x = np.nonzero(skel)
    coords = np.column_stack([y, x])
    diameters = 2.0 * dist[skel]  # Convert radius to diameter
    return skel, dist, coords, diameters

def find_main_vessel_path(skeleton):
    """Find main vessel path using simple distance-based approach"""
    # Get skeleton coordinates
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        return np.array([]), np.array([])

    coords = np.column_stack([ys, xs])

    # Find endpoints (points with only one neighbor)
    endpoints = []
    for i, (y, x) in enumerate(coords):
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx]:
                        neighbors += 1
        if neighbors <= 1:
            endpoints.append(i)

    if len(endpoints) < 2:
        # If no clear endpoints, find two most distant points
        if len(coords) < 2:
            return np.arange(len(coords)), coords

        # Find the two points that are farthest apart
        max_dist = 0
        best_pair = (0, len(coords)-1)
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (i, j)

        # Create simple path from start to end
        start_idx, end_idx = best_pair
        path_indices = np.arange(min(start_idx, end_idx), max(start_idx, end_idx) + 1)
        return path_indices, coords

    # Find path between two furthest endpoints
    ep1, ep2 = endpoints[0], endpoints[-1]
    if len(endpoints) > 2:
        max_dist = 0
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                dist = np.linalg.norm(coords[endpoints[i]] - coords[endpoints[j]])
                if dist > max_dist:
                    max_dist = dist
                    ep1, ep2 = endpoints[i], endpoints[j]

    # Create path by sorting points along the line between endpoints
    start_pt = coords[ep1]
    end_pt = coords[ep2]

    # Project all points onto the line between start and end
    line_vec = end_pt - start_pt
    line_length = np.linalg.norm(line_vec)
    if line_length == 0:
        return np.arange(len(coords)), coords

    line_unit = line_vec / line_length

    projections = []
    for i, pt in enumerate(coords):
        vec_to_pt = pt - start_pt
        projection = np.dot(vec_to_pt, line_unit)
        projections.append((projection, i))

    # Sort by projection to get ordered path
    projections.sort(key=lambda x: x[0])
    path_indices = np.array([idx for _, idx in projections])

    return path_indices, coords

def calculate_rolling_reference(diameters, window=ROLL_WINDOW, percentile=ROLL_PERCENT):
    """Calculate rolling reference diameter for stenosis quantification"""
    s = pd.Series(diameters)
    ref = s.rolling(window, min_periods=max(10, window // 4)).quantile(percentile)
    ref = ref.bfill().ffill().to_numpy()
    return ref

def measure_stenosis_profile(coords, diameters, path_order):
    """Measure stenosis along the vessel path"""
    # Extract ordered coordinates and diameters along path
    ordered_coords = coords[path_order, :]
    ordered_diameters = diameters[path_order]

    # Calculate rolling reference diameter
    ref_diameters = calculate_rolling_reference(ordered_diameters)

    # Calculate percent diameter stenosis
    pds = (1.0 - (ordered_diameters / (ref_diameters + 1e-8))) * 100.0

    # Find minimum diameter location (excluding edges)
    lo = EDGE_EXCLUDE
    hi = len(ordered_diameters) - EDGE_EXCLUDE
    if hi > lo:
        min_idx = np.argmin(ordered_diameters[lo:hi]) + lo
    else:
        min_idx = np.argmin(ordered_diameters)

    # Ensure reference diameter is significant
    if ref_diameters[min_idx] < MIN_REF_DIAM:
        for k in np.argsort(ordered_diameters):
            if ref_diameters[k] >= MIN_REF_DIAM:
                min_idx = int(k)
                break

    return ordered_coords, ordered_diameters, ref_diameters, pds, int(min_idx)

def detect_stenoses(diameters, pds, rel_drop=0.4, abs_drop=2.0, window=15):
    """Detect stenoses based on sudden diameter drops"""
    d = np.asarray(diameters)
    stenoses = []

    # Compute gradient for V-shape detection
    grad = np.gradient(d)

    # Find local minima as potential stenosis points
    mins, _ = find_peaks(-d)

    for i in mins:
        left = max(0, i - window)
        right = min(len(d), i + window)
        d_left = np.max(d[left:i]) if i > left else d[i]
        d_right = np.max(d[i:right]) if i < right else d[i]

        drop = max(d_left - d[i], d_right - d[i])
        rel_drop_val = drop / (max(d_left, d_right) + 1e-8)

        # Stenosis criteria: significant drop + V-shape
        is_significant_drop = (drop >= abs_drop or rel_drop_val >= rel_drop)
        is_v_shape = (i > 0 and i < len(grad) - 1 and
                     grad[i-1] < 0 and grad[i+1] > 0)

        if is_significant_drop and is_v_shape:
            stenoses.append(i)

    # Sort by severity (narrowest first)
    stenoses = sorted(stenoses, key=lambda j: d[j])
    return stenoses

def create_overlay_visualization(image, coords, diameters, pds, stenosis_idx, output_path):
    """Create comprehensive overlay visualization"""
    # Convert to 8-bit RGB
    img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)

    # Draw vessel centerline
    for i in range(len(coords) - 1):
        pt1 = (int(coords[i, 1]), int(coords[i, 0]))
        pt2 = (int(coords[i+1, 1]), int(coords[i+1, 0]))
        cv2.line(overlay, pt1, pt2, (255, 255, 0), 1)  # Yellow centerline

    # Highlight stenosis location
    if stenosis_idx < len(coords):
        center_y, center_x = coords[stenosis_idx]
        center_pt = (int(center_x), int(center_y))

        # Draw stenosis marker
        cv2.drawMarker(overlay, center_pt, (0, 0, 255),
                      markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # Draw diameter measurement line
        diameter_px = diameters[stenosis_idx]
        radius = max(1, int(diameter_px / 2))

        # Calculate perpendicular direction for diameter line
        if stenosis_idx > 0 and stenosis_idx < len(coords) - 1:
            # Use neighboring points to find tangent direction
            prev_pt = coords[stenosis_idx - 1]
            next_pt = coords[stenosis_idx + 1]
            tangent = next_pt - prev_pt
            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
            normal = np.array([-tangent[1], tangent[0]])  # Perpendicular
        else:
            normal = np.array([0, 1])  # Default direction

        # Draw diameter line
        offset = normal * radius
        pt1 = (int(center_x - offset[1]), int(center_y - offset[0]))
        pt2 = (int(center_x + offset[1]), int(center_y + offset[0]))
        cv2.line(overlay, pt1, pt2, (0, 0, 255), 2)

        # Add stenosis percentage label
        pds_val = pds[stenosis_idx]
        label = f"{pds_val:.0f}% DS"
        cv2.putText(overlay, label, (int(center_x) + 8, int(center_y) - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, overlay)
    return overlay

def create_diameter_profile_plot(diameters, ref_diameters, pds, stenosis_idx, output_path, image_name):
    """Create diameter profile plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Diameter plot
    indices = np.arange(len(diameters))
    ax1.plot(indices, diameters, 'b-', linewidth=2, label='Diameter (px)')
    ax1.plot(indices, ref_diameters, 'g--', linewidth=1.5, label='Reference (P90)')

    # Mark stenosis
    if stenosis_idx < len(diameters):
        ax1.plot(stenosis_idx, diameters[stenosis_idx], 'ro', markersize=8,
                label=f'Stenosis ({pds[stenosis_idx]:.1f}% DS)')

    ax1.set_xlabel('Centerline Index')
    ax1.set_ylabel('Diameter (pixels)')
    ax1.set_title(f'Vessel Diameter Profile: {image_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Percent diameter stenosis plot
    ax2.plot(indices, pds, 'r-', linewidth=2, label='% Diameter Stenosis')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% DS threshold')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='70% DS threshold')

    if stenosis_idx < len(pds):
        ax2.plot(stenosis_idx, pds[stenosis_idx], 'ro', markersize=8)

    ax2.set_xlabel('Centerline Index')
    ax2.set_ylabel('% Diameter Stenosis')
    ax2.set_title('Stenosis Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_angiogram_advanced(image_path, output_dir="advanced_output", px_size=None):
    """Complete advanced angiogram analysis pipeline"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read and preprocess image
    try:
        image = img_as_float(io.imread(image_path, as_gray=True))
    except:
        print(f"Error: Could not read image {image_path}")
        return None

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing {image_path}...")

    # Create field-of-view mask
    fov_mask = build_fov_mask(image)

    # Advanced vessel segmentation
    vessel_mask, vesselness = segment_vessels_advanced(image, fov_mask)

    # Extract skeleton with diameter measurements
    skeleton, distance_map, coords, diameters = extract_skeleton_with_diameters(vessel_mask)

    # Find main vessel path
    path_indices, path_coords = find_main_vessel_path(skeleton)

    if len(path_indices) == 0:
        print(f"Warning: No valid vessel path found in {image_path}")
        return None

    # Measure stenosis profile along main vessel
    ordered_coords, ordered_diams, ref_diams, pds, min_idx = measure_stenosis_profile(
        coords, diameters, path_indices)

    # Detect stenoses
    stenosis_indices = detect_stenoses(ordered_diams, pds)
    primary_stenosis_idx = stenosis_indices[0] if stenosis_indices else min_idx

    # Save intermediate results
    cv2.imwrite(f"{output_dir}/{base_name}_vesselness.png",
                (vesselness * 255).astype(np.uint8))
    cv2.imwrite(f"{output_dir}/{base_name}_mask.png",
                (vessel_mask.astype(np.uint8)) * 255)

    # Create skeleton visualization
    img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    skel_overlay = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    ys, xs = np.nonzero(skeleton)
    skel_overlay[ys, xs] = (255, 255, 255)  # White skeleton
    cv2.imwrite(f"{output_dir}/{base_name}_skeleton.png", skel_overlay)

    # Create comprehensive overlay
    overlay_path = f"{output_dir}/{base_name}_overlay.png"
    create_overlay_visualization(image, ordered_coords, ordered_diams, pds,
                               primary_stenosis_idx, overlay_path)

    # Create diameter profile plot
    profile_path = f"{output_dir}/{base_name}_profile.png"
    create_diameter_profile_plot(ordered_diams, ref_diams, pds, primary_stenosis_idx,
                                profile_path, base_name)

    # Generate comprehensive CSV data
    curve_data = pd.DataFrame({
        'curve_index': np.arange(len(ordered_coords)),
        'y': ordered_coords[:, 0],
        'x': ordered_coords[:, 1],
        'diameter_px': ordered_diams,
        'ref_diameter_px': ref_diams,
        'percent_diameter_stenosis': pds
    })

    if px_size:
        curve_data['diameter_mm'] = curve_data['diameter_px'] * px_size
        curve_data['ref_diameter_mm'] = curve_data['ref_diameter_px'] * px_size

    csv_path = f"{output_dir}/{base_name}_analysis.csv"
    curve_data.to_csv(csv_path, index=False)

    # Generate detection summary
    detections = []
    for idx in stenosis_indices[:3]:  # Top 3 stenoses
        detections.append({
            'curve_index': idx,
            'y': int(ordered_coords[idx, 0]),
            'x': int(ordered_coords[idx, 1]),
            'diameter_px': float(ordered_diams[idx]),
            'ref_diameter_px': float(ref_diams[idx]),
            'percent_stenosis': float(pds[idx])
        })

    detection_path = f"{output_dir}/{base_name}_detections.csv"
    pd.DataFrame(detections).to_csv(detection_path, index=False)

    # Print results
    print(f"✓ Vesselness map: {output_dir}/{base_name}_vesselness.png")
    print(f"✓ Vessel mask: {output_dir}/{base_name}_mask.png")
    print(f"✓ Skeleton: {output_dir}/{base_name}_skeleton.png")
    print(f"✓ Overlay: {overlay_path}")
    print(f"✓ Profile plot: {profile_path}")
    print(f"✓ Analysis data: {csv_path}")
    print(f"✓ Detections: {detection_path}")

    if detections:
        primary = detections[0]
        print(f"Primary stenosis: {primary['percent_stenosis']:.1f}% DS at ({primary['x']}, {primary['y']})")

    return {
        'image_path': image_path,
        'num_detections': len(detections),
        'primary_stenosis': detections[0] if detections else None,
        'analysis_csv': csv_path,
        'overlay_path': overlay_path
    }

def main():
    """Process angiogram images with advanced vessel analysis"""
    print("🔬 Advanced Vessel Analysis Pipeline")
    print("=" * 50)
    print("Features:")
    print("• Frangi vesselness filtering for enhanced vessel detection")
    print("• Medial axis transform for precise diameter measurement")
    print("• Graph-based skeleton analysis for main vessel extraction")
    print("• Quantitative stenosis detection with %DS calculation")
    print("• Comprehensive outputs: overlays, profiles, and CSV data")
    print("=" * 50)

    # Process all angiogram images
    image_files = ["Angiogram_1.png", "Angiogram_2.png", "Angiogram_3.png"]
    results = []

    for image_file in image_files:
        if os.path.exists(image_file):
            result = process_angiogram_advanced(image_file, px_size=0.22)  # Assuming 0.22 mm/pixel
            if result:
                results.append(result)
        else:
            print(f"Warning: {image_file} not found")

    # Generate summary
    if results:
        print(f"\n🎉 Analysis complete! Processed {len(results)} images.")
        print("📁 Check 'advanced_output/' directory for comprehensive results.")

        summary_data = []
        for r in results:
            summary_data.append({
                'image': os.path.basename(r['image_path']),
                'num_stenoses': r['num_detections'],
                'max_stenosis_pct': r['primary_stenosis']['percent_stenosis'] if r['primary_stenosis'] else 0,
                'overlay_path': r['overlay_path'],
                'analysis_csv': r['analysis_csv']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = "advanced_output/analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Summary report: {summary_path}")
    else:
        print("❌ No images were successfully processed.")

if __name__ == "__main__":
    main()