#!/usr/bin/env python3
"""
Detailed Batch Vessel Analysis Pipeline
Generates comprehensive outputs for each image similar to advanced_output format.

Features:
- Individual overlays with stenosis markers for each image
- Diameter profile plots for each vessel
- Vesselness maps and binary masks
- Skeleton visualizations
- Complete CSV analysis data
- Multi-threaded processing with detailed outputs
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multi-threading
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import io, img_as_float
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import (closing, opening, remove_small_holes,
                               remove_small_objects, disk, medial_axis, dilation)
from skimage.segmentation import clear_border
import time
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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

ROLL_WINDOW = 80
ROLL_PERCENT = 0.90
EDGE_EXCLUDE = 15
MIN_REF_DIAM = 1.0

# Processing configuration
DEFAULT_PX_SIZE = 0.22  # mm per pixel
MAX_WORKERS = 6  # Increased workers for detailed processing
BATCH_SIZE = 10  # Smaller batches for memory management with detailed outputs

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
    """Advanced vessel segmentation using Frangi vesselness and morphological operations"""
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
    if np.any(fov_mask):
        t_bh = threshold_otsu(bh[fov_mask])
    else:
        t_bh = 0.1
    vessels_initial = bh > t_bh

    # Gate with vesselness ridges (dilated) to keep only vascular structures
    ridge = vesselness > np.percentile(vesselness[fov_mask], 70) if np.any(fov_mask) else vesselness > 0.1
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
    try:
        skel, dist = medial_axis(mask, return_distance=True)
        y, x = np.nonzero(skel)
        if len(y) == 0:
            return None, None, None, None
        coords = np.column_stack([y, x])
        diameters = 2.0 * dist[skel]  # Convert radius to diameter
        return skel, dist, coords, diameters
    except:
        return None, None, None, None

def find_main_vessel_path(skeleton):
    """Find main vessel path using distance-based approach"""
    try:
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

    except:
        return np.array([]), np.array([])

def calculate_rolling_reference(diameters, window=ROLL_WINDOW, percentile=ROLL_PERCENT):
    """Calculate rolling reference diameter for stenosis quantification"""
    try:
        s = pd.Series(diameters)
        ref = s.rolling(window, min_periods=max(10, window // 4)).quantile(percentile)
        ref = ref.bfill().ffill().to_numpy()
        return ref
    except:
        return np.full_like(diameters, np.mean(diameters))

def measure_stenosis_profile(coords, diameters, path_order):
    """Measure stenosis along the vessel path"""
    try:
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
    except:
        return coords, diameters, np.ones_like(diameters), np.zeros_like(diameters), 0

def detect_stenoses(diameters, pds, rel_drop=0.4, abs_drop=2.0, window=15):
    """Detect stenoses based on sudden diameter drops"""
    try:
        d = np.asarray(diameters)
        stenoses = []

        # Find significant stenoses (>50% DS)
        significant_idx = np.where(pds > 50)[0]
        if len(significant_idx) > 0:
            # Group nearby stenoses and pick the worst in each group
            groups = []
            current_group = [significant_idx[0]]

            for i in range(1, len(significant_idx)):
                if significant_idx[i] - significant_idx[i-1] <= window:
                    current_group.append(significant_idx[i])
                else:
                    groups.append(current_group)
                    current_group = [significant_idx[i]]
            groups.append(current_group)

            # Get worst stenosis from each group
            for group in groups:
                worst_idx = group[np.argmax(pds[group])]
                stenoses.append(worst_idx)

        # Sort by severity (worst first)
        stenoses = sorted(stenoses, key=lambda j: pds[j], reverse=True)
        return stenoses[:5]  # Return top 5 stenoses
    except:
        return []

def create_overlay_visualization(image, coords, diameters, pds, stenosis_indices, output_path):
    """Create comprehensive overlay visualization"""
    try:
        # Convert to 8-bit RGB
        img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        overlay = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)

        # Draw vessel centerline in yellow
        for i in range(len(coords) - 1):
            pt1 = (int(coords[i, 1]), int(coords[i, 0]))
            pt2 = (int(coords[i+1, 1]), int(coords[i+1, 0]))
            cv2.line(overlay, pt1, pt2, (0, 255, 255), 1)

        # Highlight stenoses
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0)]  # Red, orange, green
        for i, stenosis_idx in enumerate(stenosis_indices[:3]):
            if stenosis_idx < len(coords):
                center_y, center_x = coords[stenosis_idx]
                center_pt = (int(center_x), int(center_y))

                color = colors[i] if i < len(colors) else (0, 0, 255)

                # Draw stenosis marker
                cv2.drawMarker(overlay, center_pt, color,
                              markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

                # Draw diameter measurement line
                diameter_px = diameters[stenosis_idx]
                radius = max(1, int(diameter_px / 2))

                # Calculate perpendicular direction for diameter line
                if stenosis_idx > 0 and stenosis_idx < len(coords) - 1:
                    prev_pt = coords[stenosis_idx - 1]
                    next_pt = coords[stenosis_idx + 1]
                    tangent = next_pt - prev_pt
                    tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                    normal = np.array([-tangent[1], tangent[0]])
                else:
                    normal = np.array([0, 1])

                # Draw diameter line
                offset = normal * radius
                pt1 = (int(center_x - offset[1]), int(center_y - offset[0]))
                pt2 = (int(center_x + offset[1]), int(center_y + offset[0]))
                cv2.line(overlay, pt1, pt2, color, 2)

                # Add stenosis percentage label
                pds_val = pds[stenosis_idx]
                label = f"{pds_val:.0f}% DS"
                cv2.putText(overlay, label, (int(center_x) + 8, int(center_y) - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imwrite(output_path, overlay)
        return True
    except:
        return False

def create_diameter_profile_plot(diameters, ref_diameters, pds, stenosis_indices, output_path, image_name):
    """Create diameter profile plot"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Diameter plot
        indices = np.arange(len(diameters))
        ax1.plot(indices, diameters, 'b-', linewidth=2, label='Diameter (px)')
        ax1.plot(indices, ref_diameters, 'g--', linewidth=1.5, label='Reference (P90)')

        # Mark stenoses
        colors = ['red', 'orange', 'green']
        for i, stenosis_idx in enumerate(stenosis_indices[:3]):
            if stenosis_idx < len(diameters):
                color = colors[i] if i < len(colors) else 'red'
                ax1.plot(stenosis_idx, diameters[stenosis_idx], 'o', color=color, markersize=8,
                        label=f'Stenosis {i+1} ({pds[stenosis_idx]:.1f}% DS)')

        ax1.set_xlabel('Centerline Index')
        ax1.set_ylabel('Diameter (pixels)')
        ax1.set_title(f'Vessel Diameter Profile: {image_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Percent diameter stenosis plot
        ax2.plot(indices, pds, 'r-', linewidth=2, label='% Diameter Stenosis')
        ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% DS threshold')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='70% DS threshold')

        for i, stenosis_idx in enumerate(stenosis_indices[:3]):
            if stenosis_idx < len(pds):
                color = colors[i] if i < len(colors) else 'red'
                ax2.plot(stenosis_idx, pds[stenosis_idx], 'o', color=color, markersize=8)

        ax2.set_xlabel('Centerline Index')
        ax2.set_ylabel('% Diameter Stenosis')
        ax2.set_title('Stenosis Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        plt.close('all')
        return False

def process_single_image_detailed(image_path, output_dir, px_size=DEFAULT_PX_SIZE):
    """Complete detailed processing of a single angiogram image"""
    try:
        # Create individual output directory for this image
        base_name = Path(image_path).stem
        image_output_dir = Path(output_dir) / base_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Read image
        image = img_as_float(io.imread(image_path, as_gray=True))

        print(f"Processing {base_name}...")

        # Create field-of-view mask
        fov_mask = build_fov_mask(image)

        # Advanced vessel segmentation
        vessel_mask, vesselness = segment_vessels_advanced(image, fov_mask)

        # Extract skeleton with diameter measurements
        skeleton, distance_map, coords, diameters = extract_skeleton_with_diameters(vessel_mask)

        if skeleton is None or len(coords) < 5:
            print(f"Warning: Insufficient skeleton data for {base_name}")
            return None

        # Find main vessel path
        path_indices, path_coords = find_main_vessel_path(skeleton)

        if len(path_indices) < 3:
            print(f"Warning: Insufficient path data for {base_name}")
            return None

        # Measure stenosis profile along main vessel
        ordered_coords, ordered_diams, ref_diams, pds, min_idx = measure_stenosis_profile(
            coords, diameters, path_indices)

        # Detect stenoses
        stenosis_indices = detect_stenoses(ordered_diams, pds)
        if not stenosis_indices:
            stenosis_indices = [min_idx]  # Use minimum diameter point as fallback

        # Save intermediate results (like advanced_output)
        cv2.imwrite(str(image_output_dir / f"{base_name}_vesselness.png"),
                    (vesselness * 255).astype(np.uint8))
        cv2.imwrite(str(image_output_dir / f"{base_name}_mask.png"),
                    (vessel_mask.astype(np.uint8)) * 255)

        # Create skeleton visualization
        img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        skel_overlay = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
        ys, xs = np.nonzero(skeleton)
        skel_overlay[ys, xs] = (255, 255, 255)  # White skeleton
        cv2.imwrite(str(image_output_dir / f"{base_name}_skeleton.png"), skel_overlay)

        # Create comprehensive overlay
        overlay_path = str(image_output_dir / f"{base_name}_overlay.png")
        create_overlay_visualization(image, ordered_coords, ordered_diams, pds,
                                   stenosis_indices, overlay_path)

        # Create diameter profile plot
        profile_path = str(image_output_dir / f"{base_name}_profile.png")
        create_diameter_profile_plot(ordered_diams, ref_diams, pds, stenosis_indices,
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

        csv_path = str(image_output_dir / f"{base_name}_analysis.csv")
        curve_data.to_csv(csv_path, index=False)

        # Generate detection summary
        detections = []
        for idx in stenosis_indices:
            detections.append({
                'curve_index': idx,
                'y': int(ordered_coords[idx, 0]),
                'x': int(ordered_coords[idx, 1]),
                'diameter_px': float(ordered_diams[idx]),
                'ref_diameter_px': float(ref_diams[idx]),
                'percent_stenosis': float(pds[idx])
            })

        detection_path = str(image_output_dir / f"{base_name}_detections.csv")
        pd.DataFrame(detections).to_csv(detection_path, index=False)

        return {
            'image': base_name,
            'status': 'success',
            'num_stenoses': len(detections),
            'max_stenosis': float(np.max(pds)) if len(pds) > 0 else 0,
            'primary_stenosis': detections[0] if detections else None,
            'output_dir': str(image_output_dir),
            'files_created': [
                f"{base_name}_vesselness.png",
                f"{base_name}_mask.png",
                f"{base_name}_skeleton.png",
                f"{base_name}_overlay.png",
                f"{base_name}_profile.png",
                f"{base_name}_analysis.csv",
                f"{base_name}_detections.csv"
            ]
        }

    except Exception as e:
        print(f"Error processing {Path(image_path).stem}: {str(e)}")
        return {
            'image': Path(image_path).stem,
            'status': 'error',
            'error': str(e),
            'num_stenoses': 0,
            'max_stenosis': 0
        }

def process_batch_detailed(image_paths, output_dir, px_size=DEFAULT_PX_SIZE, max_workers=MAX_WORKERS):
    """Process a batch of images with detailed outputs"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    errors = []

    print(f"Processing batch of {len(image_paths)} images with detailed outputs using {max_workers} workers...")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_path = {
            executor.submit(process_single_image_detailed, path, output_dir, px_size): path
            for path in image_paths
        }

        # Collect results with progress tracking
        completed = 0
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    if result['status'] == 'error':
                        errors.append((path, result.get('error', 'unknown')))
                completed += 1

                # Progress update
                if completed % 5 == 0 or completed == len(image_paths):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(image_paths) - completed) / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{len(image_paths)} ({completed/len(image_paths)*100:.1f}%) - "
                          f"Rate: {rate:.1f} img/sec - ETA: {eta:.0f}s")

            except Exception as e:
                print(f"Error with {path}: {str(e)}")
                errors.append((path, str(e)))

    elapsed = time.time() - start_time
    print(f"Detailed batch completed in {elapsed:.1f}s ({len(image_paths)/elapsed:.1f} img/sec)")
    print(f"Success: {len([r for r in results if r.get('status') == 'success'])}, "
          f"Failed: {len([r for r in results if r.get('status') != 'success'])}")

    return results, errors

def main():
    """Main detailed batch processing function"""
    print("🔬 Detailed Batch Vessel Analysis Pipeline")
    print("Generating comprehensive outputs similar to advanced_output format")
    print("=" * 70)

    # Find all images in the images directory
    images_dir = Path("images")
    if not images_dir.exists():
        print(f"Error: {images_dir} directory not found!")
        return

    image_files = list(images_dir.glob("*.png"))
    image_files.extend(images_dir.glob("*.jpg"))
    image_files.extend(images_dir.glob("*.jpeg"))

    if not image_files:
        print("No image files found!")
        return

    print(f"Found {len(image_files)} images to process with detailed outputs")

    # Create main output directory
    output_dir = "detailed_vessel_output"
    os.makedirs(output_dir, exist_ok=True)

    # Process images in smaller batches for detailed processing
    all_results = []
    all_errors = []

    start_time = time.time()

    # Process all 300 images
    test_files = image_files  # Processing all images

    for i in range(0, len(test_files), BATCH_SIZE):
        batch_files = test_files[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(test_files) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n--- Detailed Batch {batch_num}/{total_batches} ---")

        batch_results, batch_errors = process_batch_detailed(
            [str(f) for f in batch_files],
            output_dir,
            px_size=DEFAULT_PX_SIZE,
            max_workers=MAX_WORKERS
        )

        all_results.extend(batch_results)
        all_errors.extend(batch_errors)

    total_time = time.time() - start_time

    # Generate summary
    success_results = [r for r in all_results if r.get('status') == 'success']

    print(f"\n🎉 Detailed processing completed!")
    print(f"Total time: {total_time:.1f}s ({len(test_files)/total_time:.1f} img/sec)")
    print(f"Total images: {len(test_files)}")
    print(f"Successful: {len(success_results)}")
    print(f"Errors: {len(all_errors)}")

    if success_results:
        # Save master summary
        summary_df = pd.DataFrame(success_results)
        summary_df.to_csv(f"{output_dir}/detailed_summary.csv", index=False)

        avg_stenosis = np.mean([r['max_stenosis'] for r in success_results])
        severe_count = len([r for r in success_results if r['max_stenosis'] > 70])

        print(f"\n📊 DETAILED ANALYSIS SUMMARY:")
        print(f"Average max stenosis: {avg_stenosis:.1f}%")
        print(f"Severe stenoses (>70%): {severe_count}")
        print(f"\n📁 Individual detailed outputs saved to: {output_dir}/")
        print(f"Each image has its own folder with:")
        print(f"   • *_vesselness.png - Frangi vesselness map")
        print(f"   • *_mask.png - Binary vessel mask")
        print(f"   • *_skeleton.png - Skeleton visualization")
        print(f"   • *_overlay.png - Stenosis overlay with markers")
        print(f"   • *_profile.png - Diameter profile plot")
        print(f"   • *_analysis.csv - Complete measurement data")
        print(f"   • *_detections.csv - Stenosis detection summary")

if __name__ == "__main__":
    main()