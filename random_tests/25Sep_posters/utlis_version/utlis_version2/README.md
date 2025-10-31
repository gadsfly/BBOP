# 3D Keypoint Analysis - Updated

## Files

### 1. `keypoint_3d_analysis.py` (64 KB, 1781 lines)
**Complete Python module with ALL analysis and visualization functions.**

Contains:
- Original functions from base module
- `find_approach_success()` - replaces `detect_approaches()` 
- `plot_incidents_3d_grid()` - 3D pose grid visualization
- `plot_ca_heatmap_and_distance()` - neural activity + distance
- `plot_incident_windows_newset()` - comprehensive per-event analysis (4 figures)
- `visualize_frames_allcams()` - multi-camera synchronized videos
- All necessary helper functions

### 2. `3d_keypoint_analysis_clean.ipynb` (24 KB)
**Clean tutorial notebook that ONLY imports and demonstrates.**

Structure:
- Setup & imports
- Data loading
- Distance analysis
- Motion analysis  
- Approach detection (using `find_approach_success`)
- 3D visualization
- Neural activity analysis
- Per-event detailed analysis
- Summary statistics

## Key Changes from Original

### Replaced Function
**OLD:** `detect_approaches()` - velocity-based approach detection
**NEW:** `find_approach_success()` - distance decrease + contact achievement

### New Visualization Functions
1. **`plot_incidents_3d_grid()`** - Grid of 3D skeletal poses
2. **`plot_ca_heatmap_and_distance()`** - Neural heatmap + distance trace
3. **`plot_incident_windows_newset()`** - 4 figure types per event
4. **`visualize_frames_allcams()`** - Multi-camera synchronized video

## Usage

### Basic Event Detection
```python
from keypoint_3d_analysis import find_approach_success

# Detect events
mask, events = find_approach_success(
    frames,
    contact_mm=50.0,
    dD_dt_thresh=0.0,
    min_len=10,
    min_drop_mm=10.0
)

print(f"Found {len(events)} approach→contact events")
```

### 3D Grid Visualization
```python
from keypoint_3d_analysis import plot_incidents_3d_grid

# Show first 12 events
event_indices = [e['contact_idx'] for e in events[:12]]
plot_incidents_3d_grid(
    merged, event_indices, COLOR, CONNECTIVITY,
    ncols=6, zoom_mode="local", dpi=300
)
```

### Per-Event Analysis
```python
from keypoint_3d_analysis import plot_incident_windows_newset

# Generate 4 figures per event
plot_incident_windows_newset(
    merged, frames, mask,
    pre_s=2.0, post_s=2.0,
    save=True, out_dir="events"
)
```

### Neural Activity Overview
```python
from keypoint_3d_analysis import plot_ca_heatmap_and_distance

plot_ca_heatmap_and_distance(
    merged,
    variance_drop_pct=5.0,
    cmap="RdBu_r",
    save=True
)
```

## What's in Each File

### keypoint_3d_analysis.py Structure
```
Original Functions (lines 1-710)
├─ Distance calculations
├─ Motion analysis
├─ detect_approaches (kept for compatibility)
├─ Proximity analysis
└─ Basic visualization

New Functions (lines 711-1781)
├─ find_approach_success
├─ Helper functions (_time_from_index_or_col, _segments_from_mask, etc.)
├─ plot_incidents_3d_grid
├─ visualize_frames_allcams
├─ plot_ca_heatmap_and_distance
└─ plot_incident_windows_newset
```

### Notebook Structure
```
1. Title & Overview
2. Setup & Imports (imports from keypoint_3d_analysis)
3. Configuration
4. Data Loading (load_flat_with_frame_map, merge_pred_with_miniscope)
5. Distance Analysis (compute_com_distance)
6. Motion Analysis (compute_motion_direction)
7. Approach Detection (find_approach_success)
8. 3D Visualization (plot_incidents_3d_grid)
9. Neural Activity (plot_ca_heatmap_and_distance)
10. Per-Event Analysis (plot_incident_windows_newset)
11. Summary Statistics
```

## Design Pattern

**Module (.py):** Contains ALL function implementations
**Notebook (.ipynb):** Tutorial that imports and demonstrates

This follows the original clean pattern where:
- Functions live in the module
- Notebook is educational/demonstrative
- No function definitions in notebook cells
- All code in notebook is usage/application

## Quick Start

1. Place `keypoint_3d_analysis.py` in your Python path
2. Open `3d_keypoint_analysis_clean.ipynb` in Jupyter
3. Update the config cell with your data paths
4. Run cells sequentially
5. Adjust parameters as needed

## Function Summary

| Function | Purpose | Output |
|----------|---------|--------|
| `find_approach_success` | Detect approach→contact events | mask + event list |
| `plot_incidents_3d_grid` | 3D pose grid snapshots | matplotlib figure |
| `plot_ca_heatmap_and_distance` | Neural activity + distance | 2-panel figure |
| `plot_incident_windows_newset` | Comprehensive event analysis | 4 figures per event |
| `visualize_frames_allcams` | Multi-camera video | MP4 video file |

## Notes

- All functions have detailed docstrings
- Notebook follows tutorial pattern (no function definitions)
- Module is self-contained (all helpers included)
- Backward compatible (original `detect_approaches` still available)
