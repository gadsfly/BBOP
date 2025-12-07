# BBOP: Bryan Building Open Field Pipeline

Dataset Project page [SocialCa3D](https://gadsfly.github.io/social-neural-imaging/)

Behavioral analysis pipeline for multi-animal social experiments combining [s-DANNCE](https://github.com/tqxli/sdannce) 3D pose estimation with calcium imaging from miniscope recordings. Calcium imaging processing is customized from [MiniAn](https://github.com/denisecailab/minian).
Calibration is done from [newsdannce](https://github.com/tdunnlab/newsdannce). Recording is handled by [campy](https://github.com/data-hound/campy).

![Demo Video](demo/side_by_side.gif)

## Installation

We recommend installing BBOP using the following steps:

```bash
# Clone the repository
git clone https://github.com/gadsfly/BBOP.git
cd BBOP

# Create conda environment with Python 3.9+
conda env create -f bbop250707.yml 
conda activate bbop

# Install BBOP
pip install -e .
```

## Pipeline Overview

BBOP processes multi-camera behavioral recordings through several stages:

```
Raw Videos → Generate Caliberation files → sync 6 cameras → COM Tracking → s-DANNCE Pose → Sync with Miniscope data (processed separately, also included in this repository) → Loading with synced data
```

**Run the complete pipeline tutorial:**
```bash
jupyter demo/bbop_preprocess_demo.ipynb
```

The tutorial walks through the complete pipeline from raw videos to analysis-ready data.

**Tutorials included:**
- `demo/bbop_preprocess_demo.ipynb` - Complete pipeline (calibration → pose → sync) [To-Do: sync is not yet added... should do sometime later...]
- `tutorial_miniscope_processing.ipynb` - Miniscope analysis with MiniAn [coming soon]
- `demo/3d_keypoint_social_analysis.ipynb` - Load merged data and analyze, this demo is with social animals (2 mice) and miniscope analyzed data


The Main demonstrates:
1. Folder scanning and status tracking
2. Calibration parameter generation
3. Camera synchronization  
4. COM tracking and visualization + Validation
5. s-DANNCE pose estimation + Validation
6. Miniscope processing (separate tutorial)
7. Miniscope-camera synchronization (produces frame_mapping.json)
8. Loading merged data for analysis (separate tutorial)



See [demo/bbop_preprocess_demo.ipynb](demo/bbop_preprocess_demo.ipynb) for complete workflow.

## Data Structure (after processing and clean up)

Each recording session follows this structure:
```
session_folder/
├── label3d_dannce.mat           # Camera calibration parameters
├── metadata/
│   ├── folder_log.parquet       # Processing status log
│   └── frame_mapping.json       # Miniscope-camera alignment (if miniscope present)
├── videos/
│   ├── Camera1/
│   │   ├── frametimes.mat       # Frame timestamps (or .npy)
│   │   ├── metadata.csv         # Recording metadata
│   │   └── 0.mp4
│   ├── Camera2-6/               # Similar structure for other cameras
├── COM/
│   └── predict00/
│       ├── COM_pred.mat
│       └── vis/
│           └── com_circle.png
├── SDANNCE/                     # s-DANNCE outputs
│   └── predict01/
│       └── save_data_AVG0.mat
└── miniscope/                   # If miniscope recording exists
    ├── 0.mp4                    # Raw miniscope videos (one or more)
    ├── timeStamps.csv           # Miniscope frame timestamps  
    └── miniscope.nc             # Processed miniscope data (MiniAn output)
```


### Processing Status Fields

The pipeline tracks these stages automatically:

| Field | Description |
|-------|-------------|
| `mir_generate_param` | step on camera parameter generation |
| `sync` | Multi-camera synchronization |
| `mini_6cam_map` | Miniscope-camera mapping, if exist means a session is recorded with miniscope regardless of quality |
| `dropf_handle` | Dropped frame handling |
| `com` | Center of mass tracking |
| `com_vis` | COM visualization |
| `social` | Social behavior or single running, when 1 means social, 0 means single |
| `dannce` | s-DANNCE 3D pose estimation |
| `dannce_vis` | s-DANNCE validation/visualization |

## Camera Calibration

We use chuachoboard calibration for intrinsics and checkerboard calibration for extrinsics. Calibration instruction is shown in recording protocol (find here. #TO-DO: some instruction of installing that lol)

Calibration protocol and tools can be found in the recording protocol documentation.

Note: `*label3d_dannce.mat` contains the calibration parameters for all cameras in DANNCE-compatible format.

view details in: **Protocol of Recording** [View PDF](https://drive.google.com/file/d/1_cB2Yxkpt41zbyjxM-VQth-bNA1zPu_v/view?usp=sharing)  

## Synchronization

For multi-camera recordings, frame synchronization is critical. BBOP expects light switrched for sync, follow the steps in teh recording protocol. [View PDF](https://drive.google.com/file/d/1_cB2Yxkpt41zbyjxM-VQth-bNA1zPu_v/view?usp=sharing)

Synchronization with miniscope recordings is handled in a similar way.

## Analysis Examples

After pipeline completion, data can be loaded for analysis:

```python
from utlis.sync_utlis.general_loader import merge_pred_with_miniscope

# Load merged data
data = merge_pred_with_miniscope(
    rec_path="/path/to/session",
    dannce_folder='SDANNCE/predict01'
)
```

See [demo/3d_keypoint_social_analysis.ipynb](demo/3d_keypoint_social_analysis.ipynb) for analysis workflows including:
- 3D trajectory analysis
- Social interaction detection
- Neural activity correlation

## Preprocessing Methods
**Protocol:** [View PDF](https://drive.google.com/file/d/1_cB2Yxkpt41zbyjxM-VQth-bNA1zPu_v/view?usp=sharing)  



## License

MIT

## Troubleshooting

For additional help, please open an issue on GitHub.


## Credits & Acknowledgments
# Acknowledgments

## Core Development

**BBOP** was developed by **Mir Qi** as part of a project funded and directed by **Dr. Timothy W. Dunn**.

## Key Contributors

### Prediction Pipeline (sDANNCE)
Developed by **Tianqing Li** ([s-DANNCE](https://github.com/tqxli/sdannce)). BBOP manages command-level execution.

### Demo Data
Provided through the surgical and recording work of **Dr. Renzhi Zhan** and **Mir Qi**.

### Calibration Pipeline
Adapted from **Chris Axon** [newsdannce](https://github.com/tdunnlab/newsdannce).

### Hardware and Recording Improvements
**Anshuman Sabath** revised the campy to enable recording [campy](https://github.com/data-hound/campy), and contributed to arena design improvements, debugging, and technical advice.

### Visualization and Validation Code
Some components adapted from **Sihan Lyu** (https://github.com/Siwudi) and **Sophie Shi** (https://github.com/Sooophy/dannce/tree/stroke_analysis/trace_protocol).

## Advisory Support

- **Repository Development:** Anshuman Sabath, Dr. Timothy W. Dunn
- **Repository Presentation, and data organizations:** Tianqing Li, Dr. Timothy W. Dunn, Anshuman Sabath

## Inspiration

- **Concept Inspiration:** Jim Roach's pipeline for STUDIO
- **sDANNCE Approaching Behavior Function:** Minji Jang
