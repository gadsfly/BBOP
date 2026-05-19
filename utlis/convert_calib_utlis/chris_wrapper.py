"""
Batch Calibration Script with Environment Wrapper
Original calibration system by: Chris Axon
Batch processing script by: Mir Qi
Environment wrapper by: Assistant
"""
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any


def run_calibration_in_env(
    test_path: str,
    intrinsics_dir: str,
    conda_env_name: str,
    chris_calib_path: str = "/data/big_rim/rsync_dcc_sum/chirs_calib/chirs_calib/aug_newsdannce/newsdannce",
    rows: int = 6,
    cols: int = 9,
    square_size_mm: float = 23.0,
    extrinsics_subdir: str = "calib_before/extrinsics",
    output_subdir: str = "calib_before_newintrinsics"
) -> Dict[str, Any]:
    """
    Run calibration in a specific conda environment.
    
    Args:
        test_path: Path to the test directory
        intrinsics_dir: Path to intrinsics directory
        conda_env_name: Name of the conda environment to use
        chris_calib_path: Path to Chris's calibration code
        rows: Number of checkerboard rows
        cols: Number of checkerboard columns
        square_size_mm: Size of checkerboard squares in mm
        extrinsics_subdir: Subdirectory for extrinsics (relative to test_path parent)
        output_subdir: Subdirectory for output (relative to test_path parent)
    
    Returns:
        Dictionary with status and results
    """
    # Create the Python script to execute
    script_content = f'''
import sys
import os

# Set the working directory and add to path
chris_calib_path = "{chris_calib_path}"
os.chdir(chris_calib_path)
sys.path.insert(0, chris_calib_path)

# NOW import
from src.calibration.do_calibrate_stateful import do_calibrate_stateful

test_path = "{test_path}"
base_path = os.path.dirname(test_path)
exc = os.path.join(base_path, "{extrinsics_subdir}")
outt = os.path.join(base_path, "{output_subdir}")
intrin = "{intrinsics_dir}"

results = do_calibrate_stateful(
    intrinsics_dir=intrin,
    extrinsics_dir=exc,
    output_dir=outt,
    override_intrinsics_dir=intrin,
    rows={rows},
    cols={cols},
    square_size_mm={square_size_mm}
)

print("\\n=== CALIBRATION COMPLETE ===")
print(results.report_summary)
print("=== END RESULTS ===")
'''
    
    # Write script to temporary file
    temp_script = "/tmp/temp_calibration_script.py"
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    # Build the command to run in the specific environment
    # Using conda run for direct environment execution
    cmd = [
        "conda", "run", "-n", conda_env_name,
        "python", temp_script
    ]
    
    print(f"Running calibration in environment: {conda_env_name}")
    print(f"Test path: {test_path}")
    
    try:
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("\n=== STDOUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("\n=== STDERR ===")
            print(result.stderr)
        
        # Clean up temporary script
        os.remove(temp_script)
        
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "test_path": test_path
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n=== ERROR ===")
        print(f"Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)
        
        return {
            "status": "failed",
            "stdout": e.stdout,
            "stderr": e.stderr,
            "return_code": e.returncode,
            "test_path": test_path
        }


def batch_calibrate(
    test_paths: list,
    intrinsics_dir: str,
    conda_env_name: str,
    **kwargs
) -> list:
    """
    Run calibration on multiple test paths.
    
    Args:
        test_paths: List of test directory paths
        intrinsics_dir: Path to intrinsics directory
        conda_env_name: Name of the conda environment to use
        **kwargs: Additional arguments to pass to run_calibration_in_env
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    for i, test_path in enumerate(test_paths, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(test_paths)}: {test_path}")
        print(f"{'='*60}")
        
        result = run_calibration_in_env(
            test_path=test_path,
            intrinsics_dir=intrinsics_dir,
            conda_env_name=conda_env_name,
            **kwargs
        )
        
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH CALIBRATION SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    return results