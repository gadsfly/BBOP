import subprocess
from manual_config import sessions

for sess in sessions:
    session_dir = sess["session_dir"]
    animal = sess["animal"]
    wnd_size = sess["wnd_size"]
    stp_size = sess["stp_size"]
    max_wnd = sess["max_wnd"]
    diff_thres = sess["diff_thres"]
    pnr_thresh = sess["pnr_thresh"]

    cmd = [
        # "python", "selective_rerun.py",
        "conda", "run", "-n", "minian", "python", "selective_rerun.py",
        "--session_dir", session_dir,
        "--animal", animal,
        "--wnd_size", str(wnd_size),
        "--stp_size", str(stp_size),
        "--max_wnd", str(max_wnd),
        "--diff_thres", str(diff_thres),
        "--pnr_thresh", str(pnr_thresh)
    ]
    print("---------------------------------------------------")
    print(f"Running selective re-run for session: {session_dir}")
    print("Command:", " ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print("[STDOUT]", result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)
