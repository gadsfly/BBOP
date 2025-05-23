import os
import csv
import argparse

def cleanup_session(session_dir, keep_combination):
    """
    Deletes all output files in session_dir except the one matching keep_combination.
    
    Parameters:
        session_dir (str): The session directory.
        keep_combination (str): The combination identifier to keep 
                                (e.g., "wnd700_stp700_max25_diff4.0_pnrauto").
    """
    log_file = os.path.join(session_dir, "param_search_log.csv")
    if not os.path.exists(log_file):
        print("Log file not found!")
        return

    with open(log_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            combination = row["combination"]
            output_file = row["result_file"]
            if combination != keep_combination:
                if os.path.exists(output_file):
                    os.remove(output_file)
                    print(f"Removed: {output_file}")
                else:
                    print(f"Output file not found for combination {combination}.")

    print("Cleanup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup session outputs by keeping one combination.")
    parser.add_argument("--session_dir", required=True, help="Path to the session directory.")
    parser.add_argument("--keep_combination", required=True, help="The combination identifier to keep (e.g., 'wnd1500_stp700_max15_diff3.5_pnrauto').")
    args = parser.parse_args()
    cleanup_session(args.session_dir, args.keep_combination)
