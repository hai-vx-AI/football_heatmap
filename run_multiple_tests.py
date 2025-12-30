"""
Script to run multiple video tests sequentially
"""
import subprocess
import sys
from pathlib import Path

def run_video_test(video_path, output_name, config="config_cpu.yaml"):
    """Run a single video test"""
    python_exe = r"C:\Users\DungLe\miniconda3\python.exe"
    
    cmd = [
        python_exe,
        "main.py",
        video_path,
        "--config", config,
        "--output-name", output_name
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {output_name}")
    print(f"Video: {video_path}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ Successfully processed: {output_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to process: {output_name}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user")
        return False

def main():
    """Run multiple tests"""
    # Define videos to test (total: 3 videos x 750 frames = 2250 frames)
    tests = [
        ("input/SoccerNet/tracking-2023/test/SNMOT-127", "snmot127_test_cpu"),
        ("input/SoccerNet/tracking-2023/test/SNMOT-128", "snmot128_test_cpu"),
        ("input/SoccerNet/tracking-2023/test/SNMOT-129", "snmot129_test_cpu"),
    ]
    
    print(f"\n{'#'*80}")
    print(f"# Running {len(tests)} video tests (Total: {len(tests) * 750} frames)")
    print(f"{'#'*80}\n")
    
    results = []
    for video_path, output_name in tests:
        success = run_video_test(video_path, output_name)
        results.append((output_name, success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print(f"\nTotal: {successful}/{total} successful")
    print(f"Total frames processed: {successful * 750}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
