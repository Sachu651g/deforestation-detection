import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n🚀 Running {script_name}...\n")
    subprocess.run([sys.executable, script_name], check=True)

if __name__ == "__main__":
    print("\n==============================")
    print(" DEFoRESTATION ANALYSIS SUITE ")
    print("==============================")

    print("\n1️⃣ Generating Confusion Matrix (Test Dataset)")
    run_script("confusion_matrix.py")

    print("\n2️⃣ Launching Smart Dashboard GUI")
    run_script("smart_dashboard_gui.py")

    print("\n✅ All modules executed successfully")
