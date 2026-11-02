"""Run the whole project end-to-end (Task 4 takes a few minutes because of grid search)."""
import subprocess, sys
for script in ["task1_data_preparation.py", "task2_eda.py", "task3_segmentation.py",
               "task4_churn_model.py", "task5_model_evaluation.py", "task6_business_recommendations.py"]:
    print(f"\n{'='*70}\n>>> {script}\n{'='*70}", flush=True)
    subprocess.run([sys.executable, script], check=True)
