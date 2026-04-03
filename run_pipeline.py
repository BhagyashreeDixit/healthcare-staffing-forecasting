import subprocess, sys

steps = [
    ("Generating synthetic healthcare data...",
     ["python", "data/generate_data.py"]),
    ("Engineering time-series features...",
     ["python", "models/feature_engineering.py"]),
    ("Training Prophet model and generating forecasts...",
     ["python", "models/train_prophet.py"]),
]

print("\n" + "="*60)
print("  Healthcare Staffing Demand Forecasting Pipeline")
print("="*60)

for msg, cmd in steps:
    print(f"\n▶  {msg}\n" + "-"*50)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n❌ Failed: {' '.join(cmd)}")
        sys.exit(1)

print("\n" + "="*60)
print("  ✅ Pipeline complete!")
print("  📊 Open dashboard/index.html in your browser")
print("  📁 Check outputs/ folder for CSVs and plots")
print("="*60 + "\n")