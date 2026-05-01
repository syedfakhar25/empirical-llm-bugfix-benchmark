This repository benchmarks open-source LLMs on real-world Python bugs (BugsInPy), measuring:

- Functional correctness (tests passing)
- Inference time
- Energy consumption (GPU via NVML)

1. Repo Clone
#Clone this repo
git clone https://github.com/syedfakhar25/empirical-llm-bugfix-benchmark
cd your_repo
# Clone BugsInPy dataset
git clone https://github.com/soarsmu/BugsInPy.git

2. Setup Python Environment

Make sure you have:
- Python ≥ 3.8
- Virtualenv support
- GPU (optional but recommended)
- Install all required libraries


Final Structure should be:
your-repo/
│
├── pipeline/
│   ├── run_experiment.py
│   ├── run_llm.py
│
├── results/
│   └── results.csv
│
└── BugsInPy/
    └── projects/



3. To run the experiments:
python run_experiment.py \
  --mode single \
  --project fastapi \
  --bug 12 \
  --model bigcode/starcoder2-7b \
  --bug-python python3.8

to run multiple:
python run_experiment.py \
  --mode multi_run \
  --project fastapi \
  --bug 12 \
  --model bigcode/starcoder2-7b \
  --runs 10 \
  --sleep 60 \
  --bug-python python3.8


4. Check results:
results/results.csv

  
