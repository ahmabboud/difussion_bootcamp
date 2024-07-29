# ClavaDDPM Refactoring Notes
ToDo:
[x] Clean up python venv for conda
[] Test python venv on vector, especially with poetry
[x] Add notes and visualization to notebook
[] Measure runtime for training and sampling
[] Hyperparameters tuning for custom dataset


## Setup python venv
```bash
conda create -n clava python=3.9 -y
conda activate clava
pip install pip==24.0.0
pip install -r requirements.txt
```

## Run the demo
Please refer to the `ClavaDDPM.ipynb` notebook for the demo.