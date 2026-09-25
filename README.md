# DeepLearning HW03

CSCI E-89 Deep Learning, Assignment 03 — a PyTorch image classifier for Fashion
MNIST (the "Building an Image Classifier with PyTorch" section of Chapter 10,
*Hands-On Machine Learning with Scikit-Learn and PyTorch*, Aurelien Geron, 2025),
extended with a required training-accuracy plot, built with Claude Code.

## Problem 1 (25%) — incremental build, scripts captured, notebook calls them

- `e89_Stokes_Jazmyn_HW03_Prob1.ipynb` — cells run top to bottom, sharing one
  namespace. Each section corresponds to one incremental prompt given to Claude
  Code (see the notebook's "Prompt log" table); each section's code cell is a
  `%run scripts/0N_....py` that runs the actual generated script below, so the
  notebook and the captured scripts are the same code, not a separate copy of it.
- `scripts/` — the individual scripts as they were generated, one per prompt, in
  the order requested:
  - `01_setup.py` — imports, device selection, seed, matplotlib defaults, sanity check
  - `02_load_data.py` — Fashion MNIST download/transform, re-seed, 55k/5k split
  - `03_dataloaders.py` — DataLoaders for train/valid/test, sample inspection
  - `04_model.py` — `ImageClassifier` MLP (784/300/100/10) + loss
  - `05_train_eval_functions.py` — `evaluate_tm()` / `train2()` definitions only
  - `06_run_training.py` — the actual 20-epoch SGD(lr=0.1) training run
  - `07_plot_accuracy.py` — required training/validation accuracy plot + loss plot
  - `08_predict.py` — predictions on 3 validation images, softmax/top-4, display
  - `09_evaluate.py` — final held-out test-set evaluation

## Problem 2 (5%) — the same process wrapped into one notebook

- `problem2/e89_Stokes_Jazmyn_HW03_Prob2.ipynb` — all nine scripts merged into one
  self-contained notebook that reads start to finish: imports consolidated into a
  single Section 0 instead of scattered per section, and one numbered, markdown-
  explained section per step. No external script files — everything is inline.

## Other files

- `DIALOG_SUMMARY.md` — summary of the Claude Code dialog that produced Problem 1
  (the incremental build) and Problem 2 (wrapping it into one notebook).
- `requirements.txt` — Python dependencies for running either notebook.

## Running it

```bash
pip install -r requirements.txt

# Problem 1 (calls scripts/*.py via %run — must be run from the repo root)
jupyter nbconvert --to notebook --execute --inplace e89_Stokes_Jazmyn_HW03_Prob1.ipynb
jupyter nbconvert e89_Stokes_Jazmyn_HW03_Prob1.ipynb --to html

# Problem 2 (fully self-contained)
jupyter nbconvert --to notebook --execute --inplace problem2/e89_Stokes_Jazmyn_HW03_Prob2.ipynb
jupyter nbconvert problem2/e89_Stokes_Jazmyn_HW03_Prob2.ipynb --to html
```
