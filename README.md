# DeepLearning HW03 — Problem 1

CSCI E-89 Deep Learning, Assignment 03, Problem 1: use Claude Code to incrementally
build a PyTorch image classifier for Fashion MNIST (the "Building an Image
Classifier with PyTorch" section of Chapter 10, *Hands-On Machine Learning with
Scikit-Learn and PyTorch*, Aurelien Geron, 2025), adding a required training-accuracy
plot.

## Contents

- `e89_Stokes_Jazmyn_HW03_Prob1.ipynb` — the assembled notebook. Cells run top to
  bottom, sharing one namespace; each section corresponds to one incremental prompt
  given to Claude Code (see the notebook's "Prompt log" table).
- `scripts/` — the individual scripts as they were generated, one per prompt, in the
  order requested:
  - `01_setup.py` — imports, device selection, seed, matplotlib defaults, sanity check
  - `02_load_data.py` — Fashion MNIST download/transform, re-seed, 55k/5k split
  - `03_dataloaders.py` — DataLoaders for train/valid/test, sample inspection
  - `04_model.py` — `ImageClassifier` MLP (784/300/100/10) + loss
  - `05_train_eval_functions.py` — `evaluate_tm()` / `train2()` definitions only
  - `06_run_training.py` — the actual 20-epoch SGD(lr=0.1) training run
  - `07_plot_accuracy.py` — required training/validation accuracy plot + loss plot
  - `08_predict.py` — predictions on 3 validation images, softmax/top-4, display
  - `09_evaluate.py` — final held-out test-set evaluation
- `DIALOG_SUMMARY.md` — summary of the Claude Code session that produced this notebook.

## Running it

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace e89_Stokes_Jazmyn_HW03_Prob1.ipynb
jupyter nbconvert e89_Stokes_Jazmyn_HW03_Prob1.ipynb --to html
```
