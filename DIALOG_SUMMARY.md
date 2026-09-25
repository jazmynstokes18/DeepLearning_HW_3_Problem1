# Dialog summary — Claude Code session

**Assignment:** CSCI E-89 Deep Learning, Assignment 03, Problem 1 — use Claude Code
to incrementally build a PyTorch image classifier for Fashion MNIST, adding a
training-accuracy plot, and capture the generated scripts in a GitHub repository.

**Approach:** each step below was requested as a separate prompt in a single Claude
Code session. Claude generated one script per prompt, which was then assembled, in
order, into `e89_Stokes_Jazmyn_HW03_Prob1.ipynb`, with a markdown cell ahead of each
code cell explaining what it does and why. Nothing was generated in one large batch
— each step built on the variables/functions defined by the previous one, in the
same running session/namespace.

## Prompt-by-prompt log

1. **Setup** — "imports (numpy, torch, nn, F, torchmetrics, matplotlib), pick the
   best device (Mac), seed 42, set some matplotlib defaults, and print the torch
   version and device." → `scripts/01_setup.py`. Claude also added a short sanity
   check (tensor-to-device, seed-reproducibility, torchmetrics import) to confirm
   the setup was solid before building on it.
2. **Load & split data** — "load Fashion MNIST from torchvision into a datasets/
   folder, convert to float tensors scaled to [0,1] with transforms.v2. Re-seed to
   42, then split the 60k training images into 55k train / 5k validation. Print the
   split sizes and the class names." → `scripts/02_load_data.py`.
3. **DataLoaders** — "DataLoaders for all three splits, batch_size=32, shuffle the
   training one only. Then print one sample's shape, dtype and label." →
   `scripts/03_dataloaders.py`.
4. **Model + loss** — "an nn.Module called ImageClassifier: Flatten, Linear, ReLU,
   Linear, ReLU, Linear, no activation at the end since CrossEntropyLoss wants
   logits. Build it with 784/300/100/10, move it to device, make the loss, and
   print the model plus the parameter count." → `scripts/04_model.py`.
5. **Training/eval functions (definitions only)** — "just the functions, don't run
   anything yet. An evaluate_tm() that runs a loader under no_grad and returns the
   metric, and a train2() that trains and returns a history dict with
   train_losses, train_metrics and valid_metrics. ... Print the metrics each
   epoch too." → `scripts/05_train_eval_functions.py`.
6. **Run training** — "train for 20 epochs with SGD lr=0.1 and a torchmetrics
   multiclass accuracy on the same device. Keep the history." →
   `scripts/06_run_training.py`.
7. **Required plot** — "add code that will plot the training accuracy per epoch,
   with validation accuracy on the same plot. Put training at the epoch midpoint
   since it's an average over the epoch. Labels, grid, legend, y range 0.7-1.0.
   Print the final numbers, and add a second figure for the training loss." →
   `scripts/07_plot_accuracy.py`. This is the one addition Assignment 03 requires
   beyond the book's reference pipeline.
8. **Predict & visualize** — "predict on 3 images from the validation loader, print
   predicted vs actual class names. Then softmax the logits and show the
   probabilities rounded, plus the top 4 per image. Heads up: round() isn't
   implemented on mps, so move to cpu first. Then show the 3 images with their
   labels." → `scripts/08_predict.py`.
9. **Final test evaluation** — "one final run on the test set, print the test
   accuracy and the parameter count." → `scripts/09_evaluate.py`.

## Notable corrections/considerations made along the way

- Device selection checks CUDA, then Apple Silicon's `mps` backend, then CPU, so the
  notebook is portable off of the author's Mac as well.
- The seed is set once in setup and explicitly re-applied before the data split (as
  requested) so the 55k/5k partition is reproducible independent of what else has run.
- `nn.CrossEntropyLoss` expects raw logits, so the model's final layer intentionally
  has no activation function.
- `Tensor.round(decimals=...)` is not implemented on the `mps` backend; probabilities
  are moved to `cpu` first whenever `device == "mps"` before rounding, so the
  notebook doesn't crash on Apple Silicon.
- Training accuracy is plotted at the epoch midpoint (it's a running average over
  the epoch) while validation accuracy is plotted at the epoch's end (a single
  measurement taken after the epoch completes).

## Repository (Problem 1)

- Notebook: `e89_Stokes_Jazmyn_HW03_Prob1.ipynb`
- Generated scripts, one per prompt above: `scripts/`
- This summary: `DIALOG_SUMMARY.md`

---

## Part 2 of the dialog — pushing to GitHub, then Problem 2

**Push to GitHub.** Asked to push the dialog to a new, separate GitHub repository
(`DeepLearning_HW_3_Problem1`). Claude Code's GitHub integration can attach and push
to an existing repo but cannot create one, so the user created the empty repo on
github.com; Claude then hit a second snag — the session's GitHub credential didn't
yet have push access — resolved by the user installing the Claude GitHub App on the
repo. After that, Claude cloned it, added the Problem 1 notebook, the `scripts/`
files, this summary, a `README.md`, and a `requirements.txt`, and pushed the
initial commit.

**Problem 1 revision — call the scripts, don't duplicate them.** Asked to update
Problem 1 so its cells call the captured scripts rather than embedding the same code
twice. Every code cell in `e89_Stokes_Jazmyn_HW03_Prob1.ipynb` is now
`%run scripts/0N_....py` (run from the repo root), preceded by the same per-prompt
markdown explanation as before — so the notebook and `scripts/` are one copy of the
code, not two.

**Problem 2 — wrap it all into one notebook.** Asked to merge the nine scripts into
a single notebook that reads start to finish: one deduplicated Section 0 for every
import used anywhere (rather than each import appearing where it was first needed),
followed by nine numbered sections (device/seed/plot defaults; load & split data;
DataLoaders; model + loss; train/eval function definitions; the 20-epoch training
run; the required accuracy + loss plots; prediction & visualization; final test
evaluation), each with its own markdown explanation and no external script
dependency. Saved as `problem2/e89_Stokes_Jazmyn_HW03_Prob2.ipynb`.

**Execution.** Both notebooks were synced to the user's Mac (Week3 folder) along
with `verify_prob1.command` / `verify_prob2.command` / `run_both.command` —
double-clickable scripts that run `jupyter nbconvert --execute` (with
`allow_errors=True` so every failing cell is surfaced in one pass, not just the
first) against the user's own Python/conda environment, then export each notebook to
HTML. Claude could not execute them directly: computer-use is enabled but macOS
Accessibility/Screen Recording permission for the Claude desktop app was still
pending, so double-clicking these scripts and reporting back any errors was left to
the user (or Claude will pick it up once that permission is granted).
