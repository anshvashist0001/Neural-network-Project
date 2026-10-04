# Neural network explorer

A Streamlit app for inspecting six small models: linear regression, an AND-gate
perceptron, an XOR multilayer perceptron, an RNN, an LSTM and a convolutional digit
classifier. Training scripts save weights and chart data as JSON; the dashboard
loads those files and runs predictions from the saved weights.

## Run locally

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.
The checked-in model files let you run the app without retraining.

## Models and datasets

| Model | Task | Implementation | Evaluation |
|---|---|---|---|
| Linear regression | Size-to-price relationship | NumPy gradient descent | Training MSE on 25 synthetic points |
| Perceptron | AND gate | Perceptron update rule | Four training inputs |
| MLP | XOR gate | 2–4–1 network, sigmoid, backpropagation | Four training inputs |
| RNN | Next value of a noisy sine series | Eight hidden units, backpropagation through time | Last 20% of the series |
| LSTM | Next value of a synthetic price series | Twelve hidden units; all gates trained | Last 20% of the series |
| CNN | Handwritten digits | Eight learned 3×3 filters, ReLU, 2×2 pooling, dense softmax | Separate validation and test sets |

The digit data is scikit-learn's **8×8 digits dataset**, not MNIST. It contains
1,797 images with pixel values from 0 to 16. The CNN uses 1,149 training images,
288 validation images and 360 test images. The saved checkpoint is selected by
validation cross-entropy; the test set is evaluated afterward.

## Reproduce the results

```bash
python train_mse.py
python train_perceptron.py
python train_backprop.py
python train_rnn.py
python train_lstm.py
python train_cnn.py
```

Scripts write `out_*.json` beside their source files, regardless of the shell's
working directory. They use fixed seeds. Small numerical differences are possible
across NumPy and BLAS versions.

Results from the checked-in run:

| Experiment | Result | Interpretation |
|---|---|---|
| AND / XOR | 100% on four inputs each | These are training scores, not generalization estimates |
| CNN | 96.39% test accuracy | 347 of 360 held-out images |
| RNN | Holdout R² 0.966; MSE 0.00320 | Persistence baseline MSE 0.00522 |
| LSTM | Holdout R² 0.257; MSE 23.04 | Persistence baseline MSE 14.16; the baseline performs better |

Recurrent inputs use the preceding ten observations. Scaling is fitted on the
training portion only. Holdout scores use one-step predictions with observed
history, while the displayed forecasts recursively feed predictions back in.
Those two evaluations are different, and recursive errors can accumulate.

## Using the dashboard

- Select a model from the sidebar to inspect its curves and examples.
- Change the AND/XOR inputs to check the truth tables.
- Enter ten finite numbers for RNN/LSTM prediction. The entered values are used
  in the forward pass; invalid lengths are rejected.
- Select a digit sample to classify its pixels with the saved CNN weights.

## Project layout

```text
app.py                 Streamlit interface and chart helpers
models.py              Shared forward passes, gradients and Adam optimizer
train_*.py             Training entrypoints
train_sequences.py     Shared RNN/LSTM training and evaluation
out_*.json             Weights, curves, sample predictions and split metadata
test_models.py         Gradient and exported-inference checks
```

## Tests

```bash
python -m pip install pytest
python -m pytest -q
```

Tests compare analytical gradients with finite differences, check exported
predictions against saved results, and verify that sequence predictions change
when their inputs change.

## Limitations

This is an educational project. The regression and sequence data are synthetic;
the LSTM is not a financial forecasting system. Its current result is deliberately
reported alongside the stronger baseline. AND and XOR are tiny demonstrations.
The CNN interface classifies provided samples, rather than accepting arbitrary
image uploads. Training runs separately from the dashboard.

If a model file is missing, run its matching training script. After retraining,
refresh the app or clear Streamlit's cache to reload the files.
