# ML Algorithms From Scratch

Implementing machine-learning algorithms with NumPy to understand the mathematics behind model training, rather than treating models as black boxes.

This repository contains hands-on implementations of **multivariate linear regression** and **binary logistic regression**. Each notebook walks through the model, derives the gradients, trains the parameters with gradient descent, visualizes training loss, evaluates performance, and compares the results with scikit-learn.

The goal is to learn the mechanics of the algorithms and build them step by step.

## What's in the repository?

| Notebook | Problem type | Dataset | Main concepts |
|---|---|---|---|
| `Linear_Regression_From_Scratch.ipynb` | Regression | California Housing | Mean squared error, vectorized gradients, gradient descent, feature standardization, MAE, R² |
| `Logistic_Regression_From_Scratch.ipynb` | Binary classification | Breast Cancer Wisconsin (Diagnostic) | Sigmoid, binary cross-entropy, gradient descent, classification metrics, confusion matrix |

## Repository structure

```text
ML-Algorithms-From-Scratch/
├── README.md
├── Linear_Regression_From_Scratch.ipynb
├── Logistic_Regression_From_Scratch.ipynb
└── .gitignore
```

The notebooks include the code, explanations, visualizations, evaluation, and scikit-learn comparisons.

---

## 1. Multivariate Linear Regression

**Notebook:** `Linear_Regression_From_Scratch.ipynb`

### Objective

Predict a continuous target using several input features. The model parameters are learned using gradient descent implemented with NumPy.

### Model

For a dataset with \(n\) observations and \(p\) features:

\[
\hat{y} = Xm + c
\]

where:

- \(X\) is the feature matrix with shape \((n, p)\).
- \(m\) is the coefficient vector with shape \((p,)\).
- \(c\) is the intercept.
- \(\hat{y}\) contains the predictions.

### Loss function

The implementation minimizes Mean Squared Error (MSE):

\[
J(m,c) = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Let the residual vector be \(r = y - \hat{y}\). The gradients are:

\[
\frac{\partial J}{\partial m} = -\frac{2}{n}X^T r
\]

\[
\frac{\partial J}{\partial c} = -\frac{2}{n}\sum_{i=1}^{n}r_i
\]

The coefficients and intercept are updated iteratively:

\[
m \leftarrow m-\alpha\frac{\partial J}{\partial m},
\qquad
c \leftarrow c-\alpha\frac{\partial J}{\partial c}
\]

Here, \(\alpha\) is the learning rate. The gradient calculations are vectorized with NumPy.

### Dataset and workflow

- **Dataset:** California Housing, loaded using `sklearn.datasets.fetch_california_housing`.
- **Inputs:** Eight numeric housing and geographic features.
- **Target:** Median house value, expressed in units of $100,000 in the source dataset.
- **Workflow:** Split the data into training and test sets, standardize features using statistics calculated from the training set, train the scratch model, inspect the loss curve, and evaluate predictions on held-out data.
- **Reference comparison:** Compare the learned parameters and results with scikit-learn's `LinearRegression`.

### Held-out test results

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.5332 |
| R² score | 0.5757 |

An MAE of `0.5332` corresponds to approximately $53,320 in the dataset's target-value units. An R² of `0.5757` means the model explains about 57.6% of the variance in the held-out targets.

These results provide a baseline for this simple linear model; they are not intended for real-world property valuation.

---

## 2. Logistic Regression

**Notebook:** `Logistic_Regression_From_Scratch.ipynb`

### Objective

Predict the probability of a binary class. The model calculates a linear score, passes it through the sigmoid function, and converts the probability to a class using a threshold of 0.5.

### Model

First, calculate the linear score:

\[
z = Xm+c
\]

Then apply the sigmoid function:

\[
\sigma(z) = \frac{1}{1+e^{-z}}
\]

The output is interpreted as a probability:

\[
\hat{p} = \sigma(z)
\]

The predicted class is:

\[
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \geq 0.5\\
0 & \text{otherwise}
\end{cases}
\]

### Loss function and gradients

The model minimizes binary cross-entropy (log loss):

\[
J(m,c) = -\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)
\right]
\]

The gradients for the logistic regression model are:

\[
\frac{\partial J}{\partial m} = \frac{1}{n}X^T(\hat{p}-y)
\]

\[
\frac{\partial J}{\partial c} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i-y_i)
\]

Gradient descent updates the coefficient vector and intercept to reduce the loss. The implementation clips probabilities when calculating log loss to avoid taking the logarithm of zero.

### Dataset and workflow

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), loaded using `sklearn.datasets.load_breast_cancer`.
- **Inputs:** 30 numeric features computed from digitized images of breast-mass samples.
- **Target encoding:** `0 = malignant`, `1 = benign`.
- **Workflow:** Use a stratified 80/20 train/test split with `random_state=42`. Calculate feature means and standard deviations from the training set only, then apply those same statistics to the test set.
- **Evaluation:** Accuracy, precision, recall, specificity, F1-score, and a confusion matrix.
- **Reference comparison:** Compare scratch-model predictions with scikit-learn's `LogisticRegression`.

### Held-out test results

| Metric | Result |
|---|---:|
| Accuracy | 0.97 |
| Precision | 0.99 |
| Recall | 0.97 |
| Specificity | 0.98 |
| F1-score | 0.98 |
| Prediction agreement with scikit-learn | 99.12% |

The test confusion matrix is:

| Actual \ Predicted | Malignant (0) | Benign (1) |
|---|---:|---:|
| Malignant (0) | 41 | 1 |
| Benign (1) | 2 | 70 |

The scratch model correctly classified 111 of the 114 test samples.

**Metric interpretation:** In the notebook, class `1` (benign) is treated as the positive class for the reported precision, recall, and F1-score. The confusion matrix makes the class outcomes explicit: one malignant case was predicted as benign, and two benign cases were predicted as malignant. For a medical classification problem, it is important to examine per-class performance and the cost of different errors rather than relying on accuracy alone.

The scikit-learn model uses regularization by default, while the scratch implementation does not. Their predictions can be compared, but their learned coefficients are not expected to match exactly.

**Important:** This notebook is an educational exercise, not a medical tool. It must not be used to diagnose or make clinical decisions.

---

## Setup and requirements

You will need Python and the following libraries:

- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

Python 3.10 or later is a reasonable starting point.

### 1. Clone the repository

Replace `<YOUR-USERNAME>` with your GitHub username.

```bash
git clone https://github.com/<YOUR-USERNAME>/ML-Algorithms-From-Scratch.git
cd ML-Algorithms-From-Scratch
```

### 2. Create a virtual environment (recommended)

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open either notebook and run the cells from top to bottom.

The California Housing dataset may be downloaded the first time it is loaded, so an internet connection may be needed for the initial run.

---

## What these implementations cover

- Translating mathematical definitions into NumPy code.
- Implementing vectorized gradient calculations.
- Training parameters with gradient descent.
- Tracking and plotting the loss over training.
- Standardizing features without using test-set statistics.
- Evaluating regression and classification models with appropriate metrics.
- Comparing scratch implementations with scikit-learn reference models.
- Interpreting results on held-out data and recognizing the limitations of simple models.

## Limitations and possible next steps

These notebooks are learning exercises, not production-ready machine-learning libraries. Some natural extensions for the repository are:

- Add unit tests for predictions, loss calculations, and gradients.
- Package the implementations into reusable Python modules.
- Compare models on the same splits and document hyperparameters consistently.
- Add cross-validation and experiments with different learning rates.
- Explore regularization for logistic regression.
- Implement additional algorithms, such as K-Nearest Neighbors, polynomial regression, decision trees, and K-Means.

## About

This repository is part of my effort to build a stronger understanding of machine learning by implementing algorithms, examining their mathematics, and validating the results against established libraries.

