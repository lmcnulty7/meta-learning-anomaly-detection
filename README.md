# Meta-Learning for Automated Anomaly Detection Model Selection

Which anomaly detector should you use on a new time series? Instead of trying them all, this project trains a
**meta-learner** that predicts, from 9 statistical meta-features of the series, which of five unsupervised
detectors (**Isolation Forest, PCA reconstruction error, One-Class SVM, K-means, and an autoencoder**) will score
the best F1. It was my final project at Loyola Marymount University; the write-up is in
[`docs/paper.pdf`](docs/paper.pdf).

**At a glance**

- **Data:** 58 labeled time series from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB),
  included as a git submodule.
- **Base detectors:** 5 unsupervised models. Each flags the top 1% of its anomaly scores, scored by point-level F1.
- **Meta-learner:** Random Forest on 9 meta-features, evaluated with **leave-one-dataset-out cross-validation**
  (train on 57 series, predict the best detector for the 58th).
- **Result:** 44.8% accuracy in the paper (one random seed), 39.4% averaged over 20 seeds, against 32.8% for always
  picking the most common winner (PCA) and 20% for a random guess. The [Results](#results) section explains
  what these numbers do and do not show.

## How it works

1. **Clean** (`scripts/clean_nab.py`): parse timestamps, standardize the value column, drop invalid and duplicate
   rows, fill small gaps. Writes `data/nab_cleaned/`.
2. **Label:** points inside NAB's anomaly windows (`data/NAB/labels/combined_windows.json`) get label 1.
3. **Meta-features** (per series): mean, standard deviation, skewness, kurtosis, lag-1 autocorrelation,
   augmented Dickey-Fuller statistic and p-value, anomaly ratio, and length.
4. **Base detectors:** each detector scores every point; the top 1% of scores become predicted anomalies; F1 is
   computed against the labels. This gives a 58 x 5 performance matrix (`performance_matrix.csv`).
   - Isolation Forest (200 trees), One-Class SVM (RBF, nu = 0.01) and K-means (k = 3, distance to centroid) on
     the standardized values.
   - PCA reconstruction error on sliding windows of 20 points (1 component).
   - Autoencoder (PyTorch, windows of 20 points, 20-32-8-32-20, 40 epochs) in `Autoencoder.ipynb`, run on a
     Google Colab GPU. Its scores are committed as `ae_performance.csv`.
5. **Meta-target:** the detector with the highest F1 on each series.
6. **Meta-learner:** `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)`,
   evaluated with leave-one-out over the 58 series, plus a confusion matrix and classification report.

## Repository layout

```
meta_learner.py              meta-features, 4 scikit-learn detectors, performance matrix, meta-learner + LOOCV
scripts/clean_nab.py         cleans the raw NAB CSVs into data/nab_cleaned/
scripts/robustness_check.py  seed sweep, baselines and anomaly_ratio ablation (numbers in Results)
Autoencoder.ipynb            PyTorch autoencoder detector (Google Colab), writes ae_performance.csv
ae_performance.csv           autoencoder precision / recall / F1 per series
performance_matrix.csv       F1 of all 5 detectors on all 58 series (the matrix behind the paper)
docs/paper.pdf               project paper
data/NAB/                    NAB git submodule (numenta/NAB, pinned at commit ea702d7)
```

## Setup

Requires Python 3.12 and git.

```bash
git clone --recurse-submodules https://github.com/lmcnulty7/meta-learning-anomaly-detection.git
cd meta-learning-anomaly-detection
# Cloned without --recurse-submodules? Run: git submodule update --init

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The NAB submodule is about a 90 MB download (about 300 MB on disk). `requirements.txt` pins the exact versions used to verify the results below
(scikit-learn 1.8.0, pandas 2.3.3, numpy 1.26.4, scipy 1.17.1, statsmodels 0.14.1). PyTorch is only needed for
the optional notebook.

## Run

From the repository root:

```bash
# 1. Clean the 58 NAB series into data/nab_cleaned/
#    (prints one harmless pandas deprecation warning per file)
python scripts/clean_nab.py --input-dir data/NAB/data --output-dir data/nab_cleaned

# 2. Base detectors + meta-learner (about a minute on a laptop)
python meta_learner.py

# 3. Optional: the seed sweep, baselines and ablation behind the Results section (a few minutes)
python scripts/robustness_check.py
```

The autoencoder step is optional because `ae_performance.csv` is committed. To regenerate it, copy
`data/nab_cleaned/` and `data/NAB/labels/` to Google Drive, set `ROOT` in the marked block at the top of
`Autoencoder.ipynb`, run it on a GPU runtime, and copy the resulting `ae_performance.csv` back to the repository
root. The autoencoder is not seeded, so a new run gives slightly different scores.

## Results

### Base detectors

F1 with a fixed top-1% threshold, from `performance_matrix.csv` (58 series):

| Detector                  | Mean F1 | Median F1 | Best detector on |
|---------------------------|--------:|----------:|-----------------:|
| PCA reconstruction error  |   0.085 |     0.080 |       19 series  |
| Autoencoder               |   0.083 |     0.083 |       12         |
| One-Class SVM             |   0.068 |     0.049 |        8         |
| Isolation Forest          |   0.060 |     0.052 |       17 *       |
| K-means (k = 3)           |   0.052 |     0.044 |        2         |
| Best detector per series  |   0.122 |           |                  |

\* 6 of the 17 are series where every detector scores 0 (see below).

Why the F1 scores are low: NAB's anomaly windows cover about 10% of each labeled series, but each detector flags
only 1% of points, so even perfect precision gives a recall of about 0.1 and an F1 of about 0.18. The scores say
which detector ranks the labeled region highest, not how well it would detect anomalies with a tuned threshold.
(The paper lists a mean F1 of 0.084 for the autoencoder, from the run whose output is shown in
`Autoencoder.ipynb`; the committed `ae_performance.csv` comes from another training run, mean 0.083.)

### Meta-learner

Leave-one-dataset-out accuracy over the 58 series, using the committed `performance_matrix.csv` and the pinned
environment above:

| Setting                                                         | Accuracy                       |
|-----------------------------------------------------------------|--------------------------------|
| Random Forest, seed 42 (**the paper's number**)                 | **44.8%** (26 / 58)            |
| Random Forest, mean over 20 seeds (0 to 19)                     | **39.4%** (range 36.2 to 44.8) |
| Same, without the `anomaly_ratio` feature (20-seed mean)        | 38.8% (seed 42: 39.7%)         |
| Majority-class baseline (always predict PCA)                    | 32.8% (19 / 58)                |
| Uniform random guess over 5 detectors                           | 20.0%                          |

With seed 42 the confusion matrix and classification report match the paper's Tables II and III exactly.

How to read these numbers:

- **The headline is one seed.** 44.8% is the top of the 20-seed range. The 20-seed mean, 39.4%, is about
  6.6 points above the majority-class baseline, a fairer comparison than the 20% random baseline the paper used.
- **`anomaly_ratio` uses the ground-truth labels** (the fraction of points inside labeled anomaly windows), so it
  would not be available for a new, unlabeled series. Without it the 20-seed mean is 38.8%, so the result does
  not rest on this feature, but it should be dropped in a real deployment.
- **6 series have F1 = 0 for every detector:** the 5 `artificialNoAnomaly` series and
  `realAWSCloudwatch/ec2_cpu_utilization_c6585a`, which are exactly the 6 series with no labeled anomalies, so
  every flagged point is a false positive. Their "best detector" is Isolation Forest only because ties go to the
  first column, and `anomaly_ratio = 0` identifies them exactly. Another 12 series have a tie for the top F1
  (PCA = autoencoder on 8, Isolation Forest = K-means on 4), so 18 of the 58 targets are decided by tie-breaking.
- **Re-running end to end gives 43.1%, not 44.8%.** `python meta_learner.py` recomputes the four scikit-learn
  detectors. In the pinned environment they reproduce the committed matrix except for small K-means differences
  on 6 series (up to 0.037 F1, none of which changes a winner) and one exact tie: on
  `realAdExchange/exchange-4_cpm_results` PCA and the autoencoder both score F1 = 0.1878, and a last-bit
  floating-point difference sends that series to the autoencoder in the committed matrix and to PCA in the rerun.
  That single label moves seed 42 to 43.1% (25 / 58) and the 20-seed mean to 44.6% (range 39.7 to 50.0), with a
  34.5% majority baseline.
- **Row order matters too.** The Random Forest results also depend on the order of the 58 rows, which comes from
  the filesystem's file listing. All numbers here were verified on macOS; with the rows sorted alphabetically,
  seed 42 gives 36.2% (20-seed mean 38.8%). On another operating system, expect a value within these ranges.

**Takeaway:** simple meta-features carry some signal about which detector will win (on average 5 to 10 points
above always picking the most common winner), but with 58 tasks and noisy, tie-heavy targets, the exact accuracy
moves by several points with the random seed, the row order, or a single tie-break.

## What I would do next

- Replace `anomaly_ratio` with label-free meta-features (seasonality and trend strength, spectral entropy,
  change-point counts) and add more series from other benchmarks.
- Score detectors with threshold-free or window-aware metrics (PR-AUC, the NAB score) to reduce ties, and treat
  remaining ties as multiple correct answers instead of breaking them by column order.
- Report accuracy as a mean over repeated seeds and row orders rather than a single run.

## License and credits

- Code: MIT License, (c) 2026 Lucien McNulty (see [`LICENSE`](LICENSE)).
- The paper, [`docs/paper.pdf`](docs/paper.pdf), is (c) Lucien McNulty.
- NAB data: MIT License, (c) Numenta. Included as a submodule pinned at commit `ea702d7`, the commit where NAB
  moved to the MIT license (earlier NAB commits are AGPL-3.0). Benchmark reference: A. Lavin and S. Ahmad,
  "Evaluating Real-time Anomaly Detection Algorithms: the Numenta Anomaly Benchmark," IEEE ICMLA 2015.
