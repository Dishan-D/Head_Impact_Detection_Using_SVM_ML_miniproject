# Head Impact Detection Using SVM Classifier

A comprehensive machine learning project for detecting American football head impacts using Support Vector Machine (SVM) classification. This project implements a complete ML pipeline including statistical analysis, feature selection, model training, and performance evaluation.

## 📋 Project Overview

This project addresses the critical need for automated head impact detection in American football using sensor data. The implementation uses advanced machine learning techniques to classify head impacts with high accuracy and provides comprehensive analysis of the underlying data patterns.

### Key Features
- **Statistical Feature Analysis**: Wilcoxon Rank-Sum test with Bonferroni correction
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Automated Feature Selection**: Sequential Feature Selection with cross-validation
- **Dual Optimization**: AUC and F-measure optimized classifiers
- **Robust Validation**: 10-fold cross-validation and independent testing
- **Comprehensive Visualization**: ROC curves, precision-recall plots, and correlation analysis

## 🎯 Problem Statement

American football players are at risk of head impacts that can lead to concussions and long-term brain injuries. Traditional impact detection methods rely on simple acceleration thresholds, which often miss subtle but significant impacts. This project develops a sophisticated machine learning approach using multiple sensor features to improve detection accuracy and reduce false negatives.

## 📊 Dataset Description

The project uses a comprehensive dataset (`head_impact_dataset.xlsx`) containing:

- **Training Data**: Collegiate-level football sensor data
- **Testing Data**: Independent youth-level football sensor data
- **Features**: Multiple sensor-derived features including:
  - Linear acceleration measurements
  - Power Spectral Density (PSD) features
  - Wavelet Transform (WT) features
  - Frequency-domain characteristics (10Hz, 20Hz, 30Hz bands)
- **Labels**: Binary classification (1 = head impact, 0 = no impact)

### Dataset Structure
```
dataset/
└── head_impact_dataset.xlsx
    ├── Feature Matrix Training (X_train)
    ├── Label Vector Training (y_train)
    ├── Feature Matrix Testing (X_test)
    └── Label Vector Testing (y_test)
```

## 🏗️ Implementation Architecture

### 1. Data Preprocessing Pipeline
```python
# Data Loading and Cleaning
load_dataset() → clean_dataset() → standardize_features()
```

**Key Steps:**
- **Loading**: Reads Excel sheets into structured pandas DataFrames
- **Cleaning**: Removes columns with all NaN values, imputes missing values with mean
- **Standardization**: Applies StandardScaler for SVM compatibility

### 2. Statistical Analysis Module
```python
# Statistical Feature Analysis
run_wilcoxon_test() → run_pca_analysis() → plot_correlation_matrix()
```

**Components:**
- **Wilcoxon Rank-Sum Test**: Identifies statistically significant features between impact/no-impact groups
- **Bonferroni Correction**: Controls for multiple comparison errors
- **PCA Analysis**: Analyzes variance explained by principal components
- **Correlation Analysis**: Visualizes feature relationships and multicollinearity

### 3. Feature Selection Engine
```python
# Automated Feature Selection
select_features(scoring="roc_auc") → select_features(scoring="f1")
```

**Methodology:**
- **Sequential Feature Selection (SFS)**: Forward selection with 10-fold cross-validation
- **Dual Optimization Strategy**:
  - **AUC-Optimized**: Selects 6 features maximizing ROC-AUC
  - **F-measure-Optimized**: Selects 7 features maximizing F1-score
- **Cross-Validation**: Ensures robust feature selection

### 4. Model Training & Validation
```python
# Model Development
train_and_validate() → test_on_independent_data()
```

**Implementation:**
- **SVM Classifier**: RBF kernel Support Vector Machine
- **Cross-Validation**: 10-fold CV for performance estimation
- **Metrics**: Sensitivity, Precision, Accuracy, F-measure, Specificity
- **Independent Testing**: Generalization assessment on youth data

### 5. Visualization & Analysis
```python
# Performance Visualization
plot_roc_pr_curves() → generate_output_plots()
```

**Outputs:**
- ROC curves and Precision-Recall curves
- Feature significance plots
- PCA variance explanation plots
- Correlation heatmaps

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.13+ (as specified in `pyproject.toml`)
- `uv` package manager (modern Python package manager)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Dishan-D/Head_Impact_Detection_Using_SVM_ML_miniproject.git
cd Head-Impact-Classifier-SVM
```

2. **Install uv package manager**
```bash
pip install uv
```

3. **Install project dependencies**
```bash
uv sync
```

4. **Activate virtual environment**
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

## 🏃‍♂️ Running the Project

### Python Script Execution
```bash
cd src
python main.py
```

**Expected Output:**
- Dataset loading confirmation with shapes
- Feature cleaning and standardization messages
- Wilcoxon test results with top 50 features
- PCA analysis showing variance explained
- Sequential Feature Selection progress for both AUC and F1 optimization
- Cross-validation results for both feature sets
- Independent test results on youth data
- Generated plots saved to `../output/` directory

## 📈 Output Files

The project generates comprehensive visualizations in the `output/` directory:

| File | Description |
|------|-------------|
| `wilcoxon-test.png` | Top 20 statistically significant features with adjusted p-values |
| `pca-analysis.png` | Cumulative variance explained by PCA components |
| `correlation-matrix.png` | Feature correlation heatmap showing multicollinearity |
| `roc-precision-recall.png` | ROC and Precision-Recall curves comparison |

## 🔧 Configuration Parameters

Key parameters you can modify in `main.py`:

```python
# Feature selection parameters
top_n = 50                    # Number of top features from Wilcoxon test
n_features_auc = 6            # Features for AUC optimization
n_features_f1 = 7             # Features for F1 optimization

# Cross-validation
cv_folds = 10                 # Number of CV folds

# SVM parameters
kernel = "rbf"                # SVM kernel type
```

## 📊 Performance Metrics

The project evaluates models using comprehensive metrics:

- **Sensitivity (Recall)**: True Positive Rate - ability to detect actual impacts
- **Precision**: Positive Predictive Value - accuracy of impact predictions
- **Accuracy**: Overall classification accuracy
- **F-measure**: Harmonic mean of precision and recall
- **Specificity**: True Negative Rate - ability to correctly identify non-impacts

## 🧪 Methodology Details

### Statistical Analysis
- **Wilcoxon Rank-Sum Test**: Non-parametric test for comparing distributions
- **Bonferroni Correction**: Controls family-wise error rate in multiple comparisons
- **Effect Size**: Measures practical significance of differences

### Feature Selection
- **Sequential Feature Selection**: Greedy forward selection algorithm
- **Cross-Validation**: Prevents overfitting during feature selection
- **Multiple Scoring Metrics**: Optimizes for different performance aspects

### Model Validation
- **Cross-Validation**: Robust performance estimation
- **Independent Testing**: Tests generalization on unseen data
- **Multiple Metrics**: Comprehensive performance assessment

## 🎓 Educational Value

This project demonstrates:
- **Complete ML Pipeline**: From data preprocessing to model evaluation
- **Statistical Analysis**: Proper hypothesis testing and multiple comparison correction
- **Feature Engineering**: Automated feature selection techniques
- **Model Validation**: Cross-validation and independent testing
- **Performance Visualization**: ROC curves, precision-recall plots
- **Code Organization**: Modular, well-documented implementation

## 🚨 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   uv sync  # Reinstall all dependencies
   ```

2. **Path Errors**
   - Ensure you're running from the correct directory (`src/` for Python script)
   - Verify `head_impact_dataset.xlsx` is in the `dataset/` folder

3. **Memory Issues**
   - The dataset is large; ensure sufficient RAM
   - Consider reducing `n_features` for limited hardware

4. **Performance Issues**
   - Sequential Feature Selection can take several minutes
   - RBF SVM kernel is computationally intensive
   - Consider using fewer CV folds for faster execution

### Performance Notes
- **Computational Requirements**: Moderate to high (depending on feature count)
- **Execution Time**: 5-15 minutes for complete pipeline
- **Memory Usage**: ~2-4GB RAM recommended

## 👥 Team Members

- **Dishan D** (PES1UG23CS196)
- **Samruddhi Patil** (PES1UG24CS828)

## 📖 Citation

This implementation follows the methodology in the following paper:

- Wu, L.C., Kuo, C., Loza, J. et al. Detection of American Football Head Impacts Using Biomechanical Features and Support Vector Machine Classification. Sci Rep 8, 855 (2018). [`https://doi.org/10.1038/s41598-017-17864-3`](https://doi.org/10.1038/s41598-017-17864-3)

Key alignment points:
- Low-frequency PSD and WT features (10–30 Hz) prioritized
- Forward feature selection with fewer than ten features
- Performance reported via sensitivity, precision, accuracy, F-measure, specificity

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | ≥3.10.7 | Data visualization |
| `numpy` | ≥2.3.3 | Numerical computations |
| `pandas` | ≥2.3.3 | Data manipulation |
| `scikit-learn` | ≥1.7.2 | Machine learning algorithms |
| `seaborn` | ≥0.13.2 | Statistical visualization |
| `statsmodels` | ≥0.14.5 | Statistical analysis |
| `openpyxl` | ≥3.1.5 | Excel file reading |



## 📄 Course Information

**Course**: Machine Learning (UE23CS352A)  
**Institution**: PES University  
**Project Type**: Mini Project



## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome. Please feel free to:
- Report issues or bugs
- Suggest enhancements
- Contribute to documentation
- Share performance improvements

## 📄 License

This project is created for educational purposes as part of the Machine Learning course at PES University.

---

**Note**: This project implements a complete machine learning pipeline for head impact detection in American football. The methodology combines statistical analysis, automated feature selection, and robust model validation to achieve high classification performance while maintaining scientific rigor.