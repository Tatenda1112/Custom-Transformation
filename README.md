# 🧠 Custom Data Transformation Library

This project provides a suite of **custom Scikit-learn-compatible data transformers** for preprocessing tasks in machine learning pipelines. These transformations enhance control and flexibility over numeric and categorical data preparation.

---

## 📦 Included Transformers

| Category              | Transformer Name             | Description |
|-----------------------|------------------------------|-------------|
| **Scaling**           | `StandardScaler`             | Standardizes numeric features (mean=0, std=1) |
|                       | `MinMaxScaler`               | Scales numeric data to a specified range |
| **Outlier Handling**  | `winsorizer`                 | Caps outliers based on IQR and coefficient |
| **Missing Values**    | `MeanMedianImputer`          | Imputes missing values using mean or median |
|                       | `categoricalImputer`         | Imputes missing categorical values using mode |
| **Encoding**          | `count_frequency_encoder`    | Encodes categories using their normalized frequency |
|                       | `OneHotEncoder`              | Performs one-hot encoding |
|                       | `OrdinalEncoder`             | Applies ordinal encoding using auto or user-defined mappings |
| **Feature Reduction** | `DropConstantFeatures`       | Removes features with a constant value ratio above a threshold |
|                       | `DropDuplicateFeatures`      | Removes duplicate columns |
|                       | `DropCorrelatedFeatures`     | Drops highly correlated features |
| **High Cardinality**  | `HighCardinalityImputer`     | Groups infrequent categories under a common label |
| **Power Transform**   | `BoxCoxTransformer`          | Applies Box-Cox transformation to numeric features |
|                       | `YeoJohnsonTransformer`      | Applies Yeo-Johnson transformation to both positive and negative values |

---

## 🚀 Getting Started

1. **Clone this repository**:
   ```bash
   git clone https://github.com/Tatenda1112/Custom-Transformation.git
   cd Custom-Transformation
