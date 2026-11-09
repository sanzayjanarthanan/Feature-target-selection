# Task 31 – Features and Target Selection (Iris Dataset)

**Veda Technology · AI & ML Track · Level 1 · Day 31**

## Objective
Select suitable input features and a target variable from a simple machine learning dataset and explain the reasoning behind the selection, to understand the basic structure of supervised machine learning problems.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Seaborn (only to load the dataset)
- Jupyter Notebook

## Dataset
Iris dataset (`sns.load_dataset("iris")`) – 150 flowers, 5 columns, no missing values.

| Column | Type | Description |
|--------|------|-------------|
| sepal_length | float | Sepal length in cm |
| sepal_width | float | Sepal width in cm |
| petal_length | float | Petal length in cm |
| petal_width | float | Petal width in cm |
| species | category/text | Flower species (setosa, versicolor, virginica) |

Each species has 50 samples, so the classes are balanced.

## Files in this Repository
| File | Description |
|------|-------------|
| `features_target_selection.ipynb` | Notebook with feature and target selection |
| `iris.csv` | Dataset used |
| `README.md` | Selection and reasoning |

## Problem Type
**Supervised classification.** Given the measurements of a flower, the goal is to predict which species it belongs to. The dataset contains the correct answer (species) for every row, which is what makes it supervised.

## Selection

```python
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_col = "species"

X = df[feature_cols]
y = df[target_col]

assert target_col not in X.columns   # safety check
```

| Item | Selection |
|------|-----------|
| **Features (X)** | sepal_length, sepal_width, petal_length, petal_width |
| **Target (y)** | species |
| **X shape** | (150, 4) |
| **y shape** | (150,) |

## Reasoning

**Why `species` is the target.** It is the value the model should predict. The task is to identify the type of flower, so the species column is the output (label).

**Why these four columns are features.** They are measurable properties of each flower that differ between species, so they carry useful information for telling the species apart. All four are numerical and complete, so they can be used directly as model input.

**Why the target is not used as a feature.** If `species` were included in X, the model would be given the answer it is supposed to predict. This is called **data leakage**. The model would look very accurate during training but would be useless on new flowers, where the species is unknown. The code includes an assertion to make sure the target is not in X.

**No extra columns to exclude.** The dataset has no ID or irrelevant columns, so all remaining columns are used as features.

## Key Learnings
- Always decide clearly what is being predicted (target) before choosing inputs (features).
- Never include the target, or anything that directly reveals it, in the features.
- Checking shapes (`X` is 2-dimensional, `y` is 1-dimensional) helps catch mistakes early.

## Interview Questions – Key Answers
1. **What is a feature?** An input column used by the model to make a prediction, such as `petal_length`. It is a measurable property of the data.
2. **What is a target variable?** The value the model tries to predict, also called the label or output. Here it is `species`.
3. **What is the difference between X and y in machine learning?** `X` holds the input features (rows × columns, 2D) and `y` holds the target values (one value per row, 1D). The model learns the relationship between X and y.

## How to Run
```bash
pip install pandas scikit-learn seaborn jupyter
jupyter notebook features_target_selection.ipynb
```
