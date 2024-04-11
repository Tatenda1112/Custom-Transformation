import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance.

    This transformer scales each feature to have a mean of 0 and a standard deviation of 1.

    Parameters:
    -----------
    variables : list or None, default=None
        List of column names to standardize. If None, all numeric columns will be standardized.

    Attributes:
    -----------
    numeric_columns_ : pd.Index
        Index of numeric columns to be standardized.
    mean_ : pd.Series
        Series containing the mean of each numeric column.
    std_ : pd.Series
        Series containing the standard deviation of each numeric column.
    learned_params_ : pd.DataFrame
        DataFrame containing the mean and standard deviation of each numeric column learned during fitting.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the StandardScaler to X.

        Parameters:
        -----------
        X : pd.DataFrame
            The training input samples.

        Returns:
        --------
        self : object
            Returns self.

    transform(X, y=None):
        Perform standardization on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be standardized.

        Returns:
        --------
        data : pd.DataFrame
            The standardized data.

    Example:
    --------
    from sklearn.datasets import load_iris
    from sklearn.pipeline import Pipeline

    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> y = iris.target

    >>> preprocessor = Pipeline(
        steps=[
            ('num', StandardScaler())
        ]
    )

    >>> X_processed = preprocessor.fit_transform(X)
    """

    def __init__(self, variables: list | None = None):
        self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the StandardScaler to X.

        Parameters:
        -----------
        X : pd.DataFrame
            The training input samples.

        Returns:
        --------
        self : object
            Returns self.
        """
        if self.variables is not None:
            self.numeric_columns_ = pd.Index(self.variables)
        else:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            self.numeric_columns_ = numeric_columns
        self.mean_ = X[self.numeric_columns_].mean()
        self.std_ = X[self.numeric_columns_].std()
        learned_params_ = pd.concat(
            [self.mean_, self.std_], axis=1, keys=["mean", "std"]
        )
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """Perform standardization on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be standardized.

        Returns:
        --------
        data : pd.DataFrame
            The standardized data.
        """
        data = X.copy()
        try:
            data[self.numeric_columns_] = (
                data[self.numeric_columns_] - self.mean_
            ) / self.std_
        except Exception as e:
            self.errors_ = e
        return data
