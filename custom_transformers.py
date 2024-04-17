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



class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Scale features to a specified range.

    This transformer scales each feature to a given range.

    Parameters:
    -----------
    feature_range : tuple, default=(0, 1)
        The desired range of transformed data.
    variables : list or None, default=None
        List of column names to scale. If None, all numeric columns will be scaled.

    Attributes:
    -----------
    numeric_columns_ : pd.Index
        Index of numeric columns to be scaled.
    min_ : pd.Series
        Series containing the minimum value of each numeric column.
    max_ : pd.Series
        Series containing the maximum value of each numeric column.
    learned_params_ : pd.DataFrame
        DataFrame containing the min and max values of each numeric column learned during fitting.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the MinMaxScaler to X.

        Parameters:
        -----------
        X : pd.DataFrame
            The training input samples.

        Returns:
        --------
        self : object
            Returns self.

    transform(X, y=None):
        Perform scaling on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be scaled.

        Returns:
        --------
        data : pd.DataFrame
            The scaled data.

    Example:
    --------
    from sklearn.datasets import load_iris
    from sklearn.pipeline import Pipeline

    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> y = iris.target

    >>> preprocessor = Pipeline(
        steps=[
            ('num', MinMaxScaler())
        ]
    )

    >>> X_processed = preprocessor.fit_transform(X)
    """

    def __init__(self, feature_range=(0, 1), variables: list | None = None):
        """
        Initialize MinMaxScaler.

        Args:
        ----
        feature_range (tuple, optional): The desired range of transformed data. Default is (0, 1).
        variables (list|None): List of column names to scale. If None, all numeric columns will be scaled.
        """
        self.feature_range = feature_range
        self.variables = variables
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the MinMaxScaler to X.

        Args:
        ----
        X (pd.DataFrame): The training input samples.
        y: Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        self: Returns an instance of self.
        """
        if self.variables is not None:
            self.numeric_columns_ = pd.Index(self.variables)
        else:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            self.numeric_columns_ = numeric_columns 
        self.min_ = X[self.numeric_columns_].min()
        self.max_ = X[self.numeric_columns_].max()
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Perform scaling on X.

        Args:
        ----
        X (pd.DataFrame): The data to be scaled.
        y: Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data (pd.DataFrame): The scaled data.
        """
        data = X.copy()
        try:
            min_val, max_val = self.feature_range
            data[self.numeric_columns_] = (data[self.numeric_columns_] - self.min_) / (self.max_ - self.min_) * (max_val - min_val) + min_val
        except Exception as e:
            self.errors_ = e
        return data
    
    


class winsorizer(BaseEstimator, TransformerMixin):
    """Winsorize features to limit extreme values.

    This transformer limits extreme values (outliers) in each feature by replacing them with the lower or upper threshold.

    Parameters:
    -----------
    variables : list or None, default=None
        List of column names to winsorize. If None, all numeric columns will be winsorized.
    lower_quantile : float, default=0.25
        Lower quantile for winsorization.
    upper_quantile : float, default=0.75
        Upper quantile for winsorization.
    K : float, default=5
        Coefficient to calculate the winsorization thresholds.

    Attributes:
    -----------
    numeric_columns_ : pd.Index
        Index of numeric columns to be winsorized.
    q1 : pd.Series
        Series containing the lower quantile of each numeric column.
    q3 : pd.Series
        Series containing the upper quantile of each numeric column.
    iqr : pd.Series
        Series containing the interquartile range (IQR) of each numeric column.
    lower_threshold : pd.Series
        Series containing the lower winsorization threshold of each numeric column.
    upper_threshold : pd.Series
        Series containing the upper winsorization threshold of each numeric column.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the winsorizer to X.

        Parameters:
        -----------
        X : pd.DataFrame
            The training input samples.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        self : object
            Returns self.

    transform(X, y=None):
        Perform winsorization on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be winsorized.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The winsorized data.
    """

    def __init__(self, variables: list | None = None, lower_quantile=0.25, upper_quantile=0.75, K=5):
        """
        Initialize winsorizer.

        Args:
        ----
        variables (list|None): List of column names to winsorize. If None, all numeric columns will be winsorized.
        lower_quantile (float, optional): Lower quantile for winsorization. Default is 0.25.
        upper_quantile (float, optional): Upper quantile for winsorization. Default is 0.75.
        K (float, optional): Coefficient to calculate the winsorization thresholds. Default is 5.
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.variables = variables
        self.K = K

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the winsorizer to X.

        Args:
        ----
        X (pd.DataFrame): The training input samples.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        self : object
            Returns an instance of self.
        """
        if self.variables is not None:
            self.numeric_columns_ = pd.Index(self.variables)
        else:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            self.numeric_columns_ = numeric_columns 
            
        self.q1 = X[self.numeric_columns_].quantile(self.lower_quantile)
        self.q3 = X[self.numeric_columns_].quantile(self.upper_quantile)
        self.iqr = self.q3 - self.q1
            
        # calculate thresholds
        self.lower_threshold = self.q1 - self.K * self.iqr
        self.upper_threshold = self.q3 + self.K * self.iqr
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Perform winsorization on X.

        Args:
        ----
        X (pd.DataFrame): The data to be winsorized.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The winsorized data.
        """
        data = X.copy()
        try:
            data[self.numeric_columns_] = data[self.numeric_columns_].clip(lower=self.lower_threshold, upper=self.upper_threshold)
        except Exception as e:
            self.errors_ = e
        return data

