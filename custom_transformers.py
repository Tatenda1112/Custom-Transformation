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
        self.errors_ = None

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
        self.errors_ = None
    
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
        self.errors_=None

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
            numeric_columns_ = X.select_dtypes(include=[np.number]).columns
            self.numeric_columns_ = numeric_columns_
            
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

class MeanMedianImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with mean or median.

    This transformer replaces missing values in each feature with either the mean or median of that feature.

    Parameters:
    -----------
    variables : list or None, default=None
        List of column names to impute. If None, all numeric columns will be imputed.
    imputation_type : {'mean', 'median'}, default='mean'
        Type of imputation to use. 'mean' for mean imputation and 'median' for median imputation.

    Attributes:
    -----------
    numeric_columns_ : pd.Index
        Index of numeric columns to be imputed.
    mean_ : pd.Series
        Series containing the mean of each numeric column.
    median_ : pd.Series
        Series containing the median of each numeric column.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the MeanMedianImputer to X.

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
        Perform imputation on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be imputed.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The imputed data.
    """

    def __init__(self, variables: list | None = None, imputation_type='mean'):
        """
        Initialize MeanMedianImputer.

        Args:
        ----
        variables (list|None): List of column names to impute. If None, all numeric columns will be imputed.
        imputation_type (str, optional): Type of imputation to use. 'mean' for mean imputation and 'median' for median imputation. Default is 'mean'.
        """
        self.imputation_type = imputation_type
        self.variables = variables
        self.errors_ =None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the MeanMedianImputer to X.

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
            numeric_columns_ = X.select_dtypes(include=[np.number]).columns
            self.numeric_columns_ = numeric_columns_
        self.mean_ = X[self.numeric_columns_].mean()
        self.median_ = X[self.numeric_columns_].median()
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Perform imputation on X.

        Args:
        ----
        X (pd.DataFrame): The data to be imputed.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The imputed data.
        """
        data = X.copy()
        try:
            if self.imputation_type == 'mean':
                data[self.numeric_columns_] = data[self.numeric_columns_].fillna(self.mean_)
            else:
                data[self.numeric_columns_] = data[self.numeric_columns_].fillna(self.median_)
        except Exception as e:
            self.errors_ = e
        return data
    
    
class categoricalImputer(BaseEstimator, TransformerMixin):
    """Impute missing categorical values with the most frequent value.

    This transformer replaces missing categorical values in each feature with the most frequent value of that feature.

    Parameters:
    -----------
    variables : list or None, default=None
        List of column names to impute. If None, all categorical columns will be imputed.
    strategy : {'most_frequent'}, default='most_frequent'
        Strategy to use for imputation. Currently, only 'most_frequent' is supported.

    Attributes:
    -----------
    categorical_columns_ : pd.Index
        Index of categorical columns to be imputed.
    fill_values_ : pd.Series
        Series containing the most frequent value of each categorical column.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the categoricalImputer to X.

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
        Perform imputation on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be imputed.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The imputed data.
    """

    def __init__(self, variables: list | None = None, strategy='most_frequent'):
        """
        Initialize categoricalImputer.

        Args:
        ----
        variables (list|None): List of column names to impute. If None, all categorical columns will be imputed.
        strategy (str, optional): Strategy to use for imputation. Currently, only 'most_frequent' is supported. Default is 'most_frequent'.
        """
        self.variables = variables
        self.strategy = strategy
        self.errors_=None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the categoricalImputer to X.

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
            self.categorical_columns_ = pd.Index(self.variables)
        else:
            categorical_columns = X.select_dtypes(include=['object']).columns
            self.categorical_columns_ = categorical_columns
        self.fill_values_ = X[self.categorical_columns_].mode().iloc[0]
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Perform imputation on X.

        Args:
        ----
        X (pd.DataFrame): The data to be imputed.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The imputed data.
        """
        data = X.copy()
        try:
            data[self.categorical_columns_] = data[self.categorical_columns_].fillna(self.fill_values_)
        except Exception as e:
            self.errors_ = e
        return data
    
class count_frequency_encoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables based on frequency counts.

    This transformer encodes categorical variables by replacing categories with their frequency counts normalized by the total count.

    Parameters:
    -----------
    variables : list or None, default=None
        List of column names to encode. If None, all categorical columns will be encoded.

    Attributes:
    -----------
    categorical_variables : pd.Index
        Index of categorical variables to be encoded.
    encoding_dict_ : dict
        Dictionary containing encoding mappings for each categorical variable.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the count_frequency_encoder to X.

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
        Perform encoding on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be encoded.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The encoded data.
    """

    def __init__(self, variables: list | None = None):
        """
        Initialize count_frequency_encoder.

        Args:
        ----
        variables (list|None): List of column names to encode. If None, all categorical columns will be encoded.
        """
        self.variables = variables
        self.errors_ =None

    def fit(self, X, y=None):
        """
        Fit the count_frequency_encoder to X.

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
            self.categorical_variables = self.variables
        else:
            categorical_columns_ = X.select_dtypes(include=['object']).columns
            self.categorical_variables = categorical_columns_
        
        self.encoding_dict_ = {}
        for var in self.categorical_variables:
            value_counts = X[var].value_counts()
            total_count = value_counts.sum()
            self.encoding_dict_[var] = value_counts / total_count 
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Perform encoding on X.

        Args:
        ----
        X (pd.DataFrame): The data to be encoded.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The encoded data.
        """
        data = X.copy()
        try:
            for var in self.categorical_variables:
                data[var] = data[var].map(self.encoding_dict_[var])
        except Exception as e:
            self.errors_ = e
        return data
    
    
class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables using one-hot encoding.

    This transformer replaces categorical variables with binary columns indicating the presence of each category.

    Parameters:
    -----------
    variables : list or None, default=None
        List of column names to encode. If None, all categorical columns will be encoded.

    Attributes:
    -----------
    categorical_columns : pd.Index
        Index of categorical columns to be encoded.
    categorical_variables : dict
        Dictionary containing unique categories for each categorical variable.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the OneHotEncoder to X.

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
        Perform one-hot encoding on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be encoded.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The encoded data.
    """

    def __init__(self, variables: list | None = None):
        """
        Initialize OneHotEncoder.

        Args:
        ----
        variables (list|None): List of column names to encode. If None, all categorical columns will be encoded.
        """
        self.variables = variables
        self.categorical_variables = {}
        self.errors_=None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the OneHotEncoder to X.

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
            self.categorical_columns = self.variables
        else:
            self.categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in self.categorical_columns:
            self.categorical_variables[col] = X[col].unique()
        
        print("OneHotEncoder fitted successfully!")
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Perform one-hot encoding on X.

        Args:
        ----
        X (pd.DataFrame): The data to be encoded.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The encoded data.
        """
        data = X.copy()
        
        try:
            for col in self.categorical_variables:
                categories = self.categorical_variables[col]
                for category in categories:
                    new_col_name = f'{col}_{category}'
                    data[new_col_name] = (data[col] == category).astype(int)
                
                # Drop the original categorical column
                data.drop(col, axis=1, inplace=True)
        except Exception as e:
            self.errors_ = e    
        return data


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables using ordinal encoding.

    This transformer replaces categorical variables with their ordinal mappings based on provided mapping or unique values.

    Parameters:
    -----------
    variables : list or None, default=None
        List of column names to encode. If None, all categorical columns will be encoded.
    mapping : dict or None, default=None
        Dictionary containing ordinal mappings for categorical variables. If None, mappings will be generated based on unique values.

    Attributes:
    -----------
    categorical_columns : pd.Index
        Index of categorical columns to be encoded.
    ordinal_mapping : dict
        Dictionary containing ordinal mappings for each categorical variable.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the OrdinalEncoder to X.

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
        Perform ordinal encoding on X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be encoded.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The encoded data.
    """

    def __init__(self, variables: list | None = None, mapping: dict = None):
        """
        Initialize OrdinalEncoder.

        Args:
        ----
        variables (list|None): List of column names to encode. If None, all categorical columns will be encoded.
        mapping (dict, optional): Dictionary containing ordinal mappings for categorical variables. Default is None.
        """
        self.variables = variables
        self.mapping = mapping
        self.ordinal_mapping = {}
        self.errors_= None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the OrdinalEncoder to X.

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
            self.categorical_columns = self.variables
        else:
            self.categorical_columns = X.select_dtypes(include=['object']).columns
        
        if self.mapping is None:
            self.generate_mapping(X)
        else:
            self.ordinal_mapping = self.mapping
        
        return self
    
    def generate_mapping(self, X: pd.DataFrame, y=None):
        """
        Generate ordinal mappings based on unique values in X.

        Args:
        ----
        X (pd.DataFrame): The training input samples.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.
        """
        for col in self.categorical_columns:
            unique_values = X[col].unique()
            self.ordinal_mapping[col] = {value: i for i, value in enumerate(unique_values)}
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Perform ordinal encoding on X.

        Args:
        ----
        X (pd.DataFrame): The data to be encoded.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The encoded data.
        """
        data = X.copy()
        
        try:
            for col, mapping in self.ordinal_mapping.items():
                data[col] = data[col].map(mapping)
        except Exception as e:
            self.errors_ = e  
        return data



class DropConstantFeatures(BaseEstimator, TransformerMixin):
    """Drop constant features from a DataFrame.

    This transformer drops columns from the DataFrame that have a constant value exceeding a specified threshold.

    Parameters:
    -----------
    threshold : float, default=0.9
        The threshold value to consider a feature as constant. Features with a value count normalized by the total count greater than or equal to this threshold will be dropped.

    Attributes:
    -----------
    columns_to_drop : list
        List of columns to be dropped from the DataFrame.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the DropConstantFeatures to X.

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
        Drop constant features from X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data from which constant features are to be dropped.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The data with constant features dropped.
    """

    def __init__(self, threshold=0.9):
        """
        Initialize DropConstantFeatures.

        Args:
        ----
        threshold (float, optional): The threshold value to consider a feature as constant. Default is 0.9.
        """
        self.threshold = threshold
        self.errors_ =None
    
    def fit(self, X, y=None):
        """
        Fit the DropConstantFeatures to X.

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
        self.columns_to_drop_ = X.columns[X.apply(lambda col: col.value_counts(normalize=True).values[0] >= self.threshold)].tolist()
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Drop constant features from X.

        Args:
        ----
        X (pd.DataFrame): The data from which constant features are to be dropped.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The data with constant features dropped.
        """
        data = X.copy()
        try:
            data = data.drop(columns=self.columns_to_drop)
        
        except Exception as e:
            self.errors_ = e
        return data



class HighCardinalityImputer(BaseEstimator, TransformerMixin):
    """Impute infrequent categories in high cardinality categorical variables.

    This transformer replaces infrequent categories in high cardinality categorical variables with a specified fill value.

    Parameters:
    -----------
    variables : list or None, default=None
        List of column names to consider for imputation. If None, all categorical columns will be considered.
    threshold : float, default=0.3
        The threshold value to identify infrequent categories based on cumulative frequency.
    fill_value : str or int, default='Other'
        The value to replace infrequent categories with.

    Attributes:
    -----------
    categories_to_replace : dict
        Dictionary containing categories to be replaced for each specified column.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the HighCardinalityImputer to X.

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
        Impute infrequent categories in X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data from which infrequent categories are to be replaced.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The data with infrequent categories replaced.
    """

    def __init__(self, variables: list | None = None, threshold=0.3, fill_value='Other'):
        """
        Initialize HighCardinalityImputer.

        Args:
        ----
        variables (list|None): List of column names to consider for imputation. If None, all categorical columns will be considered.
        threshold (float, optional): The threshold value to identify infrequent categories based on cumulative frequency. Default is 0.3.
        fill_value (str|int, optional): The value to replace infrequent categories with. Default is 'Other'.
        """
        self.threshold = threshold
        self.variables = variables
        self.fill_value = fill_value
        self.categories_to_replace = {}
        self.errors_ =None
    
    def fit(self, X, y=None):
        """
        Fit the HighCardinalityImputer to X.

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
            for col in self.variables:
                freq = X[col].value_counts(normalize=True)
                cumulative_freq = freq.cumsum()
                infrequent_categories = cumulative_freq[cumulative_freq < self.threshold].index.tolist()
                self.categories_to_replace[col] = infrequent_categories
        else:
            for col in X.columns:
                freq = X[col].value_counts(normalize=True)
                cumulative_freq = freq.cumsum()
                infrequent_categories = cumulative_freq[cumulative_freq < self.threshold].index.tolist()
                self.categories_to_replace[col] = infrequent_categories
        
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Impute infrequent categories in X.

        Args:
        ----
        X (pd.DataFrame): The data from which infrequent categories are to be replaced.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The data with infrequent categories replaced.
        """
        data = X.copy()
        try:
            for col, categories in self.categories_to_replace.items():
                data[col] = data[col].apply(lambda x: self.fill_value if x in categories else x)
        except Exception as e:
            self.errors_ = e
        return data

class DropDuplicateFeatures(BaseEstimator, TransformerMixin):
    """Drop duplicate features from a DataFrame.

    This transformer drops columns from the DataFrame that are duplicates of other columns.

    Attributes:
    -----------
    duplicate_columns : list
        List of duplicate columns to be dropped from the DataFrame.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the DropDuplicateFeatures to X.

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
        Drop duplicate features from X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data from which duplicate features are to be dropped.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The data with duplicate features dropped.
    """

    def __init__(self):
        """
        Initialize DropDuplicateFeatures.
        """
        self.duplicate_columns = []
        self.errors_ =None
    
    def fit(self, X, y=None):
        """
        Fit the DropDuplicateFeatures to X.

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
        self.duplicate_columns = self.find_duplicate_columns(X)
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Drop duplicate features from X.

        Args:
        ----
        X (pd.DataFrame): The data from which duplicate features are to be dropped.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The data with duplicate features dropped.
        """
        data = X.copy()
        try:
            data = data.drop(columns=self.duplicate_columns)
        except Exception as e:
            self.errors_ = e
        return data
    
    def find_duplicate_columns(self, X: pd.DataFrame):
        """
        Find duplicate columns in X.

        Args:
        ----
        X (pd.DataFrame): The data to check for duplicate columns.

        Returns:
        --------
        duplicate_columns : list
            List of duplicate columns found in X.
        """
        duplicate_columns = []
        for i in range(X.shape[1]):
            col1 = X.iloc[:, i]
            for j in range(i+1, X.shape[1]):
                col2 = X.iloc[:, j]
                if col1.equals(col2):
                    duplicate_columns.append(X.columns[j])
                    
        return duplicate_columns


class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """Drop correlated features from a DataFrame.

    This transformer drops columns from the DataFrame that are highly correlated with other columns based on a correlation threshold.

    Parameters:
    -----------
    threshold : float, default=0.95
        The threshold value to identify highly correlated features. Features with a correlation coefficient greater than this threshold will be dropped.

    Attributes:
    -----------
    correlated_features : list
        List of correlated features to be dropped from the DataFrame.
    errors_ : Exception or None
        Stores any exception occurred during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the DropCorrelatedFeatures to X.

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
        Drop correlated features from X.

        Parameters:
        -----------
        X : pd.DataFrame
            The data from which correlated features are to be dropped.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The data with correlated features dropped.
    """

    def __init__(self, threshold=0.95):
        """
        Initialize DropCorrelatedFeatures.

        Args:
        ----
        threshold (float, optional): The threshold value to identify highly correlated features. Default is 0.95.
        """
        self.threshold = threshold
        self.correlated_features = []
        self.errors_ =None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the DropCorrelatedFeatures to X.

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
        corr_matrix = X.corr().abs()
        self.correlated_features = self.find_correlated_features(corr_matrix)
        
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Drop correlated features from X.

        Args:
        ----
        X (pd.DataFrame): The data from which correlated features are to be dropped.
        y : array-like, default=None
            Ignored parameter. Present for compatibility with scikit-learn's pipeline.

        Returns:
        --------
        data : pd.DataFrame
            The data with correlated features dropped.
        """
        data = X.copy()
        try:
            data = data.drop(columns=self.correlated_features)
        except Exception as e:
            self.errors_ = e
        return data
    
    def find_correlated_features(self, corr_matrix):
        """
        Find correlated features based on correlation matrix.

        Args:
        ----
        corr_matrix (pd.DataFrame): The correlation matrix of the input data.

        Returns:
        --------
        correlated_features : list
            List of correlated features found based on the correlation matrix.
        """
        correlated_features = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)
        
        return list(correlated_features)

class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Box-Cox transformation.

    Parameters:
    -----------
    lambda_range : tuple, optional (default=(-3.0, 3.0))
        The range of lambda values to search for during fitting.

    Attributes:
    -----------
    lambda_ : float
        The lambda value that maximizes the log-likelihood during fitting.
    log_likelihood_ : float
        The log-likelihood of the transformed data using the best lambda value.
    errors_ : exception
        Any errors encountered during fitting or transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the BoxCoxTransformer to the data.
    
    transform(X):
        Transform the input data using the best lambda value found during fitting.

    _boxcox_transform(X, lambda_):
        Apply the Box-Cox transformation to the input data using the specified lambda value.

    _log_likelihood(X):
        Calculate the log-likelihood of the transformed data.

    """

    def __init__(self, lambda_range=(-3.0, 3.0)):
        """
        Initialize the BoxCoxTransformer.

        Parameters:
        -----------
        lambda_range : tuple, optional (default=(-3.0, 3.0))
            The range of lambda values to search for during fitting.
        """
        self.lambda_range = lambda_range
        self.lambda_ = None
        self.log_likelihood_ = None
        self.errors_ = None
        
    def fit(self, X, y=None):
        """
        Fit the BoxCoxTransformer to the data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.
 
        y : array-like, shape (n_samples,), optional (default=None)
            The target labels.

        Returns:
        --------
        self : object
            Returns self.
        """
        try:
            best_lambda = None
            best_log_likelihood = float('-inf')

            for lambda_value in np.linspace(*self.lambda_range, num=100):
                transformed_data = self._boxcox_transform(X, lambda_value)
                log_likelihood = self._log_likelihood(transformed_data)

                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_lambda = lambda_value

            self.lambda_ = best_lambda
            self.log_likelihood_ = best_log_likelihood
        except Exception as e:
            self.errors_ = e
        return self
   
    def transform(self, X):
        """
        Transform the input data using the best lambda value found during fitting.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        try:
            return self._boxcox_transform(X, self.lambda_)
        except Exception as e:
            self.errors_ = e
            return None
   
    def _boxcox_transform(self, X, lambda_):
        """
        Apply the Box-Cox transformation to the input data using the specified lambda value.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        lambda_ : float
            The lambda value for the Box-Cox transformation.

        Returns:
        --------
        transformed_data : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        try:
            if lambda_ == 0:
                return np.log(X)
            else:
                return (X**lambda_ - 1) / lambda_
        except Exception as e:
            self.errors_ = e
            return None
   
    def _log_likelihood(self, X):
        """
        Calculate the log-likelihood of the transformed data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The transformed data.

        Returns:
        --------
        log_likelihood : float
            The log-likelihood of the transformed data.
        """
        try:
            n = X.shape[0]
            if self.lambda_ is None or self.lambda_ == 0:
                return -n / 2 * (1 + np.log(2 * np.pi) + np.log(np.mean(X ** 2)))
            else:
                return n / 2 * (np.log(self.lambda_ / (2 * np.pi)) + (self.lambda_ - 1) * np.mean(np.log(X)))
        except Exception as e:
            self.errors_ = e
            return None


class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Yeo-Johnson transformation.

    Parameters:
    -----------
    lambda_range : tuple, optional (default=(-1.0, 2.0))
        The range of lambda values to search for during fitting.

    Attributes:
    -----------
    lambda_ : float
        The lambda value that maximizes the log-likelihood during fitting.
    log_likelihood_ : float
        The log-likelihood of the transformed data using the best lambda value.
    errors_ : exception
        Any errors encountered during fitting or transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the YeoJohnsonTransformer to the data.
    
    transform(X):
        Transform the input data using the best lambda value found during fitting.

    _yeo_johnson_transform(X, lambda_):
        Apply the Yeo-Johnson transformation to the input data using the specified lambda value.

    _log_likelihood(X):
        Calculate the log-likelihood of the transformed data.

    """

    def __init__(self, lambda_range=(-1.0, 2.0)):
        """
        Initialize the YeoJohnsonTransformer.

        Parameters:
        -----------
        lambda_range : tuple, optional (default=(-1.0, 2.0))
            The range of lambda values to search for during fitting.
        """
        self.lambda_range = lambda_range
        self.lambda_ = None
        self.log_likelihood_ = None
        self.errors_ = None
        
    def fit(self, X, y=None):
        """
        Fit the YeoJohnsonTransformer to the data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.
 
        y : array-like, shape (n_samples,), optional (default=None)
            The target labels.

        Returns:
        --------
        self : object
            Returns self.
        """
        try:
            best_lambda = None
            best_log_likelihood = float('-inf')

            for lambda_value in np.linspace(*self.lambda_range, num=100):
                transformed_data = self._yeo_johnson_transform(X, lambda_value)
                log_likelihood = self._log_likelihood(transformed_data)

                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_lambda = lambda_value

            self.lambda_ = best_lambda
            self.log_likelihood_ = best_log_likelihood
        except Exception as e:
            self.errors_ = e
        return self
   
    def transform(self, X):
        """
        Transform the input data using the best lambda value found during fitting.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        try:
            return self._yeo_johnson_transform(X, self.lambda_)
        except Exception as e:
            self.errors_ = e
            return None
   
    def _yeo_johnson_transform(self, X, lambda_):
        """
        Apply the Yeo-Johnson transformation to the input data using the specified lambda value.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        lambda_ : float
            The lambda value for the Yeo-Johnson transformation.

        Returns:
        --------
        transformed_data : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        try:
            if lambda_ == 0:
                return np.log1p(X)
            else:
                if lambda_ < 0:  
                    offset = 0.5  
                    X += offset  
                    transformed = ((X)**lambda_ - 1) / lambda_
                    transformed -= transformed.min() + 1  
                    return transformed
                else:  
                    return (X**lambda_ - 1) / lambda_
        except Exception as e:
            self.errors_ = e
            return None
   
    def _log_likelihood(self, X):
        """
        Calculate the log-likelihood of the transformed data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The transformed data.

        Returns:
        --------
        log_likelihood : float
            The log-likelihood of the transformed data.
        """
        try:
            n = X.shape[0]
            if self.lambda_ is None or self.lambda_ == 0:
                return -n / 2 * (1 + np.log(2 * np.pi) + np.log(np.mean(X ** 2)))
            else:
                return n / 2 * (np.log(self.lambda_ / (2 * np.pi)) + (self.lambda_ - 1) * np.mean(np.log(X)))
        except Exception as e:
            self.errors_ = e
            return None
