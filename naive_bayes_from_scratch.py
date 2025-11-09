import numpy as np
from collections import Counter


def accuracy_score(y_true, y_pred):
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='binary'):
    """Calculate precision score."""
    classes = np.unique(y_true)
    
    if average == 'binary' and len(classes) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        precisions = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.mean(precisions)


def recall_score(y_true, y_pred, average='binary'):
    """Calculate recall score."""
    classes = np.unique(y_true)
    
    if average == 'binary' and len(classes) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.mean(recalls)


def f1_score(y_true, y_pred, average='binary'):
    """Calculate F1 score."""
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return matrix


class GaussianNB:
    """
    Gaussian Naive Bayes Classifier.
    
    Assumes features follow a Gaussian (normal) distribution.
    Best for continuous features.
    
    P(x_i|y) = (1/√(2πσ²)) * exp(-(x_i - μ)² / (2σ²))
    """
    
    def __init__(self, var_smoothing=1e-9):
        """
        var_smoothing: portion of largest variance added to all variances for stability
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # mean of each feature per class
        self.sigma_ = None  # variance of each feature per class
        self.epsilon_ = None  # smoothing value
    
    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes classifier.
        
        Calculates:
        - Prior probabilities P(y)
        - Mean and variance for each feature per class
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.sigma_[idx, :] = X_c.var(axis=0)
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]
        
        # Add smoothing to variance
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.sigma_ += self.epsilon_
        
        return self
    
    def _calculate_likelihood(self, X):
        """Calculate log likelihood for each class."""
        likelihoods = []
        
        for idx in range(len(self.classes_)):
            # Log of prior probability
            prior = np.log(self.class_prior_[idx])
            
            # Log of Gaussian probability density function
            # log P(x|y) = -0.5 * sum(log(2π*σ²) + ((x-μ)²/σ²))
            likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * self.sigma_[idx, :]))
            likelihood -= 0.5 * np.sum(((X - self.theta_[idx, :]) ** 2) / self.sigma_[idx, :], axis=1)
            
            likelihoods.append(prior + likelihood)
        
        return np.array(likelihoods).T
    
    def predict(self, X):
        """Predict class labels."""
        X = np.array(X)
        likelihoods = self._calculate_likelihood(X)
        return self.classes_[np.argmax(likelihoods, axis=1)]
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        likelihoods = self._calculate_likelihood(X)
        
        # Convert log likelihoods to probabilities using softmax
        # Subtract max for numerical stability
        likelihoods = likelihoods - np.max(likelihoods, axis=1, keepdims=True)
        exp_likelihoods = np.exp(likelihoods)
        return exp_likelihoods / np.sum(exp_likelihoods, axis=1, keepdims=True)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        return accuracy_score(y, self.predict(X))


class MultinomialNB:
    """
    Multinomial Naive Bayes Classifier.
    
    Assumes features are discrete counts (e.g., word counts).
    Best for text classification with count features.
    
    P(x_i|y) = (N_yi + α) / (N_y + α*n)
    where N_yi is count of feature i in class y, N_y is total count in class y
    """
    
    def __init__(self, alpha=1.0):
        """
        alpha: Laplace smoothing parameter (additive smoothing)
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
    
    def fit(self, X, y):
        """
        Fit Multinomial Naive Bayes classifier.
        
        Calculates:
        - Prior probabilities P(y)
        - Feature probabilities P(x_i|y) for each class
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            
            # Feature counts for this class (with smoothing)
            feature_count = X_c.sum(axis=0) + self.alpha
            total_count = feature_count.sum()
            
            # Log probabilities
            self.feature_log_prob_[idx, :] = np.log(feature_count / total_count)
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]
        
        return self
    
    def _calculate_likelihood(self, X):
        """Calculate log likelihood for each class."""
        # log P(X|y) = sum(x_i * log(P(x_i|y)))
        return (X @ self.feature_log_prob_.T) + np.log(self.class_prior_)
    
    def predict(self, X):
        """Predict class labels."""
        X = np.array(X)
        likelihoods = self._calculate_likelihood(X)
        return self.classes_[np.argmax(likelihoods, axis=1)]
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        likelihoods = self._calculate_likelihood(X)
        
        # Convert log likelihoods to probabilities
        likelihoods = likelihoods - np.max(likelihoods, axis=1, keepdims=True)
        exp_likelihoods = np.exp(likelihoods)
        return exp_likelihoods / np.sum(exp_likelihoods, axis=1, keepdims=True)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        return accuracy_score(y, self.predict(X))


class BernoulliNB:
    """
    Bernoulli Naive Bayes Classifier.
    
    Assumes features are binary (0 or 1).
    Best for binary/boolean features (e.g., word presence/absence).
    
    P(x_i|y) = P(i|y)*x_i + (1 - P(i|y))*(1 - x_i)
    """
    
    def __init__(self, alpha=1.0, binarize=0.0):
        """
        alpha: Laplace smoothing parameter
        binarize: threshold for binarizing features (None = no binarization)
        """
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
    
    def fit(self, X, y):
        """
        Fit Bernoulli Naive Bayes classifier.
        
        Calculates:
        - Prior probabilities P(y)
        - Feature probabilities P(x_i=1|y) for each class
        """
        X = np.array(X)
        y = np.array(y)
        
        # Binarize if threshold is set
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.feature_log_prob_ = []
        self.class_prior_ = np.zeros(n_classes)
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            
            # Count of feature occurrences (with smoothing)
            feature_count = X_c.sum(axis=0) + self.alpha
            total_count = X_c.shape[0] + 2 * self.alpha
            
            # Log probabilities for feature = 1 and feature = 0
            feature_prob = feature_count / total_count
            self.feature_log_prob_.append([
                np.log(feature_prob),  # P(x_i=1|y)
                np.log(1 - feature_prob)  # P(x_i=0|y)
            ])
            
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]
        
        self.feature_log_prob_ = np.array(self.feature_log_prob_)
        
        return self
    
    def _calculate_likelihood(self, X):
        """Calculate log likelihood for each class."""
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        
        likelihoods = []
        
        for idx in range(len(self.classes_)):
            # log P(X|y) = sum(x_i * log P(x_i=1|y) + (1-x_i) * log P(x_i=0|y))
            likelihood = np.sum(
                X * self.feature_log_prob_[idx, 0, :] + 
                (1 - X) * self.feature_log_prob_[idx, 1, :],
                axis=1
            )
            likelihood += np.log(self.class_prior_[idx])
            likelihoods.append(likelihood)
        
        return np.array(likelihoods).T
    
    def predict(self, X):
        """Predict class labels."""
        X = np.array(X)
        likelihoods = self._calculate_likelihood(X)
        return self.classes_[np.argmax(likelihoods, axis=1)]
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        likelihoods = self._calculate_likelihood(X)
        
        # Convert log likelihoods to probabilities
        likelihoods = likelihoods - np.max(likelihoods, axis=1, keepdims=True)
        exp_likelihoods = np.exp(likelihoods)
        return exp_likelihoods / np.sum(exp_likelihoods, axis=1, keepdims=True)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        return accuracy_score(y, self.predict(X))

