from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class GradientBoostingClassifier:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None,
        subsample: float | int = 1.0,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: Optional[str] = 'Bernoulli',
        goss: Optional[bool] = False,
        goss_k: float | int = 0.2,
        rsm: float | int = 1.0,
        quantization_type: Optional[str] = None,
        nbins: int = 255,
        random_seed: int = 42
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list) # store metrics evolution
        
        self.early_stopping_rounds = early_stopping_rounds

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z) 
        
        # store iteration number with best validation metric value (useful for early stopping)
        self.best_iteration: Optional[int] = None
        
        # bootstrap-related attributes
        self.subsample: float | int = subsample
        self.bagging_temperature: float | int = bagging_temperature
        self.bootstrap_type: Optional[str] = bootstrap_type

        # GOSS-related attrubutes
        self.goss: Optional[bool] = goss
        self.goss_k: float | int = goss_k
        
        # feature selection
        self.rsm: float | int = rsm
        self.selected_features: list = []
        
        # quantization
        self.quantization_type: Optional[str] = quantization_type
        self.nbins: int = nbins
        self.bins: Optional[np.ndarray] = None
        
        self.features_n_ : Optional[int] = None
        
        self.random_seed: int = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

    def partial_fit(self, X, y, old_predictions=None):
        if old_predictions is None:
            old_predictions = self.predict_logit(X)
        
        residuals = -self.loss_derivative(y, old_predictions) 
        
        # Bootstrap and GOSS
        sample_weight = None
        if self.goss:
            X_train, residuals_train, sample_weight = self.apply_goss(X, -residuals) # negate, because we need gradients, not residuals
        elif self.bootstrap_type is not None:
            X_train, residuals_train, sample_weight = self.apply_bootstrap(X, residuals)
        else:
           X_train, residuals_train = X, residuals
        
        self.apply_rsm(X_train)
           
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_train[:, self.selected_features[-1]], residuals_train, sample_weight=sample_weight)
     
        new_residuals = model.predict(X[:, self.selected_features[-1]])
        gamma = self.find_optimal_gamma(y, old_predictions, new_residuals)
        
        self.models.append(model)
        self.gammas.append(gamma)
        
        # return new_residuals to the caller fit to avoid redundant computing
        return new_residuals


    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        self._feature_importances = None # reset feature importances
        
        self.features_n_ = X_train.shape[1]
        
        # quantization
        if self.quantization_type is not None:
            X_train = self.quantize(X_train)
            if X_val is not None:
                # apply quantization on validation set based on the existing bins
                X_val = self.quantize_on_cur_bins(X_val)
        
        train_predictions = np.zeros(y_train.shape[0])
        val_predictions = np.zeros(y_val.shape[0]) if X_val is not None else None
        
        rounds_no_imrovement = 0
        best_val_roc_auc = -1 

        for i in range(self.n_estimators):
            # pass new_residuals to partial_fit to avoid redundant computing
            new_residuals = self.partial_fit(X_train, y_train, train_predictions)
            
            train_predictions += self.learning_rate * self.gammas[-1] * new_residuals
            
            self.history["train_loss"].append(self.loss_fn(y_train, train_predictions))
            self.history["train_roc_auc"].append(roc_auc_score(y_train == 1, self.sigmoid(train_predictions)))
            
            if X_val is not None and y_val is not None:
                val_predictions += self.learning_rate * self.gammas[-1] * \
                                   self.models[-1].predict(X_val[:, self.selected_features[-1]])
                cur_val_loss = self.loss_fn(y_val, val_predictions)
                cur_val_roc_auc = roc_auc_score(y_val == 1, self.sigmoid(val_predictions))
                
                self.history["val_loss"].append(cur_val_loss)
                self.history["val_roc_auc"].append(cur_val_roc_auc)
                
                # early stopping
                if cur_val_roc_auc > best_val_roc_auc:
                    best_val_roc_auc = cur_val_roc_auc
                    rounds_no_imrovement = 0 
                    self.best_iteration = i
                else:
                    rounds_no_imrovement += 1
                    if self.early_stopping_rounds is not None and rounds_no_imrovement >= self.early_stopping_rounds:
                        break
           
        # plot metrics evolution based on the existing history     
        if plot:
            self.plot_history(based_on_history=True)

    
    def predict_logit(self, X):
        logit_predictions = np.zeros(X.shape[0])
        for model, gamma, sel_feat in zip(self.models, self.gammas, self.selected_features):
            logit_predictions += self.learning_rate * gamma * model.predict(X[:, sel_feat])
        return logit_predictions
    
    
    def predict_proba(self, X):
        logit_predictions = self.predict_logit(X)
        proba_predictions = self.sigmoid(logit_predictions)
        # sklearn format
        return np.vstack((1 - proba_predictions, proba_predictions)).T
        

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
    
    
    def apply_bootstrap(self, X, y):
        if self.bootstrap_type == 'Bernoulli':
            # if subset is int, then it is number of objects, convert to ratio
            threshold = self.subsample if isinstance(self.subsample, float) else self.subsample / X.shape[0]
            mask = self.random_state.rand(X.shape[0]) < threshold
            return X[mask], y[mask], None # None, since we don't use weights
        elif self.bootstrap_type == 'Bayesian':
            weights = (-np.log(self.random_state.rand(X.shape[0]))) ** self.bagging_temperature
            return X, y, weights
            
    
    def apply_goss(self, X, gradients):
        large_grad_n = int(self.goss_k * X.shape[0]) # number of objects with large gradients
        small_grad_init_n = X.shape[0] - large_grad_n # initial number of objects with small gradients
        sorted_indices = np.argsort(np.abs(gradients))[::-1] # sort in descending order
        
        # for large gradients
        large_indices = sorted_indices[:large_grad_n]
        X_large_grad = X[large_indices]
        large_gradients = gradients[large_indices]
        
        # for small gradients
        small_indices_init = sorted_indices[large_grad_n:]
        # compute number of objects with small gradients after applying subsample and their indices
        small_share = self.subsample if isinstance(self.subsample, float) else self.subsample / small_grad_init_n
        small_grad_n = int(small_share * small_grad_init_n)
        small_indices = self.random_state.choice(small_indices_init, size=small_grad_n, replace=False)
        
        X_small_grad = X[small_indices]
        small_gradients = gradients[small_indices]
        
        # concatenate large and small gradients
        X_final = np.vstack((X_large_grad, X_small_grad))
        gradients_final = np.hstack((large_gradients, small_gradients))
        
        sample_weight = np.ones(X.shape[0])
        sample_weight[small_indices] *= (1 - self.goss_k) / small_share # multiply by the factor
       
        sample_weight_final = np.hstack((sample_weight[large_indices], sample_weight[small_indices]))
        
        # negate gradients_final, because we need residuals
        return X_final, -gradients_final, sample_weight_final
    
    
    def quantize(self, X):
        if self.quantization_type == 'uniform':
            min_vals = np.nanmin(X, axis=0) 
            max_vals = np.nanmax(X, axis=0)
            
            self.bins = np.linspace(min_vals, max_vals, self.nbins + 1)
            return np.array([np.digitize(X[:, i], self.bins[:, i], right=True) for i in range(X.shape[1])]).T
        
        elif self.quantization_type == 'quantile':
            X_sorted_features = np.sort(X, axis=0) # sort features (along columns)
            # compute bins for each feature
            self.bins = np.nanquantile(X_sorted_features, np.linspace(0, 1, self.nbins + 1), axis=0) 
            return np.array([np.digitize(X[:, i], self.bins[:, i], right=True) for i in range(X.shape[1])]).T
    
    
    def quantize_on_cur_bins(self, X):
        return np.array([np.digitize(X[:, i], self.bins[:, i], right=True) for i in range(X.shape[1])]).T
    
    
    def apply_rsm(self, X):
        threshold = self.rsm if isinstance(self.rsm, float) else self.rsm / X.shape[1]
        mask = self.random_state.rand(X.shape[1]) < threshold
        # compute indices of features to take
        self.selected_features.append(np.argwhere(mask).flatten())
    
    
    @property
    def feature_importances_(self):
        if self._feature_importances is not None:
            return self._feature_importances
        
        self._feature_importances = self.compute_feature_importances()
        return self._feature_importances

    
    def compute_feature_importances(self):
        importances = np.zeros(self.features_n_)
        for model, selected_features in zip(self.models, self.selected_features):
            importances[selected_features] += model.feature_importances_
        importances /= importances.sum()
        return importances
    
    
    def make_plot_history(self):
        fig, ax = plt.subplots(2, 1, figsize=(14, 14))
        base_models = np.linspace(1, len(self.models), len(self.models), dtype=np.int64)
        
        sns.lineplot(x=base_models, y=self.history["train_loss"], ax=ax[0], label="train_loss")
        if "val_loss" in self.history:
            sns.lineplot(x=base_models, y=self.history["val_loss"], ax=ax[0], label="val_loss")
              
        sns.lineplot(x=base_models, y=self.history["train_roc_auc"], ax=ax[1], label="train_roc_auc")
        if "val_roc_auc" in self.history:
            sns.lineplot(x=base_models, y=self.history["val_roc_auc"], ax=ax[1], label="val_roc_auc")
        
        ax[0].set_title("Base Model vs Loss")
        ax[0].set_xlabel("Base Model")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        
        ax[1].set_title("Base Model vs ROC-AUC")
        ax[1].set_xlabel("Base Model")
        ax[1].set_ylabel("ROC-AUC")
        ax[1].legend()
        
        plt.show()
        
    
    def make_plot_new_data(self, X, y):
        losses = []
        roc_aucs = []
        logit_predictions_init = np.zeros(X.shape[0])
        for model, gamma, selec_feat in zip(self.models, self.gammas, self.selected_features):
            logit_predictions_init += self.learning_rate * gamma * model.predict(X[:, selec_feat])
        
            loss = self.loss_fn(y, logit_predictions_init)
            roc_auc = roc_auc_score(y == 1, self.sigmoid(logit_predictions_init))
            losses.append(loss)
            roc_aucs.append(roc_auc)
        
        fig, ax = plt.subplots(2, 1, figsize=(14, 14))
        base_models = np.linspace(1, len(self.models), len(self.models), dtype=np.int64)
        
        sns.lineplot(x=base_models, y=losses, ax=ax[0], label="loss")
        sns.lineplot(x=base_models, y=roc_aucs, ax=ax[1], label="roc_auc")
        
        ax[0].set_title("Base Model vs Loss")
        ax[0].set_xlabel("Base Model")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        
        ax[1].set_title("Base Model vs ROC-AUC")
        ax[1].set_xlabel("Base Model")
        ax[1].set_ylabel("ROC-AUC")
        ax[1].legend()
        
        plt.show()
    
        
    def plot_history(self, X=None, y=None, based_on_history=False):
        # two options for plotting:
        # - based on the existing history
        # - based on new calculations for new data
        if not based_on_history:
            self.make_plot_new_data(X, y)
        else:
            self.make_plot_history()
