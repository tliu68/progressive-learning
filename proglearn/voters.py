'''
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
'''
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KernelDensity

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import check_classification_targets

from .base import BaseVoter


class EmpiricalTreeClassificationVoter(BaseVoter):
    def __init__(self, finite_sample_correction=False):
        """
        Doc strings here.
        """

        self.finite_sample_correction = finite_sample_correction
        self._is_fitted = False

    def fit(self, X, y, transformer=None):
        """
        Doc strings here.
        """
        check_classification_targets(y)
        
        self.transformer = transformer
        X = self.transformer.transform(X)

        num_classes = len(np.unique(y))
        self.uniform_posterior = np.ones(num_classes) / num_classes

        self.leaf_to_posterior = {}

        for leaf_id in np.unique(X):
            idxs_in_leaf = np.where(X == leaf_id)[0]
            class_counts = [
                len(np.where(y[idxs_in_leaf] == y_val)[0]) for y_val in np.unique(y)
            ]
            posteriors = np.nan_to_num(np.array(class_counts) / np.sum(class_counts))

            if self.finite_sample_correction:
                posteriors = self._finite_sample_correction(
                    posteriors, len(idxs_in_leaf), len(np.unique(y))
                )

            self.leaf_to_posterior[leaf_id] = posteriors

        self._is_fitted = True

        return self

    def vote(self, X):
        """
        Doc strings here.
        """

        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this voter."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
            
        X = self.transformer.transform(X)

        votes_per_example = []
        for x in X:
            if x in list(self.leaf_to_posterior.keys()):
                votes_per_example.append(self.leaf_to_posterior[x])
            else:
                votes_per_example.append(self.uniform_posterior)
        return np.array(votes_per_example)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def _finite_sample_correction(posteriors, num_points_in_partition, num_classes):
        """
        encourage posteriors to approach uniform when there is low data
        """
        correction_constant = 1 / (num_classes * num_points_in_partition)

        zero_posterior_idxs = np.where(posteriors == 0)[0]
        posteriors[zero_posterior_idxs] = correction_constant

        posteriors /= sum(posteriors)

        return posteriors
    
class KDETreeClassificationVoter(BaseVoter):
    def __init__(self, finite_sample_correction=False):
        """
        Doc strings here.
        """
        self._is_fitted = False

    def fit(self, X, y, transformer):
        """
        Doc strings here.
        """
        check_classification_targets(y)
        _, y = np.unique(y, return_inverse = True)
        
        X_nodes = transformer.transform(X)
        
        #an array of all of the X-values of the class-wise means in each leaf
        leaf_means_X = []
        
        #an array of all of the y-values (i.e. class values) of the class-wise means in each leaf
        leaf_means_y = []
        
        #an array of all of the number of points that comprise the means in each leaf
        leaf_means_weight = []
        
        #an array of all of the average variances of the points in each leaf corresponding to 
        #a single class from the class-wise mean in that leaf
        leaf_means_var = []

        #loop over all of the given leaf nodes
        for leaf_id in np.unique(X_nodes):
            #identify all of the y values in that leaf 
            y_vals_in_leaf = np.unique(y[np.where(X_nodes == leaf_id)[0]])
            #loop over these y values
            for y_val_in_leaf in y_vals_in_leaf:
                #gather all the points of that given y value in that leaf
                idxs_of_y_val_in_leaf = np.where((y == y_val_in_leaf) & (X_nodes == leaf_id))[0]
                #compute the mean X-value of the points in that leaf corresponding to that y value
                #and append to the aggregate array
                leaf_mean_X = np.mean(X[idxs_of_y_val_in_leaf], axis = 0)
                leaf_means_X.append(leaf_mean_X)
                #we already know the y value, so just append it 
                leaf_means_y.append(int(y_val_in_leaf))
                #compute the number of points in that leaf corresponding to that y value
                #and append to the aggregate array
                leaf_means_weight.append(len(idxs_of_y_val_in_leaf))
                #compute the distances of all the points in that leaf corresponding to that y value
                #from the mean X-value of the points in that leaf corresponding to that y value
                dists_in_leaf = np.sqrt(np.sum((X[idxs_of_y_val_in_leaf] - leaf_mean_X)**2, axis = 1))
                #compute the variance as the average distance of the class-wise points in that leaf 
                #and append to the aggregate array
                leaf_means_var.append(np.mean(dists_in_leaf))
                
        #convert to numpy array so we can easily refer to multiple sets of points as ra[idxs]
        leaf_means_y, leaf_means_X, leaf_means_weight = np.array(leaf_means_y), np.array(leaf_means_X), np.array(leaf_means_weight)
        #compute the bandwidth as the average variance across the class-wise leaf means, weighted
        #by the number of points
        bandwidth=np.average(leaf_means_var, weights = leaf_means_weight)
              
        #an array of all of the KDEs. Each KDE will be respondsible for computing the probability 
        #that a given set of inference points belongs to that class. Thus, we have separate KDEs 
        #for each class. The KDE at index i is the KDE that is responsible for computations on 
        #y_value = i
        self.KDEs = []
        #loop over the y values in the leaf means 
        for y_val in np.unique(leaf_means_y):
            #gather all of the leaf means corresponding to the given y value
            leaf_means_X_of_y_val = leaf_means_X[np.where(leaf_means_y == y_val)[0]]
            #father all of the weights corresponding to the given y value
            leaf_means_weight_of_y_val = leaf_means_weight[np.where(leaf_means_y == y_val)[0]]
            #train an sklearn KDE with a gaussian kernel with the bandwidth calculated above
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            #fit the KDE on the leaf means corresponding to the given y value, weighted by
            #the weights corresponding to the given y value
            kde.fit(X = leaf_means_X_of_y_val, sample_weight = leaf_means_weight_of_y_val)
            #append the KDE to the aggregate array
            self.KDEs.append(kde)
        
        #we are done fitting
        self._is_fitted = True

        return self

    def vote(self, X):
        """
        Doc strings here.
        """

        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this voter."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
        
        #instantiate the posterior predictions
        y_proba = np.zeros((len(X), len(self.KDEs)))
        
        #loops over the KDEs (and thus, implicitly, the y-values)
        for y_val in range(len(self.KDEs)):
            #compute the (unnormalized) posterior for the KDE corresponding
            #to y_val. NOTE: we perform np.exp since the score_samples 
            #function returns the log of the probability of belonging to that 
            #class, so we must invert the log through an exponential
            y_proba[:, y_val] = np.exp(self.KDEs[y_val].score_samples(X))
            
        #normalize the posteriors per example so the posterior per example
        #sums to 1
        y_proba_sum_per_idx = np.sum(y_proba, axis = 1)
        for y_val in range(np.shape(y_proba)[1]):
            y_proba[:, y_val] /= y_proba_sum_per_idx
            
        return y_proba

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

class KNNClassificationVoter(BaseVoter):
    def __init__(self, k=None, kwargs={}):
        """
        Doc strings here.
        """
        self._is_fitted = False
        self.k = k
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Doc strings here.
        """
        X, y = check_X_y(X, y)
        k = int(np.log2(len(X))) if self.k == None else self.k
        self.knn = KNeighborsClassifier(k, **self.kwargs)
        self.knn.fit(X, y)
        self._is_fitted = True

        return self

    def vote(self, X):
        """
        Doc strings here.
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)
        return self.knn.predict_proba(X)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

