"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
import numpy as np

from .base import BaseClassificationDecider

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import type_of_target
import ardent

class SimpleArgmaxAverage(BaseClassificationDecider):
    """
    Doc string here.
    """

    def __init__(self, classes=[]):
        self.classes = classes
        self._is_fitted = False

    def fit(
        self,
        X,
        y,
        transformer_id_to_transformers,
        transformer_id_to_voters,
        classes=None,
    ):
        if not isinstance(self.classes, (list, np.ndarray)):
            if len(y) == 0:
                raise ValueError(
                    "Classification Decider classes undefined with no class labels fed to fit"
                )
            else:
                self.classes = np.unique(y)
        else:
            self.classes = np.array(self.classes)
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        self._is_fitted = True
        return self

    def predict_proba(self, X, transformer_ids=None):
        vote_per_transformer_id = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):  
            if not self.is_fitted():
                msg = (
                    "This %(name)s instance is not fitted yet. Call 'fit' with "
                    "appropriate arguments before using this decider."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})

            vote_per_bag_id = []
            for bag_id in range(
                len(self.transformer_id_to_transformers[transformer_id])
            ):
                transformer = self.transformer_id_to_transformers[transformer_id][
                    bag_id
                ]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                vote = voter.predict_proba(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict_proba_reg(self, X, transformer_ids=None):
        # if apply lddmm registration (only for 2-class)
        # for cases where not considering cross-task posteriors

        vote_per_transformer_id = []

        # here transformer_ids be a list of a single task id
        # reference_id will be another task id that's not transformer_id
        transformer_id = transformer_ids[0]
        reference_id = [i for i in [0, 1] if i != transformer_id][0]

        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this decider."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
        
        vote_per_bag_id_ref = []
        for bag_id in range(
            len(self.transformer_id_to_transformers[reference_id])
        ):
            transformer = self.transformer_id_to_transformers[reference_id][
                bag_id
            ]
            X_transformed = transformer.transform(X)
            voter = self.transformer_id_to_voters[reference_id][bag_id]
            vote = voter.predict_proba(X_transformed)
            vote_per_bag_id_ref.append(vote)

        vote_per_bag_id = []
        for bag_id in range(
            len(self.transformer_id_to_transformers[transformer_id])
        ):
            transformer = self.transformer_id_to_transformers[transformer_id][
                bag_id
            ]
            X_transformed = transformer.transform(X)
            voter = self.transformer_id_to_voters[transformer_id][bag_id]
            vote = voter.predict_proba(X_transformed)
            
            # lddmm
            transform = ardent.Transform()
            reference = vote_per_bag_id_ref[bag_id]
            moving = vote
            transform.register(target=moving, template=reference)
            vote = transform.transform_image(
                subject=moving,
                output_shape=moving.shape,
                deform_to='template')

            vote_per_bag_id.append(vote)
        
        vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict(self, X, transformer_ids=None, registration=False):
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this decider."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        if registration is False:
            vote_overall = self.predict_proba(X, transformer_ids=transformer_ids)
        else:
            vote_overall = self.predict_proba_reg(X, transformer_ids=transformer_ids)
        return vote_overall

    def is_fitted(self):
        """
        Doc strings here.
        """
        return self._is_fitted
