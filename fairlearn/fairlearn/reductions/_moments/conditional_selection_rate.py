# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from .moment import ClassificationMoment
from .moment import _GROUP_ID, _LABEL, _PREDICTION, _ALL, _EVENT, _SIGN, _SW
from .error_rate import ErrorRate

_DIFF = "diff"
_WPREDICTION = "wprediction"

class ConditionalSelectionRate(ClassificationMoment):
    """Generic fairness metric including DemographicParity and EqualizedOdds"""

    def default_objective(self):
        return ErrorRate()

    def load_data(self, X, y, sw, event=None, **kwargs):
        super().load_data(X, y, sw, **kwargs)
        self.tags[_EVENT] = event
        self.prob_event = self.tags.groupby(_EVENT)[_SW].sum() / self.sw.sum()
        self.prob_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID])[_SW].sum() / self.sw.sum()
        signed = pd.concat([self.prob_group_event, self.prob_group_event],
                           keys=["+", "-"],
                           names=[_SIGN, _EVENT, _GROUP_ID])
        self.index = signed.index
        self.default_objective_lambda_vec = None
        

        # fill in the information about the basis
        event_vals = self.tags[_EVENT].dropna().unique()
        group_vals = self.tags[_GROUP_ID].unique()
        self.pos_basis = pd.DataFrame()
        self.neg_basis = pd.DataFrame()
        self.neg_basis_present = pd.Series()
        zero_vec = pd.Series(0.0, self.index)
        i = 0
        for event_val in event_vals:
            for group in group_vals[:-1]:
                self.pos_basis[i] = 0 + zero_vec
                self.neg_basis[i] = 0 + zero_vec
                self.pos_basis[i]["+", event_val, group] = 1
                self.neg_basis[i]["-", event_val, group] = 1
                self.neg_basis_present.at[i] = True
                i += 1
    def gamma(self, predictor):
        """ Calculates the degree to which constraints are currently violated by
        the predictor.
        """
        pred = predictor(self.X)
        self.tags[_PREDICTION] = pred
        self.tags[_WPREDICTION] = self.tags[_PREDICTION].multiply(self.sw)
        expect_event = self.tags.groupby(_EVENT).mean()
        expect_event[_WPREDICTION] = expect_event[_WPREDICTION]/expect_event[_SW]
        expect_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).mean()
        expect_group_event[_WPREDICTION] = expect_group_event[_WPREDICTION]/expect_group_event[_SW]
        expect_group_event[_DIFF] = expect_group_event[_WPREDICTION] - expect_event[_WPREDICTION]

        g_unsigned = expect_group_event[_DIFF]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+", "-"],
                             names=[_SIGN, _EVENT, _GROUP_ID])
        self._gamma_descr = str(expect_group_event[[_WPREDICTION, _DIFF]])
        return g_signed

    # TODO: this can be further improved using the overcompleteness in group membership
    def project_lambda(self, lambda_vec):
        lambda_pos = lambda_vec["+"] - lambda_vec["-"]
        lambda_neg = -lambda_pos
        lambda_pos[lambda_pos < 0.0] = 0.0
        lambda_neg[lambda_neg < 0.0] = 0.0
        lambda_projected = pd.concat([lambda_pos, lambda_neg],
                                     keys=["+", "-"],
                                     names=[_SIGN, _EVENT, _GROUP_ID])
        return lambda_projected

    def signed_weights(self, lambda_vec):
#         lambda_signed = lambda_vec["+"] - lambda_vec["-"]
#         adjust = lambda_signed.sum(level=_EVENT) / self.prob_event \
#             - lambda_signed / self.prob_group_event
#         signed_weights = self.tags.apply(
#             lambda row: adjust[row[_EVENT], row[_GROUP_ID]], axis=1
#         )
#         return signed_weights
        lambda_event = (lambda_vec["+"] - lambda_vec["-"]).sum(level=_EVENT) / \
            self.prob_event
        lambda_group_event = (lambda_vec["+"] - lambda_vec["-"]) / \
            self.prob_group_event
        adjust = lambda_event - lambda_group_event
        signed_weights = self.tags.apply(
            lambda row: 0 if pd.isna(row[_EVENT]) else adjust[row[_EVENT], row[_GROUP_ID]], axis=1
        )
#         utility_diff = self.utilities[:, 1] - self.utilities[:, 0]
#         signed_weights = utility_diff.T * signed_weights
        return signed_weights


# Ensure that ConditionalSelectionRate shows up in correct place in documentation
# when it is used as a base class
ConditionalSelectionRate.__module__ = "fairlearn.reductions"


class   DemographicParity(ConditionalSelectionRate):
    """ Demographic parity
    A classifier h satisfies DemographicParity if
    Prob[h(X) = y' | A = a] = Prob[h(X) = y'] for all a, y'
    """
    short_name = "DemographicParity"

    def load_data(self, X, y, sw, **kwargs):
        super().load_data(X, y, sw, event=_ALL, **kwargs)


class EqualizedOdds(ConditionalSelectionRate):
    """ Equalized odds
    Adds conditioning on label compared to Demographic parity, i.e.
    Prob[h(X) = y' | A = a, Y = y] = Prob[h(X) = y' | Y = y] for all a, y, y'
    """
    short_name = "EqualizedOdds"

    def load_data(self, X, y, sw, **kwargs):
        super().load_data(X, y, sw,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(y)),
                          **kwargs)

class EqualOpportunity(ConditionalSelectionRate):
    """ Equal Opportunity
    Adds conditioning on label compared to Demographic parity, i.e.
    Prob[h(X) = y' | A = a, Y = y] = Prob[h(X) = y' | Y = y] for all a, y=1, y'
    """
    short_name = "EqualOpportunity"

    def load_data(self, X, y, sw, **kwargs):
        super().load_data(X, y, sw,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(y)).where(y == 1),
                          **kwargs)

