import numpy as np
from operator import itemgetter

class MarginalScenario():
    """
    A marginal scenario

    Args:
        outcomes: list of cardinality of the variables (sets self.variables to length of this list)
        contexts: list of list of indices pointing to subsets of outcomes
    """
    def __init__(self, outcomes, contexts):
        self.num_variables = len(outcomes)
        self.outcomes = outcomes
        for context in contexts:
            for index in context:
                if index > self.num_variables:
                    raise ValueError("Index {} in context {} must be less than the number of variables {}.".format(index,context,self.num_variables))
        self.contexts = contexts

    def display(self, variable_labels):
        """
        Displays the marginal scenario in a readable format.

        Args:
            variable_labels: list of labels for each of the labels (same order as self.outcomes)
        """
        assert len(variable_labels) == self.num_variables, "Number of variable labels must equal the number of variables: {}".format(self.num_variables)
        return ', '.join(list('{{{}}}'.format(', '.join(itemgetter(*context)(variable_labels))) for context in self.contexts))