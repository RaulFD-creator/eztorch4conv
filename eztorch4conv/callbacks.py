# Copyright by Raúl Fernández Díaz

"""
Callbacks module within the eztorch4conv library. It contains the basic class blueprint
for constructing callbacks for the DCNN and MC-DCNN models, and 2 of the most common implementations:
early stopping and checkpoint.
"""

from abc import abstractmethod


class Callback():
    """
    Basic class to easily construct callbacks in eztorch4conv environment.

    Methods
    -------
    check_condition(epoch) : Verifies whether a certain condition has been met

    run(epoch) : Performs a certain action if the condition was met

    Attributes
    ----------
    model : DNN or MCDNN class object
            model for which the callback is created

    metric : str
            Name of the parameter that has to be checked to verify the condition

    target : int or float
            Value the metric has to achieve to satisfy the condition
    """

    def __init__(self, metric, target, model, action=None):
        """
        Instanciates the class object

        Parameters
        ----------
        model : DCNN or MCDCNN class object
                model for which the callback is created

        metric : str
                Name of the parameter that has to be checked to verify the condition

        target : int or float
                Value the metric has to achieve to satisfy the condition            
        """
        self.metric = metric
        self.target = target
        self.model = model

    def check_condition(self, epoch):
        """
        Verifies whether the condition has been met

        Returns
        -------
        True : when the condition has been satisfied

        False : when the condition was not satisfied
        """
        return (self.metric != 'loss' and self.model.history['validate'][self.metric][epoch] >= self.target or
            self.metric == 'loss' and self.model.history['validate'][self.metric][epoch] <= self.target):

    @abstractmethod
    def action(self, epoch):
        """
        Customizable action, that can be introduced to create custom callbacks
        """

    def run(self, epoch):
        """
        Checks whether the condition has been met, and if so performs a predefined
        action.
        """
        if self.check_condition(epoch):
            self.action(epoch)
        
class early_stop(Callback):
    """
    Early stop callback class. Inherits methods from Callback class.
    At the end of an epoch checks whether a condition has been met and if so it saves
    the model and stops the training.

    Methods
    -------
    action() : Saves the model and stops training
    """    
    def action(self, epoch):
        self.model._save_model(epoch, final=True)

    def check_condition(self, epoch):
        """
        Verifies whether the condition has been met

        Returns
        -------
        True : Metric has changed more than target from one
               epoch to the next.

        False : Metric has not changed more than target from
                one epoch to the next.
        """
        if epoch >= 2:
            return self.model.history['train'][self.metric][epoch] - self.model.history['train'][self.metric][epoch-1] > self.target
        

class checkpoint(early_stop):
    """
    Checkpoint callbcak class. Inherits methods from Callback class.
    At the end of an epoch checks whether a condition has been met and if so it saves
    the model without stopping training.

    Methods
    -------
    action() : Saves the model
    """    
    def action(self, epoch):
        self.model._save_model(epoch, final=False)
