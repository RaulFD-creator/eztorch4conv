# Copyright by Raúl Fernández Díaz

"""Callbacks: utilities called at certain points during model training."""

from abc import abstractmethod


class Callback():
    """
    Basic wrapper to easily wrap callbacks in eztorch4conv environment.

    Methods
    -------
    __init__() : Creates an instance of the class

    check_condition() : Verifies whether a certain condition has been met

    run() : Performs a certain action if the condition was met

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
        if (self.metric != 'loss' and self.model.history['validate'][self.metric][epoch] >= self.target or
            self.metric == 'loss' and self.model.history['validate'][self.metric][epoch] <= self.target):
            return True
        elif self.metric == 'epochs':
                if epoch % self.target == 0:
                        return True
        else:
            return False

    @abstractmethod
    def action(self):
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
    Wrapper for an early stop callback class. Inherits methods from Callback class.
    At the end of an epoch checks whether a condition has been met and if so it saves
    the model.

    Methods
    -------
    action() : Saves the model and stops training
    """    
    def action(self, epoch):
        self.model._save_model(epoch, final=True)

class checkpoint(early_stop):
    """
    Wrapper for a checkpoint callbcak class. Inherits methods from Callback class.
    At the end of an epoch checks whether a condition has been met and if so it saves
    the model.

    Methods
    -------
    action() : Saves the model
    """    
    def action(self, epoch):
        self.model._save_model(epoch, final=False)
