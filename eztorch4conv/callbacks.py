# Copyright by Raúl Fernández Díaz

"""Callbacks: utilities called at certain points during model training."""

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
    model : DCNN or MCDCNN class object
            model for which the callback is created

    metric : str
            Name of the parameter that has to be checked to verify the condition

    target : int or float
            Value the metric has to achieve to satisfy the condition

    action : function object
            Function that describes the action the callback is supposed to perform
            when the condition has been met
    """

    def __init__(self, model, metric, target, action=None):
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
            
        action : function object
                Function that describes the action the callback is supposed to perform
                when the condition has been met
        """
        self.model = model
        self.metric = metric
        self.target = target
        self.action = action

    def check_condition(self):
        """
        Verifies whether the condition has been met

        Returns
        -------
        True : when the condition has been satisfied

        False : when the condition was not satisfied
        """
        if (self.metric != 'loss' and self.model.params[self.metric] >= self.target or
            self.metric == 'loss' and self.model.params[self.metric] <= self.target):
            return True

        else:
            return False

    def run(self):
        """
        Checks whether the condition has been met, and if so performs a predefined
        action.
        """
        if self.check_condition():
            self.action()
        
class early_stop(Callback):
    """
    Wrapper for an early stop callback class. Inherits methods from Callback class.
    At the end of an epoch checks whether a condition has been met and if so it saves
    the model.

    Methods
    -------
    action() : Saves the model and stops training
    """    
    def action(self):
        self.model.save_model(final=True)

    

class checkpoint(early_stop):
    """
    Wrapper for a checkpoint callbcak class. Inherits methods from Callback class.
    At the end of an epoch checks whether a condition has been met and if so it saves
    the model.

    Methods
    -------
    action() : Saves the model
    """    
    def action(self):
        self.model.save_model(final=False)
