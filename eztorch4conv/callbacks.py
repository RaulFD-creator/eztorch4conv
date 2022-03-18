class early_stop():

    def __init__(self, metric, target):
        self.metric = metric
        self.target = target

    def check_condition(self, params):

        if self.metric != 'loss' and params[self.metric] >= self.target:
            return True

        elif self.metric == 'loss' and params[self.metric] <= self.target:
            return True

        else:
            return False

    def __str__(self):
        return 'early_stop'

class checkpoint(early_stop):

    def __str__(self):
        return 'checkpoint'