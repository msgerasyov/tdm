class MetricMeter:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.values = []

    def update(self, metric_value):
        pass

    def reset(self):
        pass