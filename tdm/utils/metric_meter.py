class MetricMeter:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.values = []

    def update(self, metric_value):
        self.values.append(metric_value)

    def reset(self):
        self.values = []

    def get_value(self):
        return sum(self.values) / len(self.values)
    
    def __str__(self) -> str:
        return f'Average {self.metric_name}: {self.get_value():.4f}'