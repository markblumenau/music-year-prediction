class Logger:
    def __init__(self):
        pass

    def update(self, metric: str = None, value: float = 0.0):
        print(metric, value)
