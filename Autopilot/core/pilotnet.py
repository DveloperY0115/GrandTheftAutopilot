class pilotNet:

    def __init__(self):
        """
        Create and initialize pilotNet for vehicle control
        """
        return

    def __call__(self, *args, **kwargs):
        """
        An alternative to call function 'predict' via Python special method
        :param args:
        :param kwargs:
        :return:
        """
        return self.predict(args, kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict the steering angle based on the input image
        :param args:
        :param kwargs:
        :return:
        """
        return