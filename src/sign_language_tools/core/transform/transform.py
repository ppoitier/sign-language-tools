from sklearn.base import BaseEstimator, TransformerMixin


class Transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        pass
