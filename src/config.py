from argparse import Namespace, ArgumentParser
from sklearn import base, linear_model, naive_bayes, neighbors, svm, tree
import yaml
import os


def yaml_to_namespace(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return Namespace(**config)


class BaseConfiguration(Namespace):
    def __init__(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        if parser is not None:
            self.args = parser.parse_args()
            super().__init__(**vars(self.args))
        elif yaml_file:
            self.args = yaml_to_namespace(yaml_file)
        elif kwargs:
            self.args = Namespace(**kwargs)

        super().__init__(**vars(self.args))

    def set_configuration(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        self.__init__(parser, yaml_file, **kwargs)

    def reset_configuration(self):
        self.__init__(**vars(self.args))


class ModelConfiguration(BaseConfiguration):
    def __init__(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        super.__init__(parser, yaml_file, **kwargs)

    def create_model(self):
        raise NotImplementedError(
            "`create_model()` method is not implemented. Please implement it in your subclass."
        )


class SKLearnModelConfiguration:
    SKLEARN_MODELS = [linear_model, naive_bayes, neighbors, svm, tree]

    def __init__(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        super.__init__(parser, yaml_file, **kwargs)
        self.model: base.BaseEstimator = self.create_model()
        self._base_model = getattr(self.model_name)

    def __get_base_model(self):
        for attr in self.SKLEARN_MODELS:
            return getattr(attr, self.model_name)

    def create_model(self):
        self.model = self._base_model(self.__get_base_model)
