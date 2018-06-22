import yaml


class Config(object):

    def __init__(self, config):
        self.config_file = config
        with open(config) as f:
            self.configs = yaml.load(f)

    def config_file(self):
        return self.config_file

    def save_config(self):
        pass

    def load_config(self):
        pass
