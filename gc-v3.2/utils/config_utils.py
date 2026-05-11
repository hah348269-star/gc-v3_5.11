import os


class ConfigUtils:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_data_path(self):
        return os.path.join(self.base_dir, 'data')

    def get_model_path(self):
        return os.path.join(self.base_dir, 'data', 'models')

    def get_log_path(self):
        return os.path.join(self.base_dir, 'logs')