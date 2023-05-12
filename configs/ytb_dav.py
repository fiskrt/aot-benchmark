import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'YTB_DAV'
        self.DATASETS = ['youtubevos', 'davis2017']

        self.init_dir()
