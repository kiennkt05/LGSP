import abc
from dataloader.data_utils import *

from utils import Timer


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = [0] * args.sessions
        self.trlog['max_acc'] = [0.0] * args.sessions
        self.trlog['novel_acc'] = [0.0] * (args.sessions - 1)

    @abc.abstractmethod
    def train(self):
        pass