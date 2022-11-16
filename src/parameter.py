import torch


class Parameter(object):
    def __init__(self):
        # data
        self.result_dir = './user_data/'
        self.data_dir = '../input/AI4Code/'
        self.k_folds = 5
        self.n_jobs = 4
        self.random_seed = 27
        self.seq_length = 512
        self.cell_count = 128
        self.cell_max_length = 128
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model
        self.use_cuda = torch.cuda.is_available()
        self.gpu = 0
        self.print_freq = 100
        self.lr = 0.003
        self.weight_decay = 0
        self.optim = 'Adam'
        self.base_epoch = 30

    def get(self, name):
        return getattr(self, name)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    parameter = Parameter()
    print(parameter)
