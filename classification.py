import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from visdom import Visdom
from tqdm import tqdm

from networks import Coder
from datasets import MetaBatchEffect
import transfer as T
from metrics import Loss, Accuracy


class OrdinalCls:
    def __init__(
        self, in_f, out_f, batch_size, epoch, lr=0.001,
        device=torch.device("cuda:0")
    ):
        self.model = Coder(in_f, out_f, 500, block_num=2, norm='BN').to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.bs = batch_size
        self.epoch = epoch
        self.device = device
        self.vis = Visdom()

    def fit(self, dats):
        dataloaders = {
            key: data.DataLoader(dat, batch_size=self.bs, shuffle=True)
            for key, dat in dats.items()
        }
        self.history = {key: {'loss': [], "acc": []} for key in dats.keys()}
        for e in tqdm(range(self.epoch)):
            for phase, dataloader in dataloaders.items():
                loss_obj = Loss()
                acc_obj = Accuracy(proba2int=True)
                if phase == 'train':
                    self.model.train()
                    sequence = tqdm(dataloader)
                else:
                    self.model.eval()
                    sequence = dataloader
                for batch_x, batch_y in sequence:
                    batch_x = batch_x.to(self.device, torch.float)
                    batch_y = batch_y.to(self.device, torch.long).squeeze(1)
                    with torch.set_grad_enabled(phase == 'train'):
                        logit = self.model(batch_x)
                        loss = self.criterion(logit, batch_y)
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        loss_obj.add(loss, batch_x.size(0))
                        acc_obj.add(logit, batch_y)

                self.history[phase]['loss'].append(loss_obj.value())
                self.history[phase]['acc'].append(acc_obj.value())
            self.vis.line(
                X=[e], Y=[[
                    self.history[phase]['loss'][-1]
                    for phase in dats.keys()
                ]],
                update=None if e == 0 else 'append', win='loss',
                opts={'title': "Loss", 'legend': list(dats.keys())}
            )
            self.vis.line(
                X=[e], Y=[[
                    self.history[phase]['acc'][-1]
                    for phase in dats.keys()
                ]],
                update=None if e == 0 else 'append', win='acc',
                opts={'title': "Accuracy", 'legend': list(dats.keys())}
            )
        return self.history


def main():
    from config import Config

    # config
    config = Config()
    config.show()

    # ----- 读取数据 -----
    meta_data = MetaBatchEffect.from_csv(
        config.sample_file, config.meta_file, 'batch',
        pre_transfer=T.MultiCompose(
            T.Normalization(), lambda x, y: (x, y - 1)
        )
    )
    subject_dat, qc_dat = meta_data.split_qc()
    train_dat, test_dat = subject_dat.split(0.2)

    # ----- 训练网络 -----
    estimator = OrdinalCls(train_dat.num_features, 4, 64, 100)
    hist = estimator.fit({'train': train_dat, 'test': test_dat, 'qc': qc_dat})
    print(hist)


if __name__ == '__main__':
    main()
