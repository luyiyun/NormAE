import os

import torch
import torch.utils.data as data
import progressbar as pb
import pandas as pd
import argparse

from datasets import MetaBatchEffect
import transfer as T


def predict(models, dataloader, device=torch.device('cuda:0')):
    results = []
    for batch_x, batch_y in pb.progressbar(dataloader):
        batch_x = batch_x.to(device, torch.float)
        batch_y = batch_y.to(device, torch.float)
        with torch.no_grad():
            no_batch_num = models['discriminator'].in_f
            hidden = models['encoder'](batch_x)
            hidden[:, no_batch_num:] = 0.
            batch_x_recon = models['decoder'](hidden)
            results.append(batch_x_recon)
    results = torch.cat(results, dim=0)
    return results


def main():
    # config
    from config import Config
    config = Config(True)

    # ----- 读取数据 -----
    meta_data = MetaBatchEffect.from_csv(
        config.sample_file, config.meta_file, ['injection.order', 'batch'],
        pre_transfer=T.Normalization()
    )
    dataloader = data.DataLoader(
        meta_data, batch_size=config.args.batch_size,
        num_workers=config.args.num_workers, shuffle=False
    )

    # ----- 读取训练好的网络 -----
    models = torch.load(os.path.join(config.args.dir, 'models.pth'))

    # ----- 得到预测的结果 -----
    results = predict(models, dataloader)
    print('')

    # 保存结果
    df_ori = meta_data.X_df_trans
    df_res = pd.DataFrame(
        results.cpu().numpy(), index=df_ori.index, columns=df_ori.columns)
    df_ori['index'] = 'ori'
    df_res['index'] = 'res'
    df = pd.concat([df_ori, df_res])
    df.to_csv(os.path.join(config.args.dir, 'result_data.csv'))


if __name__ == "__main__":
    main()
