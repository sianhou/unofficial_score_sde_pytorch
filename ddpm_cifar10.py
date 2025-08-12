import copy
import os

import torch
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from tqdm import trange

from gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from unet import Unet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class config:
    batch_size = 128
    num_workers = 4
    nf = 128
    ch_mult = (1, 2, 2, 2)
    num_res_blocks = 2
    attn_resolutions = (False, True, False, False)
    dropout = 0.1
    num_groups = 32
    resamp_with_conv = True
    lr = 0.0002
    warmup = 5000
    beta_1 = 0.0001
    beta_T = 0.02
    T = 1000
    img_size = 32
    var_type = "fixedlarge"
    mean_type = "epsilon"
    logdir = "./logs/DDPM_CIFAR10_EPS"
    sample_size = 64
    save_step = 5000
    total_steps = 800000
    ema_decay = 0.9999
    grad_clip = 1.0
    sample_step = 1000


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


# This function warmup_lr(step) implements a linear learning rate warm-up schedule -
# a common technique in training neural networks,
# especially in deep learning setups like transformers or diffusion models.
# When step <=  warmup, the function returns a value between 0 and 1, increasing linearly.
# When step >  warmup, the function returns 1.0 (i.e., full learning rate).
def warmup_lr(step):
    return min(step, config.warmup) / config.warmup


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def main():
    # dataset
    dataset = CIFAR10(
        root='./data/cifar10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model
    net_model = Unet(in_channels=3, nf=config.nf, ch_mult=config.ch_mult, num_res_blocks=config.num_res_blocks,
                     attn_resolutions=config.attn_resolutions, dropout=config.dropout, num_groups=config.num_groups,
                     resamp_with_conv=config.resamp_with_conv)
    ema_model = copy.deepcopy(net_model).to(device)

    # opt
    optim = torch.optim.Adam(net_model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(net_model, config.beta_1, config.beta_T, config.T).to(device)
    net_sampler = GaussianDiffusionSampler(net_model, config.beta_1, config.beta_T, config.T, config.img_size,
                                           config.mean_type, config.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(ema_model, config.beta_1, config.beta_T, config.T, config.img_size,
                                           config.mean_type, config.var_type).to(device)

    # log setup
    os.makedirs(os.path.join(config.logdir, 'sample'), exist_ok=True)
    x_T = torch.randn(config.sample_size, 3, config.img_size, config.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:config.sample_size]) + 1) / 2
    writer = SummaryWriter(config.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(config.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), config.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, config.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if config.sample_step > 0 and step % config.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(config.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if config.save_step > 0 and step % config.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(config.logdir, 'ckpt.pt'))

            # # evaluate
            # if eval_step > 0 and step % eval_step == 0:
            #     net_IS, net_FID, _ = evaluate(net_sampler, net_model)
            #     ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
            #     metrics = {
            #         'IS': net_IS[0],
            #         'IS_std': net_IS[1],
            #         'FID': net_FID,
            #         'IS_EMA': ema_IS[0],
            #         'IS_std_EMA': ema_IS[1],
            #         'FID_EMA': ema_FID
            #     }
            #     pbar.write(
            #         "%d/%d " % (step, total_steps) +
            #         ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
            #     for name, value in metrics.items():
            #         writer.add_scalar(name, value, step)
            #     writer.flush()
            #     with open(os.path.join(logdir, 'eval.txt'), 'a') as f:
            #         metrics['step'] = step
            #         f.write(json.dumps(metrics) + "\n")
    writer.close()


if __name__ == '__main__':
    main()
