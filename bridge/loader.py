import copy
import os
import unittest
import urllib
from os import path
import os.path as osp
import random
from torchviz import make_dot, make_dot_from_trace

import mmcv
import torch
from IPython.core.display import Image
from fvcore.nn import FlopCountAnalysis, flop_count_table
from matplotlib import pyplot as plt
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from torchinfo import summary

from bridge.utils import image_plots
from mmseg import __version__

from bridge.executor import execute_now
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv import Config, get_git_hash
from mmseg.apis import init_segmentor, init_random_seed, set_random_seed, train_segmentor, single_gpu_test
from mmseg.utils import get_device, setup_multi_processes, build_dp


def display_tool(x):
    image_url = x
    filename = "display.jpg"

    req = urllib.request.build_opener()
    req.addheaders = [('User-Agent',
                       'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.1.2 Safari/605.1.15')]
    urllib.request.install_opener(req)

    urllib.request.urlretrieve(image_url, filename)

    return Image(filename="display.jpg")


class GlobalBridgeController:
    class MMSegConfig(object):
        def __init__(self, model_name, config_file, work_dir=None, checkpoint_file=None, checkpoint_url=None):
            self.model_name = model_name
            self.config_file = config_file
            self.checkpoint_file = self.check_checkpoint_download(checkpoint_url) if checkpoint_url else checkpoint_file
            self.checkpoint_url = checkpoint_url
            pass

        def check_checkpoint_download(self, checkpoint_url):
            if not path.exists(osp.join('checkpoints', f'{self.model_name}.pth')):
                mmcv.mkdir_or_exist(osp.join('checkpoints'))
                execute_now('wget -nv ' + checkpoint_url + ' -O checkpoints/' + self.model_name + '.pth')
                return 'checkpoints/' + self.model_name + '.pth'
            else:
                print(f'checkpoint file already exists at {osp.join("checkpoints", f"{self.model_name}.pth")}')
                return 'checkpoints/' + self.model_name + '.pth'

    def __init__(self):
        pass

    def download_data(self):
        # download the data
        if not path.exists("data/kitti-seg-competition/seg_data_processed"):
            execute_now('mkdir -p data/')
            execute_now(f'wget -nv http://vmv.re/E7kMX -O data/kitti-seg-competition.tar.gz')
            execute_now('tar -zxvf data/kitti-seg-competition.tar.gz -C data/')
            execute_now('mv data/kitti-seg-competition-zip data/kitti-seg-competition')
            execute_now('rm -rf data/kitti-seg-competition.tar.gz')

    def convert_data(self):
        pass

    def summarize_network(self, config, image_shape, transformer=False, viz=False):
        cfg = Config.fromfile(config)
        cfg.model.pretrained = None

        model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')).cuda()
        model.eval()
        model.forward = model.forward_dummy

        if not transformer:
            print(summary(model, (1, 3, image_shape[0], image_shape[1])))
        else:
            inputs = (torch.randn((1, 3, image_shape[0], image_shape[1])).to('cuda'),)

            # flops, params = get_model_complexity_info(model, input_shape)
            flops = FlopCountAnalysis(model, inputs)
            print(flop_count_table(flops))
            print('{:<30}  {:<8}'.format('FLOPs: ', flops.total()))

        return

        if viz:
            y = model(inputs[0])
            make_dot(y, params=dict(list(model.named_parameters()) + [('x', inputs[0])]))

            print('{:<30}  {:<8}'.format('FLOPs: ', flops.total()))
            return

    def get_network(self, config):
        cfg = Config.fromfile(config)
        model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')).cuda()
        cfg.log_config['hooks'] = [
            dict(type='TextLoggerHook', by_epoch=False),
        ]
        model.eval()

        model.forward = model.forward_dummy

        return model

    def train_network(self, config, load_from=None):
        cfg = Config.fromfile(config)

        # clean gpu memory when starting a new evaluation.
        torch.cuda.empty_cache()

        # disable log
        cfg.log_config['hooks'] = [
            dict(type='TextLoggerHook', by_epoch=False),
        ]

        # set random seeds
        cfg.device = get_device()
        seed = init_random_seed(None, device=cfg.device)
        set_random_seed(seed)
        cfg.seed = seed

        # set only use on gpu
        cfg.gpu_ids = [0]

        model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        # model.init_weights()

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmseg version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                config=cfg.pretty_text,
                CLASSES=datasets[0].CLASSES,
                PALETTE=datasets[0].PALETTE)

        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES

        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(config))[0])

        meta = dict()

        # passing checkpoint meta for saving best checkpoint
        meta.update(cfg.checkpoint_config.meta)
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=False,
            validate=True,
            meta=meta)

    def evaluation_network(self, config, work_dir, opacity=0.5, load_from=None, multi_scale=False):

        cfg = Config.fromfile(config)

        # disable log
        cfg.log_config['hooks'] = [
            dict(type='TextLoggerHook', by_epoch=False),
        ]

        # set multi-process settings
        # setup_multi_processes(cfg)

        if multi_scale:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

        else:
            cfg.data.test.pipeline[1].img_ratios = None
            cfg.data.test.pipeline[1].flip = False

            if cfg.data.test.pipeline[1].transforms[1].type == 'ResizeToMultiple':
                cfg.data.test.pipeline[1].transforms.remove(cfg.data.test.pipeline[1].transforms[1])

        cfg.gpu_ids = [0]

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        # The default loader config
        loader_cfg = dict(
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=False,
            shuffle=False)
        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.data.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader'
            ]
        })
        test_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('test_dataloader', {})
        }

        data_loader = build_dataloader(dataset, **test_loader_cfg)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

        checkpoint = load_checkpoint(model, load_from, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            print('"CLASSES" not found in meta, use dataset.CLASSES instead')
            model.CLASSES = dataset.CLASSES
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        else:
            print('"PALETTE" not found in meta, use dataset.PALETTE instead')
            model.PALETTE = dataset.PALETTE

        # clean gpu memory when starting a new evaluation.
        torch.cuda.empty_cache()

        cfg.device = get_device()

        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        results = single_gpu_test(
            model,
            data_loader,
            False,
            work_dir,
            False,
            opacity,
            grey_out=True
        )

    def show_results(self, prediction_dir, image_dir='data/kitti-seg-competition/seg_data_processed/img_dir/val',
                     show_all=False):
        """
        Show the results of the prediction.
        """
        # load the prediction results

        original_dataset = sorted(list(
            mmcv.scandir(osp.join(image_dir), recursive=False)))

        prediction_dataset = sorted(list(
            mmcv.scandir(osp.join(prediction_dir), recursive=False)))

        if not show_all:
            # random select three image to show
            original_dataset = sorted(random.sample(original_dataset, 3))

            prediction_dataset = sorted([image for image in prediction_dataset if image in original_dataset])

        display_array = []
        display_cations = []

        for i, pred, orig in zip(range(len(prediction_dataset)), prediction_dataset, original_dataset):
            display_array.append(
                mmcv.imread(osp.join(image_dir, orig), channel_order='rgb'))
            display_cations.append('Original ' + orig)

            display_array.append(
                mmcv.imread(osp.join(prediction_dir, pred)
                            , channel_order='rgb'))
            display_cations.append('Prediction ' + pred)

        # plt.figure(figsize=(20, 60))

        image_plots(display_array, display_cations, len(prediction_dataset), 2)

    def printer(self, txt):
        with open(txt) as f:
            lines = f.readlines()

        for line in lines:
            print(line, end='')


class LoaderTest(unittest.TestCase):

    def test_evaluate(self):
        bridge = GlobalBridgeController()

        baseline_config = bridge.MMSegConfig(
            model_name='baseline',
            config_file='configs/basicNet/basicnet_8x1_1024x1024_180e_kitti.py',
        )

        bridge.evaluation_network(
            baseline_config.config_file,
            work_dir='data/kitti-seg-competition/output',
            load_from='work_dirs/basicnet_8x1_1024x1024_180e_kitti/best_mIoU_epoch_180.pth'
        )

    def test_show_results(self):
        bridge = GlobalBridgeController()

        bridge.show_results(
            'data/kitti-seg-competition/output',
        )
