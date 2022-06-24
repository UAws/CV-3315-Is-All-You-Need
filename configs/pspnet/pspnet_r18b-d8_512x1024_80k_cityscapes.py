_base_ = './pspnet_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    test=dict(
        img_dir='leftImg8bit/test',
        ann_dir='gtFine/test'
    ),
)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
runner = dict(type='IterBasedRunner', max_iters=10)
workflow = [('train', 100), ('val', 1)]
evaluation = dict(interval=5, metric='mIoU', pre_eval=True, save_best='mIoU')

