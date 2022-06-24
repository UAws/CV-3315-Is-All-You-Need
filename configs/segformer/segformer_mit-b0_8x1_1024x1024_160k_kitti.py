_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/kitti_seg_basic.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
    '../_base_/wandb_logger_mmseg_training_kitti_segFormer.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b0.pth')),
    # test_cfg=dict(mode='slide', crop_size=(368, 368), stride=(245, 245)),
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', min_kept=100000),
        loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True,
                # Kitti
                class_weight=[0.75952312, 0.8523161, 0.80858376, 0.94312681, 0.95249373, 0.91668514,
                              1.02670926, 0.99855901, 0.74849044, 0.8041853, 0.79715573, 1.14388026,
                              1.28290288, 0.82965688, 1.04470749, 1.17929034, 1.32624767, 1.40595936,
                              1.17952672])
    )
)

# optimizer

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-5,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# optimizer
crop_size = (864, 256)

val_interval = 500

# runner = dict(
#     _delete_=True,
#     type='EpochBasedRunner', max_epochs=100)
workflow = [('train', val_interval), ('val', 1)]
evaluation = dict(interval=val_interval, metric='mIoU', pre_eval=True, save_best='mIoU')
runner = dict(type='IterBasedRunner', max_iters=10000)

# runner = dict(type='EpochBasedRunner', max_epochs=100)
# workflow = [('train', 2000), ('val', 1)]
# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
# checkpoint_config = dict(by_epoch=False, interval=4000)
data = dict(samples_per_gpu=8, workers_per_gpu=4)

log_config = {{_base_.customized_log_config}}
# test_cfg = dict(mode='slide', crop_size=368, stride=245)
