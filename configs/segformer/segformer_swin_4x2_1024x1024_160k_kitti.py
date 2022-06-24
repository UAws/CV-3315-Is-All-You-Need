_base_ = [
    '../_base_/models/segformer_swin.py',
    '../_base_/datasets/kitti_seg_basic.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
    '../_base_/wandb_logger_mmseg_training_kitti_segFormer.py'
]

checkpoint_file = 'checkpoints/swin_tiny_patch4_window7_224_22k.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        in_channels=[96, 192, 384, 768], num_classes=19,
        sampler=dict(type='OHEMPixelSampler', min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True,
            # Kitti
            class_weight=[0.74928016, 0.84599227, 0.80983893, 0.94461, 0.94403714, 0.91394077,
                          1.03971502, 0.94605021, 0.73242514, 0.79119748, 0.78673501, 1.14044977,
                          1.30447229, 0.81121395, 1.11445713, 1.18535591, 0.95353713, 1.72195258,
                          1.2647391])
    ),
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
    warmup_iters=5000,
    warmup_ratio=1e-5,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# optimizer
crop_size = (864, 256)

val_interval = 500

runner = dict(type='IterBasedRunner', max_iters=20000)

# runner = dict(
#     _delete_=True,
#     type='EpochBasedRunner', max_epochs=100)
workflow = [('train', val_interval), ('val', 1)]
evaluation = dict(interval=val_interval, metric='mIoU', pre_eval=True, save_best='mIoU')
checkpoint_config = dict(_delete_=True)
data = dict(samples_per_gpu=2, workers_per_gpu=4)

# workflow = [('train', 4000), ('val', 1)]
# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
# checkpoint_config = dict(by_epoch=False, interval=4000)
# data = dict(samples_per_gpu=2, workers_per_gpu=4)


log_config = {{_base_.customized_log_config}}
