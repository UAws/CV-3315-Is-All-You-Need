_base_ = [
    '../_base_/models/segformer_swin.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py','../_base_/schedules/schedule_160k.py',
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
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=19),
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# optimizer

val_interval = 1600

# runner = dict(
#     _delete_=True,
#     type='EpochBasedRunner', max_epochs=100)
workflow = [('train', val_interval), ('val', 1)]
evaluation = dict(interval=val_interval, metric='mIoU', pre_eval=True, save_best='mIoU')
checkpoint_config = dict(_delete_=True)
data = dict(samples_per_gpu=2, workers_per_gpu=4)

# print(_base_)

log_config={{_base_.customized_log_config}}



