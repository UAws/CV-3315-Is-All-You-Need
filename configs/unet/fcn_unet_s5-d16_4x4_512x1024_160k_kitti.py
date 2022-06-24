_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/kitti_seg_basic.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    decode_head=dict(num_classes=19),
    auxiliary_head=dict(num_classes=19),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer_config = dict()
# optimizer = dict(
#     _delete_=True,
#     type='Adam',
#     lr=3e-4,
# )
#
# lr_config = dict(
#     _delete_=True,
#     policy='fixed',
#     warmup='linear',
#     warmup_iters=1,
#     warmup_ratio=3e-4,
#     by_epoch=False)

# runner = dict(
#     _delete_=True,
#     type='EpochBasedRunner', max_epochs=6000)
workflow = [('train', 1600), ('val', 1)]
# evaluation = dict(interval=200, metric='mIoU', pre_eval=True, save_best='mIoU')
evaluation = dict(save_best='mIoU',interval=1600)
# checkpoint_config = dict(by_epoch=True, interval=100)
# data = dict(samples_per_gpu=24, workers_per_gpu=4)
# runner = dict(type='IterBasedRunner',
#               max_iters=2000)
# workflow = [('train', 2000), ('val', 1)]
# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
# checkpoint_config = dict(by_epoch=False, interval=4000)
data = dict(samples_per_gpu=4, workers_per_gpu=4)

# trace_config = dict(type='tb_trace', dir_name='work_dir')
# profiler_config = dict(on_trace_ready=trace_config)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook',
             init_kwargs={
                 'entity': 'ak6',
                 'project': 'mmseg_training_kitti_segFormer',
             },
             out_suffix=('.log', '.log.json', '.pth', '.py')
             ),
        # dict(type='TensorboardLoggerHook')
    ])
