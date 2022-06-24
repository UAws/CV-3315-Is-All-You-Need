_base_ = [
    '../_base_/models/basicnet.py',
    '../_base_/datasets/kitti_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# model = dict(
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b0.pth')),
#     test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# optimizer

optimizer_config = dict()
optimizer = dict(
    _delete_=True,
    type='Adam',
    lr=3e-4,
)

lr_config = dict(
    _delete_=True,
    policy='fixed',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=3e-4,
    by_epoch=False)

# optimizer
cudnn_benchmark = True
runner = dict(
    _delete_=True,
    type='EpochBasedRunner', max_epochs=1000)
# # workflow = [('train', 10), ('val', 1)]
evaluation = dict(interval=20, metric='mIoU', pre_eval=True, save_best='mIoU')
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
