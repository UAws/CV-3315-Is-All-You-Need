_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/kitti_seg_basic.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
    '../_base_/wandb_logger_mmseg_training_kitti_segFormer.py'
]

val_interval = 400

# runner = dict(
#     _delete_=True,
#     type='EpochBasedRunner', max_epochs=1500)
workflow = [('train', val_interval), ('val', 1)]
evaluation = dict(interval=val_interval, metric='mIoU', pre_eval=True, save_best='mIoU')
# evaluation = dict(save_best='mIoU',interval=1600)
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
