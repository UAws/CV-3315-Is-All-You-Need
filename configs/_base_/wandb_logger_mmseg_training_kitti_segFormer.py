customized_log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook',
             init_kwargs={
                 'entity': 'ak6',
                 'project': 'mmseg_training_kitti_segFormer',
             },
             out_suffix=('.log', '.log.json', '.pth', '.py'),
             commit=False
             ),
        # dict(type='TensorboardLoggerHook')
    ])