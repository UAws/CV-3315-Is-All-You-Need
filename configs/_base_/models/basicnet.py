norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BasicSegNet',
        n_class=19
    ),
    decode_head=dict(
        type='BasicDecoder',
        num_classes=19,
        n_class=19,
        input_size=(256, 864),
        in_channels=19,
        channels=256,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)

    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
