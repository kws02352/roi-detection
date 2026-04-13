"""
Training configuration for Card/ID ROI Detection Model.

Key design decisions
--------------------
- input_shape (180, 180): balanced between accuracy and mobile inference speed
- patch_size 6: 180/6 = 30×30 = 900 tokens (manageable for attention)
- embed_dim 40: keeps model under 1MB for on-device deployment
- global_attn_type 'linear': O(N·d²) vs softmax O(N²), ONNX-friendly
- crop_min_size_ratio 0.75: corrects Train-Eval area_ratio gap (0.18 → 0.33)
- dist_saturation (0.9, 1.2): corrects 2× saturation gap between Train/Eval
"""


class TrainConfig:
    """
    Training hyperparameters and augmentation settings.

    Augmentation parameters were derived from a full distribution analysis
    of 160,774 training images vs 11,052 eval images using PSI metric.
    """

    def __init__(self):

        # ------------------------------------------------------------------
        # Model architecture
        # ------------------------------------------------------------------
        self.input_shape     = (180, 180, 3)
        self.patch_size      = 6              # token grid: 30×30
        self.embed_dim       = 40             # ~238K params, <1MB ONNX
        self.num_layers      = 8
        self.in_channels     = self.input_shape[2]
        self.output_channels = 3              # card_mask, title_mask, corner_mask
        self.global_attn_type = 'linear'      # 'softmax' | 'linear' | 'performer'
        self.global_num_heads = 4

        # Card categories (11 classes)
        self.card_type = {
            'idcard': 0, 'driver': 1, 'passport': 2,
            'alien': 3, 'alien_back': 4, 'veteran': 5,
            'welfare_A': 6, 'welfare_B': 7,
            'passport_alien': 8, 'credit': 9, 'unknown': 10,
        }
        self.classes     = list(self.card_type.keys())
        self.num_classes = len(self.classes)

        # Output map types (channel definitions)
        self.map_types = [
            ('card',  'cylinder',  1.5),   # ch0: card mask
            ('title', 'cylinder',  2.0),   # ch1: title region (rotation cue)
            ('card',  'heat_pts',  1.25),  # ch2: corner keypoints
        ]
        self.map_channels = 3
        self.target_size  = (180, 180)

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        self.seed        = 22
        self.device      = 'cuda'
        self.num_workers = 8
        self.epochs      = 5000
        self.batch_size  = 512            # 4 GPU × 128 = effective 512
        self.distributed = False
        self.max_norm    = 1.0

        # ------------------------------------------------------------------
        # Optimiser (AdamW + polynomial decay)
        # ------------------------------------------------------------------
        self.lr               = 1e-4
        self.lr_backbone_ratio = 0.0
        self.weight_decay     = 1e-5
        self.warmup_steps     = 0
        self.max_steps        = 636024    # ~6 epochs over 160K images
        self.decay_power      = 1
        self.end_lr           = 1e-6

        # ------------------------------------------------------------------
        # Augmentation
        # (parameters derived from Train-Eval distribution analysis)
        # ------------------------------------------------------------------

        self.train_min_size = 180
        self.train_max_size = 180

        # RandomCrop — corrects area_ratio gap (Train 0.18 vs Eval 0.33)
        self.crop_min_size_ratio = 0.75   # was 0.3 → increased to fix small-card bias
        self.crop_max_size_ratio = 1.0
        self.crop_prob           = 0.15   # was 0.3 → reduced to match eval distribution

        # Rotation
        self.rotate_max_angle = 90
        self.rotate_prob      = 0.5

        # Color distortion — all ranges tuned to match eval statistics
        # brightness: Train 130.01 vs Eval 140.21 → bias toward brighter
        self.dist_brightness  = (1.0, 1.15)
        # contrast: Train 37.70 vs Eval 51.96 → reduce downward range
        self.dist_contrast    = (0.9, 1.3)
        # saturation: Train 26.98 vs Eval 55.87 (2× gap, main culprit)
        self.dist_saturation  = (0.9, 1.2)
        # hue: small range to avoid excessive color distortion
        self.dist_hue         = 0.15
        self.distortion_prob  = 0.5

        # ------------------------------------------------------------------
        # Normalisation
        # ------------------------------------------------------------------
        self.normalize = False
        self.mean      = [0.485, 0.456, 0.406]
        self.std       = [0.229, 0.224, 0.225]

        # ------------------------------------------------------------------
        # Dataset & checkpoint (override these for your environment)
        # ------------------------------------------------------------------
        self.train_dataset   = ['card3']
        self.val_dataset     = ['card3']
        self.data_root       = './data'
        self.output_folder   = './output/roi_detection'
        self.resume          = None        # path to checkpoint to resume from
        self.pretrain_ckpt   = None        # path to pretrained checkpoint
        self.continue_train  = False

        # ------------------------------------------------------------------
        # Misc
        # ------------------------------------------------------------------
        self.aug_ratio      = 0.5
        self.shuffle        = True
        self.save_freq      = 1
        self.print_freq     = 1
        self.checkpoint_freq = 1
        self.startidx       = 0
        self.nsamples       = -1
        self.backbone       = ''
        self.pretrained_file = None
        self.train          = True
        self.infer_vie      = False
        self.train_vie      = False
        self.mask_ratio     = 0.0