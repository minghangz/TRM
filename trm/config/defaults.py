import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "TRM"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.NAME = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.TRM = CN()
_C.MODEL.TRM.NUM_CLIPS = 128
_C.MODEL.TRM.JOINT_SPACE_SIZE = 256
_C.MODEL.TRM.RESIDUAL = 1.0

_C.MODEL.TRM.FEATPOOL = CN()
_C.MODEL.TRM.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.TRM.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.TRM.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.TRM.FEAT2D = CN()
_C.MODEL.TRM.FEAT2D.NAME = "pool"
_C.MODEL.TRM.FEAT2D.POOLING_COUNTS = [15, 8, 8, 8]

_C.MODEL.TRM.TEXT_ENCODER = CN()
_C.MODEL.TRM.TEXT_ENCODER.NAME = 'BERT'
_C.MODEL.TRM.TEXT_ENCODER.USE_PHRASE = True
_C.MODEL.TRM.TEXT_ENCODER.DROP_PHRASE = False

_C.MODEL.TRM.PREDICTOR = CN() 
_C.MODEL.TRM.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.TRM.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.TRM.PREDICTOR.NUM_STACK_LAYERS = 8

_C.MODEL.TRM.LOSS = CN()
_C.MODEL.TRM.LOSS.MIN_IOU = 0.3
_C.MODEL.TRM.LOSS.MAX_IOU = 0.7
_C.MODEL.TRM.LOSS.BCE_WEIGHT = 1
_C.MODEL.TRM.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL = 1
_C.MODEL.TRM.LOSS.NEGATIVE_VIDEO_IOU = 0.5
_C.MODEL.TRM.LOSS.SENT_REMOVAL_IOU = 0.5
_C.MODEL.TRM.LOSS.PAIRWISE_SENT_WEIGHT = 0.0
_C.MODEL.TRM.LOSS.CONTRASTIVE_WEIGHT = 0.05
_C.MODEL.TRM.LOSS.TAU_VIDEO = 0.2
_C.MODEL.TRM.LOSS.TAU_SENT = 0.2
_C.MODEL.TRM.LOSS.MARGIN = 0.2
_C.MODEL.TRM.LOSS.SENT_NEG_IOU = 0.2
_C.MODEL.TRM.LOSS.NEGATIVE_PROP_IOU = 0.2
_C.MODEL.TRM.LOSS.PHRASE_ONLY = False
_C.MODEL.TRM.LOSS.USE_SCORE_MAP_LOSS = False
_C.MODEL.TRM.LOSS.USE_FOCAL_LOSS = False
_C.MODEL.TRM.LOSS.CONTRASTIVE = False
_C.MODEL.TRM.LOSS.CONSIS_WEIGHT = 1.0
_C.MODEL.TRM.LOSS.EXC_WEIGHT = 1.0
_C.MODEL.TRM.LOSS.THRESH = 0.001

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_EPOCH = 1
_C.SOLVER.FREEZE_BERT = 4
_C.SOLVER.ONLY_IOU = 7
_C.SOLVER.SKIP_TEST = 0

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.5
_C.TEST.CONTRASTIVE_SCORE_POW = 0.5
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
