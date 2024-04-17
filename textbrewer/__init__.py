__version__ = "0.2.1.post1"

from .distillers import BasicTrainer
from .distillers import BasicDistiller
from .distillers import GeneralDistiller


from .configurations import TrainingConfig, DistillationConfig

from .presets import FEATURES
from .presets import ADAPTOR_KEYS
from .presets import KD_LOSS_MAP, MATCH_LOSS_MAP, PROJ_MAP
from .presets import WEIGHT_SCHEDULER, TEMPERATURE_SCHEDULER
from .presets import register_new

Distillers = {
    "Basic": BasicDistiller,
    "General": GeneralDistiller,
    "Train": BasicTrainer,
}
