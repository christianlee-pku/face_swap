from registry import AUGMENTATIONS
from data.transforms import LightAugmentation

AUGMENTATIONS.register("LightAugmentation")(LightAugmentation)
