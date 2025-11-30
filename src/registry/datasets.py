from registry import DATASETS
from data.lfw_dataset import LFWDataset

# Register dataset
DATASETS.register("LFWDataset")(LFWDataset)
