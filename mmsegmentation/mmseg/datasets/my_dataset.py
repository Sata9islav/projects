from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("background", "class_1", "class_2"),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
