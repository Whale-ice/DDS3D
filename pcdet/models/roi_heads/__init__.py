from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate
from .pvrcnn_head_dl import PVRCNNHead_dl
from .pvrcnn_head_mask import PVRCNNHead_mask
from .pvrcnn_head_db import PVRCNNHead_db
from .pvrcnn_head_dbv4 import PVRCNNHead_dbv4

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'PVRCNNHead_dl': PVRCNNHead_dl,
    'PVRCNNHead_mask': PVRCNNHead_mask,
    'PVRCNNHead_db': PVRCNNHead_db,
    'PVRCNNHead_dbv4': PVRCNNHead_dbv4
}
