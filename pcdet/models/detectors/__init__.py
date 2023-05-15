from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .pv_rcnn_ssl import PVRCNN_SSL
from .second_net import SECONDNet
from .second_ssl import SECOND_SSL
from .second_ssl_db import SECOND_SSL_db
from .voxel_rcnn import VoxelRCNN
from .voxel_rcnn_ssl import VoxelRCNN_SSL
from .voxel_rcnn_ssl_soft import VoxelRCNN_SSL_SOFT
from .pv_rcnn_ssl_soft import PVRCNN_SSL_SOFT
from .pv_rcnn_ssl_child import PVRCNN_SSL_CHILD
from .voxel_rcnn_ssl_child_v2 import VoxelRCNN_SSL_CHILD_v2
from .pv_rcnn_ssl_child_v2 import PVRCNN_SSL_CHILD_v2
from .pv_rcnn_ssl_child_v3 import PVRCNN_SSL_CHILD_v3
from .pv_rcnn_ssl_dl import PVRCNN_SSL_dl
from .pv_rcnn_dl import PVRCNN_dl
from .pv_rcnn_ssl_mask import PVRCNN_SSL_mask
from .pv_rcnn_ssl_db import PVRCNN_SSL_db
from .pv_rcnn_ssl_db_denseloss import PVRCNN_SSL_db_denseloss
from .pv_rcnn_ssl_dbv2 import PVRCNN_SSL_dbv2
from .pv_rcnn_ssl_dbv3 import PVRCNN_SSL_dbv3
from .pv_rcnn_ssl_dbv4 import PVRCNN_SSL_dbv4
from .pv_rcnn_ssl_db_nms import PVRCNN_SSL_db_nms
from .pv_rcnn_ssl_db_grad import PVRCNN_SSL_db_grad
from .point_rcnn_ssl import PointRCNN_ssl

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PVRCNN_SSL': PVRCNN_SSL,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECOND_SSL': SECOND_SSL,
    'SECOND_SSL_db': SECOND_SSL_db,
    'VoxelRCNN': VoxelRCNN,
    'VoxelRCNN_SSL': VoxelRCNN_SSL,
    'VoxelRCNN_SSL_SOFT': VoxelRCNN_SSL_SOFT,
    'PVRCNN_SSL_SOFT': PVRCNN_SSL_SOFT,
    'PVRCNN_SSL_CHILD': PVRCNN_SSL_CHILD,
    'PVRCNN_SSL_CHILD_v2': PVRCNN_SSL_CHILD_v2,
    'PVRCNN_SSL_CHILD_v3': PVRCNN_SSL_CHILD_v3,
    'VoxelRCNN_SSL_CHILD_v2': VoxelRCNN_SSL_CHILD_v2,
    'PVRCNN_SSL_dl': PVRCNN_SSL_dl,
    'PVRCNN_dl': PVRCNN_dl,
    'PVRCNN_SSL_mask': PVRCNN_SSL_mask,
    'PVRCNN_SSL_db': PVRCNN_SSL_db,
    'PVRCNN_SSL_db_denseloss': PVRCNN_SSL_db_denseloss,
    'PVRCNN_SSL_dbv2': PVRCNN_SSL_dbv2,
    'PVRCNN_SSL_dbv3': PVRCNN_SSL_dbv3,
    'PVRCNN_SSL_dbv4': PVRCNN_SSL_dbv4,
    'PVRCNN_SSL_db_nms': PVRCNN_SSL_db_nms,
    'PVRCNN_SSL_db_grad': PVRCNN_SSL_db_grad
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
