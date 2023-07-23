import kornia.feature as KF
import kornia as K
import cv2
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches
from enum import Enum


class ModelType(str, Enum):
    indoor = 'indoor'
    outdoor = 'outdoor'


max_input_shape = 640


class Service:
    def __init__(self) -> None:
        self.matchers = {
            ModelType.indoor: KF.LoFTR(pretrained="indoor_new"),
            ModelType.outdoor: KF.LoFTR(pretrained="outdoor")
        }

    @staticmethod
    def __ready_img(img_data):
        img = cv2.imdecode(np.fromstring(
            img_data, dtype="uint8"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_tensor = torch.Tensor(np.array([img])).permute(
            0, 3, 1, 2).float() / 255.0
        img_width, img_hieght = img_tensor.shape[2], img_tensor.shape[3]
        resize_factor = max(img_width/max_input_shape,
                            img_hieght/max_input_shape)
        return K.geometry.resize(img_tensor, (int(img_width/resize_factor), int(img_hieght/resize_factor)), antialias=True)

    def process(self, model_type: ModelType, img1_data, img2_data):
        img1 = Service.__ready_img(img1_data)
        img2 = Service.__ready_img(img2_data)
        input_dict = {
            # LofTR works on grayscale images only
            "image0": K.color.rgb_to_grayscale(img1),
            "image1": K.color.rgb_to_grayscale(img2),
        }

        with torch.inference_mode():
            correspondences = self.matchers[model_type](input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        Fm, inliers = cv2.findFundamentalMat(
            mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0

        fig, ax = draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                torch.from_numpy(mkpts0).view(1, -1, 2),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                torch.from_numpy(mkpts1).view(1, -1, 2),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1),
            ),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(img1),
            K.tensor_to_image(img2),
            inliers,
            draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (
                0.2, 0.5, 1), "vertical": False},
            return_fig_ax=True
        )
        return fig