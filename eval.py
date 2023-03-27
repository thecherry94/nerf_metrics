import torch
import numpy as np
import pandas as pd
from skimage import metrics as skmetrics
from skimage import util as skutil
import lpips as lpips_metric

from dataset.dataset import MultiViewDataset


class NeRFEvaluator:
    def __init__(self, data_path: str, data_name: str):
        self.data_path = data_path
        self.data_name = data_name
        self.dataset = MultiViewDataset(data_path, data_name, 50)
        self.lpips_vgg = lpips_metric.LPIPS(net='vgg').to(device='cuda')

    @staticmethod
    def random_noise(image: np.ndarray):
        mode = np.random.choice(["gaussian", "s&p", "speckle", "poisson"])
        noisy_image = skutil.random_noise(image, mode=mode).astype(np.float32)  # dtype np.float64, range [0.0, 1.0]
        return noisy_image

    def get_metrics(self, ref_image: np.ndarray, mod_image: np.ndarray):
        assert ref_image.dtype == np.float32
        assert mod_image.dtype == np.float32

        psnr = skmetrics.peak_signal_noise_ratio(ref_image, mod_image, data_range=1)
        ssim = skmetrics.structural_similarity(ref_image, mod_image, channel_axis=-1, data_range=1)
        lpips = self.lpips_vgg(
            torch.from_numpy(ref_image).permute(2, 0, 1).to(device="cuda"),
            torch.from_numpy(mod_image).permute(2, 0, 1).to(device="cuda"),
            normalize=True).item()
        return [psnr, ssim, lpips]

    def evaluate(self, sensor_type: str = "color_cam", image_type: str = "color"):
        global_metrics = list()
        n_scenes = len(self.dataset)
        for scene_index in range(n_scenes):
            print(f"Calculating metrics for scene {scene_index+1} / {n_scenes}")
            observation = self.dataset[scene_index]["observation"]
            # src_image = observation[10][sensor_type][image_type].astype(np.float32) / 255.0
            scene_metrics = list()
            for view_index in range(len(observation)):
                if view_index == 10:
                    continue
                tgt_image = observation[view_index][sensor_type][image_type].astype(np.float32) / 255.0
                # TODO: use previously reconstructed images (by trained nerf model) instead of adding random noise to gt
                reconstr_image = self.random_noise(tgt_image)
                scene_metrics.append(self.get_metrics(tgt_image, reconstr_image))
            global_metrics.append(np.mean(scene_metrics, axis=0))
            print(f"\t[PSNR, SSIM, LPIPS]: {global_metrics[-1]}")
        df_index = list(range(n_scenes))
        df_index.append("mean")
        global_metrics.append(np.mean(global_metrics, axis=0))
        df = pd.DataFrame(global_metrics, columns=["PSNR", "SSIM", "LPIPS"], index=df_index)
        print(df)
        df.to_csv(self.dataset.log_path + "/metrics.csv")


if __name__ == '__main__':
    evaluator = NeRFEvaluator(
        data_path="/opt/project/data/00_data/research/NeRF/suction_multi",
        data_name="valid")
    evaluator.evaluate()
