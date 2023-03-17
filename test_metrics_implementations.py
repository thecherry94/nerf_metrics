from skimage import metrics, util
from typing import List, Tuple
import pandas as pd
import numpy as np
import lpips
import torch
import cv2
import os

SHOW_TEST_IMAGES = True
SHOW_SSIM_IMAGES = True

project_dir = "/home/iras-student/projects/alka1020/isaac_envirnoment"
nerf_dir = project_dir + "/10_experiments/research/NeRF_b2/suction_grasp_2500"

PATH_REF_IMG = nerf_dir + "/tgt/tgt-0.png"  # os.path.join("nerf_images", "tgt", "tgt-0.png")
# if PATH_TEST_IMGS None: test images are created by adding noise to the reference image
PATH_TEST_IMGS = nerf_dir + "/output"  # os.path.join("nerf_images", "output")


def get_reference_image(scale_percent: int = 30) -> np.ndarray:
    """Load original image (reference image) and resize it by scale_percent percent.

    Parameters
    ----------
    scale_percent : int, default=30
        Percentage of the original image width and height to scale the image to.

    Returns
    -------
    ndarray
        The resized reference image.

    """
    img = cv2.imread(PATH_REF_IMG)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def get_test_images(
        img: np.ndarray,
        img_names: List[str] = ["gaussian", "s&p", "speckle", "poisson", ]) -> List[np.ndarray]:
    """Create noisy test images (modified / reconstructed images) from the reference image.

    Parameters
    ----------
    img : ndarray
        The reference image to add noise to.
    img_names : List[str], default=["gaussian", "s&p", "speckle", "poisson"]
        The test images' file names. If no modified images given, list of noise types to add to the reference image.

    Returns
    -------
    List[ndarray]
        List of test images.

    """
    img_list = list()
    if PATH_TEST_IMGS is None:
        for mode in img_names:
            img_noisy = util.random_noise(img, mode=mode)  # dtype np.float64, range [0.0, 1.0]
            img_list.append(img_noisy)
    else:
        for file in img_names:
            f = os.path.join(PATH_TEST_IMGS, f"{file}.png")
            if os.path.exists(f):
                img_reconstr = cv2.resize(
                    cv2.imread(f),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                img_list.append(img_reconstr)
    return img_list


def show_test_images(images: List[np.ndarray], img_names: List[str]) -> None:
    """Show the test images in a window.

    Parameters
    ----------
    images : List[ndarray]
        List of test images to show.
    img_names : List[str]
        List of the test images' names.

    """
    if SHOW_TEST_IMAGES:
        for idx, img in enumerate(images):
            cv2.imshow(str(img_names[idx]), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def calculate_metrics(img_ref: np.ndarray, img_mod: List[np.ndarray]) -> Tuple[List, List, np.ndarray]:
    """Calculate the peak signal-to-noise ratio (PSNR) and the (mean) structural similarity (SSIM) index between a
    reference image and modified or reconstructed test images.

    Parameters
    ----------
    img_ref : numpy.ndarray
        The reference image.
    img_mod : List[numpy.ndarray]
        The modified / reconstructed images.

    Returns
    -------
    Tuple[List, List, List[np.ndarray]]
        The PSNR and mean SSIM values per modified image and the full SSIM images if SHOW_SSIM_IMAGES (else None).

    """
    lpips_vgg = lpips.LPIPS(net='vgg').to(device='cuda')
    psnr_vals = list()
    ssim_vals = list()
    ssim_images = list()
    lpips_vals = list()
    img_ref_lpips = torch.from_numpy(img_ref).permute(2, 0, 1).to(device="cuda")
    for idx, img in enumerate(img_mod):
        psnr_vals.append(metrics.peak_signal_noise_ratio(img_ref, img, data_range=1))  # 10 * np.log10((data_range ** 2) / mse)
        if SHOW_SSIM_IMAGES:
            mssim, ssim_image = metrics.structural_similarity(img_ref, img, channel_axis=-1, full=True, data_range=1)
            ssim_images.append(ssim_image)
        else:
            ssim_images.append(None)
            mssim = metrics.structural_similarity(img_ref, img, channel_axis=-1, data_range=1)
        ssim_vals.append(mssim)
        img_lpips = torch.from_numpy(img).permute(2, 0, 1).to(device="cuda")
        lpips_vals.append(lpips_vgg(img_ref_lpips, img_lpips, normalize=True).item())
    return psnr_vals, ssim_vals, ssim_images, lpips_vals


def show_results(
        img_names: List[str],
        psnr_list: List[float],
        ssim_list: List[float],
        ssim_images: List[np.ndarray],
        lpips_list: List[float]) -> None:
    """Show the results in a table and optionally the SSIM images.

    Parameters
    ----------
    img_names : List[str]
        List of image names.
    psnr_list : List[float]
        List of PSNR values.
    ssim_list : List[float]
        List of mean SSIM values.
    ssim_images : List[np.ndarray]
        List of full SSIM images.
    lpips_list : List[float]
       List of LPIPS values.

    """
    df = pd.DataFrame(
        list(zip(img_names, psnr_list, ssim_list, lpips_list)),
        columns=["Noise", "PSNR", "Mean SSIM", "LPIPS"])
    df.sort_values(
        by=["LPIPS"],
        ascending=True,
        inplace=True)  # inplace: update df instead of creating a new one (?)
    print(df)
    if SHOW_SSIM_IMAGES:
        for idx, img in enumerate(ssim_images):
            cv2.imshow(f"{img_names[idx]} ssim image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Calculate the peak signal-to-noise ratio (PSNR) and the (mean) structural similarity (SSIM) index between a
    # reference image and a modified or reconstructed image

    # 1. Load original image (reference image) and resize it
    img_true = get_reference_image(scale_percent=100)

    # 2. Create noisy test images (modified / reconstructed image)
    if PATH_TEST_IMGS is None:
        modes = ["gaussian", "speckle", "poisson", "s&p"]
    else:
        modes = ["output-0", "output-92", "output-93"]
    img_test = get_test_images(img_true, img_names=modes)
    # Add the original image to the list of test images to compare the metrics with the modified images
    img_test.append(img_true)
    modes.append("reference")

    # 3. Show the images if desired
    show_test_images(img_test, modes)

    # 4. Calculate PSNR and (mean) SSIM index
    psnr, ssim, img_ssim, lpips_score = calculate_metrics(img_true, img_test)

    # 5. Show results
    show_results(modes, psnr, ssim, img_ssim, lpips_score)
    # as reference --> average metrics in VisionNeRF, PixelNeRF, SRN for chairs and cars (category-specific):
    # PSNR: 23.23 (22.25, 23.72) | SSIM: 0.91 (0.89, 0.93) | LPIPS: 0.111 (0.146, 0.077))
