"""Fixed RGB image metrics; no learned weights or network access required."""

IMAGE_PROTOCOL = {
    'input': '64x64 RGB float images, data range 1; all pixels, no foreground mask.',
    'prediction': 'Render in the predicted input camera, before geometric ICP; clip RGB to [0,1] for image metrics.',
    'psnr': '-10*log10(max(per-object RGB MSE,1e-12)); cap 120dB, count floor use explicitly.',
    'ssim': 'Mean across RGB channels and valid spatial centers; 11x11 Gaussian sigma1.5, K1=.01,K2=.03, population covariance.',
    'aggregation': 'Arithmetic mean of per-object PSNR and SSIM; PSNR of aggregate MSE is a different quantity.',
    'lpips': 'Not computed by this bundle; no learned metric weights are downloaded on the cluster.',
}


def image_metrics(prediction, target):
    import numpy as np
    from scipy.ndimage import convolve1d
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != (64, 64, 3) or target.shape != prediction.shape:
        raise ValueError('Expected two 64x64 RGB images')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite image')
    if target.min() < 0 or target.max() > 1:
        raise ValueError('Target must be in [0,1]')
    x, y = prediction.clip(0, 1), target
    mse = float(np.mean((x-y)**2))
    kernel = np.exp(-np.arange(-5, 6, dtype=np.float64)**2/(2*1.5**2))
    kernel /= kernel.sum()
    def blur(a):
        return convolve1d(convolve1d(a, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    ux, uy = blur(x), blur(y)
    vx, vy, cov = blur(x*x)-ux*ux, blur(y*y)-uy*uy, blur(x*y)-ux*uy
    score = ((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    ssim = float(score[5:-5, 5:-5].mean())
    if not np.isfinite(ssim) or not -1.000001 <= ssim <= 1.000001:
        raise ValueError('Invalid SSIM')
    return {'mse': mse, 'psnr_db': float(-10*np.log10(max(mse, 1e-12))), 'ssim': ssim,
            'prediction_raw_min': float(prediction.min()), 'prediction_raw_max': float(prediction.max()),
            'clip_fraction': float(((prediction<0)|(prediction>1)).mean()),
            'psnr_floor_applied': bool(mse<1e-12)}
