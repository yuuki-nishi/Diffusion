from torchmetrics import ScaleInvariantSignalNoiseRatio
def metrics(y_true,y_pred,mixtured=None):
    metrics = {}
    si_snr = ScaleInvariantSignalNoiseRatio()
    metrics["si-snr"] = si_snr(y_true,y_pred)
    return metrics