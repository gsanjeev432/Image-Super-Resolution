import models
import tensorflow as tf

suffix = "scaled"

mode = "fast"

scale_factor = 2
save = "True"

patch_size = 8

with tf.device('/CPU:0'):
    path = "barbara.bmp"
    model = models.DistilledResNetSR(scale_factor)

    model.upscale(path, save_intermediate=save, mode=mode, patch_size=patch_size, suffix=suffix)
