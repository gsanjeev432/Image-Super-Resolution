from __future__ import print_function, division

from keras.utils.vis_utils import plot_model
import models
import img_utils

if __name__ == "__main__":
    val_path = "val_images/"

    scale = 2

    """
    Plot the model
    """

    model = models.DistilledResNetSR(scale).create_model()
    plot_model(model, to_file='distilled_resnet_sr.png', show_layer_names=True, show_shapes=True)


    """
    Distilled ResNetSR
    """

    distilled_rnsr = models.DistilledResNetSR(scale)
    distilled_rnsr.create_model(None, None, 3, load_weights=True)
    distilled_rnsr.evaluate(val_path)
