from .layers.activations import *
from .layers.affine_layers import *
from .layers.noise_injection import *
from .layers.normalization_layers import *

"""
Used for linking the class names of keras layers with the corresponding
variance propagation layers.
Therefore importat by uncertainty_propagator.py
"""

noise_layers = {
    'Dropout': DropoutVarPropagationLayer
}

activation_layers = {
    'linear': LinearActivationVarPropagationLayer,
    'relu': ReLUActivationVarPropagationLayer,
    'softmax': SoftmaxActivationVarPropagationLayer,
    'tanh': TanhActivationVarPropagationLayer
}

affine_layers = {
    'Dense': DenseVarPropagationLayer,
    'Conv2D': Conv2DVarPropagationLayer,
    'Conv2DTranspose': Conv2DTransposeVarPropagationLayer
}

