# Import dependencies
import tensorflow as tf
from keras.layers import Layer

# Custom L1 Distance Layer - Fixed to match saved model expectations
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, input_embedding, validation_embedding=None, **kwargs):
        # Handle different input formats the saved model might use
        if validation_embedding is not None:
            # Called with separate arguments
            return tf.math.abs(input_embedding - validation_embedding)
        elif isinstance(input_embedding, list) and len(input_embedding) == 2:
            # Called with list of two tensors
            return tf.math.abs(input_embedding[0] - input_embedding[1])
        else:
            # Fallback - assume it's a concatenated input that needs splitting
            # This shouldn't happen but just in case
            return tf.math.abs(input_embedding)
    
    def compute_output_shape(self, input_shape):
        # Return proper shape tuple instead of None
        if isinstance(input_shape, list):
            return input_shape[0]
        else:
            return input_shape