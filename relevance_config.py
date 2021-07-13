import os
import sys
import numpy as np

class RelevanceParams16Comp1Trust:
    def __init__(self):
        
        #embedding
        self.architecture = 'ResNet50' #also used by model
        self.color_means = np.array([0.485, 0.456, 0.406], dtype='float32')
        self.color_SDs = np.array([0.229, 0.224, 0.225], dtype='float32')
        self.default_smaller_size = 224
        self.size_multiple = 32

        #data generator
        self.batch_size = 32 #for training, half of these will be negative and half will be positive

        #model + training
        self.num_components = 16
        self.trust_multiplier = 1.0 #multiply this by avg(||w_c||_2^2) where w_c are the linear readout weights of the model specified by self.architecture
        self.learning_rate = 1e-3 #look up default value for this
        self.num_training_steps = 2000 #for now
        self.val_prop = 0.1
        self.val_freq = 50

class RelevanceParams1Comp1Trust(RelevanceParams16Comp1Trust):
    def __init__(self):
        super(RelevanceParams1Comp1Trust, self).__init__()
        self.num_components = 1
        assert(self.trust_multiplier == 1.0)

class RelevanceParams32Comp1Trust(RelevanceParams16Comp1Trust):
    def __init__(self):
        super(RelevanceParams32Comp1Trust, self).__init__()
        self.num_components = 32
        assert(self.trust_multiplier == 1.0)

class RelevanceParams16CompHalfTrust(RelevanceParams16Comp1Trust):
    def __init__(self):
        super(RelevanceParams16CompHalfTrust, self).__init__()
        self.trust_multiplier = 0.5
        assert(self.num_components == 16)

class RelevanceParams1CompHalfTrust(RelevanceParams16CompHalfTrust):
    def __init__(self):
        super(RelevanceParams1CompHalfTrust, self).__init__()
        self.num_components = 1
        assert(self.trust_multiplier == 0.5)

class RelevanceParams32CompHalfTrust(RelevanceParams16CompHalfTrust):
    def __init__(self):
        super(RelevanceParams32CompHalfTrust, self).__init__()
        self.num_components = 32
        assert(self.trust_multiplier == 0.5)

class RelevanceParams16CompInfTrust(RelevanceParams16Comp1Trust):
    def __init__(self):
        super(RelevanceParams16CompInfTrust, self).__init__()
        self.trust_multiplier = float('+inf')

class RelevanceParams1CompInfTrust(RelevanceParams16CompInfTrust):
    def __init__(self):
        super(RelevanceParams1CompInfTrust, self).__init__()
        self.num_components = 1

class RelevanceParams32CompInfTrust(RelevanceParams16CompInfTrust):
    def __init__(self):
        super(RelevanceParams32CompInfTrust, self).__init__()
        self.num_components = 32

def grab_params(params_key):
    return eval(params_key + '()')
