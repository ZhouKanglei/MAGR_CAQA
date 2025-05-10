# -*- coding: utf-8 -*-
# @Time: 2023/6/22 23:01
from pydoc import locate
from backbone import Backbone, xavier

class AQAMLP(Backbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, args):
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(AQAMLP, self).__init__()

        feature_extractor = locate(args.feature_extractor)
        self.feature_extractor = feature_extractor(**args.feature_extractor_args)

        regressor = locate(args.regressor)
        self.regressor = regressor(**args.regressor_args)

        if args.projector is not None:
            projector = locate(args.projector)
            self.projector = projector(**args.projector_args)

        self.args = args
        # self.reset_parameters()

    def reset_parameters(self):
        """
        Calls the Xavier parameter initialization function.
        """
        self.regressor.apply(xavier)

    def forward(self, x, returnt='out', replay=False):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        feats = self.feature_extractor(x)

        # this step is non-standard, but it is used to ensure that all the models have been forward-passed
        if self.args.projector is not None:
            feats = feats + 0 * self.projector(feats)

        if returnt == 'features':
            return feats

        out = self.regressor(feats)
        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)

        raise NotImplementedError("Unknown return type")
