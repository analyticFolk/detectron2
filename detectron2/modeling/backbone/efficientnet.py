import torch
from torch import nn
from torch.nn import functional as F

from .efficientnet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

from detectron2.layers import (
    ShapeSpec,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(Backbone):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, out_features=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        self._swish = MemoryEfficientSwish()
        self._out_feature_strides = {"stem": 2,
                                     'res1': 2,
                                     'res2': 4,
                                     'res3': 8,
                                     'res4': 16,
                                     'res5': 32}
        self._out_feature_channels = dict()
        block_indices = [0, 1, 2, 4, 6] # block indices to use for the FPN

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self.stem = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False),
                                  nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
                                  self._swish)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        repeats = [r.num_repeat for r in self._blocks_args]
        round_reps = [round_repeats(r, self._global_params) for r in repeats]
        block_end_indices = []
        for i in range(len(round_reps)):
            block_end_indices.append(sum(round_reps[:i + 1]) - 1)
        self._out_feature_indices = {'res'+str(i+1): block_end_indices[block_indices[i]] for i in range(len(block_indices))}
        self._out_indices_to_feature = {v:k for k, v in self._out_feature_indices.items()}
        self._out_feature_channels = {'res'+str(i+1): round_filters(b.output_filters, self._global_params) for i,b in enumerate(self._blocks_args[:3]+self._blocks_args[4:7:2])}
        if out_features is None:
            out_features = ['res5']
        self._out_features = out_features

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def freeze(self, layer_idx):
        if layer_idx > 0:
            for p in self.stem.parameters():
                p.requires_grad = False
            for i in range(self._out_feature_indices['res'+str(layer_idx)]):
                for p in self._blocks[i].parameters():
                    p.requires_grad = False

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        outputs = {}
        x = self.stem(inputs)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self._out_indices_to_feature:
                name = self._out_indices_to_feature[idx]
                if name in self._out_features:
                    outputs[name] = x
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x, outputs

    def forward(self, input):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = input.size(0)
        # Convolution layers
        x, outputs = self.extract_features(input)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        if "linear" in self._out_features:
            outputs["linear"] = x
        return outputs

    @classmethod
    def from_name(cls, model_name, override_params=None, out_features=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, out_features=out_features)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3, out_features=None):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes}, out_features=out_features)
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model
    
    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """ 
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg, input_shape):
    """
    Create an EfficientNet instance from config.

    Returns:
        EfficientNet: a :class:`EfficientNet` instance.
    """
    # need registration of new blocks/stems?
    #norm = cfg.MODEL.RESNETS.NORM
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    # fmt: off
    out_features        = cfg.MODEL.EFFICIENTNET.OUT_FEATURES
    num_classes         = cfg.MODEL.RETINANET.NUM_CLASSES
    pretrained_weights  = cfg.MODEL.EFFICIENTNET.WEIGHTS
    model_name          = cfg.MODEL.EFFICIENTNET.NAME
    # fmt: on

    stages = []

    model = EfficientNet.from_pretrained(model_name,
                                         num_classes=num_classes,
                                         out_features=out_features)
    model.freeze(freeze_at)
    if len(pretrained_weights) > 0:
        model.load_state_dict(torch.load(pretrained_weights))
    return model
