from tflite_model_maker.config import QuantizationConfig

OPTIMIZATION_NONE = "none"
OPTIMIZATION_FLOAT16 = "float16"
OPTIMIZATION_DYNAMIC = "dynamic"

OPTIMIZATIONS = [
    OPTIMIZATION_NONE,
    OPTIMIZATION_FLOAT16,
    OPTIMIZATION_DYNAMIC,
]


def configure_optimization(optimization):
    """
    Returns the optimization configuration to use.

    :param optimization: the optimization name
    :type optimization: str
    :return: the configuration
    :rtype: QuantizationConfig
    """
    if optimization == OPTIMIZATION_NONE:
        return None
    elif optimization == OPTIMIZATION_FLOAT16:
        return QuantizationConfig.for_float16()
    elif optimization == OPTIMIZATION_DYNAMIC:
        return QuantizationConfig.for_dynamic()
    else:
        raise "Unhandled optimization: %s" % optimization
