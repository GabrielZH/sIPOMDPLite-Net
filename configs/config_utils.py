from distutils.util import strtobool


def boolean_arg(arg):
    """
    Convert a string argument to boolean.
    """
    return bool(strtobool(arg))
