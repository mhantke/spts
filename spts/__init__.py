

def _init():
    import logging, sys

    global logger
    logger = logging.getLogger('spts')

    import pkg_resources
    global __version__
    __version__ = pkg_resources.require("spts")[0].version

    
_init()
