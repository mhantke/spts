import os
import sys
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, "%s/../" % this_dir)

import copy

# SPTS modules
import config

from PyQt5 import QtCore, QtGui, uic

import logging
logger = logging.getLogger("SPTS_GUI")

DEFAULT_SPTS_CONF = this_dir + "/spts_default.conf"

class Conf(dict):
    def __init__(self, mainWindow, *args):
        dict.__init__(self, args)
        self.w = mainWindow
        self.load_default()
        self.filename = None
        self._saved = None
        
    def open(self):
        filename = QtGui.QFileDialog.getOpenFileName(self.w, "Open SPTS configuration file", "", "CONF Files (*.conf)")
        if isinstance(filename, tuple):
            if filename[0]:
                filename = filename[0]
        if filename:
            self.filename = filename
            self.load(filename)

    def load(self, filename):
        self.filename = filename
        dict_new = config.read_configfile(filename)
        for sec_name, sec_dict in dict_new.items():
            if sec_name not in self:
                logger.warning("\"%s\" is not an accepted name for a configuration section. Ignoring this input." % sec_name)
            else:
                for opt_name, opt_value in sec_dict.items():
                    if opt_name not in self[sec_name]:
                        logger.warning("\"%s\" is not an accepted name for an option in the configuration section \"%s\". Ignoring this input." % (opt_name, sec_name))
                    else:
                        self[sec_name][opt_name] = opt_value
        self._saved = copy.deepcopy(dict(self))
        self.w.options.load_all()
        self.w._init_worker()
        
    def any_unsaved_changes(self):
        if self._saved is None:
            return True
        for sec_name, sec_dict in self._saved.items():
            if sec_name not in self:
                return True
            for opt_name, opt_value in sec_dict.items():
                if opt_name not in self[sec_name] or self[sec_name][opt_name] != opt_value:
                    return True
        return False        
        
    def load_default(self, default_spts_conf=DEFAULT_SPTS_CONF):
        for sec_name, sec_dict in config.read_configfile(default_spts_conf).items():
            self[sec_name] = sec_dict

    def save_as(self):
        filename = QtGui.QFileDialog.getSaveFileName(self.w, "Save SPTS configuration file", "", "CONF Files (*.conf)")
        if(isinstance(filename, tuple)):
            if(filename[0]):
                filename = filename[0]
        if filename:
            self.filename = filename
            self.save()
        
    def save(self):
        if self.filename is None:
            self.save_as()
        with open(self.filename, "w") as f:
            for sec_name, sec_dict in self.items():
                f.write("[%s]\n" % sec_name)
                for opt_name, opt_value in sec_dict.items():
                    f.write("%s = %s\n" % (opt_name, str(opt_value)))
                f.write("\n")
        self._saved = copy.deepcopy(dict(self))
            
                

