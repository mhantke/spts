import os
import sys
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, "%s/../" % this_dir)

import logging
logging.basicConfig()
logger = logging.getLogger("MSI_GUI")

import numpy as np
from matplotlib import pyplot as pypl

from PyQt5 import QtCore, QtGui, uic
import pyqtgraph as pg

from expiringdict import ExpiringDict

import ui

# MSI modules 
import worker

from spts_conf import Conf
from options import Options
from view import View, ViewOptions
from preferences import Preferences

from dummy_worker import DummyWorker

#from IPython.core.debugger import Tracer
#Tracer()()

class MainWindow(ui.MainUI, ui.MainBaseUI):

    def __init__(self, parent=None):
        ui.MainBaseUI.__init__(self, parent)
        self.ui = ui.MainUI()
        self.ui.setupUi(self)

        self.settings = QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, "fxihub", "spts")

        self.preferences = Preferences(self)
            
        self.view = View(self)
        self.ui.horizontalLayout.addWidget(self.view)
        
        self.ui.dataTypeTabWidget.setCurrentIndex(0)
        
        self.clear_cache()
        
        self.conf = Conf(self)
        self.i_frame = 0
        self.data_type = "1_raw"
        self._create_worker(silent=True)
        
        self.options = Options(self)
        self.options.load_all()
        self.options.connect_all()
        
        self.view_options = ViewOptions(self)
        self.view_options.connect_all()

        self.ui.actionOpen.triggered.connect(self.conf.open)
        self.ui.actionSave.triggered.connect(self.conf.save)
        self.ui.actionSaveAs.triggered.connect(self.conf.save_as)
        self.ui.actionPreferences.triggered.connect(self.preferences.open_preferences_dialog)
        self.ui.dataTypeTabWidget.currentChanged.connect(self._on_tab_changed)

        self.ui.previousPushButton.clicked.connect(self._show_previous)
        self.ui.nextPushButton.clicked.connect(self._show_next)
        self.ui.iFrameSpinBox.valueChanged.connect(self._on_i_frame_changed)
        
        #from IPython.terminal.debugger import set_trace
        #set_trace()
        pgDown = QtGui.QShortcut(QtGui.QKeySequence('PgDown'), self)
        pgDown.activated.connect(self._show_next)
        pgUp = QtGui.QShortcut(QtGui.QKeySequence('PgUp'), self)
        pgUp.activated.connect(self._show_previous)

        self._show_frame()
        
    def clear_cache(self):
        self.cache = ExpiringDict(max_len=10, max_age_seconds=1000)

    def clear_cache_types(self, data_types):
        for i in self.cache.keys():              
            for dt in data_types:
                if dt in self.cache[i]:
                    del self.cache[i][dt]
                    if i == self.i_frame and dt == self.data_type:
                        self._show_frame()
        
    def _init_worker(self):
        self._create_worker()
        self._set_index(0)
        self.clear_cache()
        self._show_frame()

    def _create_worker(self, silent=False):
        try:
            self.worker = worker.Worker(self.conf, pipeline_mode=True, data_mount_prefix=self.preferences.data_mount_prefix)
        except IOError:
            if not silent:
                logger.warning("Cannot create worker instance (IOError). Data might not be mounted.")
            self.worker = DummyWorker(self.conf)

    def _on_i_frame_changed(self):
        i_frame = self.ui.iFrameSpinBox.value()
        if self.i_frame is not None and i_frame == self.i_frame:
            return
        elif i_frame is None or self.worker.N is None or (i_frame+1 >= self.worker.N) or (i_frame < 0):
            self._set_index(0)
        else:
            self.i_frame = i_frame
        self._show_frame()
            
    def _set_index(self, i_frame):
        self.i_frame = i_frame
        self.ui.iFrameSpinBox.setValue(i_frame)
                    
    def _on_tab_changed(self, tab_index):
        data_types = ["1_raw", "2_process", "3_denoise", "4_threshold", "5_detect", "6_analyse"]
        self.data_type = data_types[tab_index]
        self._show_frame()

    def _get_data(self, i_frame, data_type):
        data = None
        if i_frame in self.cache:
            if self.cache[i_frame] is not None and data_type in self.cache[i_frame]:
                data = self.cache[i_frame][data_type]
                return data
        else:
            self.cache[i_frame] = None
        out_package = self.cache[i_frame]
        data = self.worker.work(work_package={"i": i_frame}, tmp_package=out_package, target=data_type)
        self.cache[i_frame] = data
        return data[data_type]

    def _get_frame(self, i_frame, data_type):
        tdata = self._get_data(i_frame, data_type)
        if data_type == "1_raw":
            return tdata["image_raw"]
        elif data_type == "2_process":
            return tdata["image"]
        elif data_type == "3_denoise":
            return tdata["image_denoised"]
        elif data_type == "4_threshold":
            return tdata["image_thresholded"]
        elif data_type == "5_detect":
            return tdata["image_labels"]
        elif data_type == "6_analyse":
            return tdata["masked_image"]

    def _get_xy(self, i_frame):
        tdata = self._get_data(i_frame, "5_detect")
        x = tdata["x"]
        y = tdata["y"]
        s = (x!=-1)*(y!=-1)
        return s, x, y
        
    def _get_peaks(self, i_frame):
        tdata = self._get_data(i_frame, "6_analyse")
        s = (tdata["peak_sum"]!=-1) * (tdata["peak_size"]!=-1)

        c = np.zeros_like(tdata["peak_size"])
        c[s] = np.sqrt(tdata["peak_size"][s]/np.pi)/(tdata["peak_circumference"][s]/(2*np.pi))

        s *= np.isfinite(c)
        
        return s, tdata["peak_sum"], c
        
    def _show_next(self):
        if self.i_frame is None or self.worker.N is None or (self.i_frame+1 >= self.worker.N):
            return
        else:
            self._set_index(self.i_frame+1)
            self._show_frame()

    def _show_previous(self):
        if self.i_frame is None or self.worker.N is None or (self.i_frame-1 <= -1):
            return
        else:
            self._set_index(self.i_frame-1)
            self._show_frame()
            
    def _show_frame(self):
        if self.i_frame is None:
            return
        frame = self._get_frame(self.i_frame, self.data_type)

        cmap = None
        auto_range = None
        if self.data_type == "4_threshold":
            auto_range = True
        elif self.data_type == "5_detect":
            cmap = pypl.cm.gnuplot
            auto_range = True
        self.view.show_image(frame.T, force_auto_range=auto_range, force_cmap=cmap)
        self.view.remove_all_circles_and_annotations()
        if (self.data_type == "5_detect") or (self.data_type == "6_analyse"):
            s_xy, x, y = self._get_xy(self.i_frame)
            if (self.data_type == "6_analyse"):
                s_p, intensities, circlescore = self._get_peaks(self.i_frame)
                s = s_xy * s_p
                if s.sum() > 0:
                    self.view.draw_new_annotations(x[s], y[s], intensities[s], circlescore[s])
            else:
                s = s_xy
            if s.sum() > 0:
                self.view.draw_new_circles(x[s], y[s])

    def closeEvent(self, event):
        if self.conf.any_unsaved_changes():
            saveChanges = QtGui.QMessageBox(QtGui.QMessageBox.Question, "Save changes?",
                                            "Would you like to save changes?",
                                            QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel).exec_()
            if saveChanges == QtGui.QMessageBox.Save:
                self.conf.save()
            if saveChanges == QtGui.QMessageBox.Cancel:
                return event.ignore()
        self.preferences.closeEvent()
        self.view_options.closeEvent()
        QtGui.QMainWindow.closeEvent(self, event)

def main():
    app = QtGui.QApplication(sys.argv)
    #app.processEvents()    

    mainWindow = MainWindow()
    mainWindow.show()

    if len(sys.argv) > 1:
        mainWindow.conf.load(sys.argv[1])

    sys.exit(app.exec_())
    
        
if __name__ == "__main__":
    main()
