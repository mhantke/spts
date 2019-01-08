import numpy as np

from PyQt5 import QtCore, QtGui, uic

import ui

class Preferences:

    def __init__(self, mainWindow):

        self.w = mainWindow

        if not self.w.settings.contains("dataMountPrefix"):
            self.w.settings.setValue("dataMountPrefix", "")
        self.data_mount_prefix = str(self.w.settings.value("dataMountPrefix"))
        
    def open_preferences_dialog(self):
        diag = PreferencesDialog(self.w)
        if(diag.exec_()):
            self.data_mount_prefix = str(diag.dataMountPrefixLineEdit.text())
            self.w.settings.setValue("dataMountPrefix", self.data_mount_prefix)
            self.w._init_worker()

    def closeEvent(self):
        self.w.settings.setValue("dataMountPrefix", self.data_mount_prefix)
        

class PreferencesDialog(QtGui.QDialog, ui.PreferencesUI):

    def __init__(self, mainWindow):
        self.w = mainWindow
        
        QtGui.QDialog.__init__(self, mainWindow, QtCore.Qt.WindowTitleHint)
        self.setupUi(self)

        self.dataMountPrefixLineEdit.setText(self.w.preferences.data_mount_prefix)
