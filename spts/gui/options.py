
import os.path

import logging
logger = logging.getLogger("MSI_GUI")

from PyQt5 import QtCore, QtGui

class Options:
    def __init__(self, mainWindow):
        self.general_box = GeneralBox(mainWindow)
        self.raw_tab = RawTab(mainWindow)
        self.process_tab = ProcessTab(mainWindow)
        self.denoise_tab = DenoiseTab(mainWindow)
        self.threshold_tab = ThresholdTab(mainWindow)
        self.detect_tab = DetectTab(mainWindow)
        self.analyse_tab = AnalyseTab(mainWindow)

    def load_all(self):
        self.general_box.load_all()
        self.raw_tab.load_all()
        self.process_tab.load_all()
        self.denoise_tab.load_all()
        self.threshold_tab.load_all()
        self.detect_tab.load_all()
        self.analyse_tab.load_all()

    def connect_all(self):
        self.general_box.connect_all()
        self.raw_tab.connect_all()
        self.process_tab.connect_all()
        self.denoise_tab.connect_all()
        self.threshold_tab.connect_all()
        self.detect_tab.connect_all()
        self.analyse_tab.connect_all()        
        

class GeneralBox:

    def __init__(self, mainWindow):
        self.w = mainWindow

        self.dataFilenameLineEdit = self.w.ui.dataFilenameLineEdit
        self.i0SpinBox = self.w.ui.i0SpinBox
        self.nFramesSpinBox = self.w.ui.nFramesSpinBox
        self.outputLevelComboBox = self.w.ui.outputLevelComboBox

    def connect_all(self):
        self.dataFilenameLineEdit.returnPressed.connect(self._set_filename)
        self.i0SpinBox.valueChanged.connect(self._on_i0_changed)
        self.nFramesSpinBox.valueChanged.connect(self._on_n_frames_changed)
        self.outputLevelComboBox.currentIndexChanged.connect(self._on_output_level_changed)

    def load_all(self):
        c = self.w.conf["general"]
        self.dataFilenameLineEdit.setText(c["filename"])
        self.i0SpinBox.setValue(c["i0"])
        self.nFramesSpinBox.setValue(c["n_images"])
        self.outputLevelComboBox.setCurrentIndex(c["output_level"])
        
    def _set_filename(self):
        filename = str(self.dataFilenameLineEdit.text())
        self.w.conf["general"]["filename"] = filename
        filename_full = self.w.preferences.data_mount_prefix + filename
        if os.path.isfile(filename_full):
            self.w._init_worker()
        else:
            logger.warning("File %s cannot be found. You might want to check data mount prefix under preferences." % filename_full)
            
    def _on_i0_changed(self):
        self.w.conf["general"]["i0"] = self.i0SpinBox.value()

    def _on_n_frames_changed(self):
        n_images = self.nFramesSpinBox.value()
        self.w.conf["general"]["n_images"] = n_images

    def _on_output_level_changed(self):
        self.w.conf["general"]["output_level"] = self.outputLevelComboBox.currentIndex()

names_data_types = ["1_raw", "2_process", "3_denoise", "4_threshold", "5_detect", "6_analyse"]
        
class RawTab:

    def __init__(self, mainWindow):
        self.w = mainWindow

        self.dataSetNameRawLineEdit = self.w.ui.dataSetNameRawLineEdit
        self.saturationLevelSpinBox = self.w.ui.saturationLevelSpinBox
        self.xMinSpinBox = self.w.ui.xMinSpinBox
        self.xMaxSpinBox = self.w.ui.xMaxSpinBox
        self.yMinSpinBox = self.w.ui.yMinSpinBox
        self.yMaxSpinBox = self.w.ui.yMaxSpinBox
        self.skipSaturatedCheckBox = self.w.ui.skipSaturatedCheckBox
        self.subtractConstRawSpinBox = self.w.ui.subtractConstRawSpinBox

    def connect_all(self):
        self.dataSetNameRawLineEdit.editingFinished.connect(self._on_data_set_name_changed)
        self.saturationLevelSpinBox.valueChanged.connect(self._on_saturation_level_changed)
        self.xMinSpinBox.valueChanged.connect(self._on_xmin_changed)
        self.xMaxSpinBox.valueChanged.connect(self._on_xmax_changed)
        self.yMinSpinBox.valueChanged.connect(self._on_ymin_changed)
        self.yMaxSpinBox.valueChanged.connect(self._on_ymax_changed)
        self.skipSaturatedCheckBox.stateChanged.connect(self._on_skip_saturated_changed)
        self.subtractConstRawSpinBox.valueChanged.connect(self._on_subtract_const_changed)
        
    def load_all(self):
        c = self.w.conf["raw"]
        self.dataSetNameRawLineEdit.setText(c["dataset_name"])
        self.saturationLevelSpinBox.setValue(c["saturation_level"])
        self.xMinSpinBox.setValue(c["xmin"])
        self.xMaxSpinBox.setValue(c["xmax"])
        self.yMinSpinBox.setValue(c["ymin"])
        self.yMaxSpinBox.setValue(c["ymax"])
        self.skipSaturatedCheckBox.setChecked(c["skip_saturated_frames"])
        self.subtractConstRawSpinBox.setValue(c["subtract_constant"])
        
    def _clear_dt_cache(self):
        self.w.clear_cache_types(names_data_types)
        
    def _on_data_set_name_changed(self):
        self.w.conf["raw"]["dataset_name"] = str(self.dataSetNameRawLineEdit.text())
        self._clear_dt_cache()

    def _on_saturation_level_changed(self):
        self.w.conf["raw"]["saturation_level"] = self.saturationLevelSpinBox.value()
        self._clear_dt_cache()

    def _on_xmin_changed(self):
        xmin = self.xMinSpinBox.value()
        xmax = self.w.conf["raw"]["xmax"]
        if xmin >= xmax:
            self.w.conf["raw"]["xmin"] = xmax - 1
        else:
            self.w.conf["raw"]["xmin"] = xmin
        self._clear_dt_cache()
        
    def _on_xmax_changed(self):
        xmin = self.w.conf["raw"]["xmin"]
        xmax = self.xMaxSpinBox.value()
        if xmax <= xmin:
            self.w.conf["raw"]["xmax"] = xmin + 1
        else:
            self.w.conf["raw"]["xmax"] = xmax
        self._clear_dt_cache()

    def _on_ymin_changed(self):
        ymin = self.yMinSpinBox.value()
        ymax = self.w.conf["raw"]["ymax"]
        if ymin >= ymax:
            self.w.conf["raw"]["ymin"] = ymax - 1
        else:
            self.w.conf["raw"]["ymin"] = ymin
        self._clear_dt_cache()
        
    def _on_ymax_changed(self):
        ymin = self.w.conf["raw"]["ymin"]
        ymax = self.yMaxSpinBox.value()
        if ymax <= ymin:
            self.w.conf["raw"]["ymax"] = ymin + 1
        else:
            self.w.conf["raw"]["ymax"] = ymax
        self._clear_dt_cache()

    def _on_skip_saturated_changed(self):
        self.w.conf["raw"]["skip_saturated_frames"] = self.skipSaturatedCheckBox.isChecked()
        self._clear_dt_cache()

    def _on_subtract_const_changed(self):
        self.w.conf["raw"]["subtract_constant"] = self.subtractConstRawSpinBox.value()
        self._clear_dt_cache()

        
class ProcessTab:

    def __init__(self, mainWindow):
        self.w = mainWindow

        self.dataSetNameLineEdit = self.w.ui.dataSetNameLineEdit
        self.subtractConstSpinBox = self.w.ui.subtractConstSpinBox
        self.floorCutCheckBox = self.w.ui.floorCutCheckBox
        self.floorCutSpinBox = self.w.ui.floorCutSpinBox
        self.cmcXCheckBox = self.w.ui.cmcXCheckBox
        self.cmcYCheckBox = self.w.ui.cmcYCheckBox
        
    def connect_all(self):
        self.dataSetNameLineEdit.editingFinished.connect(self._on_data_set_name_changed)
        self.subtractConstSpinBox.valueChanged.connect(self._on_subtract_const_changed)
        self.floorCutCheckBox.stateChanged.connect(self._on_floor_cut_toggled)
        self.floorCutSpinBox.valueChanged.connect(self._on_floor_cut_changed)
        self.cmcXCheckBox.stateChanged.connect(self._on_cmcx_changed)
        self.cmcYCheckBox.stateChanged.connect(self._on_cmcy_changed)
        
    def load_all(self):
        c = self.w.conf["process"]
        self.dataSetNameLineEdit.setText(c["dataset_name"])
        self.subtractConstSpinBox.setValue(c["subtract_constant"])
        if c["floor_cut_level"] is None:
            self.floorCutCheckBox.setChecked(False)
            self.floorCutSpinBox.setReadOnly(True)
            self.floorCutSpinBox.setValue(0.)
        else:
            self.floorCutCheckBox.setChecked(True)
            self.floorCutSpinBox.setReadOnly(False)
            self.floorCutSpinBox.setValue(c["floor_cut_level"])
        self.cmcXCheckBox.setChecked(c["cmcx"])
        self.cmcYCheckBox.setChecked(c["cmcy"])
        
    def _clear_dt_cache(self):
        self.w.clear_cache_types(names_data_types[1:])
        
    def _on_data_set_name_changed(self):
        self.w.conf["process"]["dataset_name"] = str(self.dataSetNameLineEdit.text())
        self._clear_dt_cache()

    def _on_subtract_const_changed(self):
        self.w.conf["process"]["subtract_constant"] = self.subtractConstSpinBox.value()
        self._clear_dt_cache()

    def _on_floor_cut_toggled(self):
        checked = self.floorCutCheckBox.isChecked()
        self.floorCutSpinBox.setReadOnly(not checked)
        if checked:
            self._on_floor_cut_changed()
        else:
            self.w.conf["process"]["floor_cut_level"] = None
            self._clear_dt_cache()
        
    def _on_floor_cut_changed(self):
        self.w.conf["process"]["floor_cut_level"] = self.floorCutSpinBox.value()
        self._clear_dt_cache()       

    def _on_cmcx_changed(self):
        self.w.conf["process"]["cmcx"] = self.cmcXCheckBox.isChecked()
        self._clear_dt_cache()

    def _on_cmcy_changed(self):
        self.w.conf["process"]["cmcy"] = self.cmcYCheckBox.isChecked()
        self._clear_dt_cache()

        
denoise_methods = ["gauss", "gauss2"]#, "hist"]
class DenoiseTab:

    def __init__(self, mainWindow):
        self.w = mainWindow

        self.methodComboBox = self.w.ui.methodComboBox
        self.sigmaDoubleSpinBox = self.w.ui.sigmaDoubleSpinBox

    def connect_all(self):
        self.methodComboBox.currentIndexChanged.connect(self._on_method_changed)
        self.sigmaDoubleSpinBox.valueChanged.connect(self._on_sigma_changed)

    def load_all(self):
        c = self.w.conf["denoise"]
        self.methodComboBox.setCurrentIndex(denoise_methods.index(c["method"]))
        self.sigmaDoubleSpinBox.setValue(c["sigma"])
        
    def _clear_dt_cache(self):
        self.w.clear_cache_types(names_data_types[2:])
        
    def _on_method_changed(self):
        self.w.conf["denoise"]["method"] = denoise_methods[self.methodComboBox.currentIndex()]
        self._clear_dt_cache()

    def _on_sigma_changed(self):
        self.w.conf["denoise"]["sigma"] = self.sigmaDoubleSpinBox.value()
        self._clear_dt_cache()

class ThresholdTab:

    def __init__(self, mainWindow):
        self.w = mainWindow

        self.thresholdDoubleSpinBox = self.w.ui.thresholdDoubleSpinBox
        self.fillHolesCheckBox = self.w.ui.fillHolesCheckBox

    def connect_all(self):
        self.thresholdDoubleSpinBox.valueChanged.connect(self._on_threshold_changed)
        self.fillHolesCheckBox.stateChanged.connect(self._on_threshold_changed)
        
    def load_all(self):
        c = self.w.conf["threshold"]
        self.thresholdDoubleSpinBox.setValue(c["threshold"])
        self.fillHolesCheckBox.setChecked(c["fill_holes"])
        
    def _clear_dt_cache(self):
        self.w.clear_cache_types(names_data_types[3:])

    def _on_threshold_changed(self):
        self.w.conf["threshold"]["threshold"] = self.thresholdDoubleSpinBox.value()
        self.w.conf["threshold"]["fill_holes"] = self.fillHolesCheckBox.isChecked()
        self._clear_dt_cache()

peak_centering_methods = ["center_to_max", "center_of_mass"]
        
class DetectTab:

    def __init__(self, mainWindow):
        self.w = mainWindow

        self.minDistDoubleSpinBox = self.w.ui.minDistDoubleSpinBox
        self.methodComboBox = self.w.ui.methodComboBox_2
        self.nParticlesMaxSpinBox = self.w.ui.nParticlesMaxSpinBox

    def connect_all(self):
        self.minDistDoubleSpinBox.valueChanged.connect(self._on_min_dist_changed)
        self.methodComboBox.currentIndexChanged.connect(self._on_method_changed)
        self.nParticlesMaxSpinBox.valueChanged.connect(self._on_n_particles_max_changed)
        
    def _clear_dt_cache(self):
        self.w.clear_cache_types(names_data_types[4:])

    def load_all(self):
        c = self.w.conf["detect"]
        self.minDistDoubleSpinBox.setValue(c["min_dist"])
        self.methodComboBox.setCurrentIndex(peak_centering_methods.index(c["peak_centering"]))
        self.nParticlesMaxSpinBox.setValue(c["n_particles_max"])
        
    def _on_min_dist_changed(self):
        self.w.conf["detect"]["min_dist"] = self.minDistDoubleSpinBox.value()
        self._clear_dt_cache()

    def _on_method_changed(self):
        self.w.conf["detect"]["peak_centering"] = peak_centering_methods[self.methodComboBox.currentIndex()]
        self._clear_dt_cache()       

    def _on_n_particles_max_changed(self):
        self.w.conf["detect"]["n_particles_max"] = self.nParticlesMaxSpinBox.value()
        self._clear_dt_cache()


        
integration_modes = ["windows", "labels"]
        
class AnalyseTab:

    def __init__(self, mainWindow):
        self.w = mainWindow
    
        self.modeComboBox = self.w.ui.modeComboBox
        self.windowSizeSpinBox = self.w.ui.windowSizeSpinBox
        self.circleWindowCheckBox = self.w.ui.circleWindowCheckBox

    def connect_all(self):
        self.modeComboBox.currentIndexChanged.connect(self._on_mode_changed)
        self.windowSizeSpinBox.valueChanged.connect(self._on_window_size_changed)
        self.circleWindowCheckBox.stateChanged.connect(self._on_circle_window_changed)

    def load_all(self):
        c = self.w.conf["analyse"]
        self.modeComboBox.setCurrentIndex(integration_modes.index(c["integration_mode"]))
        self.windowSizeSpinBox.setValue(c["window_size"])
        self.circleWindowCheckBox.setChecked(c["circle_window"])
        
    def _clear_dt_cache(self):
        self.w.clear_cache_types(names_data_types[5:])

    def _on_mode_changed(self):
        self.w.conf["analyse"]["integration_mode"] = integration_modes[self.modeComboBox.currentIndex()]
        self._clear_dt_cache()

    def _on_window_size_changed(self):
        self.w.conf["analyse"]["window_size"] = self.windowSizeSpinBox.value()
        self._clear_dt_cache()

    def _on_circle_window_changed(self):
        self.w.conf["analyse"]["circle_window"] = self.circleWindowCheckBox.isChecked()
        self._clear_dt_cache()
