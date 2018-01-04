import numpy as np

from PyQt4 import QtCore, QtGui
import pyqtgraph as pg

class ViewOptions:

    def __init__(self, mainWindow):
        self.w = mainWindow

        if(not self.w.settings.contains("viewOptionsVMin")):
            self.w.settings.setValue("viewOptionsVMin", 0.)
        if(not self.w.settings.contains("viewOptionsVMax")):
            self.w.settings.setValue("viewOptionsVMax", 300.)
        if(not self.w.settings.contains("viewOptionsAutoRange")):
            self.w.settings.setValue("viewOptionsAutoRange", True)

        self.vmin = self.w.settings.value("viewOptionsVMin").toFloat()[0]
        self.vmax = self.w.settings.value("viewOptionsVMax").toFloat()[0]
        self.auto_range = self.w.settings.value("viewOptionsAutoRange").toBool()   
        
        self.vMinDoubleSpinBox = self.w.ui.vMinDoubleSpinBox
        self.vMaxDoubleSpinBox = self.w.ui.vMaxDoubleSpinBox
        self.autoScaleCheckBox = self.w.ui.autoScaleCheckBox
        self.vMinDoubleSpinBox.setValue(self.vmin)
        self.vMaxDoubleSpinBox.setValue(self.vmax)
        self.autoScaleCheckBox.setChecked(self.auto_range)
        
    def connect_all(self):
        self.vMinDoubleSpinBox.editingFinished.connect(self.on_vlim_changed)
        self.vMaxDoubleSpinBox.editingFinished.connect(self.on_vlim_changed)
        self.autoScaleCheckBox.stateChanged.connect(self.on_vlim_changed)

    def on_vlim_changed(self):
        self.w.view.update_display_options(vmin=self.vMinDoubleSpinBox.value(),
                                           vmax=self.vMaxDoubleSpinBox.value(),
                                           auto_range=self.autoScaleCheckBox.isChecked())

    def closeEvent(self):
        self.w.settings.setValue("viewOptionsVMin", self.vmin)
        self.w.settings.setValue("viewOptionsVMax", self.vmax)
        self.w.settings.setValue("viewOptionsAutoRange", self.auto_range)


class View(pg.GraphicsView):

    def __init__(self, mainWindow):
        self.w = mainWindow
        
        pg.GraphicsView.__init__(self)
        self.layout = pg.GraphicsLayout()
        self.box = self.layout.addViewBox(lockAspect=True)
        self.box.invertY()
        self.image = pg.ImageItem()
        self.setCentralItem(self.layout)
        self.box.addItem(self.image)

        self.circles = []
        
    def show_image(self, image, force_auto_range=None, force_cmap=None):
        if image is None:
            return
        self._force_auto_range = force_auto_range
        self._force_cmap = force_cmap
        args = {}
        if not self.w.view_options.auto_range and (force_auto_range is None or not force_auto_range):
            args["levels"] = [self.w.view_options.vmin, self.w.view_options.vmax]
        else:
            args["autoLevels"] = True
        self.image.setImage(np.asarray(image, dtype=np.float64), **args)

        if force_cmap is not None:
            Ncol = 256
            pos = np.linspace(0., 1., Ncol)
            lut = pg.ColorMap(pos=pos, color=force_cmap(pos)).getLookupTable()
        else:
            lut = None
        self.image.setLookupTable(lut)
        self.box.autoRange()

    def remove_all_circles_and_annotations(self):
        # This while loop is needed because of a weird try-pass section in the removeItem method of ViewBox
        while len([i for i in self.box.addedItems if (isinstance(i, CircleOverlay) or isinstance(i, pg.TextItem))]) > 0:
            for i in self.box.addedItems:
                if isinstance(i, CircleOverlay) or isinstance(i, pg.TextItem):
                    self.box.removeItem(i)
        
    def draw_new_circles(self, xs, ys):
        for x, y in zip(xs, ys):
            size = 20
            c = CircleOverlay(x=x-size/2, y=y-size/2, size=size, pen=QtGui.QPen(QtCore.Qt.red, 0.1), movable=False)
            self.box.addItem(c)
            
    def draw_new_annotations(self, xs, ys, intensities, circlescores):
        for x, y, i, c in zip(xs, ys, intensities, circlescores):
            t = pg.TextItem("%i" % i, color='g')
            t.setPos(x+12, y-20)
            self.box.addItem(t)
            t = pg.TextItem("%.2f" % c, color='y')
            t.setPos(x+12, y)
            self.box.addItem(t)
            
    def update_display_options(self, vmin, vmax, auto_range):
        self.w.view_options.vmin = vmin
        self.w.view_options.vmax = vmax
        self.w.view_options.auto_range = auto_range
        if self._force_auto_range:
            return
        if auto_range:
            self.show_image(self.image.image)
        else:
            self.image.setLevels([vmin, vmax])


class CircleOverlay(pg.EllipseROI):
    def __init__(self, x, y, size, **args):
        pg.ROI.__init__(self, (x, y), size, **args)
        self.aspectLocked = True

