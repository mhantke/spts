from PyQt4 import uic
import os

uidir = os.path.dirname(os.path.realpath(__file__))
MainUI, MainBaseUI = uic.loadUiType(uidir + '/main.ui')
PreferencesUI, PreferencesBaseUI = uic.loadUiType(uidir + '/preferences.ui')
