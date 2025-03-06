from PyQt5.QtWidgets import *
from mocap_ui import Ui_Mocap
import sys
import tkinter as tk
class LMainWindow(QMainWindow):
    def __init__(self,parent = None):
        super(LMainWindow,self).__init__(parent)


class LMain():
    def __init__(self):
        self.__mainWindow = None
        self.__mainform = None
 
    def _init_mainWindow(self):
        "Initialize main window"
        self.__mainWindow = LMainWindow()  # main window
        self.__mainform = Ui_Mocap()  # main interface
        self.__mainform.setupUi(self.__mainWindow)    
   
    def run(self):
        #创建应用程序实例
        app = QApplication(sys.argv)
        #初始化主窗口
        self._init_mainWindow()
        #显示主窗口
        self.__mainWindow.show()
        #运行应用程序
        sys.exit(app.exec_())
 
if __name__ == '__main__':
    app = LMain()
    app.run()
