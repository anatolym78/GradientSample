# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGraphicsView, QGridLayout, QPushButton,
    QSizePolicy, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(951, 936)
        self.gridLayout = QGridLayout(Widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.graphicsView = QGraphicsView(Widget)
        self.graphicsView.setObjectName(u"graphicsView")

        self.gridLayout.addWidget(self.graphicsView, 5, 0, 1, 2)

        self.buttonMakeDatabase = QPushButton(Widget)
        self.buttonMakeDatabase.setObjectName(u"buttonMakeDatabase")

        self.gridLayout.addWidget(self.buttonMakeDatabase, 0, 0, 1, 1)

        self.learnButton = QPushButton(Widget)
        self.learnButton.setObjectName(u"learnButton")

        self.gridLayout.addWidget(self.learnButton, 0, 1, 1, 1)

        self.buttonTest = QPushButton(Widget)
        self.buttonTest.setObjectName(u"buttonTest")

        self.gridLayout.addWidget(self.buttonTest, 1, 0, 1, 1)

        self.buttonShowKernels = QPushButton(Widget)
        self.buttonShowKernels.setObjectName(u"buttonShowKernels")

        self.gridLayout.addWidget(self.buttonShowKernels, 2, 0, 1, 1)

        self.buttonTestNet2 = QPushButton(Widget)
        self.buttonTestNet2.setObjectName(u"buttonTestNet2")

        self.gridLayout.addWidget(self.buttonTestNet2, 1, 1, 1, 1)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.buttonMakeDatabase.setText(QCoreApplication.translate("Widget", u"Make database", None))
        self.learnButton.setText(QCoreApplication.translate("Widget", u"Learn Net", None))
        self.buttonTest.setText(QCoreApplication.translate("Widget", u"Test Net", None))
        self.buttonShowKernels.setText(QCoreApplication.translate("Widget", u"Show kernels", None))
        self.buttonTestNet2.setText(QCoreApplication.translate("Widget", u"Test Net 2", None))
    # retranslateUi

