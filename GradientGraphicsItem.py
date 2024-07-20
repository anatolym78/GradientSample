import math

from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter
from PySide6.QtCore import QRectF, QPointF
from PySide6.QtGui import QVector2D


class GradientGraphicsItem(QGraphicsItem):
    def __init__(self):
        super(GradientGraphicsItem, self).__init__()
        self.item_size = 500
        self.scene_rect = QRectF(0, 0, self.item_size, self.item_size)
        self.linear_gradient = QLinearGradient()
        self.linear_gradient.setStart(0, 0)
        self.linear_gradient.setFinalStop(self.item_size*0.5, 0)
        self.linear_gradient.setStops([(0, QColor(32, 32, 64)), (0.5, QColor(64, 128, 255)), (1, QColor(32, 32, 64))])


    def make_gradient(self, start, end):
        v_start = QVector2D(start)
        v_end = QVector2D(end)
        dv = v_end - v_start
        ev = dv/dv.length()
        v_start = v_end - 2*ev*dv.length()
        self.linear_gradient.setStart(v_start.toPointF())
        self.linear_gradient.setFinalStop(v_end.toPointF())

        return 2*dv.length()/self.item_size

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        painter.fillRect(self.scene_rect, self.linear_gradient)

    def boundingRect(self) -> QRectF:
        return self.scene_rect
