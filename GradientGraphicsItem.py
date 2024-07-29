import math

from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter
from PySide6.QtCore import QRectF, QPointF, QRect
from PySide6.QtGui import QVector2D, QRegion


class GradientGraphicsItem(QGraphicsItem):
    def __init__(self):
        super(GradientGraphicsItem, self).__init__()
        self.item_size = 600
        self.scene_rect = QRect(0, 0, self.item_size, self.item_size)
        dx = self.scene_rect.width() * GradientGraphicsItem.valid_area_size()
        dy = self.scene_rect.height() * GradientGraphicsItem.valid_area_size()
        self.valid_rect = QRect((self.scene_rect.width() - dx) * 0.5, (self.scene_rect.height() - dy) * 0.5, dx, dy)
        self.linear_gradient = QLinearGradient()
        self.linear_gradient.setStart(0, 0)
        self.linear_gradient.setFinalStop(self.item_size*0.5, 0)
        self.linear_gradient.setStops([(0, QColor(32, 32, 64)), (0.5, QColor(64, 128, 255)), (1, QColor(32, 32, 64))])

    def update_gradient(self, start, end):
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
        main_region = QRegion(self.scene_rect)
        valid_region = QRegion(self.valid_rect)
        invalid_region = main_region.subtracted(valid_region)
        painter.setClipRegion(invalid_region)
        painter.fillRect(self.scene_rect, QBrush(QColor(255, 0, 0, 64)))

    def boundingRect(self) -> QRectF:
        return self.scene_rect

    @staticmethod
    def valid_area_size():
        return 0.6
