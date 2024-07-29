
from PySide6.QtWidgets import QGraphicsScene, QGraphicsSceneMouseEvent

from PySide6.QtGui import QPen, QBrush, QColor, QVector2D

from PySide6.QtCore import Qt, QPointF, QLineF

import math

import torch

from GradientGraphicsItem import GradientGraphicsItem

from SimpleModule import SimpleModule
from GradientImagesCreator import GradientImagesCreator

from torchvision.transforms import v2


class GradientScene(QGraphicsScene):
    def __init__(self) -> object:
        super(GradientScene, self).__init__()
        self.gradient_item = GradientGraphicsItem()
        self.addItem(self.gradient_item)
        self.text_item = self.addText("Gradient view")
        self.text_item.setDefaultTextColor(QColor(255, 255, 255))
        self.start_pos_item = self.addEllipse(-2.5, -2.5, 5, 5, QPen(QColor(128, 128, 128), 0.5), QBrush(QColor(32, 255, 64)))
        self.end_pos_item = self.addEllipse(-2.5, -2.5, 5, 5, QPen(QColor(128, 128, 128), 0.5), QBrush(QColor(32, 255, 64)))
        self.real_direction_line_item = self.addLine(QLineF(QPointF(0, 0), QPointF(100, 1)))
        self.real_direction_line_item.setPen(QPen(QBrush(QColor(32, 255, 32)), 1.5))
        self.predict_direction_line_item = self.addLine(QLineF(QPointF(0, 0), QPointF(100, 1)))
        self.predict_direction_line_item.setPen(QPen(QBrush(QColor(255, 32, 16)), 1.5))
        self.left_button_pressed = False
        self.start_point = QPointF(0, 0)

        self.models = []
        net_suffices = ['h', 'v']
        for i_model in range(0, 2):
            check_point = torch.load(f"model_state_{net_suffices[i_model]}.pth")
            model = SimpleModule()
            model.load_state_dict(check_point["model_state"])
            model.eval()
            self.models.append(model)

    def calculate_angle_predict(self, pos, stretch, angle):
        to_float_tensor = v2.ToDtype(torch.float32, scale=False)
        to_normalize = v2.Normalize([128], [128])

        x = pos.x()/self.gradient_item.item_size
        y = pos.y()/self.gradient_item.item_size
        tensor_image, label = GradientImagesCreator.create_sample(32, torch.tensor([x, y], dtype=torch.float32), stretch, angle)

        tensor_image = to_float_tensor(tensor_image)
        tensor_image = to_normalize(tensor_image)

        angle_predicts = []
        for model in self.models:
            angle_predicts.append(model(tensor_image))

        return angle_predicts[0].item()*90, (1 - angle_predicts[1].item())*180

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        end_pos = event.scenePos()
        if self.left_button_pressed:
            dp = end_pos - self.start_point
            self.end_pos_item.setPos(event.scenePos())
            if QVector2D(end_pos).distanceToPoint(QVector2D(self.start_point)) < 2:
                return

            self.real_direction_line_item.setLine(QLineF(self.start_point, end_pos))
            angle = self.normalize_angle(math.degrees(math.atan2(dp.y(), dp.x())))
            stretch = self.gradient_item.update_gradient(self.start_point, end_pos)
            self.text_item.setPlainText(f"angle: {angle:.0f}, width: {stretch:.1f}")

            if QVector2D(end_pos).distanceToPoint(QVector2D(self.start_point)) > 25:
                angle_predict_h, angle_predict_v = self.calculate_angle_predict(self.start_point, stretch, angle)
                result_angle = GradientScene.choose_angle_predict(angle_predict_h, angle_predict_v)
                self.text_item.setPlainText(f"angle real/predicted: {angle:.0f}/{angle_predict_h:.0f},{angle_predict_v:.0f}, width: {stretch:.1f}")
        else:
            if self.gradient_item.valid_rect.contains(event.scenePos().toPoint()):
                self.start_pos_item.setPos(event.scenePos())

        self.update()
        pass

    def get_predict_line(self, pos, angle) -> QLineF:
        angle_radians = math.radians(self.normalize_angle(angle))

        ex = math.cos(angle_radians)
        ey = math.sin(angle_radians)

        size = 0.4*self.gradient_item.item_size
        size = 0.4*self.gradient_item.item_size

        return QLineF(pos - size*QPointF(ex, ey), pos + size*QPointF(ex, ey))

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.left_button_pressed = True
            self.start_point = event.scenePos()

        pass

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.left_button_pressed = False

        pass

    @staticmethod
    def normalize_angle(angle):
        if angle < -90:
            angle = angle + 180
        else:
            if angle > 90:
                angle = angle - 180

        return angle

    @staticmethod
    def choose_angle_predict(horizontal_angle, vertical_angle):

        return horizontal_angle
        pass

