import rclpy
from yolo_pkg.ros_communicator import RosCommunicator
from yolo_pkg.boundingbox_visaulizer import BoundingBoxVisualizer
from yolo_pkg.yolo_detect_model import YoloDetectionModel
from yolo_pkg.object_detect_manager import ObjectDetectManager
from yolo_pkg.camera_parameters import CameraParameters
from yolo_pkg.camera_geometry import CameraGeometry
import threading

def _init_ros_node():
    rclpy.init()
    node = RosCommunicator()
    thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread.start()
    return node, thread

def main():
    ros_communicator, ros_thread = _init_ros_node()
    yolo_model = YoloDetectionModel()
    camera_parameters = CameraParameters()
    object_detect_manager = ObjectDetectManager(ros_communicator, yolo_model)
    boundingbox_visualizer = BoundingBoxVisualizer(ros_communicator, object_detect_manager, camera_parameters)
    camera_geometry = CameraGeometry(camera_parameters, object_detect_manager)

    while(1):
        boundingbox_visualizer.draw_bounding_boxes(screenshot_mode=False, draw_crosshair=True)
        depth = object_detect_manager.get_yolo_object_depth()
        c = camera_geometry.calculate_3d_position()

if __name__ == '__main__':
    main()