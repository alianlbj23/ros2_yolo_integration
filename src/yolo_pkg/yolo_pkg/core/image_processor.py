from sensor_msgs.msg import CompressedImage, Image, Imu
import numpy as np
from cv_bridge import CvBridge

class ImageProcessor():
    def __init__(self, ros_communicator):
        self.bridge = CvBridge()
        self.ros_communicator = ros_communicator
        
    def _convert_image_from_ros_to_cv(self, img, mode):
        """Converts ROS image to OpenCV format (np.ndarray)."""
        try:
            # 檢查影像格式並選擇對應的轉換方法
            if isinstance(img, CompressedImage):
                cv_image = self.bridge.compressed_imgmsg_to_cv2(img)
            elif isinstance(img, Image):
                encoding = 'bgr8' if mode == "rgb" else '16UC1'
                cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding=encoding)
            else:
                raise TypeError("Unsupported image type. Expected CompressedImage or Image.")

            if not isinstance(cv_image, np.ndarray):
                raise TypeError("Converted image is not a valid numpy array.")

            return cv_image

        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    
    def _convert_image_from_cv_to_ros(self, img):
        return self.bridge.cv2_to_compressed_imgmsg(img)
    
    def get_rgb_ros_image(self, img):
        ros_img = self._convert_image_from_cv_to_ros(img)
        return ros_img
        
    def get_depth_cv_image(self):
        """Fetch and convert the depth image from ROS to OpenCV format."""
        image = self.ros_communicator.get_latest_depth_image()
        return self._convert_image_to_cv(image, mode="depth")

    def get_rgb_cv_image(self):
        image = self.ros_communicator.get_latest_rgb_image()
        return self._convert_image_from_ros_to_cv(image, mode="rgb")
    
    