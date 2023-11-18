import cv2
import pyrealsense2
from realsense_depth import *

point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera()

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
depth_point=[0,0,0]
while True:
    # ret, depth_frame, color_frame = dc.get_frame()

    # Show distance for a specific point
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    distance = depth_image[point[1], point[0]]
    depth_point=rs.rs2_deproject_pixel_to_point(depth_intrin, point, distance)
    
    
    cv2.circle(color_image, point, 4, (0, 0, 255))
    cv2.putText(color_image, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.imshow("depth frame", depth_image)
    cv2.imshow("Color frame", color_image)
    print(depth_point)
    key = cv2.waitKey(1)
    if key == 27:
        break