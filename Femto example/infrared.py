# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

import cv2
import numpy as np
from pyorbbecsdk import *

ESC_KEY = 27


def process_ir_frame(ir_frame, is_dual_ir=False):
    if ir_frame is None:
        return None
    ir_frame = ir_frame.as_video_frame()
    ir_data = np.asanyarray(ir_frame.get_data())
    width = ir_frame.get_width()
    height = ir_frame.get_height()
    ir_format = ir_frame.get_format()

    if ir_format == OBFormat.Y8:
        ir_data = np.resize(ir_data, (height, width, 1))
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
    elif ir_format == OBFormat.MJPG:
        ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
        if ir_data is None:
            print("decode mjpeg failed")
            return None
        ir_data = np.resize(ir_data, (height, width, 1))
    else:
        ir_data = np.frombuffer(ir_data, dtype=np.uint16)
        data_type = np.uint16
        image_dtype = cv2.CV_16UC1
        max_data = 65535
        ir_data = np.resize(ir_data, (height, width, 1))

    cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
    ir_data = ir_data.astype(data_type)
    result = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)
    
    if is_dual_ir:
        # Scale image to 640x400
        target_width = 640
        target_height = 400

        aspect_ratio = width / height
        target_ratio = target_width / target_height
        
        # Determine how to scale based on aspect ratio
        if aspect_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Determine whether to enlarge or reduce, and choose the appropriate interpolation method
        if width > new_width or height > new_height:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
            
        # Scale
        scaled = cv2.resize(result, (new_width, new_height), interpolation=interpolation)
        
        # Create a black background, place the scaled image on a black background
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled
        
    return result


def main():
    config = Config()
    pipeline = Pipeline()
    device = pipeline.get_device()
    sensor_list = device.get_sensor_list()

    has_dual_ir = False
    for sensor in range(len(sensor_list)):
        if (sensor_list[sensor].get_type() == OBSensorType.LEFT_IR_SENSOR or
                sensor_list[sensor].get_type() == OBSensorType.RIGHT_IR_SENSOR):
            has_dual_ir = True
            break

    if has_dual_ir:
        config.enable_video_stream(OBSensorType.LEFT_IR_SENSOR)
        config.enable_video_stream(OBSensorType.RIGHT_IR_SENSOR)
    else:
        config.enable_video_stream(OBSensorType.IR_SENSOR)

    pipeline.start(config)

    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            if has_dual_ir:
                left_ir_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
                right_ir_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)

                left_image = process_ir_frame(left_ir_frame, True)
                right_image = process_ir_frame(right_ir_frame, True)

                if left_image is None or right_image is None:
                    continue

                combined_ir = np.hstack((left_image, right_image))
                cv2.imshow("Dual IR", combined_ir)
            else:
                ir_frame = frames.get_frame(OBFrameType.IR_FRAME)
                ir_image = process_ir_frame(ir_frame)
                if ir_image is not None:
                    cv2.imshow("IR", ir_image)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break

    pipeline.stop()


if __name__ == "__main__":
    main()
