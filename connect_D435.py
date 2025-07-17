# connect_D435.py (출력 단위를 cm로 변경)

import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d

try:
    from picking_point import PickingProcessor
except ImportError:
    print("오류: 'picking_point.py' 파일을 찾을 수 없습니다.")
    print("수정된 picking_point.py 스크립트를 현재 디렉토리에 저장해주세요.")
    exit()

# --- 3D 시각화 및 변환을 위한 헬퍼 함수 ---

def get_rotation_matrix_from_vectors(vec1, vec2):
    """두 벡터 사이의 회전 행렬을 계산합니다."""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-6: return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def create_arrow(p_start, p_end, shaft_radius=0.001, color=[1, 0, 0]):
    """Open3D를 사용하여 화살표 지오메트리를 생성합니다."""
    vec = p_end - p_start
    length = np.linalg.norm(vec)
    if length < 1e-6: return None
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=shaft_radius, cone_radius=shaft_radius * 2,
        cylinder_height=length * 0.8, cone_height=length * 0.2)
    arrow.paint_uniform_color(color)
    rotation_matrix = get_rotation_matrix_from_vectors([0, 0, 1], vec)
    arrow.rotate(rotation_matrix, center=(0, 0, 0))
    arrow.translate(p_start)
    return arrow

def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg, order='zyx'):
    """오일러 각 (도 단위)을 3x3 회전 행렬로 변환합니다."""
    roll_rad, pitch_rad, yaw_rad = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(roll_rad), -np.sin(roll_rad)], [0, np.sin(roll_rad), np.cos(roll_rad)]])
    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)], [0, 1, 0], [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
    if order.lower() == 'zyx': return Rz @ Ry @ Rx
    raise ValueError(f"지원하지 않는 회전 순서: {order}.")


def run_robot_task_with_realsense():
    # 1. 모델 및 카메라 초기화
    processor = PickingProcessor()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    # 2. 좌표계 설정
    R_EF_CAM = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    t_EFCAM = np.array([-80, 32.5, 34.5]) / 1000.0
    T_EF_CAM = np.eye(4); T_EF_CAM[:3, :3], T_EF_CAM[:3, 3] = R_EF_CAM, t_EFCAM
    print("--- 카메라-엔드이펙터 변환 행렬(T_EF_CAM) ---\n", T_EF_CAM)

    try:
        while True:
            # 3. 프레임 획득
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame, depth_frame = aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame: continue

            color_image_bgr = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense Feed', color_image_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                print("\n'C' 키 입력 감지! 처리 시작...")
                
                # 4. 데이터 준비
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                rgb_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
                depth_image = np.asanyarray(depth_frame.get_data())

                # 5. 2D 분할 수행
                detections = processor.detect_and_segment(rgb_image=rgb_image)

                if detections:
                    object_analysis_results = []
                    # 6. 각 객체에 대해 포인트 클라우드 생성 및 3D 분석 요청
                    for det in detections:
                        mask = det['mask']
                        rows, cols = np.where(mask)
                        depth_values = depth_image[rows, cols]
                        
                        valid_indices = depth_values > 0
                        if np.sum(valid_indices) < 10: continue
                        
                        rows, cols, depth_values = rows[valid_indices], cols[valid_indices], depth_values[valid_indices]
                        
                        # 포인트 클라우드 생성
                        points = np.zeros((len(rows), 3))
                        points[:, 0] = (cols - depth_intrinsics.ppx) * depth_values / (depth_intrinsics.fx * 1000.0)
                        points[:, 1] = (rows - depth_intrinsics.ppy) * depth_values / (depth_intrinsics.fy * 1000.0)
                        points[:, 2] = depth_values / 1000.0
                        
                        # 생성된 포인트 클라우드를 전달하여 PCA 분석 요청
                        analysis_result = processor.analyze_point_cloud(points)
                        if analysis_result:
                            object_analysis_results.append(analysis_result)

                    if not object_analysis_results:
                        print("유효한 객체에 대한 3D 분석 결과를 얻지 못했습니다.")
                        continue

                    # 7. 타겟 객체 선정 및 결과 처리
                    if len(object_analysis_results) > 1:
                        all_centers = [data['center_cam'] for data in object_analysis_results]
                        distances = [min(np.linalg.norm(c - other_c) for other_c in all_centers if not np.array_equal(c, other_c)) for c in all_centers]
                        target_info = object_analysis_results[np.argmax(distances)]
                    else:
                        target_info = object_analysis_results[0]
                    
                    P_CAM = target_info['center_cam']
                    # P_CAM을 cm로 변환하여 출력
                    P_CAM_cm = P_CAM * 100
                    print(f"\n[측정] 카메라 기준 객체 좌표 (P_CAM): [{P_CAM_cm[0]:.2f}, {P_CAM_cm[1]:.2f}, {P_CAM_cm[2]:.2f}] (cm)")

                    # 8. 로봇 좌표계로 변환
                    current_ef_pos_mm = np.array([431.77, 0, 446.860 ])
                    current_ef_rot_deg = np.array([0, 160, 0])
                    
                    t_BASE_EF = current_ef_pos_mm / 1000.0
                    R_BASE_EF = euler_to_rotation_matrix(*current_ef_rot_deg)
                    T_BASE_EF = np.eye(4); T_BASE_EF[:3, :3], T_BASE_EF[:3, 3] = R_BASE_EF, t_BASE_EF
                    
                    P_BASE = (T_BASE_EF @ T_EF_CAM @ np.append(P_CAM, 1.0))[:3]
                    
                    # ##################################################################
                    # ### 수정된 부분 ###
                    # ##################################################################
                    # P_BASE의 각 요소를 cm로 변환(*100)하여 소수점 2자리까지 출력합니다.
                    print(f"\n[최종 결과] 로봇 베이스 원점(0,0,0) 기준 객체 좌표:")
                    print(f"  - X 좌표: {P_BASE[0] * 100:.2f} cm")
                    print(f"  - Y 좌표: {P_BASE[1] * 100:.2f} cm")
                    print(f"  - Z 좌표: {P_BASE[2] * 100:.2f} cm\n")
                    # ##################################################################

                    # 9. 통합 3D 시각화
                    T_BASE_CAM = T_BASE_EF @ T_EF_CAM
                    
                    # 전체 씬 포인트 클라우드
                    o3d_color = o3d.geometry.Image(rgb_image)
                    o3d_depth = o3d.geometry.Image(depth_image)
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, depth_scale=1000.0, convert_rgb_to_intensity=False)
                    intr = o3d.camera.PinholeCameraIntrinsic(depth_intrinsics.width, depth_intrinsics.height, depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)
                    full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr).voxel_down_sample(0.005)
                    
                    base_frame_geometries = [full_pcd.transform(T_BASE_CAM)]
                    
                    # 로봇 베이스 좌표계 (크기: 0.1)
                    base_frame_geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
                    
                    # 분석 결과 시각화 요소 추가
                    center, v1, arrow_vec = target_info['center_cam'], target_info['direction'], target_info['arrow_direction']
                    arrow_geom = create_arrow(center - arrow_vec * 0.05, center + arrow_vec * 0.05, shaft_radius=0.002, color=[0, 0, 1])
                    if arrow_geom: base_frame_geometries.append(arrow_geom.transform(T_BASE_CAM))

                    object_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    object_marker.paint_uniform_color([1, 0, 0])
                    object_marker.translate(P_BASE)
                    base_frame_geometries.append(object_marker)

                    # 카메라 위치를 나타내는 좌표계 추가 (크기: 0.05)
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
                    camera_frame.transform(T_BASE_CAM)
                    base_frame_geometries.append(camera_frame)

                    # 엔드이펙터 위치를 나타내는 좌표계 추가 (크기: 0.05)
                    end_effector_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
                    end_effector_frame.transform(T_BASE_EF)
                    base_frame_geometries.append(end_effector_frame)

                    print("통합 3D 시각화 창을 표시합니다...")
                    o3d.visualization.draw_geometries(base_frame_geometries, window_name="Robot Base Coordinate System View")
                else:
                    print("탐지된 객체가 없습니다.")
            
            elif key == ord('q'):
                print("\n'Q' 키 입력 감지. 프로그램 종료.")
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_robot_task_with_realsense()