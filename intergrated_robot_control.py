# intergrated_robot_control_with_fk_validation.py (순기구학 검증 기능 추가)

import os
import time
import math
import numpy as np
import threading
import ctypes
import cv2
import pyrealsense2 as rs
import open3d as o3d

# PyTorch, YOLO, SAM 관련 라이브러리
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from sklearn.decomposition import PCA

# Scipy 라이브러리 임포트
from scipy.spatial.transform import Rotation as ScipyRotation

# --- [추가] 순기구학(Forward Kinematics) 계산 클래스 ---
class ForwardKinematics:
    """
    Doosan A0509 모델의 DH 파라미터를 사용하여 순기구학을 계산합니다.
    """
    def __init__(self):
        # PDF의 A0509 사양(mm)을 미터(m) 단위로 정의 
        # d1=155.5, d2=409, a1=367, d3=0, d4=124
        d1 = 0.1555
        a1 = 0.367
        d2 = 0.409 # PDF의 d2는 두 번째 링크의 길이(a2)에 해당
        d3 = 0.0
        d4 = 0.124

        # Doosan Robot A0509의 표준 DH 파라미터 (Standard DH)
        # alpha, a, d, theta 순서
        self.dh_params = [
            # alpha(i-1), a(i-1),    d(i),      theta(i) - joint variable
            [0,             0,          d1,         0],
            [np.pi/2,       a1,         0,          0],
            [0,             d2,         0,          np.pi/2], # d2가 a2 역할
            [np.pi/2,       0,          d3 + d4,    0], # d3, d4는 Z축 offset
            [-np.pi/2,      0,          0,          0],
            [np.pi/2,       0,          0,          0]
        ]

    def _create_dh_matrix(self, alpha, a, d, theta):
        """ 단일 DH 변환 행렬을 생성하는 함수 """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        T = np.array([
            [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
            [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0,          sin_alpha,             cos_alpha,            d],
            [0,          0,                     0,                    1]
        ])
        return T

    def compute_fk(self, joint_angles_deg):
        """
        주어진 6축 관절 각도(degree)를 사용하여 최종 변환 행렬을 계산합니다.
        """
        if len(joint_angles_deg) != 6:
            raise ValueError("관절 각도는 6개여야 합니다.")
            
        joint_angles_rad = np.deg2rad(joint_angles_deg)
        
        T_final = np.identity(4)
        for i in range(6):
            alpha, a, d, theta_offset = self.dh_params[i]
            # Joint variable 추가 (A0509 모델의 특정 theta offset 적용)
            theta = joint_angles_rad[i]
            if i == 2: # Joint 3
                theta += theta_offset
            
            T_i = self._create_dh_matrix(alpha, a, d, theta)
            T_final = T_final @ T_i
            
        return T_final
    
    def get_pose_from_matrix(self, T):
        """
        4x4 변환 행렬에서 위치(mm)와 오일러 각(XYZ, degree)을 추출합니다.
        """
        position = T[:3, 3] * 1000.0  # 미터를 밀리미터로 변환
        
        rotation_matrix = T[:3, :3]
        r = ScipyRotation.from_matrix(rotation_matrix)
        # 로봇 제어기와 동일한 'xyz' 순서의 오일러 각 사용
        euler_angles = r.as_euler('xyz', degrees=True)
        
        return list(position) + list(euler_angles)

# ----------------------------------------------------------------

# --- 1. 비전 처리 클래스 ---
class PickingProcessor:
    def __init__(self, sam_checkpoint_name: str = "sam_vit_h.pth", yolo_model_name: str = "yolov11m.pt"):
        self.sam_checkpoint_name = sam_checkpoint_name
        self.yolo_model_name = yolo_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor_sam = None
        self.model_yolo = None
        self._load_models()

    def _load_models(self):
        try:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                script_dir = os.getcwd()

            sam_model_path = os.path.join(script_dir, self.sam_checkpoint_name)
            if not os.path.exists(sam_model_path):
                raise FileNotFoundError(f"오류: SAM 모델 '{sam_model_path}'을 찾을 수 없습니다.")

            sam = sam_model_registry["vit_h"](checkpoint=sam_model_path)
            sam.to(self.device)
            self.predictor_sam = SamPredictor(sam)
            self.model_yolo = YOLO(self.yolo_model_name)
            print("✅ AI 모델 로드 완료.")
        except Exception as e:
            print(f"❌ AI 모델 로드 중 오류 발생: {e}")
            exit(1)

    def detect_and_segment(self, rgb_image: np.ndarray):
        self.predictor_sam.set_image(rgb_image)
        results = self.model_yolo(rgb_image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        detections = []
        if boxes.size == 0:
            return detections

        for box in boxes:
            masks, scores, _ = self.predictor_sam.predict(box=np.array(box), multimask_output=True)
            mask = masks[np.argmax(scores)]
            detections.append({'box': box, 'mask': mask})
            
        return detections

    def analyze_point_cloud(self, points: np.ndarray):
        if points is None or len(points) < 10:
            return None
        
        try:
            points = points[np.any(points != 0, axis=1)]
            if len(points) < 10:
                print("  ⚠️ PCA 분석을 위한 유효 포인트 부족")
                return None
                
            pca = PCA(n_components=3)
            pca.fit(points)
            center = pca.mean_
            long_axis_vec = pca.components_[0]
            return {'center_cam': center, 'arrow_direction': long_axis_vec}
        except Exception as e:
            print(f"❌ PCA 분석 중 오류 발생: {e}")
            return None

# --- 2. 로봇 제어 클래스 ---
class DoosanRobot:
    GRIPPER_PIN_A = 0 
    GRIPPER_PIN_B = 1

    def __init__(self, dll_path): 
        self.robot_ctrl = None
        try:
            print(f"DLL 로드 시도: {dll_path}")
            self.dsr = ctypes.WinDLL(dll_path)
            self._define_functions()
            self.robot_ctrl = self.dsr._CreateRobotControl()
            if not self.robot_ctrl:
                raise ConnectionError("로봇 제어 핸들 생성 실패")
            print("✅ 로봇 제어 핸들 생성 성공.")
        except Exception as e:
            print(f"❌ DRFL DLL 로드 또는 핸들 생성 실패: {e}")
            
    def _define_functions(self):
        self.dsr._CreateRobotControl.restype = ctypes.c_void_p
        self.dsr._OpenConnection.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self.dsr._OpenConnection.restype = ctypes.c_bool
        self.dsr._CloseConnection.argtypes = [ctypes.c_void_p]
        self.dsr._DestroyRobotControl.argtypes = [ctypes.c_void_p]
        self.dsr._ManageAccessControl.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dsr._ManageAccessControl.restype = ctypes.c_bool
        self.dsr._SetRobotMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dsr._SetRobotMode.restype = ctypes.c_bool
        self.dsr._SetRobotControl.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dsr._SetRobotControl.restype = ctypes.c_bool
        self.dsr._GetRobotState.argtypes = [ctypes.c_void_p]
        self.dsr._GetRobotState.restype = ctypes.c_int
        self.dsr._MoveL.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
        self.dsr._MoveL.restype = ctypes.c_bool
        self.dsr._GetCurrentPose.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dsr._GetCurrentPose.restype = ctypes.POINTER(ctypes.c_float * 6)
        self.dsr._SetToolDigitalOutput.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
        self.dsr._SetToolDigitalOutput.restype = ctypes.c_bool
        
        # --- [추가] _get_current_posj 함수 정의 ---
        # 이 함수는 현재 로봇의 6축 관절 각도를 반환합니다. 
        self.dsr._get_current_posj.argtypes = [ctypes.c_void_p]
        self.dsr._get_current_posj.restype = ctypes.POINTER(ctypes.c_float * 6)
        # -------------------------------------------

    def connect(self, ip="192.168.137.100", port=12345):
        if not self.robot_ctrl: return False
        if not self.dsr._OpenConnection(self.robot_ctrl, ip.encode('utf-8'), port):
            print(f"❌ 로봇 연결 실패! (IP: {ip})")
            return False
        print(f"✅ 로봇 연결 성공. (IP: {ip})")
        if not self.dsr._ManageAccessControl(self.robot_ctrl, 0):
            print("❌ 접근 권한 설정 실패!")
            self.disconnect()
            return False
        print("✅ 접근 권한 설정 성공.")
        time.sleep(1)
        if not self.dsr._SetRobotControl(self.robot_ctrl, 3):
            print("❌ 서보 온 시도 실패.")
            self.disconnect()
            return False
        start_time = time.time()
        while time.time() - start_time < 7.0:
            if self.dsr._GetRobotState(self.robot_ctrl) == 1:
                print("✅ 서보 활성화 완료.")
                if self.dsr._SetRobotMode(self.robot_ctrl, 1):
                    print("✅ 자동 모드 설정 성공.")
                    return True
                else:
                    print("❌ 자동 모드 설정 실패.")
                    self.disconnect()
                    return False
            time.sleep(0.1)
        print("❌ 서보 활성화 시간 초과!")
        self.disconnect()
        return False

    def disconnect(self):
        if self.robot_ctrl:
            self.dsr._CloseConnection(self.robot_ctrl)
            self.dsr._DestroyRobotControl(self.robot_ctrl)
            print("✅ 로봇 연결 종료 및 핸들 해제 완료.")
    
    def get_current_pose_tcp(self):
        if not self.robot_ctrl: return None
        pose_ptr = self.dsr._GetCurrentPose(self.robot_ctrl, 1)
        if pose_ptr:
            return list(pose_ptr.contents)
        return None

    # --- [추가] 현재 관절 각도(joint)를 가져오는 메서드 ---
    def get_current_joint_angles(self):
        if not self.robot_ctrl: return None
        # _get_current_posj 함수를 호출하여 관절 각도를 가져옵니다.
        joint_ptr = self.dsr._get_current_posj(self.robot_ctrl)
        if joint_ptr:
            return list(joint_ptr.contents)
        print("❌ 현재 관절 각도 읽기 실패.")
        return None
    # ----------------------------------------------------

    def move_l(self, pos, vel=100.0, acc=100.0, wait=True):
        if not self.robot_ctrl or len(pos) != 6: return False
        pos_ctype = (ctypes.c_float * 6)(*pos)
        vel_ctype = (ctypes.c_float * 2)(vel, vel)
        acc_ctype = (ctypes.c_float * 2)(acc, acc)
        res = self.dsr._MoveL(self.robot_ctrl, pos_ctype, vel_ctype, acc_ctype, 0.0, 0, 0, 0.0, 0)
        
        if res and wait:
            # 이동 명령이 완료될 때까지 기다립니다.
            time.sleep(0.2) 
            while self.dsr._GetRobotState(self.robot_ctrl) == 2: # STATE_MOVING
                time.sleep(0.1)
        elif not res:
            print("❌ MoveL 실행 실패.")
        return res

    def move_to_home(self):
        print("\n🤖 초기 위치(준비 자세)로 이동을 시작합니다...")
        
        home_pos_l = [487,51,448,8,164,7] 

        print(f"  -> 명령 좌표 (X,Y,Z,R,P,Y): {home_pos_l}")
        self.move_l(home_pos_l, vel=150, acc=150)
        print("✅ 초기 위치 이동 명령 완료.")
        time.sleep(0.5)
        self.open_gripper()

    def open_gripper(self):
        self.dsr._SetToolDigitalOutput(self.robot_ctrl, self.GRIPPER_PIN_A, False)
        self.dsr._SetToolDigitalOutput(self.robot_ctrl, self.GRIPPER_PIN_B, False)
        time.sleep(0.5)
        
        self.dsr._SetToolDigitalOutput(self.robot_ctrl, self.GRIPPER_PIN_A, True)
        time.sleep(0.5)
        self.dsr._SetToolDigitalOutput(self.robot_ctrl, self.GRIPPER_PIN_B, False)
        print("👐 그리퍼 열기 완료.")

    def close_gripper(self):
        self.dsr._SetToolDigitalOutput(self.robot_ctrl, self.GRIPPER_PIN_A, False)
        self.dsr._SetToolDigitalOutput(self.robot_ctrl, self.GRIPPER_PIN_B, False)
        time.sleep(0.5)
        
        self.dsr._SetToolDigitalOutput(self.robot_ctrl, self.GRIPPER_PIN_A, False)
        time.sleep(0.5)
        self.dsr._SetToolDigitalOutput(self.robot_ctrl, self.GRIPPER_PIN_B, True)
        print("🤏 그리퍼 닫기 완료.")


# --- 3. 통합 제어기 클래스 (메인 로직) ---
class IntegratedController:
    APPROACH_HEIGHT_M = 0.20 
    Z_LOWER_LIMIT_MM = 160

    def __init__(self, robot_dll_path): 
        self.vision = PickingProcessor()
        self.robot = DoosanRobot(robot_dll_path) 
        # --- [추가] 순기구학 계산기 인스턴스 생성 ---
        self.fk_calculator = ForwardKinematics()
        # -----------------------------------------
        self.target_base_pose = None
        self.approach_pose = None 
        self.last_object_yaw_deg = None
        self.pc = rs.pointcloud()
        self._initialize_realsense()
        
        self.R_EF_CAM = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        self.t_EF_CAM = np.array([-80, 32.5, 34.5]) / 1000.0
        
        self.T_EF_CAM = np.eye(4)
        self.T_EF_CAM[:3, :3], self.T_EF_CAM[:3, 3] = self.R_EF_CAM, self.t_EF_CAM
        self.gripper_offset_vector_ef = np.array([0, 0, 0.146])

    def _initialize_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        print("✅ RealSense 카메라 초기화 완료.")

    def _euler_to_rotation_matrix(self, roll_deg, pitch_deg, yaw_deg):
        try:
            euler_angles_XYZ = [roll_deg, pitch_deg, yaw_deg]
            r = ScipyRotation.from_euler('xyz', euler_angles_XYZ, degrees=True)
            rotation_matrix = r.as_matrix()
            return rotation_matrix
        except Exception as e:
            print(f"❌ Scipy('xyz') 변환 중 오류 발생: {e}")
            return np.eye(3)

    def _calculate_target_pose(self, color_frame, depth_frame):
        print("\n'c' 키 입력 감지! 객체 탐지 및 최종 목표 자세 계산 시작...")
        
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        rgb_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        depth_image = np.asanyarray(depth_frame.get_data())
        
        print("  - 1. 객체 탐지 및 세그멘테이션...")
        detections = self.vision.detect_and_segment(rgb_image=rgb_image)
        if not detections: 
            print("  ❌ 1단계 실패: 탐지된 객체 없음")
            return
        
        print("  - 2. 포인트 클라우드 생성 및 분석...")
        mask = detections[0]['mask']

        self.pc.map_to(color_frame)
        points_rs = self.pc.calculate(depth_frame)
        
        vertices = np.asanyarray(points_rs.get_vertices()).reshape(depth_frame.get_height(), depth_frame.get_width())
        points_struct = vertices[mask]
        
        points = np.array([list(p) for p in points_struct])
        
        analysis_result = self.vision.analyze_point_cloud(points)
        if not analysis_result: 
            print("  ❌ 2단계 실패: PCA 분석 실패")
            return

        print("  - 3. 좌표계 변환 및 목표 위치/자세 계산...")
        current_pose_mm_deg = self.robot.get_current_pose_tcp()
        if not current_pose_mm_deg: 
            print("  ❌ 3단계 실패: 로봇 현재 위치 읽기 실패")
            return
            
        pos_mm, rot_deg = current_pose_mm_deg[:3], current_pose_mm_deg[3:]
        t_BASE_EF = np.array(pos_mm) / 1000.0
        R_BASE_EF = self._euler_to_rotation_matrix(*rot_deg)
        T_BASE_EF = np.eye(4); T_BASE_EF[:3, :3], T_BASE_EF[:3, 3] = R_BASE_EF, t_BASE_EF
        
        R_BASE_CAM = T_BASE_EF[:3, :3] @ self.T_EF_CAM[:3, :3]
        P_CAM = analysis_result['center_cam']
        P_BASE_gripper_tip_homogeneous = T_BASE_EF @ self.T_EF_CAM @ np.append(P_CAM, 1.0)
        P_BASE_gripper_tip = P_BASE_gripper_tip_homogeneous[:3]
        offset_vector_base = R_BASE_EF @ self.gripper_offset_vector_ef
        target_pos_flange = P_BASE_gripper_tip - offset_vector_base
        target_pos_flange[0] += 0.01; target_pos_flange[1] += 0.05 ;target_pos_flange[2] +=0.0
        
        obj_dir_cam = analysis_result['arrow_direction']
        obj_dir_base = R_BASE_CAM @ obj_dir_cam

        calculated_yaw_rad = math.atan2(obj_dir_base[1], obj_dir_base[0])
        correction_offset_rad = np.deg2rad(-90.0)
        final_yaw_rad = calculated_yaw_rad + correction_offset_rad
        
        final_yaw_deg = np.rad2deg(final_yaw_rad)
        pitch_deg = 180.0
        roll_deg = 0.0
        
        target_rot_deg = [roll_deg, pitch_deg, final_yaw_deg]
        
        self.target_base_pose = list(target_pos_flange * 1000.0) + list(target_rot_deg)
        self.last_object_yaw_deg = final_yaw_deg

        original_z = self.target_base_pose[2]
        if original_z < self.Z_LOWER_LIMIT_MM:
            self.target_base_pose[2] = self.Z_LOWER_LIMIT_MM
            print(f"  ⚠️  Z축 하강 한계 적용! 계산된 Z값({original_z:.2f}mm)이 한계({self.Z_LOWER_LIMIT_MM}mm)보다 낮아 조정합니다.")

        print("✅ 최종 목표 자세 계산 완료.")
        
        print("📊 3D 포인트 클라우드 시각화를 시작합니다... (창을 닫아야 다음 단계 진행)")
        self._visualize_results(rgb_image, depth_image, depth_intrinsics, T_BASE_EF, P_BASE_gripper_tip, analysis_result)

    def _move_robot_to_target(self):
        if self.target_base_pose is None:
            print("⚠️ 먼저 'c' 키를 눌러 목표 좌표를 계산해야 합니다.")
            return

        print(f"\n'm' 키 입력 감지! 객체 상단 접근 위치로 이동합니다...")
        
        final_pos_mm = np.array(self.target_base_pose[:3])
        final_rot_deg = np.array(self.target_base_pose[3:])
        approach_pos_mm = final_pos_mm + np.array([0, 0, self.APPROACH_HEIGHT_M * 1000.0])
        
        approach_robot_target = list(approach_pos_mm) + list(final_rot_deg)
        
        self.approach_pose = approach_robot_target
        
        print(f"  - 목표 지점 +{self.APPROACH_HEIGHT_M*100:.0f}cm 로 이동합니다.")
        print(f"  - 접근 목표: Pos(mm)={np.round(self.approach_pose[:3], 2)}, Rot(deg)={np.round(self.approach_pose[3:], 2)}")
        self.robot.move_l(self.approach_pose, vel=150, acc=150)
        print(f"✅ 접근 위치로 이동 완료.")
        
    def _move_down_and_pick(self):
        if self.target_base_pose is None or self.approach_pose is None:
            print("⚠️ 먼저 'c'와 'm'키를 순서대로 눌러 목표 및 접근 좌표를 설정해야 합니다.")
            return

        print("\n'd' 키 입력 감지! 파지 동작을 시작합니다...")
        
        print("  - 1. 최종 목표 위치로 하강...")
        final_target_pose = self.target_base_pose
        print(f"    - 최종 목표: Pos(mm)={np.round(final_target_pose[:3], 2)}, Rot(deg)={np.round(final_target_pose[3:], 2)}")
        self.robot.move_l(final_target_pose, vel=80, acc=80)
        print("  ✅ 하강 완료.")
        time.sleep(0.5)

        # --- [추가] 순기구학 기반 검증 로직 ---
        self.validate_pose_with_fk()
        # ------------------------------------

        print("  - 2. 그리퍼를 닫아 파지합니다...")
        self.robot.close_gripper()
        time.sleep(1.0) 

        print("  - 3. 접근 위치로 복귀합니다...")
        print(f"    - 복귀 목표: Pos(mm)={np.round(self.approach_pose[:3], 2)}, Rot(deg)={np.round(self.approach_pose[3:], 2)}")
        self.robot.move_l(self.approach_pose, vel=150, acc=150)
        print("✅ 파지 및 복귀 동작 완료.")

    # --- [추가] 순기구학을 이용한 자세 검증 및 오차 계산 메서드 ---
    def validate_pose_with_fk(self):
        print("\n--- 🤖 순기구학(FK) 기반 좌표 검증 시작 ---")
        
        # 1. 현재 로봇의 실제 관절 각도(degree)를 가져옵니다.
        actual_joint_angles = self.robot.get_current_joint_angles()
        if actual_joint_angles is None:
            print("❌ 검증 실패: 현재 관절 각도를 읽어올 수 없습니다.")
            return
            
        # 2. FK 계산기를 사용하여 관절 각도로부터 TCP 좌표를 계산합니다.
        fk_matrix = self.fk_calculator.compute_fk(actual_joint_angles)
        fk_pose = self.fk_calculator.get_pose_from_matrix(fk_matrix)
        
        # 3. 비전으로 계산했던 목표 좌표와 비교합니다.
        vision_target_pose = self.target_base_pose
        
        # 4. 결과 출력
        print(f"  - 현재 관절 각도 (deg): {[f'{q:.2f}' for q in actual_joint_angles]}")
        print("-" * 30)
        print(f"  - 비전 목표 좌표 (X,Y,Z): {[f'{p:.2f}' for p in vision_target_pose[:3]]} (mm)")
        print(f"  - FK 계산 좌표 (X,Y,Z) : {[f'{p:.2f}' for p in fk_pose[:3]]} (mm)")
        print("-" * 30)
        print(f"  - 비전 목표 자세 (R,P,Y): {[f'{r:.2f}' for r in vision_target_pose[3:]]} (deg)")
        print(f"  - FK 계산 자세 (R,P,Y) : {[f'{r:.2f}' for r in fk_pose[3:]]} (deg)")
        print("-" * 30)
        
        # 5. 오차 계산
        pos_error = np.linalg.norm(np.array(vision_target_pose[:3]) - np.array(fk_pose[:3]))
        
        # 자세 오차는 각 축별로 단순 차이를 계산
        rot_error_r = abs(vision_target_pose[3] - fk_pose[3])
        rot_error_p = abs(vision_target_pose[4] - fk_pose[4])
        rot_error_y = abs(vision_target_pose[5] - fk_pose[5])
        
        print("🔎 오차 분석:")
        print(f"  - 위치 오차 (Euclidean Distance): {pos_error:.3f} mm")
        print(f"  - 자세 오차 (Roll, Pitch, Yaw): {rot_error_r:.3f}, {rot_error_p:.3f}, {rot_error_y:.3f} (deg)")
        print("--- 검증 종료 ---\n")
    # -------------------------------------------------------------

    def _get_rotation_matrix_from_vectors(self, vec1, vec2):
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-6: return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    def _create_arrow(self, p_start, p_end, shaft_radius=0.001, color=[1, 0, 0]):
        vec = p_end - p_start
        length = np.linalg.norm(vec)
        if length < 1e-6: return None
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=shaft_radius, cone_radius=shaft_radius * 2,
            cylinder_height=length * 0.8, cone_height=length * 0.2)
        arrow.paint_uniform_color(color)
        rotation_matrix = self._get_rotation_matrix_from_vectors([0, 0, 1], vec)
        arrow.rotate(rotation_matrix, center=(0, 0, 0))
        arrow.translate(p_start)
        return arrow

    def _visualize_results(self, rgb_image, depth_image, depth_intrinsics, T_BASE_EF, P_BASE_object, target_info):
        T_BASE_CAM = T_BASE_EF @ self.T_EF_CAM
        
        o3d_color = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        o3d_depth = o3d.geometry.Image(depth_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1000.0, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            depth_intrinsics.width, depth_intrinsics.height, depth_intrinsics.fx, 
            depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)
        full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr).voxel_down_sample(0.005)
        
        base_frame_geometries = [full_pcd.transform(T_BASE_CAM)]
        base_frame_geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
        
        if target_info and 'arrow_direction' in target_info:
            center, arrow_vec = target_info['center_cam'], target_info['arrow_direction']
            arrow_geom = self._create_arrow(center - arrow_vec * 0.05, center + arrow_vec * 0.05, shaft_radius=0.002, color=[0, 0, 1])
            if arrow_geom: 
                base_frame_geometries.append(arrow_geom.transform(T_BASE_CAM))

        object_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        object_marker.paint_uniform_color([1, 0, 0])
        object_marker.translate(P_BASE_object)
        base_frame_geometries.append(object_marker)

        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        camera_frame.transform(T_BASE_CAM)
        base_frame_geometries.append(camera_frame)

        end_effector_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        end_effector_frame.transform(T_BASE_EF)
        base_frame_geometries.append(end_effector_frame)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Robot Base Coordinate System View", width=960, height=540)
        
        for geometry in base_frame_geometries:
            vis.add_geometry(geometry)
            
        vis.run()
        vis.destroy_window()

    def run(self):
        if self.robot.robot_ctrl is None or not self.robot.connect():
            print("로봇 연결 실패. 프로그램을 종료합니다.")
            return
        time.sleep(1)
        self.robot.move_to_home()
        
        print("\n--- 조작 안내 ---\n'c': 객체 탐지 및 목표 자세 계산\n'm': 계산된 자세로 +20cm 위로 이동 (접근)\n'd': 하강 -> 파지 -> 복귀 동작 실행\n'e': 초기 위치로 복귀\n'q': 프로그램 종료\n-----------------\n")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame: continue

                color_image_bgr = np.asanyarray(color_frame.get_data())

                current_pose = self.robot.get_current_pose_tcp()
                if current_pose:
                    robot_rot_deg = current_pose[3:]
                    robot_text = f"Robot Angle (R,P,Y): {robot_rot_deg[0]:.1f}, {robot_rot_deg[1]:.1f}, {robot_rot_deg[2]:.1f}"
                    cv2.putText(color_image_bgr, robot_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if self.last_object_yaw_deg is not None:
                    obj_text = f"Target Yaw: {self.last_object_yaw_deg:.1f}"
                    cv2.putText(color_image_bgr, obj_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('RealSense Live Feed', color_image_bgr)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'): self._calculate_target_pose(color_frame, depth_frame)
                elif key == ord('m'): self._move_robot_to_target()
                elif key == ord('d'): self._move_down_and_pick()
                elif key == ord('e'): self.robot.move_to_home()
                elif key == ord('q'):
                    print("\n'q' 키 입력 감지. 프로그램을 종료합니다.")
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.robot.disconnect()


if __name__ == "__main__":
    DLL_FOLDER_PATH = r"C:\Users\robot\Desktop\Segmentation\dll"
    
    print("-" * 50)
    print("DLL 로드 전 경로 검사를 시작합니다...")
    
    try:
        if os.path.exists(DLL_FOLDER_PATH):
             os.add_dll_directory(DLL_FOLDER_PATH)
             print("✅ DLL 검색 경로 추가 완료.")
        else:
            raise FileNotFoundError
       
        full_dll_path = os.path.join(DLL_FOLDER_PATH, "DRFLWin64.dll")

        print("-" * 50)
        controller = IntegratedController(robot_dll_path=full_dll_path)
        controller.run()
            
    except FileNotFoundError:
        print(f"\n[오류] '{DLL_FOLDER_PATH}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요. ❌")
    except AttributeError:
        print("\n[오류] 'os.add_dll_directory' 함수를 찾을 수 없습니다.")
        print("   Python 3.8 이상 버전이 필요합니다. 파이썬 버전을 확인해주세요. ❌")
    except Exception as e:
        print(f"\n[오류] 경로 검사 또는 실행 중 에러 발생: {e}")
        
    print("-" * 50)m