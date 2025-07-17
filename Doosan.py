import ctypes
import time
import math
import numpy as np
import threading

# DRFL.h에 정의된 상수
MANAGE_ACCESS_CONTROL_REQUEST = 0
MANAGE_ACCESS_CONTROL_RESPONSE = 1
ROBOT_MODE_AUTO = 1
ROBOT_MODE_AUTONOMOUS = 1
CONTROL_INIT_CONFIG = 0
CONTROL_ENABLE_OPERATION = 1
CONTROL_RESET_SAFET_STOP = 2
CONTROL_RESET_SAFE_STOP = 2
CONTROL_RESET_SAFET_OFF = 3
CONTROL_RESET_SAFE_OFF = 3
CONTROL_SERVO_ON = 3
CONTROL_RECOVERY_SAFE_STOP = 5
CONTROL_RECOVERY_SAFE_OFF = 6
CONTROL_RECOVERY_BACKDRIVE = 7
CONTROL_RESET_RECOVERY = 8
CONTROL_LAST = 9
STATE_INITIAL = 0
STATE_STANDBY = 1
STATE_MOVING = 2
STATE_SAFE_OFF = 3
STATE_TEACHING = 4
STATE_RECOVERY = 5
STATE_ERROR = 6

# DLL 로드
dsr = ctypes.WinDLL(r"C:\Users\robot\Desktop\Segmentation\DRFLWin64.dll")
create_robot_control = dsr._CreateRobotControl
create_robot_control.restype = ctypes.c_void_p
create_robot_control.argtypes = []
open_connection = dsr._OpenConnection
open_connection.restype = ctypes.c_bool
open_connection.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
ManageAccessControl = dsr._ManageAccessControl
ManageAccessControl.restype = ctypes.c_bool
ManageAccessControl.argtypes = [ctypes.c_void_p, ctypes.c_int]
set_robot_mode = dsr._SetRobotMode
set_robot_mode.restype = ctypes.c_bool
set_robot_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
set_robot_control = dsr._SetRobotControl
set_robot_control.restype = ctypes.c_bool
set_robot_control.argtypes = [ctypes.c_void_p, ctypes.c_int]
get_robot_state = dsr._GetRobotState
get_robot_state.restype = ctypes.c_int
get_robot_state.argtypes = [ctypes.c_void_p]
close_connection = dsr._CloseConnection
close_connection.restype = None
close_connection.argtypes = [ctypes.c_void_p]
destroy_robot_control = dsr._DestroyRobotControl
destroy_robot_control.restype = None
destroy_robot_control.argtypes = [ctypes.c_void_p]
movej = dsr._MoveJ
movej.restype = ctypes.c_bool
movej.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_int]
movel = dsr._MoveL
movel.restype = ctypes.c_bool
movel.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int
]

get_current_pose = dsr._GetCurrentPose
get_current_pose.restype = ctypes.POINTER(ctypes.c_float)
get_current_pose.argtypes = [ctypes.c_void_p, ctypes.c_int]
set_current_tool = dsr._SetCurrentTool
set_current_tool.restype = ctypes.c_bool
set_current_tool.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
get_current_tool = dsr._GetCurrentTool
get_current_tool.restype = ctypes.c_char_p
get_current_tool.argtypes = [ctypes.c_void_p]
config_create_tool = dsr._ConfigCreateTool
config_create_tool.restype = ctypes.c_bool
config_create_tool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
config_create_tcp = dsr._ConfigCreateTCP
config_create_tcp.restype = ctypes.c_bool
config_create_tcp.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
get_current_tcp = dsr._GetCurrentTCP
get_current_tcp.restype = ctypes.c_char_p
get_current_tcp.argtypes = [ctypes.c_void_p]

# 로봇 상태 확인 함수
def get_robot_current_state(robot_ctrl):
    return get_robot_state(robot_ctrl)

# DH 파라미터 (mm 단위, degree 단위)
DH_PARAMS = [
    {'alpha': 0, 'a': 0, 'd': 155.5, 'theta_offset': 0},    # Joint 1
    {'alpha': -90, 'a': 0, 'd': 409, 'theta_offset': -90},  # Link 1
    {'alpha': 0, 'a': 367, 'd': 0, 'theta_offset': 90},     # Joint 2
    {'alpha': 90, 'a': 0, 'd': 0, 'theta_offset': 0},       # Link 2
    {'alpha': -90, 'a': 0, 'd': 0, 'theta_offset': 0},      # Joint 3
    {'alpha': 90, 'a': 0, 'd': 124, 'theta_offset': 0}      # End
]

def deg_to_rad(deg):
    return deg * math.pi / 180.0
def monitor_robot_state(robot_ctrl, stop_event):
    while not stop_event.is_set():
        current_state = get_robot_current_state(robot_ctrl)
        #print(f"\r현재 로봇 상태 (쓰레드): {current_state}", end="")
        time.sleep(1)
    #print("\n로봇 상태 모니터링 종료")
def get_transform_matrix(theta, d, a, alpha):
    alpha_rad = deg_to_rad(alpha)
    theta_rad = deg_to_rad(theta)
    transform = np.array([
        [math.cos(theta_rad), -math.sin(theta_rad), 0, a],
        [math.sin(theta_rad), math.cos(alpha_rad), 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])
    rotation_alpha = np.array([
        [1, 0, 0, 0],
        [0, math.cos(alpha_rad), -math.sin(alpha_rad), 0],
        [0, math.sin(alpha_rad), math.cos(alpha_rad), 0],
        [0, 0, 0, 1]
    ])
    return transform @ rotation_alpha

def calculate_forward_kinematics(joint_angles_deg):
    transform = np.eye(4)
    for i in range(len(DH_PARAMS)):
        theta = joint_angles_deg[i] + DH_PARAMS[i]['theta_offset']
        d = DH_PARAMS[i]['d']
        a = DH_PARAMS[i]['a']
        alpha = DH_PARAMS[i]['alpha']
        transform = transform @ get_transform_matrix(theta, d, a, alpha)
    return transform

def rpy_to_rotation_matrix(rpy):
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])

    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])

    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx

# 메인 로직
if __name__ == "__main__":
    # 제어 핸들 생성
    robot_ctrl = create_robot_control()

    if not robot_ctrl:
        print("로봇 제어 핸들 생성 실패!")
        exit()

    print("로봇 제어 핸들 생성 성공!")

    # 연결 시도
    ip_address = b"192.168.137.100"
    port = 12345

    if not open_connection(robot_ctrl, ip_address, port):
        print("로봇 연결 실패!")
        destroy_robot_control(robot_ctrl)
        exit()

    print("로봇 연결 성공!")

    # 접근 권한 요청
    if not ManageAccessControl(robot_ctrl, MANAGE_ACCESS_CONTROL_REQUEST):
        print("접근 권한 설정 실패!")
        close_connection(robot_ctrl)
        destroy_robot_control(robot_ctrl)
        exit()

    print("접근 권한 설정 성공!")
    time.sleep(1)

    # 서보 온 시도 및 자동 모드 설정
    if set_robot_control(robot_ctrl, CONTROL_SERVO_ON):
        print("서보 온 시도 성공")
    else:
        print("서보 온 시도 실패")
        close_connection(robot_ctrl)
        destroy_robot_control(robot_ctrl)
        exit()

    # 서보 온 완료 및 상태 변화 대기 (최대 5초)
    servo_on_complete = False
    start_time = time.time()
    while time.time() - start_time < 7.0:
        if get_robot_current_state(robot_ctrl) == STATE_STANDBY:
            servo_on_complete = True
            break
        time.sleep(0.1)

    if servo_on_complete:
        print("서보 활성화 완료, 자동 모드 설정 시도")
        if set_robot_mode(robot_ctrl, ROBOT_MODE_AUTONOMOUS):
            print("자동 모드 설정 성공")
        else:
            print("자동 모드 설정 실패")
            close_connection(robot_ctrl)
            destroy_robot_control(robot_ctrl)
            exit()
    else:
        print("서보 활성화 대기 시간 초과!")
        close_connection(robot_ctrl)
        destroy_robot_control(robot_ctrl)
        exit()
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_robot_state, args=(robot_ctrl, stop_event))
    monitor_thread.daemon = True  # 메인 스레드가 종료될 때 함께 종료되도록 설정
    monitor_thread.start()
    # 로봇 끝단에서 엔드 이펙터까지의 상대적인 위치 (mm 단위, 요구된 차이에 맞게 조정)
    ee_relative_position = np.array([-70.15,1.995,56.802])  # ΔX=1.995, ΔY=-56.802, ΔZ=70.15

    # 로봇 끝단에서 엔드 이펙터까지의 상대적인 회전 (Roll, Pitch, Yaw - 라디안)
    ee_relative_rotation_rpy = np.array([math.radians(0.0), math.radians(0.0), math.radians(0.0)])

    # 유효한 메인 메뉴 번호 목록
    valid_main_numbers = [0, 1, 2, 3, 4,5, 99]
    # 유효한 Setting 메뉴 번호 목록
    valid_setting_numbers = [1, 2, 3, 4, 5, 6, 99]

    while True:
        # 메인 메뉴 출력
        print("=" * 20)
        print("실행할 함수 번호를 입력하세요:")
        print("1. MoveJ 실행")
        
        print("2. 현재 로봇 위치(Joint)")
        print("3. 현재 로봇 위치(XYZ - Tool 유/무)")
        print("4. Setting")
        print("5. MoveL 실행")
        print("0. 현재 상태")
        print("99. 종료")
        print("=" * 20)

        # 메인 메뉴 입력 검증
        try:
            function_number = input("함수 번호: ").strip()
            function_number = int(function_number)
            if function_number not in valid_main_numbers:
                print("잘못된 번호입니다. 유효한 번호(0, 1~4, 99)를 입력하세요.")
                continue
        except ValueError:
            print("숫자를 입력하세요. 유효한 번호(0, 1~4, 99)를 입력하세요.")
            continue

        if function_number == 1:
            # MoveJ 파라미터 입력 받기
            try:
                target_pos = input("Target Position (6 float values separated by space, in mm): ").split()
                if len(target_pos) != 6:
                    print("6개의 값을 입력하세요.")
                    continue
                target_pos = [float(x) for x in target_pos]
                target_pos_ctype = (ctypes.c_float * 6)(*target_pos)
                target_vel = 10
                target_acc = 10
                target_time = 0.0
                move_mode = 0
                blending_radius = 0.0
                blending_type = 0

                if movej(robot_ctrl, target_pos_ctype, target_vel, target_acc, target_time, move_mode, blending_radius, blending_type):
                    print("MoveJ 실행 성공")
                else:
                    print("MoveJ 실행 실패")
            except ValueError:
                print("유효한 숫자를 입력하세요.")
                continue
        elif function_number == 2:
            # 현재 로봇 위치 가져오기 (Joint)
            current_pose = get_current_pose(robot_ctrl, 0)
            if current_pose:
                print("현재 로봇 위치 (Joint):")
                for i in range(6):
                    print(f"Joint {i+1}: {current_pose[i]:.4f} deg")
            else:
                print("현재 로봇 위치 가져오기 실패")
        elif function_number == 3:
            # 현재 로봇 위치 가져오기 (XYZ - Tool 유/무, 순기구학 기반)
            current_pose = get_current_pose(robot_ctrl, 1)  # 0: 관절 좌표, 1: 작업 좌표
            if current_pose:
                print("현재 로봇 위치:")
                for i in range(3):
                    print(f"XYZ {i+1}: {current_pose[i]}")
            else:
                print("현재 로봇 위치 가져오기 실패")
            
            # if joint_angles_ptr:
            #     joint_angles_deg = np.ctypeslib.as_array(joint_angles_ptr, (6,))
            #     # 로봇 끝단 위치 (Tool 없음)
            #     transform_wf = calculate_forward_kinematics(joint_angles_deg)
            #     robot_x = transform_wf[0, 3]
            #     robot_y = transform_wf[1, 3]
            #     robot_z = transform_wf[2, 3]

            #     # 엔드 이펙터 위치 (Tool 있음)
            #     rotation_fe = rpy_to_rotation_matrix(ee_relative_rotation_rpy)
            #     transform_fe = np.eye(4)
            #     transform_fe[:3, :3] = rotation_fe
            #     transform_fe[:3, 3] = ee_relative_position
            #     transform_we = transform_wf @ transform_fe
            #     ee_x = transform_we[0, 3]
            #     ee_y = transform_we[1, 3]
            #     ee_z = transform_we[2, 3]

            #     # 결과 출력
            #     print("\n현재 로봇 위치 (XYZ - Tool 없음, 순기구학):")
            #     print(f"X: {robot_x:.4f} mm")
            #     print(f"Y: {robot_y:.4f} mm")
            #     print(f"Z: {robot_z:.4f} mm")
            #     print("\n엔드 이펙터 위치 (XYZ - Tool 있음, 순기구학):")
            #     print(f"X: {ee_x:.4f} mm")
            #     print(f"Y: {ee_y:.4f} mm")
            #     print(f"Z: {ee_z:.4f} mm")
            #     # 차이 출력
            #     print("\nTool 유/무 간 위치 차이:")
            #     print(f"ΔX: {ee_x - robot_x:.4f} mm (요구: 1.995 mm)")
            #     print(f"ΔY: {ee_y - robot_y:.4f} mm (요구: -56.802 mm)")
            #     print(f"ΔZ: {ee_z - robot_z:.4f} mm (요구: 70.15 mm)")
            # else:
            #     print("현재 로봇 관절 각도 가져오기 실패")
        elif function_number == 4:
            # Setting 메뉴
            while True:
                print("=" * 20)
                print("Setting 메뉴:")
                print("1. Set Tool")
                print("2. Add Tool")
                print("3. Get Tool")
                print("4. Set TCP")
                print("5. Add TCP")
                print("6. Get TCP")
                print("99. 뒤로 가기")
                print("=" * 20)

                try:
                    setting_number = input("Setting 번호: ").strip()
                    setting_number = int(setting_number)
                    if setting_number not in valid_setting_numbers:
                        print("잘못된 번호입니다. 유효한 번호(1~6, 99)를 입력하세요.")
                        continue
                except ValueError:
                    print("숫자를 입력하세요. 유효한 번호(1~6, 99)를 입력하세요.")
                    continue

                if setting_number == 1:
                    tool_name = input("Tool Name to Set: ")
                    tool_name_bytes = tool_name.encode('utf-8')
                    if set_current_tool(robot_ctrl, tool_name_bytes):
                        print(f"Tool '{tool_name}' 설정 성공")
                    else:
                        print(f"Tool '{tool_name}' 설정 실패")
                elif setting_number == 2:
                    try:
                        tool_name = input("Tool Name to Add: ")
                        tool_name_bytes = tool_name.encode('utf-8')
                        tool_weight = float(input("Tool Weight (kg): "))
                        tool_cog = input("Tool Center of Gravity (3 float values in mm, separated by space): ").split()
                        if len(tool_cog) != 3:
                            print("3개의 값을 입력하세요.")
                            continue
                        tool_cog = [float(x) for x in tool_cog]
                        tool_cog_ctype = (ctypes.c_float * 3)(*tool_cog)
                        tool_inertia = [0.0] * 6
                        tool_inertia_ctype = (ctypes.c_float * 6)(*tool_inertia)
                        if config_create_tool(robot_ctrl, tool_name_bytes, tool_weight, tool_cog_ctype, tool_inertia_ctype):
                            print(f"Tool '{tool_name}' 추가 성공")
                        else:
                            print(f"Tool '{tool_name}' 추가 실패")
                    except ValueError:
                        print("유효한 숫자를 입력하세요.")
                        continue
                elif setting_number == 3:
                    current_tool = get_current_tool(robot_ctrl)
                    if current_tool:
                        print(f"현재 Tool: {current_tool.decode('utf-8')}")
                    else:
                        print("현재 Tool을 가져오지 못했습니다.")
                elif setting_number == 4:
                    tcp_name = input("TCP Name to Set: ")
                    tcp_name_bytes = tcp_name.encode('utf-8')
                    print("Set TCP 기능은 현재 지원되지 않습니다. DRFL 문서를 확인하세요.")
                elif setting_number == 5:
                    try:
                        tcp_name = input("TCP Name to Add: ")
                        tcp_name_bytes = tcp_name.encode('utf-8')
                        tcp_position = input("TCP Position (6 float values in mm, separated by space): ").split()
                        if len(tcp_position) != 6:
                            print("6개의 값을 입력하세요.")
                            continue
                        tcp_position = [float(x) for x in tcp_position]
                        tcp_position_ctype = (ctypes.c_float * 6)(*tcp_position)
                        if config_create_tcp(robot_ctrl, tcp_name_bytes, tcp_position_ctype):
                            print(f"TCP '{tcp_name}' 추가 성공")
                        else:
                            print(f"TCP '{tcp_name}' 추가 실패")
                    except ValueError:
                        print("유효한 숫자 6개를 입력하세요.")
                        continue
                elif setting_number == 6:
                    current_tcp = get_current_tcp(robot_ctrl)
                    if current_tcp:
                        print(f"현재 TCP: {current_tcp.decode('utf-8')}")
                    else:
                        print("현재 TCP를 가져오지 못했습니다.")
                elif setting_number == 99:
                    break

        if function_number == 5:
            print("MoveL 실행 중...")
            #pos = (ctypes.c_float * 6)(100.0, 200.0, 300.0, 0.0, 0.0, 0.0)
            pos = input("Target Position (6 float values separated by space, in mm): ").split()
            if len(pos) != 6:
                print("6개의 값을 입력하세요.")
                continue
            pos = [float(x) for x in pos]
            pos = (ctypes.c_float * 6)(*pos)

            vel = (ctypes.c_float * 2)(200.0, 100.0)
            acc = (ctypes.c_float * 2)(100.0, 100.0)
            res = movel(robot_ctrl, pos, vel, acc, ctypes.c_float(0.0), ctypes.c_int(0), ctypes.c_int(0), ctypes.c_float(0.0), ctypes.c_int(0))
            print("MoveL 실행 성공" if res else "MoveL 실행 실패")

        elif function_number == 0:
            current_state = get_robot_current_state(robot_ctrl)
            print(f"현재 로봇 상태: {current_state}")
        elif function_number == 99:
            break

    # 연결 종료 및 핸들 해제
    close_connection(robot_ctrl)
    destroy_robot_control(robot_ctrl)
    print("연결 종료!")
    print("로봇 제어 핸들 해제 완료!")