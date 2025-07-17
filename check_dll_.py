import ctypes
import os

dll_folder = r"C:\Users\robot\Desktop\Segmentation\dll"
dll_path = os.path.join(dll_folder, "DRFLWin64.dll")

print(f"DLL 경로: {dll_path}")
try:
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DLL 파일을 찾을 수 없습니다: {dll_path}")
    print(f"DLL 파일 존재 확인 완료: {dll_path}")
    
    os.add_dll_directory(dll_folder)
    print(f"DLL 디렉토리 추가 완료: {dll_folder}")
    
    dsr = ctypes.WinDLL(dll_path)
    print("✅ DLL 로드 성공!")
except FileNotFoundError as e:
    print(f"❌ DLL 파일 오류: {e}")
except OSError as e:
    print(f"❌ DLL 로드 실패 (종속성 문제 가능성): {e}")
    print("Dependencies 도구를 사용하여 누락된 종속성을 확인하세요.")
except Exception as e:
    print(f"❌ 알 수 없는 오류: {e}")