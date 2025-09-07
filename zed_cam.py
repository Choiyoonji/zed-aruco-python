import pyzed.sl as sl

# 상수 정의
RESOLUTION = {
    "2K": sl.RESOLUTION.HD2K,
    "1080P": sl.RESOLUTION.HD1080,
    "720P": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA
}
DEPTH_MODE = {
    "NONE": sl.DEPTH_MODE.NONE,
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "NEURAL": sl.DEPTH_MODE.NEURAL
}
UNIT = {
    "METER": sl.UNIT.METER,
    "CENTIMETER": sl.UNIT.CENTIMETER,
    "MILLIMETER": sl.UNIT.MILLIMETER,
    "INCH": sl.UNIT.INCH,
    "FOOT": sl.UNIT.FOOT
}

class ZedCam:
    """
    ZED 스테레오 카메라를 쉽게 사용하기 위한 래퍼 클래스.
    'with' 구문을 지원하여 안전한 리소스 관리가 가능합니다.
    """
    def __init__(self, resolution="1080P", depth_mode="NONE", unit="METER", fps=30):
        self.cam = sl.Camera()
        
        # 초기화 파라미터 설정
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = RESOLUTION.get(resolution, sl.RESOLUTION.HD1080)
        self.init_params.depth_mode = DEPTH_MODE.get(depth_mode, sl.DEPTH_MODE.NONE)
        self.init_params.coordinate_units = UNIT.get(unit, sl.UNIT.METER)
        self.init_params.camera_fps = fps

        # 런타임 파라미터 초기화
        self.runtime_params = sl.RuntimeParameters()
        self.set_depth(enable=(depth_mode != "NONE"))

        # 이미지 매트릭스 생성
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        if self.init_params.depth_mode != sl.DEPTH_MODE.NONE:
            self.depth_image = sl.Mat()

    def open(self) -> bool:
        """카메라를 엽니다."""
        status = self.cam.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Error: {status}")
            return False
        return True
    
    def get_intrinsics(self) -> tuple | None:
        """
        카메라 내부 파라미터(Intrinsics)와 왜곡 계수를 반환합니다.
        (fx, fy, cx, cy, distortion_coeffs)
        """
        if self.cam.is_opened():
            cam_info = self.cam.get_camera_information()
            calib = cam_info.camera_configuration.calibration_parameters.left_cam
            return calib.fx, calib.fy, calib.cx, calib.cy, list(calib.disto)
        return None

    def set_depth(self, enable=True, confidence_threshold=95):
        """런타임 시 깊이 측정 관련 파라미터를 설정합니다."""
        self.runtime_params.enable_depth = enable
        if enable:
            self.runtime_params.confidence_threshold = confidence_threshold

    def grab(self) -> sl.ERROR_CODE:
        """카메라로부터 새로운 프레임을 캡처하고 상태 코드를 반환합니다."""
        return self.cam.grab(self.runtime_params)

    def get_left_image(self):
        """획득한 왼쪽 이미지를 Numpy 배열로 반환합니다."""
        self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT)
        return self.left_image.get_data().copy()

    def close(self):
        """카메라를 닫고 리소스를 해제합니다."""
        if self.cam.is_opened():
            self.cam.close()
            
    def __enter__(self):
        """'with' 구문 진입 시 카메라를 엽니다."""
        if self.open():
            return self
        else:
            raise IOError("ZED 카메라를 열 수 없습니다.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """'with' 구문 탈출 시 카메라를 닫습니다."""
        self.close()