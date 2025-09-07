import cv2
import numpy as np
import threading
import queue
from zed_cam import ZedCam
import pyzed.sl as sl

class ZedAruco:
    """ArUco 마커 탐지 및 자세 추정을 담당하는 클래스"""
    def __init__(self, zed_cam: ZedCam, aruco_dict_type=cv2.aruco.DICT_7X7_250, marker_length=1.822):
        intrinsics = zed_cam.get_intrinsics()
        if intrinsics is None:
            raise ValueError("카메라 파라미터를 가져올 수 없습니다. 카메라가 열려있는지 확인하세요.")
        
        self.fx, self.fy, self.cx, self.cy, dist_coeffs_list = intrinsics
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32) # 왜곡이 보정되어 나오기 때문에 0으로 설정

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_length = marker_length

        # solvePnP를 위한 마커의 3D 월드 좌표 (마커 중심을 원점으로)
        self.marker_3d_points = np.array([
            [-self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0]
        ], dtype=np.float32)

    def detect(self, image):
        """이미지에서 ArUco 마커를 탐지합니다."""
        return self.detector.detectMarkers(image)

    def estimate_pose(self, corner):
        """단일 마커의 자세를 추정합니다."""
        ret, rvec, tvec = cv2.solvePnP(self.marker_3d_points, corner, self.camera_matrix, self.dist_coeffs)
        return rvec if ret else None, tvec if ret else None

    def draw_info(self, image, corner, marker_id, rvec, tvec):
        """탐지된 마커 정보를 이미지에 그립니다."""
        cv2.aruco.drawDetectedMarkers(image, [corner], np.array([[marker_id]]))
        cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

        tvec_str = f"id:{marker_id} T: {tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}m"
        top_left_corner = tuple(corner[0][0].astype(int))
        cv2.putText(image, tvec_str, (top_left_corner[0], top_left_corner[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def camera_worker(zed: ZedCam, image_queue: queue.Queue, stop_event: threading.Event):
    """
    백그라운드 스레드에서 ZED 카메라 이미지를 지속적으로 캡처하여 큐에 넣습니다.
    """
    print("카메라 스레드 시작.")
    while not stop_event.is_set():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            image = zed.get_left_image()
            
            # 큐가 가득 차 있으면 오래된 프레임을 버리고 새 프레임을 넣습니다.
            if not image_queue.full():
                image_queue.put(image)
    print("카메라 스레드 종료.")

def main():
    """
    메인 함수: 카메라 초기화, 스레드 생성, GUI 업데이트를 담당합니다.
    """
    image_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    try:
        with ZedCam(resolution="1080P", fps=30) as zed:
            aruco_detector = ZedAruco(zed, marker_length=0.1)

            # 카메라 작업을 위한 스레드 생성 및 시작
            cam_thread = threading.Thread(target=camera_worker, args=(zed, image_queue, stop_event))
            cam_thread.start()

            while True:
                try:
                    # 큐에서 최신 이미지를 가져옵니다. 1초간 응답 없으면 예외 발생.
                    # 4채널 BGRA 이미지 원본을 받음
                    image_bgra = image_queue.get(timeout=1.0)
                    # 3채널 BGR 이미지로 변환
                    image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
                except queue.Empty:
                    print("경고: 카메라로부터 새 프레임을 받지 못했습니다.")
                    continue

                corners, ids, _ = aruco_detector.detect(image_bgr)

                if ids is not None:
                    for marker_id, corner in zip(ids, corners):
                        rvec, tvec = aruco_detector.estimate_pose(corner)
                        if rvec is not None and tvec is not None:
                            aruco_detector.draw_info(image_bgr, corner, marker_id[0], rvec, tvec)

                cv2.imshow("ZED Aruco Detection", image_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("종료 신호를 보냅니다...")
                    break
    
    except Exception as e:
        print(f"오류 발생: {e}")
    
    finally:
        # 프로그램 종료 처리
        stop_event.set()
        if 'cam_thread' in locals() and cam_thread.is_alive():
            cam_thread.join() # 스레드가 완전히 종료될 때까지 대기
        cv2.destroyAllWindows()
        print("프로그램이 성공적으로 종료되었습니다.")


if __name__ == "__main__":
    main()