import cv2
import os
import signal
import time
import logging.config

try:
    from rtux_cnn.utils import SlidingWindow
except ImportError:
    from utils import SlidingWindow
class Config:
    _instance = None

    def __new__(cls, device=None, cap_ind=0, cv_window=False, calib=None):
        if device is None:
            device = os.popen('adb devices').read().split('\n')[1].split('\t')[0]
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = device
            cls._instance.cap = cv2.VideoCapture(cap_ind)
            cls._instance.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cls._instance.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cls._instance.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
            cls._instance.cap.set(cv2.CAP_PROP_FPS, 120)
            if cv_window is True:
                cls._instance.cv_window = "cam"
                cv2.namedWindow(cls._instance.cv_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(cls._instance.cv_window, 1280, 800)
            else:
                cls._instance.cv_window = None
            # cls._instance.max_br = int(os.popen(f'adb -s {device} shell cat '
                                                # f'/sys/class/backlight/panel0-backlight/max_brightness').read().strip())

            cls._instance.device = device if device is not None else ''
            cls._instance.calib = calib
            cls._instance.models = {}
            cls._instance.touchscreen = None
            cls._instance.log_all = False
            cls._instance.log_file = None
            cls._instance.log_yaml = None
            cls._instance.log_path = None
            cls._instance._ind = 0
            
            cls._instance.interrupt_count = 0
            cls._instance.last_interrupt_time = 0
            
            cls._instance.current_tasks = SlidingWindow(3)
        return cls._instance

    def handle_sigint(self, cleanup_func, force=False):
        def sigint_handler(signal, frame):
            print(f"\nCaught KeyboardInterrupt (Ctrl+C). Performing cleanup... {self.interrupt_count}")
            print(f"Queue: {self.current_tasks.queue}")
            current_time = time.time()
            if current_time - self.last_interrupt_time < 1:
                self.interrupt_count += 1
            else:
                self.interrupt_count = 1

            self.last_interrupt_time = current_time

            if self.interrupt_count == 1:
                print("\nCaught KeyboardInterrupt (Ctrl+C). Press Ctrl+C again within 1 second to force exit.")
                if cleanup_func:
                    cleanup_func()
            elif self.interrupt_count >= 2 or force:
                print("\nForce exiting...")
                os._exit(1)

        return sigint_handler

    def setup_sigint_handler(self, cleanup_func=None):
        signal.signal(signal.SIGINT, self.handle_sigint(cleanup_func))

    def setup_queue(self, max_size):
        self.current_tasks = SlidingWindow(max_size)

    def get_camera(self):
        return self.cap

    def get_cv_window(self):
        return self.cv_window

    def get_calib(self):
        return self.calib

    def get_max_br(self):
        return 500
        # return self.max_br

