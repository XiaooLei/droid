import os
import random
import threading
import time
from collections import defaultdict

from droid.camera_utils.camera_readers.zed_camera import gather_zed_cameras
from droid.camera_utils.info import get_camera_type


class MultiCameraWrapper:
    def __init__(self, camera_kwargs={}):
        # Open Cameras #
        zed_cameras = gather_zed_cameras()
        self.camera_dict = {cam.serial_number: cam for cam in zed_cameras}

        # Set Correct Parameters #
        for cam_id in self.camera_dict.keys():
            cam_type = get_camera_type(cam_id)
            curr_cam_kwargs = camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

        # Launch Camera #
        self._latest_camera_timestamp = {}
        self._latest_camera_obs = defaultdict(dict)
        self._latest_lock = threading.Lock()
        self._async_running = False
        self._async_thread = None
        self.set_trajectory_mode()

    ### Calibration Functions ###
    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def enable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        # If High Res Calibration, Only One Can Run #
        close_all = any([cam.high_res_calibration for cam in self.camera_dict.values()])

        if close_all:
            for curr_cam_id in self.camera_dict:
                if curr_cam_id != cam_id:
                    self.camera_dict[curr_cam_id].disable_camera()

        self.camera_dict[cam_id].set_calibration_mode()

    def set_trajectory_mode(self):
        # If High Res Calibration, Close All #
        close_all = any(
            [cam.high_res_calibration and cam.current_mode == "calibration" for cam in self.camera_dict.values()]
        )

        if close_all:
            for cam in self.camera_dict.values():
                cam.disable_camera()

        # Put All Cameras In Trajectory Mode #
        for cam in self.camera_dict.values():
            cam.set_trajectory_mode()

    ### Data Storing Functions ###
    def start_recording(self, recording_folderpath):
        subdir = os.path.join(recording_folderpath, "SVO")
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        for cam in self.camera_dict.values():
            filepath = os.path.join(subdir, cam.serial_number + ".svo")
            cam.start_recording(filepath)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()

    ### Basic Camera Functions ###
    def start_async_reading(self, include_images=False, frequency=None):
        if self._async_running:
            return

        self._async_running = True
        period_s = None if frequency is None or frequency <= 0 else 1.0 / frequency

        def async_loop():
            next_tick = time.monotonic()
            while self._async_running:
                obs, timestamp = self.read_cameras(include_images=include_images)
                with self._latest_lock:
                    self._latest_camera_obs = obs
                    self._latest_camera_timestamp = timestamp

                if period_s is None:
                    continue
                next_tick += period_s
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()

        self._async_thread = threading.Thread(target=async_loop, daemon=True)
        self._async_thread.start()

    def stop_async_reading(self):
        self._async_running = False
        if self._async_thread is not None:
            self._async_thread.join(timeout=2)
            self._async_thread = None

    def get_latest_camera_reading(self):
        with self._latest_lock:
            obs = defaultdict(dict)
            for key, value in self._latest_camera_obs.items():
                obs[key].update(value)
            return obs, dict(self._latest_camera_timestamp)

    def read_cameras(self, include_images=True):
        full_obs_dict = defaultdict(dict)
        full_timestamp_dict = {}

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if not self.camera_dict[cam_id].is_running():
                continue
            data_dict, timestamp_dict = self.camera_dict[cam_id].read_camera(include_images=include_images)

            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])
            full_timestamp_dict.update(timestamp_dict)

        return full_obs_dict, full_timestamp_dict

    def disable_cameras(self):
        self.stop_async_reading()
        for camera in self.camera_dict.values():
            camera.disable_camera()
