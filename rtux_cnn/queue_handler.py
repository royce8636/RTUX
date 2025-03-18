from datetime import datetime
import json
import threading
import os
import subprocess
import time
import re
import queue
import yaml

from rtux_cnn.rtux_detection import RtuxDetect
from rtux_cnn.utils import print
from rtux_cnn.model_load import ModelLoader
from rtux_cnn.config import Config
from rtux_cnn.threshold_parser import main as threshold_parser
import time
import subprocess
from datetime import datetime
import signal

MAX_DETECTORS=3


class QueueHandler:
    def __init__(self, args, exp_shape, sync_diff, touchscreen):
        self.args = args
        self.config = Config()
        self.config.touch_screen = touchscreen
        self.config.log_all = args.all
        self.sync_diff = sync_diff

        self.queue_lock = threading.Lock()

        self.touchscreen = touchscreen

        self.config.setup_sigint_handler(self.cleanup)

        self.comp_runs = 0
        self.org_path = args.log_path

        strftime = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if "[timestamp]" in self.org_path:
            self.org_path = self.org_path.replace("[timestamp]", strftime)
        print(f"Creating directory {self.org_path}")
        os.makedirs(self.org_path, exist_ok=True)
        with open(f"{self.org_path}/sync_diff.txt", 'w') as f:
            json.dump(self.sync_diff, f)

        self._queue = []
        self._ind = 0
        self.shell_running = None
        if args.file is not None:
            self.file_queue_putter(args.file)
        self.loader = ModelLoader(self._queue)
        self.config.models = self.loader.load_model(args.directory)
        self.threshold_lst = {}
        self.setup_threshold(args.file)
        print(self.threshold_lst)

        self.repeat = self.args.repeat

        self.game, self.action = 0, 0
        self.abnormal = False
        self.abnormal_message = None
        self.abnormal_count = 0
        # self.perfetto_out = f"/data/nfs/perfetto_trace_{self.comp_runs}"
        self.perfetto_out = f"/data/media/0/perfetto_trace_{self.comp_runs}"
        self.start_perfetto()
        
        self.rtux = RtuxDetect(dir, exp_shape, self.notify_detection)
        self.rtux.set_cam(self.config.device)
        self.rtux.start_threads()

        # Current Tasks: [previous, current, future]
        self.config.setup_queue(MAX_DETECTORS)

        self.setup_task_manager()

        self.detector_threads = []
        for i in range(MAX_DETECTORS):
            self.detector_threads.append(
                threading.Thread(target=self.rtux.detect, args=(i,))
            )
            self.detector_threads[i].start()

        self.rtux.frame_grabber_thread = threading.Thread(target=self.rtux.frame_grabber)
        self.rtux.frame_grabber_thread.start()

        self.queue_event = threading.Event()
        self.queue_thread = threading.Thread(target=self.queue_manager)
        self.queue_thread.start()
        print("Queue Manager started")

        self.threshold_wait_thread = None
        self.threshold_stopper = threading.Event()
        self.threshold_lock = threading.Lock()
        self.threshold_wait_event = threading.Event()
        self.threshold_processing = False
        self.threshold_queue = queue.PriorityQueue()
        self.threshold_active_ts = 0

        self.push_proc = None

        self.calibrate_thread = None
        self.calibrate_stopper = False
        self.calibrator_waiting = False

        self.reinit_window()
        self.queue_event.set()

    def cleanup(self):
        end_time = time.clock_gettime(time.CLOCK_BOOTTIME)
        self.rtux.kill, self.rtux.stop_event = True, True
        self.kill_script()
        print("Cleaning up:")
        print("Killing Perfetto")
        try:
            self.kill_perfetto()
        except (ProcessLookupError, AttributeError) as e:
            print("Perfetto already killed")
        self.rtux.write_queue.put(None)
        self.rtux.writer_thread.join()
        times = 0
        for i in range(self.comp_runs):
            start_file = f"{self.org_path}/{i}/{i}.yaml"
            with open(start_file, 'r') as f:
                start_log = yaml.safe_load(f)
            start_time = start_log[0]["Metrics"]["LINUX_MONOTONIC"]
            end_time = start_log[-1]["Metrics"]["LINUX_MONOTONIC"]
            times += (end_time - start_time)

        print(f"Number of runs: {self.comp_runs}")
        print(f"Abnormal runs: {self.abnormal_count}")
        print(f"Successful runs: {self.comp_runs - self.abnormal_count}")
        print(f"Total time: {times}s")
        
        print("Exiting program")
        os._exit(0)


    @property
    def ind(self):
        return self.config._ind

    @ind.setter
    def ind(self, value):
        self.config._ind = value

    def calibrate_window(self, org_ind, threshold=30):
        with self.queue_lock:
            queue_snapshot = self._queue[:]
        
        target_lst = [None, None, None]
        cur_task_ind = 1000
        
        for i in range(max(0, org_ind - 5), len(queue_snapshot)):
            if '0' in queue_snapshot[i]:
                key, val = self.read_queue(i)
                if val is None:
                    self.queue_event.set()
                    return
                if i < org_ind:
                    target_lst[2] = val
                elif i >= org_ind and target_lst[1] is None:
                    target_lst[1] = val
                    cur_task_ind = i
                elif i > cur_task_ind and target_lst[0] is None:
                    target_lst[0] = val
                    break

        transformed_lst = []
        for item in target_lst:
            if item is not None:
                game, action = item.split('_')
                transformed_lst.append({f"{game}_{action}": False})
            else:
                transformed_lst.append(None)

        wait_start = time.time()
        while not self.calibrate_stopper and threshold > 0:
            future, cur, prev = self.config.current_tasks.get_all_values()
            if not any([future, cur, prev]):
                break
            if time.time() - wait_start > threshold:
                print(f"(called ind: {org_ind}) Threshold reached")
                break
            self.calibrator_waiting = True
            time.sleep(0.001)

        with self.queue_lock:
            self.config.current_tasks.change_all(transformed_lst)

        self.calibrator_waiting = False
        self.calibrate_thread = None
        print(f"Calibrating window (called ind: {org_ind}): {transformed_lst} DONE")


    def kill_perfetto(self):
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        
        found = False
        pattern = re.compile(
            r'\[\d+\.\d+\]\s+perfetto_cmd\.cc:\d+\s+(Wrote\s+\d+\s+bytes into new|Trace written into the output file)'
        )
        start = time.time()
        while found is False:
            if time.time() - start > 10:
                print("Timeout reached")
                found = False
                break
            output = self.process.stdout.readline()
            if output:
                print(f"OUTPUT: {output}")

            stderr = self.process.stderr.readline()
            if stderr:
                decoded_stderr = stderr.strip().decode()
                print(f"STDERR: {decoded_stderr}")
                if pattern.search(decoded_stderr):
                    found = True
                    print("Found the expected output:", decoded_stderr)
                    break
        if found:
            print(f"Pulling perfetto trace {self.perfetto_out}")
            os.system(f"adb -s {self.config.device} pull {self.perfetto_out} {self.config.rep_log_path}/perfetto_trace_{self.comp_runs}")
            self.process.kill()
        else:
            os.system(f"adb -s {self.config.device} shell killall perfetto")
            print(f"Failed to kill perfetto cleanly {self.perfetto_out}")
            os.system(f"adb -s {self.config.device} pull {self.perfetto_out} {self.config.rep_log_path}/perfetto_trace_{self.comp_runs}")
            self.process.kill()

        os.system(f"adb -s {self.config.device} shell rm -rf {self.perfetto_out}")
        os.system(f"adb -s {self.config.device} shell rm -rf /data/media/0/perfetto_trace_*")

        return True

    def start_perfetto(self):
        self.abnormal = False
        
        cmd = f"cat {self.args.perfetto} | adb -s {self.config.device} shell perfetto -c - --txt -o {self.perfetto_out}"
        self.process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, preexec_fn=os.setsid)
        print("Started perfetto")

    def writer_process(self, model_name, img2save, boottime, frame, level, status):
        self.rtux.writer(model_name, img2save, boottime, frame, level, status)

    def notify_detection(self, thread_ind, model_name, frame, boottime, changed_time, index, update_to_true=False):

        if update_to_true:
            with self.queue_lock:
                print(f"ACQUIRED LOCK")
                self.config.current_tasks.update_to_true(model_name, thread_ind)

        if thread_ind == 0:  # Future
            level = "WARNING"
            status = "Future"
        elif thread_ind == 1:  # current (normal)
            level = "INFO"
            status = "Current"
        elif thread_ind == 2:  # Previous
            level = "WARNING"
            status = "Previous"
        else:
            print(f"Not writing for {thread_ind}")

        self.rtux.writer(model_name, self.rtux.img2save, boottime, frame, level, status, boottime - self.det_start, index)

        print(f"Finished Notifying task {thread_ind} for {model_name}")

    def reinit_window(self):
        self.config.current_tasks.reset_window()
        os.system(f"prep/killer.sh {self.config.device}")
        os.system(f"adb -s {self.config.device} shell 'swapoff /dev/block/zram0'")
        os.system(f"adb -s {self.config.device} shell 'echo 1 > /sys/block/zram0/reset'")
        os.system(f"prep/idle.sh {self.config.device}")

        if self.push_proc is not None:
            print(f"Killing push process {self.push_proc.pid}")
            os.system(f"adb -s {self.config.device} shell 'kill -s 9 {self.push_proc.pid}'")
            self.push_proc = None

        """FIO"""
        # self.push_proc = subprocess.Popen(f"""adb -s {self.config.device} shell ./data/local/tmp/fio \
        #                 --direct=0 --rw=write --bs=128k --ioengine=sync --size=25G\
        #                 --numjobs=3 --rate=,60m --rate_process=poisson\
        #                 --group_reporting --name=iops-test-job --eta-newline=1 \
        #                 --directory=/data/media/0""", # /mnt/sdcard, /data/media/0
        #                 shell=True,
        #                 preexec_fn=os.setsid,
        #                 stdout=subprocess.PIPE,
        #                 stderr=subprocess.PIPE,
        #                 text=True  # Use 'universal_newlines=True' if 'text' is not available                        
        #                 )
        # while True:
        #     output = self.push_proc.stdout.readline()
        #     if output == '' and self.push_proc.poll() is not None:
        #         break
        #     if output:
        #         print(output.strip())
        #         if f"fio-3.37" in output:
        #             print("Desired output detected.")
        #             break
        # print("FIO STARTED")
        
        count = 0
        for i in range(self.ind + 1, len(self._queue)):
            if '0' in self._queue[i]:
                key, val = self.read_queue(i)
                try:
                    game, action = val.split('_')[0], val.split('_')[1]
                except IndexError:
                    print("WRONG APP_ACTION format")
                    return 0
                game, action = game.strip(), action.strip()
                self.config.current_tasks.add(f"{game}_{action}", False)
                count += 1
            if count == 2:
                break
        self.det_start = time.clock_gettime(time.CLOCK_BOOTTIME)

    def parse_force_stop(self, adb_command):
        pattern = re.compile(r"com\.[\w\.]+")
        match = pattern.search(adb_command)
        if match:
            component_name = match.group()
            return f"adb -s {self.config.device} shell am force-stop {component_name}"
        else:
            return None

    def setup_task_manager(self):
        @self.config.current_tasks.on_change
        def task_manager(queue):
            with self.queue_lock:
                future, cur, prev = self.config.current_tasks.get_all_values()
                cur_queue = self.config.current_tasks.queue
                fut_task, cur_task, prev_task = self.config.current_tasks.get_all_tasks()
            try:
                threshold_val = self.threshold_lst[cur_task]
            except (AttributeError, KeyError) as e:
                threshold_val = 60

            threshold_val = max(threshold_val, 60)

            print(f"TASK MANAGER: {cur_queue}")
            if not fut_task and not cur_task:
                self.config.current_tasks.reset_window()
            elif cur:
                print(f"CURRENT {(self.ind)}: Setting event for queueManager")
                if self.calibrator_waiting:
                    print(f"Calibrator waiting, ignoring true signals")
                    return
                self.ind += 1
                self.queue_event.set()
                
                self.stop_calibrate_thread()

                self.calibrate_thread = threading.Thread(target=self.calibrate_window, args=(self.ind, threshold_val, )).start()
                self.threshold_wait_joiner()
            elif future and not prev:
                print("Future task detected")
                local_ind = self.ind
                while True:
                    print("Calling read queue")
                    key, val = self.read_queue(local_ind)
                    if key is None:
                        self.ind += 1
                        self.queue_event.set()
                        break
                    if val == fut_task:
                        print(f"Found future ind at {local_ind}")
                        self.ind = local_ind
                        break
                    local_ind += 1
                
                self.stop_calibrate_thread()

                self.calibrate_thread = threading.Thread(target=self.calibrate_window, args=(self.ind, -1,)).start()
                self.threshold_wait_joiner()
                print(f"FUTURE: Setting event for queueManager | ind {self.ind}")
                self.queue_event.set()
            elif prev and not future:
                print(f"Previous task detected, waiting for threshold {threshold_val}")
                self.enqueue_threshold(threshold_val, "PREVIOUS")
            else:
                print(f"No Task detected ({cur_task}), waiting for threshold {threshold_val}")
                self.enqueue_threshold(threshold_val, "ALL FALSE")


    def enqueue_threshold(self, threshold_val, action_type):
        timestamp = time.time()
        self.threshold_queue.put((-timestamp, threshold_val, action_type))
        print(f"Enqueued threshold {threshold_val} for {action_type} at {timestamp}")
        self.start_threshold()

    def stop_calibrate_thread(self):
        if self.calibrate_thread is not None:
            print("Calibrate thread already running")
            self.calibrate_stopper = True
            self.calibrate_thread.join()
            self.calibrate_stopper = False
            self.calibrate_thread = None

    def threshold_wait_joiner(self):
        
        if self.threshold_wait_thread is not None:
            self.threshold_stopper.set()
            self.threshold_wait_event.set()
            try:
                print("Waiting for threshold thread to finish")
                self.threshold_wait_thread.join()
            except (RuntimeError, AttributeError) as e:
                pass
            self.threshold_wait_event.clear()
            self.threshold_stopper.clear()
            self.threshold_wait_thread = None
            print(f"Joined threshold thread")

    def start_threshold(self):
        try:
            ts, threshold_val, action_type = self.threshold_queue.get(timeout=1)
        except queue.Empty:
            print("No threshold tasks in queue")
            return
        except Exception as e:
            print(f"Error starting threshold: {e}")
            return
        print(f"Threshold thread called for {action_type} ({threshold_val})")
        if self.threshold_active_ts < ts:
            print(f"Threshold already active for {action_type} ({threshold_val})")
            return
        else: 
            self.threshold_active_ts = ts
        self.threshold_wait_joiner()

        with self.threshold_lock:
            print(f"Acquired lock for {action_type} ({threshold_val})")
            self.threshold_wait_joiner()
            self.threshold_wait_thread = threading.Thread(target=self.wait_for_threshold, args=(threshold_val, action_type))
            self.threshold_wait_thread.start()

    def wait_for_threshold(self, threshold_val, action_type):
        start_time = time.clock_gettime(time.CLOCK_BOOTTIME)
        print(f"Waiting for threshold {threshold_val} for {action_type}")
        while not self.config.current_tasks._is_notifying:
            if start_time % 30 == 0:
                print(f"Remaining time: {threshold_val - (time.clock_gettime(time.CLOCK_BOOTTIME) - start_time)}")
            if self.threshold_stopper.is_set():
                print(f"Threshold stopped early for {action_type} ({threshold_val})")
                break
            if time.clock_gettime(time.CLOCK_BOOTTIME) - start_time > threshold_val:
                cur_queue = self.config.current_tasks.queue
                print(f"ERROR: THRESHOLD REACHED for {action_type} ({threshold_val})")
                if self.abnormal_message is None:
                    self.abnormal_message = f"Threshold reached for {action_type} ({threshold_val}), queue {cur_queue} "
                if self.args.strict:
                    print(f"Ending Queue")
                    self.end_queue()
                    self.calibrate_window(self.ind, -1)
                    break
                if action_type == "PREVIOUS":
                    self.reverse_to_last_action()
                else:  # ALL FALSE
                    self.reverse_to_app_start()
                print(f"{action_type}: Setting event for queueManager")
                self.calibrate_window(self.ind, -1)
                self.queue_event.set()
                break

            self.threshold_wait_event.wait(0.1)
            self.threshold_wait_event.clear()

        self.threshold_stopper.clear()
        self.threshold_wait_thread = None
        print(f"Threshold thread for {action_type} ({threshold_val}) finished")
        return

    def end_queue(self):
        self.abnormal = True
        local_ind = self.ind
        while True:
            print(f"Calling read queue")
            key, val = self.read_queue(local_ind)
            if key is None:
                print(f"Done putting index to end of the queue")
                self.ind = local_ind + 1
                self.queue_event.set()
                break
            local_ind += 1

    def reverse_to_last_action(self):
        self.abnormal = True
        prev_ind = self.ind
        local_ind = self.ind
        while True:
            print(f"Calling read queue")
            key, val = self.read_queue(local_ind)
            if key is None:
                self.ind += 1
                self.queue_event.set()
                break
            if key == "1":
                print(f"Queue reversal done | ind {local_ind}")
                self.ind = local_ind
                break
            local_ind -= 1

        local_ind = 0 if None else local_ind
        self.abnormal_message += f"\nRevert to action {local_ind} at index {prev_ind}"

    def reverse_to_app_start(self):
        self.abnormal = True
        local_ind = self.ind
        prev_ind = self.ind
        while True:
            print(f"Calling read queue")
            key, val = self.read_queue(local_ind)
            if key is None:
                self.ind += 1
                self.queue_event.set()
                break
            if "am start" in val:
                print(f"Queue reversal done | ind {local_ind}")
                self.ind = local_ind
                command = self.parse_force_stop(val)
                print(f"Force stopping {command}")
                os.system(command)
                break
            elif local_ind == 0:
                print("No app start found")
                break
            local_ind -= 1
        if local_ind == self.ind:
            self.ind = 0
        self.abnormal_message += f"\nRevert to app start {local_ind} at index {prev_ind}"
        
    def read_queue(self, target_ind):
        try:
            data_dict = self._queue[target_ind]
        except IndexError:
            print(f"Index {target_ind} out of range ({len(self._queue)})")
            return None, None
        if data_dict == "" or data_dict is None:
            return None, None
        try:
            val = list(data_dict.values())[0]
            key = list(data_dict.keys())[0]
        except IndexError:
            print("WRONG APP_ACTION format")
            self.cleanup()
        return key, val

    def kill_script(self):
        if self.shell_running is not None:
            print(f"Killing shell command {self.shell_running.pid}")
            try:
                os.killpg(os.getpgid(self.shell_running.pid), signal.SIGUSR1)
            except ProcessLookupError as e:
                print(f"Error killing shell command: {e}")
            self.shell_running = None

    def queue_manager(self):
        while True:
            time.sleep(0.01)
            print(f"CALLED: {self.ind} | {self.config.current_tasks.queue}")

            self.queue_event.wait()

            if self.ind >= len(self._queue):
                print("Queue finished")
                os.system(f"adb -s {self.config.device} shell am start -a android.intent.action.MAIN -c android.intent.category.HOME")

                if self.abnormal:
                    self.abnormal_count += 1
                    with open(f"{self.config.rep_log_path}/abnormal.txt", 'w') as f:
                        f.write(self.abnormal_message)
                    self.abnormal_message = None
                    
                self.kill_perfetto()
                if self.repeat > 1:
                    
                    self.threshold_wait_joiner()
                    self.stop_calibrate_thread()
                    self.config.current_tasks.reset_window()
                    if not self.abnormal:
                        self.repeat -= 1
                    else:
                        print("Abnormal task detected, Adding more repeats (strict: True, but not changing dirs)")
                        
                    print(f"Remaining repeats: {self.repeat}")
                    self.abnormal = False
                    self.ind = 0
                    self.comp_runs += 1
                    
                    self.reinit_window()

                    self.path_handler()
                    # self.perfetto_out = f"/data/nfs/perfetto_trace_{self.comp_runs}"
                    self.perfetto_out = f"/data/media/0/perfetto_trace_{self.comp_runs}"
                    self.start_perfetto()
                    time.sleep(1)

                    self.queue_event.clear()
                    self.queue_event.set()
                    continue
                print("All tasks finished, cleaning up")
                self.cleanup()
            
            with self.queue_lock:
                key, val = self.read_queue(self.ind)
            if key is None:
                self.ind += 1
                self.queue_event.set()
                continue

            if key == "0": # detection task
                self.det_start = time.clock_gettime(time.CLOCK_BOOTTIME)
                self.queue_event.clear()

            elif key == "1": # command line task
                if self.shell_running is not None:
                    print(f"Killing shell command {self.shell_running.pid}")
                    self.kill_script()
                print(f"{self.ind}: Doing {val}")
                if ".sh" in val:
                    val = val.strip().split(' ')
                    val.insert(0, '/bin/bash')
                    val.append(self.touchscreen)
                    val.append(self.config.device)
                    self.rtux.writer(val, None, time.clock_gettime(time.CLOCK_BOOTTIME), self.rtux.count, "INFO", "Touch", 0, self.ind)
                    self.shell_running = subprocess.Popen(val, preexec_fn=os.setsid)
                    print(f"Started shell command {val} ({self.shell_running.pid})")
                elif "am start" in val and "HOME" not in val:
                    if self.shell_running is not None:
                        print(f"Killing shell command {self.shell_running.pid}")
                        self.kill_script()
                    match = re.search(r'com\.(.*)', val)
                    if match:
                        pack = "com." + match.group(1)
                    if not match:
                        match = re.search(r'org\.(.*)', val)
                        pack = "org." + match.group(1)
                    else:
                        pack = val.split(' ')[-1]
                    threading.Thread(target=self.am_start, args=(pack, self.ind, self.rtux.count )).start()
                elif "force-stop" in val:
                    os.system(f"adb -s {self.config.device} shell {val.split('shell ')[1]}")
                elif "sleep" in val:
                    time.sleep(int(val.split(' ')[-1]))
                
                self.ind += 1
                
                self.queue_event.clear()
                self.queue_event.set()

            print(f"FINISHED: {self.ind} | {self.config.current_tasks.queue}")

    def file_queue_putter(self, file):
        temp_queue = []
        try:
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0 or line[0] == '#':
                        continue
                    line = line.split('#', 1)[0].strip()
                    if line:
                        lst = line.split(':')
                        temp_queue.append({lst[0].strip(): ('').join(lst[1:]).strip()})
            print(f"Loaded {len(temp_queue)} commands from {file}")
            print(f"Queue: {temp_queue}")
            
            self._queue = temp_queue
            self.ind = 0
            self.path_handler()
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print(f"{file} not found")
            self.cleanup()

    def setup_threshold(self, file):
        self.threshold_lst = threshold_parser(self.args.directory)


    def path_handler(self, dup=False):
        if dup is True:
            self._queue = self._queue[self.ind:]
            print(f"new queue: {self._queue}")
            self.ind = 0

        self.config.log_path = os.path.join(self.org_path, "all_log")
        self.config.rep_log_path = os.path.join(self.org_path, f"{self.comp_runs}")
        self.config.log_yaml = os.path.join(self.config.rep_log_path, f"{self.comp_runs}.yaml")

        self.config.log_file = os.path.join(self.config.log_path, "full_log.txt")

        print(f"Creating directory {self.config.log_path}, {self.config.rep_log_path}")
        os.makedirs(self.config.log_path, exist_ok=True)
        os.makedirs(self.config.rep_log_path, exist_ok=True)

        os.makedirs(f"{self.config.log_path}/all_frames", exist_ok=True)

    @RtuxDetect.increment_counter
    def am_start(self, package, queue_ind, start_frame, **kwargs):
        tag = kwargs.get('tag', 'default_tag') 
        logcat_command = f"adb -s {self.config.device} shell am start -W {package}"
        # print(logcat_command)
        self.rtux.start_time = time.clock_gettime(time.CLOCK_BOOTTIME)
        realtime = datetime.now()
        logcat_output = subprocess.check_output(logcat_command, shell=True).decode('utf-8')
        # print(logcat_output)
        log_entries = logcat_output.splitlines()

        wait_time = time.time()
        while True:
            if time.time() - wait_time > 10:
                try:
                    TotalTime = [entry for entry in log_entries if "WaitTime" in entry]
                    TotalTime = int(TotalTime[0].split(':')[-1])
                    break
                except IndexError:
                    break
            try:
                TotalTime = [entry for entry in log_entries if "TotalTime" in entry]
                TotalTime = int(TotalTime[0].split(':')[-1])
                break
            except IndexError:
                time.sleep(0.01)
                continue

        self.rtux.writer(package, None, self.rtux.start_time, max(0, start_frame), "INFO", "AppStart", TotalTime, queue_ind, tag.split(':')[1].strip())
        print(f"APP START DONE {package}")
        return

    @RtuxDetect.increment_counter
    def am_video(self, package, queue_ind, start_frame, **kwargs):
        tag = kwargs.get('tag', 'default_tag') 
        logcat_command = f"adb -s {self.config.device} shell am start {package}"
        print(logcat_command)
        self.rtux.start_time = time.clock_gettime(time.CLOCK_BOOTTIME)
        realtime = datetime.now()
        logcat_output = subprocess.check_output(logcat_command, shell=True).decode('utf-8')
        print(logcat_output)
        log_entries = logcat_output.splitlines()

        while True:
            try:
                TotalTime = [entry for entry in log_entries if "TotalTime" in entry]
                TotalTime = int(TotalTime[0].split(':')[-1])
                break
            except IndexError:
                time.sleep(0.01)
                continue

        self.rtux.writer(package, None, self.rtux.start_time, max(0, start_frame), "INFO", "AppStart", TotalTime, queue_ind, tag.split(':')[1].strip())

        return