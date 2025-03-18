import os
import subprocess
import argparse
import sys
import signal
import shutil
from threading import Thread
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt, QUrl, QSize
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar, QMessageBox)
import glob
import cv2
import tensorflow as tf
import subprocess
import time

THRESHOLD = 0.70
MAX_WORKERS = 16

def quit_app(*args):
    QApplication.instance().quit()
    sys.exit(0)

def create_directory(path, name):
    directory_path = os.path.join(path, name)
    os.makedirs(directory_path, exist_ok=True)
        
def encode_to_120fps(input_file, output_file):
    abs_output_path = os.path.abspath(output_file)
    abs_input_path = os.path.abspath(input_file)
    process = subprocess.Popen(f"timeout 0.3s ffprobe -v error -select_streams v:0 -show_entries frame=width,height -of csv=s=x:p=0 -read_intervals -1 '{abs_input_path}'", shell=True, stdout=subprocess.PIPE)
    output = process.stdout.read().decode('utf-8')
    resolutions = output.strip().split('\n')
    last_resolution = resolutions[-1]
    print(f"last_resolution: {last_resolution}")
    command = f"ffmpeg -i {abs_input_path} -r 120 -s {last_resolution} {abs_output_path}"
    subprocess.call(command, shell=True)
    
def extract_frames(input_file, output_folder, last_frame_number):
    abs_input_path = os.path.abspath(input_file)
    abs_output_folder = os.path.abspath(output_folder)
    
    if os.path.exists(abs_output_folder):
        shutil.rmtree(abs_output_folder)
    
    os.makedirs(abs_output_folder)
        
    output_path = os.path.join(abs_output_folder, '%d.jpg')
    vid = cv2.VideoCapture(abs_input_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command = f'ffmpeg -i "{abs_input_path}" -vf "fps=120,scale={width//2}:{height//2}" -vframes {last_frame_number}  -threads {MAX_WORKERS} "{output_path}"'
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    
    if result.returncode != 0:
        print(f"Error occurred during ffmpeg execution:\n{result.stderr}")
    else:
        print(f"ffmpeg output:\n{result.stdout.decode('utf-8')}")
    
        
class CustomSlider(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marked_positions = []
     
class VideoPlayer(QWidget):     
    def __init__(self, args, parent=None, group_name=None, directory_path=None, device=None):
        super(VideoPlayer, self).__init__(parent)
        self.group_name = group_name
        self.device = device
        self.directory_path = directory_path
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.fps = 0
        self.frame_duration = 0
        self.vid = None
        self.last_frame = 0
        self.writer_thread = []

        self.mediaPlayer = QMediaPlayer()
        self.frame_range = {'X': []}
        self.true_frames = {key: [] for key in range(16)}

        self.time_range = {}
        self.true_times = {key: [] for key in range(16)}

        self.touch_record = None
        self.grouped_infos, self.grouped_times = [], []

        btnSize = QSize(16, 16)
        videoWidget = QVideoWidget()
        videoWidget.setStyleSheet("background-color: black;")

        openButton = QPushButton("Open Video")   
        openButton.setToolTip("Open Video File")
        openButton.setStatusTip("Open Video File")
        openButton.setFixedHeight(24)
        openButton.setIconSize(btnSize)
        openButton.setFont(QFont("Noto Sans", 8))
        openButton.setIcon(QIcon.fromTheme("document-open", QIcon("D:/_Qt/img/open.png")))
        openButton.clicked.connect(self.abrir)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setFixedHeight(24)
        self.playButton.setIconSize(btnSize)
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = CustomSlider(Qt.Orientation.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        self.markFrameButton = QPushButton("No Marked")
        self.markFrameButton.setEnabled(False)
        
        self.extractButton = QPushButton("Extract Images")
        self.extractButton.setEnabled(False)
        self.extractButton.clicked.connect(self.extract_images)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(openButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.markFrameButton)
        controlLayout.addWidget(self.extractButton)
        
        self.markedInfoLabel = QLabel()
        self.markedInfoLabel.setFont(QFont("Noto Sans", 8))

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(videoWidget)
        mainLayout.addLayout(controlLayout)
        mainLayout.addWidget(self.markedInfoLabel)
        mainLayout.addWidget(self.statusBar)

        self.setLayout(mainLayout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.playbackStateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.errorChanged.connect(self.handleError)
        self.statusBar.showMessage("Ready")
        
        self.markedInfoLabel.setFixedHeight(10)

    def abrir(self, fileName=None):
        
        if not fileName:
            fileName, _ = QFileDialog.getOpenFileName(self, "Select Media",
                    ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi *.m4a)")
        
        self.markFrameButton.setEnabled(True)
        self.extractButton.setEnabled(True)

        if fileName != '':
            encoded_file = fileName.rsplit('.', 1)[0] + '_120fps.mp4' 
            encode_to_120fps(fileName, encoded_file)

            self.vid = cv2.VideoCapture(encoded_file)
            self.fps = self.vid.get(cv2.CAP_PROP_FPS)
            print(f"FPS: {self.fps}")
            self.frame_duration = float(1000/self.fps)

            self.touch_record = fileName.rsplit('.', 1)[0] + '.txt'
            print(f"Touch record: {self.touch_record}")
            infos = []
            times = []
            if (os.path.exists(self.touch_record)):
                with open(self.touch_record, 'r') as f:
                    for line in f.readlines():
                        time = float(line.split(']')[0].strip('['))
                        info = line.split(']')[1].strip()
                        infos.append(info)
                        times.append(time)
            else:
                print(f"Touch file {self.touch_record} does not exist")
                exit()
                
            cur_times, cur_infos = [], []
            for i in range(1, len(times)):
                if times[i] - times[i-1] < 0.1:
                    cur_times.append((times[i-1]))
                    cur_infos.append((infos[i-1]))
                else:
                    cur_infos.append((infos[i-1]))
                    cur_times.append((times[i-1]))
                    self.grouped_times.append(cur_times)
                    self.grouped_infos.append(cur_infos)
                    cur_times, cur_infos = [], []
            if cur_times:
                self.grouped_times.append(cur_times)
                self.grouped_infos.append(cur_infos)

            self.mediaPlayer.setSource(QUrl.fromLocalFile(encoded_file))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage(encoded_file)
            self.play()
        
    def extract_images(self):
        url = self.mediaPlayer.source()
        video_file = url.toLocalFile()

        output_folder = os.path.join(self.directory_path, "image")
        
        marks = [frame for sublist in self.true_frames.values() for frame in sublist]
        last_frame = max(marks) if marks else 0
        self.last_frame = last_frame
        if args.s is False:
            extract_frames(video_file, output_folder, last_frame)
            print(f"Images extracted to {output_folder}")
        else:
            print(f"Not extracting the images.")

        for idx, frames in self.true_frames.items():
            if len(frames) == 0:
                continue
                
            new_folder_name = str(idx) 
            new_folder_path = os.path.join(self.directory_path, 'screen', new_folder_name)

            create_directory(new_folder_path, "true")
            create_directory(new_folder_path, "false")

        self.save_extracted_images(os.path.abspath(output_folder))
        self.save_script()
        for t in self.writer_thread:
            t.join()
        self.push_videos()
        print("Extraction finished. Exiting program")
        quit_app()


    def get_frames(self, sublist, output_folder, last_true):
        if not sublist:
            return [], []
        limits = self.frame_range['X']
        filtered_frames = [x for x in sublist if not any(limits[i] <= x <= limits[i+1] for i in range(0, len(limits), 2))]
        true_frames = [os.path.join(output_folder, f"{frame}.jpg") for frame in filtered_frames]

        earliest_true_frame = min(filtered_frames)
        false_frames = [
            os.path.join(output_folder, f"{frame}.jpg") for frame in range(last_true, earliest_true_frame)
            if frame not in filtered_frames and
            not any(self.frame_range['X'][i] <= frame <= self.frame_range['X'][i+1] for i in range(0, len(self.frame_range['X']), 2))
        ]
        
        return true_frames, false_frames
    
    def save_extracted_images(self, output_folder):
        actions = {0: [self.grouped_times, self.grouped_infos]}
        last_stage_true = 1
        for idx, sublist in self.true_frames.items():
            if len(sublist) == 0:
                print(f"task is completed.")
                return
            
            if idx not in actions:
                actions[idx] = [[], []]

            first_frame_time = self.true_times[idx][0]
            last_true_frame = max(sublist)


            if idx != 0:
                for i in range(len(actions[idx - 1][0])):                                                                                              
                    subtime = actions[idx - 1][0][i]
                    if subtime[0] > first_frame_time:
                        sliced_time = actions[idx - 1][0][i:]
                        sliced_info = actions[idx - 1][1][i:]
                        actions[idx][0] = sliced_time
                        actions[idx][1] = sliced_info
                        actions[idx-1][0] = actions[idx-1][0][:i]
                        actions[idx-1][1] = actions[idx-1][1][:i]
                        break
                    else:
                        pass

            self.actions = actions
            if args.s is True:
                continue
            
            true_dir = os.path.join(os.path.abspath(self.directory_path), 'screen', str(idx), 'true')
            false_dir = os.path.join(os.path.abspath(self.directory_path), 'screen', str(idx), 'false')

            """" For using VIDEO"""
            true_frames, false_frames = self.get_frames(sublist, output_folder, last_stage_true)
            false_frames = sorted(false_frames, key=self.extract_number)
            last_stage_true = last_true_frame

            true_thread = Thread(target=self.write_to_vid, args=(true_frames, true_dir))
            self.writer_thread.append(true_thread)
            false_thread = Thread(target=self.write_to_vid, args=(false_frames, false_dir))
            self.writer_thread.append(false_thread)
            print(f"Writing to video: {true_dir}, {false_dir}")
            true_thread.start()
            false_thread.start()
        
    def extract_number(self, file_path):
        return int(file_path.split('/')[-1].split('.')[0])

    def write_to_vid(self, frames, destination_dir):
        first_frame = cv2.imread(frames[0])
        cv2.imwrite(f"{destination_dir}/thumbnail.bmp", first_frame)
        frame_size = first_frame.shape[:2]
        frame_size = (frame_size[1], frame_size[0])
        out = cv2.VideoWriter(f'{destination_dir}/vid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 120, frame_size)
        for frame in frames:
            img = cv2.imread(frame)
            img = cv2.resize(img, frame_size)
            out.write(img)
        out.release()
        print(f"video saved to {destination_dir}/vid.avi")

    def push_videos(self):
        video_folders = glob.glob(os.path.join(self.directory_path, 'screen', '*'))
        for folder in video_folders:
            command = f"adb {self.device} push {folder} /sdcard/temp_pic/{self.group_name}/"
            print(command)
            os.system(command)

    def save_script(self):
        queue = []
        start_times = [self.true_times[idx][0] for idx in self.actions.keys()]
        thresholds = [start_times[0]] + [start_times[i] - start_times[i-1] for i in range(1, len(start_times))]
        with open(os.path.join(self.directory_path, f"{self.group_name}_threshold.txt"), 'w') as f:
            for i in thresholds:
                f.write(f"{int(i * 5)}\n")
            f.flush()

        for idx in self.actions.keys():
            start_frame_time = self.true_times[idx][0]

            destination = os.path.join(self.directory_path, 'script')
            os.makedirs(destination, exist_ok=True)
            script_name = os.path.join(destination, f"{idx}.txt")
            xy_name = os.path.join(destination, f"{idx}_xy.txt")
            queue.append(f"0: {self.group_name}_{idx}")
            if len(self.actions[idx][0]) != 0:
                queue.append(f"1: prep/replay.sh {self.group_name} {idx}")

            end_frame_time = 0
            with open(script_name, 'w') as f:
                for i in range(len(self.actions[idx][0])):
                    for j in range(len(self.actions[idx][1][i])):
                        f.write(f"[ {float(self.actions[idx][0][i][j]) - start_frame_time}] {self.actions[idx][1][i][j]}\n")
                        end_time = float(self.actions[idx][0][i][j])
                        if end_time > end_frame_time:
                            end_frame_time = end_time
                f.flush()
            print(f"Script saved to {script_name}")

            x, y = None, None
            with open(xy_name, 'w') as outfile, open(script_name, 'r') as infile:
                for line in infile.readlines():
                    time, event_data = line.strip().split("]", 1)
                    event_type, event_code, event_value = event_data.split()
                    if event_code == "0035":
                        x = int(event_value, 16)
                        last_time = time
                    elif event_code == "0036":
                        y = int(event_value, 16)
                        last_time = time
                    
                    if x is not None and y is not None and last_time is not None:
                        outfile.write(f"{last_time}] {x} {y}\n")
                        x = None
                        y = None

        if self.device is not None:
            command = f"adb {self.device} push {destination} /sdcard/temp_pic/{self.group_name}/script/"
        else:
            command = f"adb push {destination} /sdcard/temp_pic/{self.group_name}/script/"
        os.system(command)
        with open(os.path.join(self.directory_path, f"{self.group_name}_queue.txt"), 'a') as f:
            for i in queue:
                f.write(f"{i}\n")
            f.flush()

    def update_marked_info(self):
        marked_info_texts = []
        exclude_info_texts = []
        for index, ranges in self.frame_range.items():
            if index == 'X':
                if len(ranges) != 0:
                    for i in range(0, len(ranges), 2):
                        try:
                            exclude_info_texts.append(f"{ranges[i]} ~ {ranges[i+1]}")
                        except IndexError:
                            exclude_info_texts.append(f"{ranges[i]} ~ ")
                continue
            ranges_count = len(ranges)
            if ranges_count == 0:
                continue
            elif ranges_count == 1:
                marked_info_texts.append(f"Mark {index-48}: {ranges[0]}~{ranges[0]}")
            else:
                marked_info_texts.append(f"Mark {index-48}: {ranges[0]}~{ranges[1]}")
        text = " | ".join(marked_info_texts) + " | " + "Exclude: " +", ".join(exclude_info_texts)
        self.markedInfoLabel.setText(text)

    def play(self):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
        
    def positionChanged(self, position):
        self.positionSlider.setValue(position)
        self.update_frame_number()
        self.update_button_text()

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)
        self.update_frame_number()

    def update_frame_number(self):
        current_frame = int(self.mediaPlayer.position() // self.frame_duration)
        if current_frame != 0:
            current_frame += 1
        total_frames = int(self.mediaPlayer.duration() // self.frame_duration)
        self.statusBar.showMessage(f"{current_frame}/{total_frames}")
            
    def update_button_text(self):
        current_position = int(self.mediaPlayer.position() // self.frame_duration)
        if current_position != 0:
            current_position += 1

        group_name = None
        for group, frames in self.true_frames.items():
            if current_position in frames:
                group_name = str(group) if group < 10 else chr(ord('A') + group - 10)
                break
        
        if group_name:
            self.markFrameButton.setText(group_name)
        else:
            self.markFrameButton.setText("No Marked")
        
    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Space:
            self.play()
            return

        if key in [Qt.Key.Key_Right, Qt.Key.Key_Left]:
            self.mediaPlayer.pause()

        if key == Qt.Key.Key_Right:
            self.adjust_position(forward=True)
        elif key == Qt.Key.Key_Left:
            self.adjust_position(forward=False)
        elif key == Qt.Key.Key_X:
            self.toggle_frame_mark('X')
            return

        frame_keys = [
            Qt.Key.Key_0, Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4,
            Qt.Key.Key_5, Qt.Key.Key_6, Qt.Key.Key_7, Qt.Key.Key_8, Qt.Key.Key_9,
            Qt.Key.Key_A, Qt.Key.Key_B, Qt.Key.Key_C, Qt.Key.Key_D, Qt.Key.Key_E, Qt.Key.Key_F,
        ]

        shifted_keys = {
            Qt.Key.Key_1: Qt.Key.Key_Exclam,       # 1 -> !
            Qt.Key.Key_2: Qt.Key.Key_At,           # 2 -> @
            Qt.Key.Key_3: Qt.Key.Key_NumberSign,   # 3 -> #
            Qt.Key.Key_4: Qt.Key.Key_Dollar,       # 4 -> $
            Qt.Key.Key_5: Qt.Key.Key_Percent,      # 5 -> %
            Qt.Key.Key_6: Qt.Key.Key_AsciiCircum,  # 6 -> ^
            Qt.Key.Key_7: Qt.Key.Key_Ampersand,    # 7 -> &
            Qt.Key.Key_8: Qt.Key.Key_Asterisk,     # 8 -> *
            Qt.Key.Key_9: Qt.Key.Key_ParenLeft,    # 9 -> (
        }

        if key in frame_keys:
            self.mark_frame(key, frame_keys.index(key))

    def adjust_position(self, forward):
        """Adjusts the media player's position."""
        position = self.mediaPlayer.position()
        offset = self.frame_duration if forward else -self.frame_duration
        new_position = max(0, position + offset)
        self.mediaPlayer.setPosition(int(new_position))

    def toggle_frame_mark(self, key):
        """Toggles a frame's marked state for the specified key."""
        current_frame = self.current_frame()
        frames = self.frame_range.get(key, [])
        if current_frame in frames:
            frames.remove(current_frame)
        else:
            frames.append(current_frame)
        self.frame_range[key] = frames
        self.update_marked_info()

    def mark_frame(self, key, index):
        """Marks or updates marked frames and times for a specified key."""
        current_time = self.mediaPlayer.position() / 1000
        current_frame = self.current_frame()
        if key not in self.frame_range:
            self.initialize_marking(key, current_frame, current_time, index)
        else:
            self.update_marking(key, current_frame, current_time, index)

        self.update_marked_info()
        self.update_button_text()

    def current_frame(self):
        """Calculates the current frame based on the media position and frame duration."""
        return int(self.mediaPlayer.position() // self.frame_duration) + 1

    def initialize_marking(self, key, current_frame, current_time, index):
        """Initializes marking structures for a new key."""
        self.time_range[key] = [current_time]
        self.frame_range[key] = [current_frame]
        self.true_times[index] = [current_time]
        self.true_frames[index] = [current_frame]

    def update_marking(self, key, current_frame, current_time, index):
        """Updates marking structures for an existing key."""
        frames = self.frame_range[key]
        times = self.time_range[key]
        if len(frames) == 1:
            insert_position = 0 if current_frame < frames[0] else 1
            frames.insert(insert_position, current_frame)
            times.insert(insert_position, current_time)
        else:
            self.resolve_frame_conflicts(frames, times, current_frame, current_time)
        self.true_frames[index] = list(range(frames[0], frames[1] + 1))
        self.true_times[index] = list((times[0], times[1]))

    def resolve_frame_conflicts(self, frames, times, current_frame, current_time):
        """Resolves conflicts when more than one frame is marked."""
        if current_frame < frames[0]:
            frames[0] = current_frame
            times[0] = current_time
        elif current_frame > frames[1]:
            frames[1] = current_frame
            times[1] = current_time


    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())

if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit_app)
    
    parser = argparse.ArgumentParser(description='Process arguments.', usage='video.py -g [name] <-s [int]>')
    parser.add_argument('-g', required=True, help='Name of the game')
    parser.add_argument('-d', default="dataset", help='Path of dataset folder, Default = dataset')
    parser.add_argument('-s', action='store_true', help='Whether to only save the scripts (no -s flag will save all)')
    parser.add_argument('-D', "--device", default=None, help='Device serial number')
    args = parser.parse_args()

    if (args.device) is None:
        result = subprocess.run("adb shell echo", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error occurred during adb execution:\n{result.stderr}")
            exit()
        elif len(result.stdout.strip().split()) > 1:
            print(f"Multiple devices connected. Please specify device serial number using -D flag.")
            exit()
        else:
            device = ""
    else:
        device = f"-s {args.device}"

    directory_path = os.path.join(os.getcwd(), args.d, args.g)

    video_files = glob.glob(os.path.join(directory_path, "*.mp4"))

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QStatusBar {
            font-size: 12px;
        }
    """)
    player = VideoPlayer(group_name=args.g, directory_path=directory_path, args=args, device=device)
    player.setWindowTitle("Player")
    player.resize(900, 600)
    
    if len(video_files) == 1:
        video_file = video_files[0]
        print(f"Video file: {video_file}")
        reply = QMessageBox.question(player, "Open Video",
                    f"Use the video {os.path.basename(video_file)}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            player.abrir(video_file)
    elif len(video_files) > 1:
        QMessageBox.warning(player, "Multiple Videos", 
                            f"Multiple video files found in directory {directory_path}. Please choose video file manually by clicking open video button.")
        
    
    player.show()
    sys.exit(app.exec())
