from collections import Counter, deque
from enum import Enum
from multiprocessing import Manager, Process

import cv2
import mediapipe as mp
import numpy as np
import pygame
import tensorflow as tf

from tetris.game import Tetris

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

######## Parameters ########
OUTPUT_HIST_LEN = 3
OUTPUT_THRESHOLD = 0.98
LABEL_HIST_LEN = 10
############################


class Label(Enum):
    LEFT = 0
    RIGHT = 1
    STACK = 2
    FAST = 3
    CLOCK = 4
    COUNTER_CLOCK = 5
    IDLE = 6


class HandMotionDetector:
    @staticmethod
    def get_landmark_list(image, landmarks):
        """
        각 joint의 좌표를 뽑아내는 method
        """
        h, w, _ = image.shape
        result = []
        for _, landmark in enumerate(landmarks.landmark):
            x = min(int(landmark.x * w), w - 1)
            y = min(int(landmark.y * h), h - 1)
            result.append([x, y])
        result = np.array(result, dtype=np.float64)
        # 21 X 2 normalized coordinates
        return result

    @staticmethod
    def show_result(image, label: Label):
        if label != "":
            cv2.putText(
                image,
                "class: " + label.name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return image

    def preprocess_point_history(self, image):
        if len(self.history[0]) != 16:
            return
        else:
            h, w, _ = image.shape
            temp = np.array(self.history, dtype=np.float64).reshape(
                1, 21, 32
            )  # 21 X 32
            temp[:, :, 0::2] -= temp[0, 0, 0]
            temp[:, :, 1::2] -= temp[0, 0, 1]
            temp[:, :, 0::2] /= w
            temp[:, :, 1::2] /= h
            history_data = np.zeros((1, 16, 42))  # 16 X 42 변환
            for i in range(32):
                k = 0
                if i % 2:
                    k = 1
                history_data[0, i // 2, k::2] = temp[0, :, i]
            return history_data  # 16 X 42 normalized coordinates

    def preprocess_before_model(self, data):
        tdata = data.copy()
        for i in range(15, 0, -1):
            tdata[0, i, :] -= tdata[0, i - 1, :]
        return tdata

    def setup(self):
        self.cap = cv2.VideoCapture(0)
        self.history = [deque(maxlen=16) for _ in range(21)]
        self.model = tf.keras.models.load_model("./model/relativemodel.h5")
        self.output_list = deque(maxlen=OUTPUT_HIST_LEN)
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def check_closed(self):
        if not self.cap.isOpened() or cv2.waitKey(5) & 0xFF == 27:
            self.hands.close()
            self.output_list = deque(maxlen=OUTPUT_HIST_LEN)
            self.history = [deque(maxlen=16) for _ in range(21)]
            self.cap.release()
            cv2.destroyAllWindows()
            return True
        return False

    def loop(self):
        """
        Run this in a loop body.
        """
        success, image = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            return
        image = cv2.flip(image, 1)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        label = Label.IDLE
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = self.get_landmark_list(image, hand_landmarks)

                for i in range(21):  # 21 개 point의 coordinate 을 history deque에 저장
                    self.history[i].append(landmark_list[i])
                # history = 21X16X2
                # history deque normalization 좀 이상함.. 왜 normalization을 두번하지..
                data = self.preprocess_point_history(image)

                if data is not None:
                    tdata = self.preprocess_before_model(data)
                    res = self.model.predict(tdata)[0]
                    output = np.argmax(res)  # model output
                    if res[output] < OUTPUT_THRESHOLD:
                        self.output_list.append(Label.IDLE)
                    else:
                        # output deque에 저장
                        self.output_list.append(Label(output))

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Flip the image horizontally for a selfie-view display.
                if len(self.output_list) > 0:
                    # deque의 최빈값 = label
                    label = Counter(self.output_list).most_common()[0][0]
                    image = self.show_result(image, label)

        cv2.imshow("MediaPipe Hands", image)
        return label

    def run(self, label_history):
        self.setup()

        while True:
            label = self.loop()

            # Act like a deque.
            if len(label_history) >= LABEL_HIST_LEN:
                label_history.pop(0)
            label_history.append(label)

            if self.check_closed():
                break


class HandMotionTetris(Tetris):
    """
    Tetris that uses HandMotionDetector
    """

    def event_from_label(self, label_history):
        """
        Fire an event that controls the game.
        Currently implemented in naive approach, firing an event only if the label is different from the previous.
        """
        if len(label_history) == 0:
            return

        label = label_history[-1]
        if hasattr(self, "prev_label") and self.prev_label == label:
            return

        self.prev_label = label
        event = pygame.event.Event(pygame.USEREVENT + label.value)
        pygame.event.post(event)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        else:
            if event.type == pygame.USEREVENT + Label.LEFT.value:
                self.dx = -1
            elif event.type == pygame.USEREVENT + Label.RIGHT.value:
                self.dx = 1
            elif event.type == pygame.USEREVENT + Label.STACK.value:
                self.stack = True
            elif event.type == pygame.USEREVENT + Label.FAST.value:
                self.anim_limit = 0
            elif event.type == pygame.USEREVENT + Label.CLOCK.value:
                self.clockwise = True
            elif event.type == pygame.USEREVENT + Label.COUNTER_CLOCK.value:
                self.counter_clockwise = True

    def run(self, label_history):
        self.setup()

        while self.running:
            self.event_from_label(label_history)
            for event in pygame.event.get():
                self.handle_event(event)
            self.loop()
            self.render()
            self.check_game_over()
            self.clock.tick(self.fps)

        self.cleanup()


if __name__ == "__main__":
    # Shared label history
    manager = Manager()
    label_history = manager.list()

    detector = HandMotionDetector()
    game = HandMotionTetris()

    # Run both detector and game, using multiprocessing.
    procs = []
    p1 = Process(target=detector.run, args=[label_history])
    p1.start()
    procs.append(p1)

    p2 = Process(target=game.run, args=[label_history])
    p2.start()
    procs.append(p2)

    # Terminate
    for p in procs:
        p.join()
