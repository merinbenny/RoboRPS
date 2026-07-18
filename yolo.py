import os
import platform
import random
import tempfile
import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk, ImageSequence

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None


# =====================================================================
# CONFIGURATION
# =====================================================================

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
MODEL_ID = "rock-paper-scissors-sxsw/14"

VALID_MOVES = ["Rock", "Paper", "Scissors"]

CAMERA_INDEX = 0
COUNTDOWN_SECONDS = 3
MIN_CONFIDENCE = 0.40
VIDEO_UPDATE_MS = 15


# =====================================================================
# GAME LOGIC
# =====================================================================

def decide_winner(player_move, computer_move):
    if player_move == computer_move:
        return "draw"

    beats = {
        "Rock": "Scissors",
        "Paper": "Rock",
        "Scissors": "Paper",
    }
    if beats[player_move] == computer_move:
        return "player"
    return "computer"


# =====================================================================
# YOLO GESTURE DETECTOR
# =====================================================================

class GestureDetector:
    def __init__(self, api_url, api_key, model_id):
        if InferenceHTTPClient is None:
            raise RuntimeError("inference-sdk not installed.")
        if not api_key:
            raise RuntimeError("Missing Roboflow API key.")

        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id

    def predict(self, frame_bgr):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, frame_bgr)
            result = self.client.infer(tmp_path, model_id=self.model_id)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        predictions = result.get("predictions", []) if result else []
        if not predictions:
            return None, 0.0, None

        best = max(predictions, key=lambda p: p.get("confidence", 0))
        raw_class = str(best.get("class", "")).strip()
        confidence = float(best.get("confidence", 0))

        move = None
        for candidate in VALID_MOVES:
            if raw_class.lower() == candidate.lower():
                move = candidate
                break

        if move is None or confidence < MIN_CONFIDENCE:
            return None, confidence, None

        box = (best.get("x"), best.get("y"), best.get("width"), best.get("height"))
        return move, confidence, box


# =====================================================================
# GUI APPLICATION
# =====================================================================

class RPSGameApp:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Rock-Paper-Scissors (YOLO)")
        self.root.geometry("760x640")
        self.root.resizable(False, False)

        # Pixel-style scaling
        self.root.tk.call('tk', 'scaling', 0.7)

        # Load backgrounds
        self._load_backgrounds()

        self.detector = None
        self.cap = None
        self.video_job = None

        self.total_rounds = 5
        self.current_round = 0
        self.player_score = 0
        self.computer_score = 0

        self.last_frame_bgr = None
        self.detecting = False

        self._build_start_screen()

    # -----------------------------------------------------------
    # Background loading
    # -----------------------------------------------------------

    def _load_backgrounds(self):
        # Animated GIF frames
        gif_path = r"C:\Users\USER\Desktop\CA_Python\opencv\Black and White Minimalist Initials Logo.gif"
        gif = Image.open(gif_path)

        self.bg_start_frames = [
            ImageTk.PhotoImage(frame.resize((760, 640)))
            for frame in ImageSequence.Iterator(gif)
        ]
        self.bg_start_index = 0

        # PNG background
        png_path = r"C:\Users\USER\Desktop\CA_Python\opencv\Black and White Minimalist Initials Logo.png"
        self.bg_other = ImageTk.PhotoImage(Image.open(png_path).resize((760, 640)))

    def _apply_background_static(self):
        bg_label = tk.Label(self.root, image=self.bg_other)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.lower()

    def _apply_background_gif(self):
        self.bg_label = tk.Label(self.root)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg_label.lower()
        self._animate_gif()

    def _animate_gif(self):
        frame = self.bg_start_frames[self.bg_start_index]
        self.bg_label.configure(image=frame)
        self.bg_start_index = (self.bg_start_index + 1) % len(self.bg_start_frames)
        self.root.after(80, self._animate_gif)

    # -----------------------------------------------------------
    # Screen management
    # -----------------------------------------------------------

    def _clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # -----------------------------------------------------------
    # Start screen
    # -----------------------------------------------------------

    def _build_start_screen(self):
        self._clear_screen()
        self._apply_background_gif()

        tk.Label(self.root, text="",
                 font=("Helvetica", 26, "bold"), bg="#000000", fg="white").pack(pady=(40, 5))

        tk.Label(self.root, text=" ",
                 font=("Helvetica", 12), bg="#000000", fg="white").pack(pady=(0, 30))

        tk.Label(self.root, text="Number of rounds:",
                 font=("Helvetica", 14), bg="#000000", fg="white").pack()

        self.rounds_var = tk.IntVar(value=5)
        rounds_frame = tk.Frame(self.root, bg="#000000")
        rounds_frame.pack(pady=10)
        for n in [3, 5, 7]:
          
            tk.Radiobutton(
                rounds_frame,
                text=str(n),
                variable=self.rounds_var,
                value=n,
                font=("Helvetica",12),
                bg="#000000",
                fg="white",
                selectcolor="#333333",
                activebackground="#000000",
                activeforeground="white"
            ).pack(side=tk.LEFT,padx=10)

        tk.Button(
            self.root, text="START GAME", font=("Helvetica", 16, "bold"),
            bg="#2e8b57", fg="white", padx=20, pady=10,
            command=self._start_game
        ).pack(pady=40)

    def _start_game(self):
        try:
            self.detector = GestureDetector(ROBOFLOW_API_URL, ROBOFLOW_API_KEY, MODEL_ID)
        except RuntimeError as e:
            messagebox.showerror("Setup error", str(e))
            return

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        )
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the webcam.")
            return

        self.total_rounds = self.rounds_var.get()
        self.current_round = 0
        self.player_score = 0
        self.computer_score = 0

        self._build_game_screen()
        self._next_round()

    # -----------------------------------------------------------
    # Game screen
    # -----------------------------------------------------------

    def _build_game_screen(self):
        self._clear_screen()
        self._apply_background_static()

        self.round_label = tk.Label(self.root, text="", font=("Helvetica", 16, "bold"), bg="#000000", fg="white")
        self.round_label.pack(pady=(15, 5))

        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.pack()

        self.status_label = tk.Label(self.root, text="Get ready...",
                                      font=("Helvetica", 20, "bold"), fg="#cc5500", bg="#000000")
        self.status_label.pack(pady=10)

        result_frame = tk.Frame(self.root, bg="#000000")
        result_frame.pack(pady=5)

        self.player_move_label = tk.Label(result_frame, text="Player: -",
                                           font=("Helvetica", 14), width=20, bg="#000000", fg="white")
        self.player_move_label.pack(side=tk.LEFT, padx=10)

        self.computer_move_label = tk.Label(result_frame, text="Computer: -",
                                             font=("Helvetica", 14), width=20, bg="#000000", fg="white")
        self.computer_move_label.pack(side=tk.LEFT, padx=10)

        self.score_label = tk.Label(self.root, text="Score  Player 0 - 0 AI",
                                     font=("Helvetica", 16, "bold"), bg="#000000", fg="white")
        self.score_label.pack(pady=15)

        self._update_video_loop()

    def _update_video_loop(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.last_frame_bgr = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.video_job = self.root.after(VIDEO_UPDATE_MS, self._update_video_loop)

    def _next_round(self):
        if self.current_round >= self.total_rounds:
            self._show_final_screen()
            return

        self.current_round += 1
        self.round_label.config(
            text=f"Round {self.current_round} / {self.total_rounds}"
        )
        self.player_move_label.config(text="Player: -")
        self.computer_move_label.config(text="Computer: -")
        self._run_countdown(COUNTDOWN_SECONDS)

    def _run_countdown(self, seconds_left):
        if seconds_left > 0:
            self.status_label.config(text=f"Show your move in... {seconds_left}")
            self.root.after(1000, self._run_countdown, seconds_left - 1)
        else:
            self.status_label.config(text="Show!")
            self.root.after(300, self._capture_and_detect)

    def _capture_and_detect(self):
        if self.detecting or self.last_frame_bgr is None:
            return
        self.detecting = True
        self.status_label.config(text="Detecting gesture...")

        frame_copy = self.last_frame_bgr.copy()
        thread = threading.Thread(target=self._detect_in_background, args=(frame_copy,))
        thread.daemon = True
        thread.start()

    def _detect_in_background(self, frame_bgr):
        try:
            move, confidence, _ = self.detector.predict(frame_bgr)
        except Exception as e:
            move, confidence = None, 0.0
            print(f"Detection error: {e}")

        self.root.after(0, self._on_detection_result, move, confidence)

    def _on_detection_result(self, move, confidence):
        self.detecting = False

        if move is None:
            self.status_label.config(
                text="Couldn't recognize a gesture -- showing this round again."
            )
            self.root.after(1500, lambda: self._run_countdown(COUNTDOWN_SECONDS))
            return

        computer_move = random.choice(VALID_MOVES)
        outcome = decide_winner(move, computer_move)

        self.player_move_label.config(text=f"Player: {move} ({confidence:.0%})")
        self.computer_move_label.config(text=f"Computer: {computer_move}")

        if outcome == "player":
            self.player_score += 1
            self.status_label.config(text="You win this round!", fg="#2e8b57")
        elif outcome == "computer":
            self.computer_score += 1
            self.status_label.config(text="Robo wins this round!", fg="#cc0000")
        else:
            self.status_label.config(text="Draw!", fg="#888888")

        self.score_label.config(
            text=f"Score   Player {self.player_score} - {self.computer_score} Robo"
        )

        self.root.after(2000, self._next_round)

    # -----------------------------------------------------------
    # Final screen
    # -----------------------------------------------------------

    def _show_final_screen(self):
        if self.video_job is not None:
            self.root.after_cancel(self.video_job)
            self.video_job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self._clear_screen()
        self._apply_background_static()

        tk.Label(self.root, text="GAME OVER...!", font=("Helvetica", 28, "bold"),
                 bg="#000000", fg="white").pack(pady=(60, 20))

        tk.Label(
            self.root,
            text=f"Player Score: {self.player_score}\nRobo Score: {self.computer_score}",
            font=("Helvetica", 18), bg="#000000", fg="white"
        ).pack(pady=10)

        if self.player_score > self.computer_score:
            winner_text, color = "You Win!", "#2e8b57"
        elif self.computer_score > self.player_score:
            winner_text, color = "Robo Wins!", "#cc0000"
        else:
            winner_text, color = "It's a Draw!", "#888888"

        tk.Label(self.root, text=winner_text, font=("Helvetica", 24, "bold"),
                 fg=color, bg="#000000").pack(pady=20)

        tk.Button(
            self.root, text="Play Again", font=("Helvetica", 14, "bold"),
            bg="#2e8b57", fg="white", padx=15, pady=8,
            command=self._build_start_screen
        ).pack(pady=20)

    def on_close(self):
        if self.video_job is not None:
            self.root.after_cancel(self.video_job)
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = RPSGameApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()