#!/usr/bin/env python3
import os
import argparse
import yaml
from pynput import keyboard
import sounddevice as sd
import numpy as np
import torch
import pyperclip
import subprocess
from pathlib import Path
import time

class STTShortcut:
    def __init__(self):
        self.config = self.load_config()
        self.model = None
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.load_model()
        
    def load_config(self):
        config_path = os.path.expanduser("~/.config/stt_shortcut/config.yaml")
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                default_config = {
                    "model": "large-v3-turbo",
                    "hotkey": "<ctrl>+<alt>+<space>",
                    "output_mode": "clipboard",
                    "audio_device": "default",
                    "use_gpu": True,
                    "record_on_press": False,
                    "compute_type": "float16"  # Added for faster-whisper
                }
                with open(config_path, "w") as f:
                    yaml.dump(default_config, f)
                return default_config
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config['hotkey'], str) and 'space' in config['hotkey'] and '<space>' not in config['hotkey']:
                    config['hotkey'] = config['hotkey'].replace('space', '<space>')
                if 'compute_type' not in config:
                    config['compute_type'] = "float16" if config.get('use_gpu', True) else "float32"
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.load_config()
    
    def load_model(self):
        try:
            from faster_whisper import WhisperModel
            device = "cuda" if (self.config["use_gpu"] and torch.cuda.is_available()) else "cpu"
            compute_type = self.config["compute_type"]
            
            print(f"Loading faster-whisper model {self.config['model']} on {device} ({compute_type})...")
            
            self.model = WhisperModel(
                model_size_or_path=self.config["model"],
                device=device,
                compute_type=compute_type,
                download_root=os.path.expanduser("~/.cache/whisper")
            )
            print("Model loaded successfully")
        except ImportError:
            print("Installing faster-whisper...")
            subprocess.run([sys.executable, "-m", "pip", "install", "faster-whisper"])
            self.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        print("\nRecording started... (Press hotkey again to stop)")
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_data.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.config["audio_device"],
            callback=callback
        )
        self.stream.start()
    
    def stop_recording(self):
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        print("Recording stopped. Processing...")
        
        audio_array = np.concatenate(self.audio_data)
        audio_array = audio_array.flatten().astype(np.float32)
        
        try:
            segments, info = self.model.transcribe(
                audio_array,
                language="en",
                beam_size=5  # Adjust for accuracy/speed tradeoff
            )
            
            # Concatenate all segments
            text = " ".join([segment.text for segment in segments])
            text = text.strip()
            
            print(f"Detected language: {info.language}, probability: {info.language_probability:.2f}")
            print(f"Transcription: {text}")
            self.handle_output(text)
        except Exception as e:
            print(f"Error during transcription: {e}")
    
    def handle_output(self, text):
        if self.config["output_mode"] == "clipboard":
            pyperclip.copy(text)
            print("Text copied to clipboard")
        elif self.config["output_mode"] == "type":
            try:
                if not text:
                    return
                if os.getenv("XDG_SESSION_TYPE") != "x11":
                    print("Warning: xdotool requires X11. Falling back to clipboard.")
                    pyperclip.copy(text)
                    return
                
                subprocess.run(["sleep", "0.1"])
                subprocess.run(["xdotool", "type", "--delay", "0", text])
            except Exception as e:
                print(f"Error using xdotool: {e}. Falling back to clipboard.")
                pyperclip.copy(text)
        elif self.config["output_mode"] == "both":
            pyperclip.copy(text)
            try:
                if os.getenv("XDG_SESSION_TYPE") == "x11":
                    subprocess.run(["sleep", "0.1"])
                    subprocess.run(["xdotool", "type", "--delay", "0", text])
            except:
                pass
    
    def run(self):
        try:
            key_combination = self.config["hotkey"]
            listener = keyboard.GlobalHotKeys({
                key_combination: self.toggle_recording
            })
            
            print(f"STT Shortcut running. Press {key_combination} to start/stop recording.")
            print(f"Configuration: {self.config}")
            
            listener.start()
            while True:
                time.sleep(0.1)
                pass
        except ValueError as e:
            print(f"\nERROR: Invalid hotkey format in config: {self.config['hotkey']}")
            print("Edit ~/.config/stt_shortcut/config.yaml to fix this")
        except KeyboardInterrupt:
            print("\nExiting...")

def configure():
    config_path = os.path.expanduser("~/.config/stt_shortcut/config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    print("Available audio devices:")
    print(sd.query_devices())
    
    print("\nAvailable Whisper models: tiny, base, small, medium, large-v1, large-v2, large-v3")
    print("For faster-whisper, you can also use custom model paths")
    
    config = {}
    model = input(f"Model [large-v3]: ") or "large-v3"
    config["model"] = model
    
    hotkey = input(f"Hotkey [<ctrl>+<alt>+<space>]: ") or "<ctrl>+<alt>+<space>"
    if 'space' in hotkey and '<space>' not in hotkey:
        hotkey = hotkey.replace('space', '<space>')
    config["hotkey"] = hotkey
    
    print("\nOutput modes:")
    print("1. Clipboard only")
    print("2. Auto-type only")
    print("3. Both")
    mode = input("Select output mode [1]: ") or "1"
    config["output_mode"] = ["clipboard", "type", "both"][int(mode)-1] if mode in ["1","2","3"] else "clipboard"
    
    config["audio_device"] = input(f"Audio device ID/name [default]: ") or "default"
    use_gpu = input(f"Use GPU if available? [Y/n]: ").lower() in ["", "y", "yes"]
    config["use_gpu"] = use_gpu
    config["compute_type"] = "float16" if use_gpu else "int8"
    config["record_on_press"] = False
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Configuration saved to {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text Shortcut Tool with Faster-Whisper")
    parser.add_argument("--start", action="store_true", help="Start the STT shortcut")
    parser.add_argument("--configure", action="store_true", help="Configure settings")
    
    args = parser.parse_args()
    
    if args.configure or not os.path.exists(os.path.expanduser("~/.config/stt_shortcut/config.yaml")):
        configure()
    
    if args.start:
        try:
            app = STTShortcut()
            app.run()
        except Exception as e:
            print(f"Fatal error: {e}")
            if "faster_whisper" in str(e):
                print("Try installing faster-whisper manually: pip install faster-whisper")

if __name__ == "__main__":
    import sys
    main()