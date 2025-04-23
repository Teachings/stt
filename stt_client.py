#!/usr/bin/env python3
import os
import argparse
import yaml
from pynput import keyboard
import sounddevice as sd
import numpy as np
import pyperclip
import subprocess
from pathlib import Path
import time
import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stt_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class STTClient:
    def __init__(self):
        self.config = self.load_config()
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        
    def load_config(self):
        config_path = os.path.expanduser("~/.config/stt_client/config.yaml")
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                default_config = {
                    "server_url": "http://localhost:8000",
                    "hotkey": "<ctrl>+<alt>+<space>",
                    "output_mode": "clipboard",
                    "audio_device": "default",
                    "api_key": None,
                    "record_on_press": False
                }
                with open(config_path, "w") as f:
                    yaml.dump(default_config, f)
                return default_config
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config['hotkey'], str) and 'space' in config['hotkey'] and '<space>' not in config['hotkey']:
                    config['hotkey'] = config['hotkey'].replace('space', '<space>')
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.load_config()
    
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        logger.info("Recording started... (Press hotkey again to stop)")
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio stream status: {status}")
            self.audio_data.append(indata.copy())
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.config["audio_device"],
                callback=callback
            )
            self.stream.start()
            logger.debug("Audio stream started successfully")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        try:
            self.stream.stop()
            self.stream.close()
            logger.info("Recording stopped. Processing...")
            
            audio_array = np.concatenate(self.audio_data)
            audio_array = audio_array.flatten().astype(np.float32)
            
            logger.debug(f"Audio data shape: {audio_array.shape}, duration: {len(audio_array)/self.sample_rate:.2f}s")
            
            self.process_audio(audio_array)
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
    
    def process_audio(self, audio_array):
        try:
            headers = {'Content-Type': 'application/octet-stream'}
            if self.config["api_key"]:
                headers["x-api-key"] = self.config["api_key"]
            
            logger.debug(f"Sending audio to {self.config['server_url']}/transcribe")
            
            start_time = datetime.now()
            response = requests.post(
                f"{self.config['server_url']}/transcribe",
                data=audio_array.tobytes(),
                headers=headers
            )
            
            response.raise_for_status()
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Server response time: {processing_time:.2f}s")
            
            result = response.json()
            text = result.get("text", "")
            logger.info(f"Transcription: {text}")
            self.handle_output(text)
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request error: {e}")
            if e.response is not None:
                logger.error(f"Server response: {e.response.text}")
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
    
    def handle_output(self, text):
        try:
            if not text:
                logger.warning("Empty transcription received")
                return
                
            if self.config["output_mode"] in ["clipboard", "both"]:
                pyperclip.copy(text)
                logger.info("Text copied to clipboard")
            
            if self.config["output_mode"] in ["type", "both"]:
                if os.getenv("XDG_SESSION_TYPE") == "x11":
                    try:
                        subprocess.run(["sleep", "0.1"])
                        subprocess.run(["xdotool", "type", "--delay", "0", text], check=True)
                        logger.info("Text typed using xdotool")
                    except Exception as e:
                        logger.error(f"Error using xdotool: {e}")
                        if self.config["output_mode"] != "both":
                            pyperclip.copy(text)
                else:
                    logger.warning("xdotool requires X11. Falling back to clipboard.")
                    if self.config["output_mode"] == "type":
                        pyperclip.copy(text)
        except Exception as e:
            logger.error(f"Error handling output: {e}")
    
    def run(self):
        try:
            key_combination = self.config["hotkey"]
            listener = keyboard.GlobalHotKeys({
                key_combination: self.toggle_recording
            })
            
            logger.info(f"STT Client running. Press {key_combination} to start/stop recording.")
            logger.info(f"Server: {self.config['server_url']}")
            logger.info(f"Configuration: {self.config}")
            
            listener.start()
            while True:
                time.sleep(0.1)
        except ValueError as e:
            logger.error(f"Invalid hotkey format in config: {self.config['hotkey']}")
            logger.error("Edit ~/.config/stt_client/config.yaml to fix this")
        except KeyboardInterrupt:
            logger.info("Exiting...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

def configure():
    config_path = os.path.expanduser("~/.config/stt_client/config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    print("Available audio devices:")
    print(sd.query_devices())
    
    config = {}
    config["server_url"] = input(f"STT Server URL [http://localhost:8000]: ") or "http://localhost:8000"
    
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
    config["api_key"] = input("API key (if required by server): ") or None
    config["record_on_press"] = False
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Configuration saved to {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text Client")
    parser.add_argument("--start", action="store_true", help="Start the STT client")
    parser.add_argument("--configure", action="store_true", help="Configure settings")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.configure or not os.path.exists(os.path.expanduser("~/.config/stt_client/config.yaml")):
        configure()
    
    if args.start:
        try:
            app = STTClient()
            app.run()
        except Exception as e:
            logger.critical(f"Fatal error: {e}")

if __name__ == "__main__":
    main()