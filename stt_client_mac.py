#!/usr/bin/env python3
import os
import argparse
import yaml
from pynput import keyboard
import sounddevice as sd
import numpy as np
import pyperclip
import subprocess
import time
import requests
import logging
from datetime import datetime
import platform
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser('~/stt_client.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MacSTTClient:
    def __init__(self):
        self.config = self.load_config()
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.accessibility_permission_warning = False
        self.python_interpreter_path = sys.executable
        self.check_system_compatibility()
        
    def check_system_compatibility(self):
        """Verify system is macOS with Apple Silicon"""
        system = platform.system()
        processor = platform.processor()
        
        if system != 'Darwin':
            logger.warning(f"Running on {system}, recommend macOS for best experience")
        
        if 'arm' in processor.lower():
            logger.info("Running on Apple Silicon processor")
        else:
            logger.warning("Not running on Apple Silicon - performance may vary")
            
    def check_accessibility_permissions(self):
        """
        Check if accessibility permissions are granted for the Python interpreter.
        Returns True if permissions seem to be granted, False otherwise.
        """
        try:
            # Create a test listener to see if we get warnings
            test_listener = keyboard.GlobalHotKeys({})
            test_listener.start()
            time.sleep(0.5)  # Give time for any warnings to be logged
            test_listener.stop()
            return not self.accessibility_permission_warning
        except Exception as e:
            logger.error(f"Error checking accessibility permissions: {e}")
            return False
            
    def show_accessibility_instructions(self):
        """
        Show detailed instructions for granting accessibility permissions
        """
        permission_msg = (
            f"⚠️ Accessibility Permission Required ⚠️\n\n"
            f"The STT client needs accessibility permissions to monitor keyboard shortcuts.\n\n"
            f"Please follow these steps:\n"
            f"1. Open System Settings > Privacy & Security > Accessibility\n"
            f"2. Click the '+' button and add this Python interpreter:\n"
            f"   {self.python_interpreter_path}\n\n"
            f"After adding permissions, restart the STT client."
        )
        
        logger.warning(permission_msg.replace('\n', ' '))
        
        # Show in notification
        self.show_notification(
            "Accessibility Permission Required",
            "Open System Settings > Privacy & Security > Accessibility"
        )
        
        # Also try to show a more detailed dialog using AppleScript
        try:
            script = f'''
            display dialog "{permission_msg}" with title "STT Client Accessibility Permissions" buttons {{"Open Settings", "OK"}} default button "Open Settings"
            if button returned of result is "Open Settings" then
                do shell script "open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'"
            end if
            '''
            subprocess.run(['osascript', '-e', script], capture_output=True)
        except Exception as e:
            logger.error(f"Failed to display accessibility dialog: {e}")

    def load_config(self):
        config_path = os.path.expanduser('~/.config/stt_client/config.yaml')
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                default_config = {
                    "server_url": "http://192.168.1.11:8000",
                    "hotkey": "<cmd>+<space>",
                    "output_mode": "clipboard",
                    "audio_device": "default",
                    "api_key": None,
                    "notifications": True,
                    "sound_effects": True
                }
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f)
                return default_config
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.load_config()
    
    def show_notification(self, title, message):
        """Show macOS notification"""
        if self.config.get('notifications', True):
            try:
                subprocess.run([
                    'osascript',
                    '-e',
                    f'display notification "{message}" with title "{title}"'
                ])
            except Exception as e:
                logger.warning(f"Couldn't show notification: {e}")

    def play_sound(self, sound_type):
        """Play system sound"""
        if self.config.get('sound_effects', True):
            try:
                if sound_type == "start":
                    subprocess.run(['afplay', '/System/Library/Sounds/Ping.aiff'])
                elif sound_type == "stop":
                    subprocess.run(['afplay', '/System/Library/Sounds/Pop.aiff'])
            except Exception as e:
                logger.warning(f"Couldn't play sound: {e}")

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        logger.info("Recording started...")
        self.show_notification("STT Client", "Recording started")
        self.play_sound("start")
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
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
        except Exception as e:
            logger.error(f"Failed to start audio: {e}")
            self.is_recording = False
            self.show_notification("STT Error", "Failed to start recording")

    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        try:
            self.stream.stop()
            self.stream.close()
            logger.info("Processing recording...")
            self.show_notification("STT Client", "Processing recording")
            
            audio_array = np.concatenate(self.audio_data)
            audio_array = audio_array.flatten().astype(np.float32)
            
            logger.debug(f"Audio length: {len(audio_array)/self.sample_rate:.2f}s")
            self.process_audio(audio_array)
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.show_notification("STT Error", "Recording failed")

    def process_audio(self, audio_array):
        try:
            headers = {'Content-Type': 'application/octet-stream'}
            if self.config["api_key"]:
                headers["x-api-key"] = self.config["api_key"]
            
            start_time = datetime.now()
            response = requests.post(
                f"{self.config['server_url']}/transcribe",
                data=audio_array.tobytes(),
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            text = result.get("text", "").strip()
            
            if text:
                self.handle_output(text)
                self.show_notification("Transcription", text[:100] + "..." if len(text) > 100 else text)
                self.play_sound("stop")
            else:
                logger.warning("Received empty transcription")
                self.show_notification("STT Error", "Received empty transcription")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            self.show_notification("STT Error", "Network connection failed")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.show_notification("STT Error", "Processing failed")

    def handle_output(self, text):
        try:
            # Always copy to clipboard
            pyperclip.copy(text)
            logger.info("Copied to clipboard")
            
            # For macOS, we can use AppleScript to type
            if self.config["output_mode"] in ["type", "both"]:
                try:
                    script = f'''
                    tell application "System Events"
                        keystroke "{text}"
                    end tell
                    '''
                    subprocess.run(['osascript', '-e', script])
                    logger.info("Text typed using AppleScript")
                except Exception as e:
                    logger.error(f"Error typing text: {e}")
                    
        except Exception as e:
            logger.error(f"Output error: {e}")
            self.show_notification("STT Error", "Couldn't output text")

    def monitor_logs_for_permission_warnings(self):
        """
        Set up a log handler to detect accessibility permission warnings
        """
        class WarningDetector(logging.Handler):
            def __init__(self, client):
                super().__init__()
                self.client = client
                
            def emit(self, record):
                if record.levelno == logging.WARNING and "not trusted" in record.getMessage() and "accessibility" in record.getMessage():
                    self.client.accessibility_permission_warning = True
        
        # Add our custom handler to the pynput.keyboard logger
        warning_detector = WarningDetector(self)
        pynput_logger = logging.getLogger("pynput.keyboard")
        pynput_logger.addHandler(warning_detector)

    def run(self):
        try:
            # Set up warning detection
            self.monitor_logs_for_permission_warnings()
            
            key_combination = self.config["hotkey"]
            listener = keyboard.GlobalHotKeys({
                key_combination: self.toggle_recording
            })
            
            logger.info(f"STT Client running - Press {key_combination} to toggle recording")
            
            # Start the listener
            listener.start()
            
            # Check for accessibility warnings
            time.sleep(1)  # Give time for warnings to appear
            if self.accessibility_permission_warning:
                logger.warning("Accessibility permissions not granted")
                self.show_accessibility_instructions()
            else:
                self.show_notification("STT Client", "Ready to transcribe")
            
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Exiting...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.show_notification("STT Error", "Client crashed")

def configure():
    config_path = os.path.expanduser('~/.config/stt_client/config.yaml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}. {dev['name']} (Inputs: {dev['max_input_channels']})")
    
    config = {
        "server_url": input("\nServer URL [http://192.168.1.11:8000]: ") or "http://192.168.1.11:8000",
        "hotkey": input("Hotkey [<cmd>+<space>]: ") or "<cmd>+<space>",
        "audio_device": input("Audio device [default]: ") or "default",
        "api_key": input("API key (if required): ") or None,
        "notifications": input("Enable notifications? [Y/n]: ").lower() in ["", "y", "yes"],
        "sound_effects": input("Enable sound effects? [Y/n]: ").lower() in ["", "y", "yes"]
    }
    
    print("\nOutput modes:")
    print("1. Clipboard only")
    print("2. Auto-type only")
    print("3. Both")
    mode = input("Select mode [1]: ") or "1"
    config["output_mode"] = ["clipboard", "type", "both"][int(mode)-1] if mode in ["1","2","3"] else "clipboard"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nConfiguration saved to: {config_path}")

def create_launch_agent():
    """Create a LaunchAgent to auto-start the app"""
    agent_dir = os.path.expanduser('~/Library/LaunchAgents')
    os.makedirs(agent_dir, exist_ok=True)
    
    plist_content = f"""
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.username.sttclient</string>
        <key>ProgramArguments</key>
        <array>
            <string>{sys.executable}</string>
            <string>{__file__}</string>
            <string>--start</string>
        </array>
        <key>RunAtLoad</key>
        <true/>
        <key>KeepAlive</key>
        <true/>
        <key>StandardOutPath</key>
        <string>{os.path.expanduser('~/stt_client.log')}</string>
        <key>StandardErrorPath</key>
        <string>{os.path.expanduser('~/stt_client.log')}</string>
    </dict>
    </plist>
    """
    
    plist_path = os.path.join(agent_dir, 'com.username.sttclient.plist')
    with open(plist_path, 'w') as f:
        f.write(plist_content)
    
    print(f"\nCreated LaunchAgent at: {plist_path}")
    print("To load it, run:")
    print(f"launchctl load {plist_path}")
    print("\nNOTE: You'll need to grant accessibility permissions to the Python interpreter:")
    print(f"System Settings > Privacy & Security > Accessibility > Add: {sys.executable}")

def main():
    parser = argparse.ArgumentParser(description='Mac STT Client')
    parser.add_argument('--start', action='store_true', help='Start the client')
    parser.add_argument('--configure', action='store_true', help='Configure settings')
    parser.add_argument('--install', action='store_true', help='Set up auto-start')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.configure:
        configure()
    elif args.install:
        create_launch_agent()
    elif args.start:
        try:
            client = MacSTTClient()
            
            # Check accessibility permissions
            if not client.check_accessibility_permissions():
                logger.warning("Initial accessibility permission check failed")
            
            client.run()
        except Exception as e:
            logger.critical(f"Failed to start: {e}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
