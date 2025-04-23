#!/usr/bin/env python3
import os
import yaml
import argparse
import sounddevice as sd
import numpy as np
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn
import logging
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stt_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class STTServer:
    def __init__(self):
        self.config = self.load_config()
        self.model = None
        self.sample_rate = 16000
        self.load_model()
    
    def load_config(self):
        config_path = os.path.expanduser("~/.config/stt_server/config.yaml")
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                default_config = {
                    "port": 8000,
                    "host": "0.0.0.0",
                    "audio_device": "default",
                    "use_gpu": True,
                    "compute_type": "float16",
                    "allowed_origins": ["*"],
                    "api_key": None
                }
                with open(config_path, "w") as f:
                    yaml.dump(default_config, f)
                return default_config
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if 'compute_type' not in config:
                    config['compute_type'] = "float16" if config.get('use_gpu', True) else "float32"
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.load_config()
    
    def load_model(self):
        try:
            use_gpu = self.config["use_gpu"] and torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"
            compute_type = self.config["compute_type"]
            
            # Model selection logic
            model_name = "large-v3-turbo" if use_gpu else "small"
            logger.info(f"Selected model: {model_name} for device: {device}")
            
            logger.info(f"Loading faster-whisper model {model_name} on {device} ({compute_type})...")
            
            self.model = WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type,
                download_root=os.path.expanduser("~/.cache/whisper")
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def transcribe_audio(self, audio_array):
        try:
            start_time = datetime.now()
            segments, info = self.model.transcribe(
                audio_array,
                language="en",
                beam_size=5
            )
            text = " ".join([segment.text for segment in segments])
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            logger.info(f"Transcription length: {len(text)} characters")
            
            return {
                "text": text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

app = FastAPI()
server = STTServer()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=server.config["allowed_origins"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(request: Request):
    # Log request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Received request from {client_ip}, content length: {request.headers.get('content-length')}")
    
    # Optional API key authentication
    if server.config["api_key"]:
        if request.headers.get("x-api-key") != server.config["api_key"]:
            logger.warning(f"Invalid API key from {client_ip}")
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Receive audio data as numpy array
        audio_data = await request.body()
        logger.debug(f"Received audio data size: {len(audio_data)} bytes")
        
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            logger.debug(f"Audio array shape: {audio_array.shape}")
            
            if audio_array.size == 0:
                logger.error("Received empty audio data")
                raise HTTPException(status_code=400, detail="Empty audio data")
                
            result = server.transcribe_audio(audio_array)
            return result
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Audio processing error: {e}")
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

def configure():
    config_path = os.path.expanduser("~/.config/stt_server/config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    print("Available audio devices:")
    print(sd.query_devices())
    
    config = {}
    config["port"] = int(input(f"Port [8000]: ") or "8000")
    config["host"] = input(f"Host [0.0.0.0]: ") or "0.0.0.0"
    
    config["audio_device"] = input(f"Audio device ID/name [default]: ") or "default"
    use_gpu = input(f"Use GPU if available? [Y/n]: ").lower() in ["", "y", "yes"]
    config["use_gpu"] = use_gpu
    config["compute_type"] = "float16" if use_gpu else "int8"
    
    origins = input("Allowed CORS origins (comma separated) [*]: ") or "*"
    config["allowed_origins"] = origins.split(",") if origins != "*" else ["*"]
    
    api_key = input("API key (leave empty for no auth): ") or None
    config["api_key"] = api_key
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Configuration saved to {config_path}")

def main():
    parser = argparse.ArgumentParser(description="STT API Server with Faster-Whisper")
    parser.add_argument("--start", action="store_true", help="Start the STT server")
    parser.add_argument("--configure", action="store_true", help="Configure settings")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.configure or not os.path.exists(os.path.expanduser("~/.config/stt_server/config.yaml")):
        configure()
    
    if args.start:
        try:
            logger.info(f"Starting STT server at {server.config['host']}:{server.config['port']}")
            uvicorn.run(
                app,
                host=server.config["host"],
                port=server.config["port"],
                log_level="debug" if args.debug else "info"
            )
        except Exception as e:
            logger.critical(f"Fatal error: {e}")
            if "faster_whisper" in str(e):
                print("Try installing faster-whisper manually: pip install faster-whisper")

if __name__ == "__main__":
    main()