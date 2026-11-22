"""配置从项目根目录 .env 加载；密钥从不进入日志或前端。"""

import sys
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", env_file_encoding="utf-8")
    camera_backend: str = "auto"
    camera_index: int = 0
    camera_width: int = 1920
    camera_height: int = 1080
    templates_dir: Path = ROOT / "data/templates"
    fixture_template: Path | None = None
    mock_captures_dir: Path = ROOT / "data/mock_captures"
    hwr_base_url: str = "http://127.0.0.1:8001/v1"
    hwr_model: str = "Qwen3.5-2B"
    hwr_api_key: SecretStr = SecretStr("")
    ocr_base_url: str = "http://127.0.0.1:8002/v1"
    ocr_model: str = "PaddleOCR-VL-1.5"
    freeform_ocr_enabled: bool = True
    freeform_ocr_provider: str = "auto"  # auto | ocr_vllm | local_vlm | cloud_vl
    freeform_vlm_model: str = "qwen3-vl-plus"
    freeform_prefer_cloud_vl: bool = False
    cloud_llm_base_url: str = "https://api.dahetao.org/v1"
    cloud_llm_model: str = "qwen3.8-27b"
    cloud_llm_api_key: SecretStr = SecretStr("")
    cloud_first_token_timeout: float = 8.0
    cloud_total_timeout: float = 40.0
    cloud_concurrency: int = Field(default=8, ge=1, le=32)  # 整页题目分批并发上限
    cloud_retries: int = Field(default=2, ge=0, le=5)  # 首包前失败（429/5xx/网络）自动重试
    hwr_timeout: float = 25.0
    stt_provider: str = "sensevoice" if sys.platform == "darwin" else "qwen3_asr"
    stt_device: str = "cpu" if sys.platform == "darwin" else "cuda"
    stt_base_url: str = "http://127.0.0.1:8003/v1"
    stt_model: str = "Qwen3-ASR-1.7B"
    stt_api_key: SecretStr = SecretStr("")
    stt_timeout: float = 10.0
    tts_provider: str = "mock"
    tts_warmup: bool = False  # 4090启动脚本开启；Mac默认不加载模型
    tts_voice: str = "Vivian"
    tts_speed: float = Field(default=1.2, ge=0.5, le=2.0)
    chat_max_chars: int = Field(default=12000, ge=1000, le=40000)  # 有界文字缓存，非默认回答目标长度
    tts_device: str = "cuda"
    tts_base_url: str = "http://127.0.0.1:8004/v1"
    tts_api_key: SecretStr = SecretStr("")
    kokoro_voice: str = "zf_xiaobei"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    auto_capture: bool = True
    auto_grade: bool = True
    auto_stable_seconds: float = Field(default=0.55, ge=0.2, le=3)
    auto_change_ratio: float = Field(default=0.045, ge=0.005, le=0.5)
    auto_min_sharpness: float = Field(default=25, ge=1, le=500)
    recording_enabled: bool = False
    recordings_dir: Path = ROOT / "data/capture_recordings"
    recording_fps: int = Field(default=15, ge=5, le=30)
    recording_max_seconds: float = Field(default=60, ge=1, le=120)
    recording_max_bytes: int = 250 * 1024 * 1024
    recording_total_bytes: int = 2 * 1024 * 1024 * 1024
    recording_min_free_bytes: int = 512 * 1024 * 1024
    voice_mode: str = "vad"
    vad_end_silence_ms: int = Field(default=700, ge=200, le=3000)
    opencv_threads: int = 2
    max_upload_bytes: int = 20 * 1024 * 1024
