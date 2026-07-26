"""TTS Voice 调试 WebUI 与 HTTP API。

Router 只负责请求校验、上传边界与 HTTP 错误转换；
合成、配置刷新与持久化分别由 Service 和 :mod:`.config_persistence` 承担。
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, cast

from fastapi import File, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field as PydanticField, ValidationError

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BasePlugin, BaseRouter

from .config import TTSStyle, TTSVoiceConfig
from .config_persistence import save_config_atomically
from .protocol import TTSPluginLike
from .services.tts_service import TTSService

logger = get_logger("tts_voice_plugin-neo.router")

_CONFIG_PATH = Path("config/plugins/tts_voice_plugin-neo/config.toml")
_WEBUI_PATH = Path(__file__).with_name("webui") / "index.html"
_ASSETS_DIR = Path("data/tts_voice_plugin-neo/assets")
_UPLOAD_CHUNK_SIZE = 1024 * 1024
_UPLOAD_STEM_LIMIT = 80


class SynthesizeRequest(BaseModel):
    """WebUI 调试合成请求。"""

    text: str = PydanticField(min_length=1, max_length=20000)
    style: str = PydanticField(default="default", min_length=1, max_length=80)
    language: str | None = PydanticField(default=None, max_length=32)
    speed_factor: float | None = PydanticField(default=None, ge=0.5, le=2.0)
    aux_refer_wav_paths: list[str] | None = PydanticField(default=None, max_length=16)


class ConfigSaveRequest(BaseModel):
    """WebUI 配置保存请求。

    ``styles`` 直接复用配置模型 :class:`~.config.TTSStyle`，
    避免在 Router 里重复声明一份会与配置定义漂移的字段约束。
    """

    server: str = PydanticField(min_length=1, max_length=2048, pattern=r"^https?://.+")
    timeout: int = PydanticField(ge=1, le=1800)
    max_text_length: int = PydanticField(ge=1, le=20000)
    wsl_mode: bool
    styles: list[TTSStyle] = PydanticField(min_length=1, max_length=64)


class TTSVoiceWebUIRouter(BaseRouter):
    """提供 GPT-SoVITS 调试、上传与配置编辑接口。"""

    name = "tts_voice_webui"
    description = "TTS Voice 语音调试与配置编辑 WebUI"
    custom_route_path = "/plugins/tts-voice"

    def __init__(self, plugin: BasePlugin) -> None:
        """初始化 Router。

        Args:
            plugin: 所属插件实例。
        """
        self.config_path = _CONFIG_PATH
        super().__init__(plugin)

    @property
    def tts_plugin(self) -> TTSPluginLike:
        """所属插件实例。"""
        return cast(TTSPluginLike, self.plugin)

    @staticmethod
    def _load_webui() -> str:
        """读取独立的 WebUI 页面资源。

        Returns:
            页面 HTML 文本。

        Raises:
            RuntimeError: 页面资源文件缺失。
        """
        try:
            return _WEBUI_PATH.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise RuntimeError("webui/index.html 不存在") from error

    def _require_service(self) -> TTSService:
        """返回可用的 Service，否则转换为 503。

        Returns:
            已初始化的 TTS Service。

        Raises:
            HTTPException: Service 尚未初始化。
        """
        service = self.tts_plugin.tts_service
        if service is None:
            raise HTTPException(status_code=503, detail="TTS Service is not initialized")
        return service

    @staticmethod
    def _safe_upload_name(file_name: str) -> str:
        """把客户端文件名转换为安全且唯一的 WAV 文件名。

        Args:
            file_name: 客户端提供的原始文件名。

        Returns:
            安全的 WAV 文件名。

        Raises:
            HTTPException: 文件名包含目录成分、扩展名不是 ``.wav``，或清洗后为空。
        """
        normalized = file_name.replace("\\", "/")
        base_name = Path(normalized).name
        if not base_name or base_name != normalized:
            raise HTTPException(status_code=400, detail="Invalid file name")
        if Path(base_name).suffix.lower() != ".wav":
            raise HTTPException(status_code=400, detail="Only WAV files are allowed")

        stem = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", Path(base_name).stem).strip("_.")
        if not stem:
            raise HTTPException(status_code=400, detail="Invalid file name")
        return f"{stem[:_UPLOAD_STEM_LIMIT]}_{uuid.uuid4().hex[:8]}.wav"

    @staticmethod
    def _merge_config(
        current: TTSVoiceConfig,
        request: ConfigSaveRequest,
    ) -> TTSVoiceConfig:
        """在保留未展示配置节的前提下构造并校验新配置。

        Args:
            current: 当前生效的配置。
            request: WebUI 提交的可编辑字段。

        Returns:
            校验通过的新配置实例。

        Raises:
            ValidationError: 合并结果不满足配置模型约束。
        """
        raw = current.model_dump(mode="python")
        raw["tts"].update(
            {
                "server": request.server.strip(),
                "timeout": request.timeout,
                "max_text_length": request.max_text_length,
                "wsl_mode": request.wsl_mode,
            }
        )
        raw["tts_styles"] = [style.model_dump(mode="python") for style in request.styles]
        return TTSVoiceConfig.model_validate(raw)

    def register_endpoints(self) -> None:
        """注册页面与 API 端点。"""

        @self.app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def get_index() -> HTMLResponse:
            """返回调试 WebUI 页面。"""

            return HTMLResponse(self._load_webui())

        @self.app.post("/api/upload")
        async def post_upload(file: UploadFile = File(...)) -> dict[str, str]:
            """限量保存 WAV 文件并返回绝对路径。"""

            safe_name = self._safe_upload_name(file.filename or "")
            max_bytes = self.tts_plugin.config.tts.upload_max_mb * 1024 * 1024
            assets_dir = _ASSETS_DIR.resolve()
            assets_dir.mkdir(parents=True, exist_ok=True)
            destination = assets_dir / safe_name

            total = 0
            try:
                with destination.open("xb") as output:
                    while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
                        total += len(chunk)
                        if total > max_bytes:
                            raise HTTPException(status_code=413, detail="WAV file is too large")
                        output.write(chunk)
            except HTTPException:
                destination.unlink(missing_ok=True)
                raise
            except OSError as error:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail="Failed to save WAV file") from error
            finally:
                await file.close()

            if total == 0:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="WAV file is empty")
            return {"file_path": destination.as_posix()}

        @self.app.get("/api/styles")
        async def get_styles() -> dict[str, dict[str, Any]]:
            """返回当前生效的风格档案，按风格名索引。"""

            return {
                name: {
                    "refer_wav_path": profile.refer_wav_path,
                    "prompt_text": profile.prompt_text,
                    "prompt_language": profile.prompt_language,
                    "speed_factor": profile.speed_factor,
                    "text_language": profile.text_language,
                    "aux_refer_wav_paths": list(profile.aux_refer_wav_paths),
                }
                for name, profile in self._require_service().styles.items()
            }

        @self.app.get("/api/config")
        async def get_config() -> dict[str, Any]:
            """返回 WebUI 可编辑的配置。"""

            config = self.tts_plugin.config
            return {
                "server": config.tts.server,
                "timeout": config.tts.timeout,
                "max_text_length": config.tts.max_text_length,
                "wsl_mode": config.tts.wsl_mode,
                "styles": [style.model_dump(mode="python") for style in config.tts_styles],
            }

        @self.app.post("/api/config/save")
        async def post_save_config(request: ConfigSaveRequest) -> dict[str, str]:
            """校验、原子保存并刷新内存中的配置。"""

            try:
                updated = self._merge_config(self.tts_plugin.config, request)
                save_config_atomically(self.config_path, updated)
            except ValidationError as error:
                raise HTTPException(status_code=422, detail=error.errors()) from error
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            except OSError as error:
                raise HTTPException(status_code=500, detail="Failed to save config") from error

            self.tts_plugin.config = updated
            self._require_service().refresh_config()
            self.tts_plugin.refresh_action_description()
            return {"status": "ok", "message": "Config saved successfully"}

        @self.app.post("/api/synthesize")
        async def post_synthesize(request: SynthesizeRequest) -> Response:
            """调用 Service 合成 WAV 字节。"""

            service = self._require_service()
            try:
                audio_bytes = await service.generate_voice_bytes(
                    text=request.text,
                    style_hint=request.style,
                    language_hint=request.language,
                    speed_factor=request.speed_factor,
                    aux_refer_wav_paths=request.aux_refer_wav_paths,
                )
            except (ValueError, TypeError) as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            except Exception as error:
                logger.error(f"WebUI 合成失败: {error}")
                raise HTTPException(status_code=502, detail="TTS synthesis failed") from error

            if not audio_bytes:
                raise HTTPException(status_code=502, detail="TTS returned empty audio")
            return Response(content=audio_bytes, media_type="audio/wav")


__all__ = [
    "ConfigSaveRequest",
    "SynthesizeRequest",
    "TTSVoiceWebUIRouter",
]
