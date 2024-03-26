import argparse
import base64
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field
import runpod
import soundfile

from voicevox_engine.core.core_adapter import CoreAdapter
from voicevox_engine.core.core_initializer import initialize_cores
from voicevox_engine.metas.Metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.preset.PresetError import PresetError
from voicevox_engine.preset.PresetManager import PresetManager
from voicevox_engine.tts_pipeline.kana_converter import create_kana
from voicevox_engine.tts_pipeline.tts_engine import (
    TTSEngine,
    make_tts_engines_from_cores,
)
from voicevox_engine.utility.core_version_utility import get_latest_core_version
from voicevox_engine.utility.path_utility import engine_root


class AudioQueryJob(BaseModel):
    action: Literal["audio_query"]
    text: str
    style_id: StyleId


class AudioQueryFromPresetJob(BaseModel):
    action: Literal["audio_query_from_preset"]
    text: str
    preset_id: int


class SynthesisJob(BaseModel):
    action: Literal["synthesis"]
    audio_query: AudioQuery
    style_id: StyleId
    enable_interrogative_upspeak: bool = True

class AutoSynthesisJob(BaseModel):
    action: Literal["auto_synthesis"]
    text: str
    style_id: StyleId

class SynthesisResult(BaseModel):
    wave: str
    size: int


JobTypes = Union[AudioQueryJob, AudioQueryFromPresetJob, SynthesisJob, AutoSynthesisJob]


class JobInput(BaseModel):
    job: JobTypes = Field(..., discriminator="action")


class App:
    def __init__(
        self,
        tts_engines: dict[str, TTSEngine],
        cores: dict[str, CoreAdapter],
        latest_core_version: str,
        preset_manager: PresetManager,
        root_dir: Optional[Path] = None,
    ):
        self.tts_engines = tts_engines
        self.cores = cores
        self.latest_core_version = latest_core_version
        self.preset_manager = preset_manager
        self.root_dir = root_dir or engine_root()

    async def handler(self, job):
        print(job)
        model = JobInput(job=job.get("input"))
        print(model)
        return self._action_switch(model.job)

    def _action_switch(self, job: JobTypes) -> BaseModel:
        if isinstance(job, AudioQueryJob):
            return self.audio_query(job.text, job.style_id)

        if isinstance(job, AudioQueryFromPresetJob):
            return self.audio_query_from_preset(job.text, job.preset_id)

        if isinstance(job, SynthesisJob):
            return self.synthesis(
                job.audio_query, job.style_id, job.enable_interrogative_upspeak
            )
        
        if isinstance(job, AutoSynthesisJob):
            return self.auto_synthesis(job.text, job.style_id)

        raise ValueError(f"不明なアクション: {job}")

    def _get_engine(self, core_version: Optional[str] = None) -> TTSEngine:
        if core_version is None:
            core_version = self.latest_core_version
        if core_version in self.tts_engines:
            return self.tts_engines[core_version]
        raise ValueError(
            f"指定されたバージョンの音声合成エンジンが見つかりませんでした: {core_version}"
        )

    def _get_core(self, core_version: Optional[str] = None) -> CoreAdapter:
        if core_version is None:
            return self.cores[self.latest_core_version]
        if core_version in self.cores:
            return self.cores[core_version]
        raise ValueError(
            f"指定されたバージョンの音声合成エンジンが見つかりませんでした: {core_version}"
        )

    def audio_query(self, text: str, style_id: StyleId) -> AudioQuery:
        engine = self._get_engine()
        core = self._get_core()
        accent_phases = engine.create_accent_phrases(text, style_id)

        return AudioQuery(
            accent_phrases=accent_phases,
            speedScale=1,
            pitchScale=0,
            intonationScale=1,
            volumeScale=1,
            prePhonemeLength=0.1,
            postPhonemeLength=0.1,
            outputSamplingRate=core.default_sampling_rate,
            outputStereo=False,
            kana=create_kana(accent_phases),
        )

    def audio_query_from_preset(self, text: str, preset_id: int) -> AudioQuery:
        engine = self._get_engine()
        core = self._get_core()
        try:
            presets = self.preset_manager.load_presets()
        except PresetError as e:
            raise ValueError(f"プリセットの読み込みに失敗しました: {e}")

        for preset in presets:
            if preset.id == preset_id:
                selected_preset = preset
                break
        else:
            raise ValueError(f"指定されたプリセットが見つかりませんでした: {preset_id}")

        accent_phases = engine.create_accent_phrases(text, selected_preset.style_id)

        return AudioQuery(
            accent_phrases=accent_phases,
            speedScale=selected_preset.speedScale,
            pitchScale=selected_preset.pitchScale,
            intonationScale=selected_preset.intonationScale,
            volumeScale=selected_preset.volumeScale,
            prePhonemeLength=selected_preset.prePhonemeLength,
            postPhonemeLength=selected_preset.postPhonemeLength,
            outputSamplingRate=core.default_sampling_rate,
            outputStereo=False,
            kana=create_kana(accent_phases),
        )

    def synthesis(
        self,
        audio_query: AudioQuery,
        style_id: StyleId,
        enable_interrogative_upspeak: bool = True,
    ) -> SynthesisResult:
        engine = self._get_engine()
        wave = engine.synthesize_wave(
            query=audio_query,
            style_id=style_id,
            enable_interrogative_upspeak=enable_interrogative_upspeak,
        )

        with NamedTemporaryFile(suffix=".wav") as f:
            soundfile.write(
                file=f,
                data=wave,
                samplerate=audio_query.outputSamplingRate,
                format="WAV",
            )

            f.seek(0)

            wavedata = f.read()
            base64_wave = base64.b64encode(wavedata).decode("utf-8")
        base64_url = f"data:audio/wav;base64,{base64_wave}"

        return SynthesisResult(wave=base64_url, size=len(wavedata))
    
    def auto_synthesis(self, text: str, style_id: StyleId) -> SynthesisResult:
        audio_query = self.audio_query(text, style_id)
        return self.synthesis(audio_query, style_id)


def setup():
    parser = argparse.ArgumentParser(description="VOICEVOX Runpod worker wrapper.")

    parser.add_argument(
        "--use_gpu", action="store_true", help="GPUを使って音声合成するようになります。"
    )
    parser.add_argument(
        "--voicevox_dir",
        type=Path,
        default=None,
        help="VOICEVOXのディレクトリパスです。",
    )
    parser.add_argument(
        "--voicelib_dir",
        type=Path,
        default=None,
        action="append",
        help="VOICEVOX COREのディレクトリパスです。",
    )
    parser.add_argument(
        "--runtime_dir",
        type=Path,
        default=None,
        action="append",
        help="VOICEVOX COREで使用するライブラリのディレクトリパスです。",
    )
    parser.add_argument(
        "--enable_mock",
        action="store_true",
        help="VOICEVOX COREを使わずモックで音声合成を行います。",
    )
    parser.add_argument(
        "--enable_cancellable_synthesis",
        action="store_true",
        help="音声合成を途中でキャンセルできるようになります。",
    )
    parser.add_argument(
        "--init_processes",
        type=int,
        default=2,
        help="cancellable_synthesis機能の初期化時に生成するプロセス数です。",
    )
    parser.add_argument(
        "--load_all_models",
        action="store_true",
        help="起動時に全ての音声合成モデルを読み込みます。",
    )

    # 引数へcpu_num_threadsの指定がなければ、環境変数をロールします。
    # 環境変数にもない場合は、Noneのままとします。
    # VV_CPU_NUM_THREADSが空文字列でなく数値でもない場合、エラー終了します。
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        default=os.getenv("VV_CPU_NUM_THREADS") or None,
        help=(
            "音声合成を行うスレッド数です。指定しない場合、代わりに環境変数 VV_CPU_NUM_THREADS の値が使われます。"
            "VV_CPU_NUM_THREADS が空文字列でなく数値でもない場合はエラー終了します。"
        ),
    )

    # parser.add_argument(
    #     "--setting_file",
    #     type=Path,
    #     default=USER_SETTING_PATH,
    #     help="設定ファイルを指定できます。",
    # )

    parser.add_argument(
        "--preset_file",
        type=Path,
        default=None,
        help=(
            "プリセットファイルを指定できます。"
            "指定がない場合、環境変数 VV_PRESET_FILE、--voicevox_dirのpresets.yaml、"
            "実行ファイルのディレクトリのpresets.yamlを順に探します。"
        ),
    )

    args, _ = parser.parse_known_args()

    return args


def create_app():
    args = setup()

    use_gpu: bool = args.use_gpu
    voicevox_dir: Path | None = args.voicevox_dir
    voicelib_dirs: list[Path] | None = args.voicelib_dir
    runtime_dirs: list[Path] | None = args.runtime_dir
    enable_mock: bool = args.enable_mock
    cpu_num_threads: int | None = args.cpu_num_threads
    load_all_models: bool = args.load_all_models

    cores = initialize_cores(
        use_gpu=use_gpu,
        voicevox_dir=voicevox_dir,
        voicelib_dirs=voicelib_dirs,
        runtime_dirs=runtime_dirs,
        enable_mock=enable_mock,
        cpu_num_threads=cpu_num_threads,
        load_all_models=load_all_models,
    )

    tts_engines = make_tts_engines_from_cores(cores)
    assert len(tts_engines) != 0, "音声合成エンジンがありません。"

    latest_core_version = get_latest_core_version(cores)

    init_processes: int = args.init_processes

    root_dir = voicevox_dir or engine_root()

    preset_path: Path | None = args.preset_file
    if preset_path is None:
        env_preset_path = os.getenv("VV_PRESET_FILE")
        if env_preset_path is not None and len(env_preset_path) != 0:
            preset_path = Path(env_preset_path)
        else:
            preset_path = root_dir / "presets.yaml"

    preset_manager = PresetManager(preset_path=preset_path)

    app = App(
        tts_engines=tts_engines,
        cores=cores,
        latest_core_version=latest_core_version,
        preset_manager=preset_manager,
        root_dir=root_dir,
    )
    return app


def main():
    app = create_app()
    runpod.serverless.start({"handler": app.handler})


if __name__ == "__main__":
    main()
