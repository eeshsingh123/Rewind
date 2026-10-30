import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from rewind.contracts import TimeSpan
from rewind.core.ports import SPEECH_RATE_HZ, VideoInfo, VideoToolsError

_SHOWINFO_PTS = re.compile(r"Parsed_showinfo.*?\bpts_time:\s*([0-9.eE+-]+)")
_ENCODE = ("-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p")
_PLAYBACK_HEIGHT = 1080


class FfmpegVideoTools:
    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe

    async def probe(self, source: Path) -> VideoInfo:
        out = await _run(
            self._ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-show_entries", "stream_side_data=rotation",
            "-show_entries", "format=duration",
            "-of", "json",
            str(source),
        )  # fmt: skip
        data = json.loads(out.stdout)
        if not data.get("streams"):
            raise VideoToolsError(f"no video stream in {source.name}")
        stream = data["streams"][0]
        fps = float(Fraction(stream["avg_frame_rate"]))
        duration = float(stream.get("duration") or data["format"]["duration"])
        frame_count = int(stream.get("nb_frames") or round(duration * fps))
        width, height = int(stream["width"]), int(stream["height"])
        # Phones store a portrait video sideways; every decoder here turns it upright.
        if _quarter_turned(stream):
            width, height = height, width
        return VideoInfo(
            duration_s=duration,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )

    async def make_proxy(
        self,
        source: Path,
        destination: Path,
        spans: Sequence[TimeSpan],
        fps: float,
        max_height: int,
    ) -> tuple[float, ...]:
        if not spans:
            raise VideoToolsError("a proxy needs at least one span")
        in_spans = "+".join(f"between(t,{span.start:.3f},{span.end:.3f})" for span in spans)
        # Keeps the first source frame at or after each 1/fps step. The fps filter is not used:
        # it stamps grid times on frames up to half a step away, so the time map would be wrong.
        new_step = f"gt(floor(t*{fps:g}+1e-6),floor(prev_t*{fps:g}+1e-6))"
        selected = f"({in_spans})*(isnan(prev_t)+{new_step})"
        # showinfo sits after select and before setpts, so it logs each kept frame's footage time.
        filters = (
            f"select='{selected}',showinfo,setpts=N/({fps:g}*TB),scale=-2:'min({max_height},ih)'"
        )
        out = await _run(
            self._ffmpeg,
            "-hide_banner", "-nostdin", "-y", "-loglevel", "info",
            "-i", str(source),
            "-an", "-vf", filters,
            "-fps_mode", "cfr", "-r", f"{fps:g}",
            *_ENCODE, "-movflags", "+faststart",
            str(destination),
        )  # fmt: skip
        frame_times = tuple(float(t) for t in _SHOWINFO_PTS.findall(out.stderr))
        if not frame_times:
            raise VideoToolsError("the spans selected no frames")
        return frame_times

    async def cut_clip(self, source: Path, destination: Path, span: TimeSpan) -> None:
        await _run(
            self._ffmpeg,
            "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-ss", f"{span.start:.3f}", "-to", f"{span.end:.3f}",
            "-i", str(source),
            "-an", *_ENCODE, "-movflags", "+faststart",
            str(destination),
        )  # fmt: skip

    async def playable(self, source: Path, destination: Path) -> None:
        await _run(
            self._ffmpeg,
            "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-i", str(source),
            "-vf", f"scale=-2:'min({_PLAYBACK_HEIGHT},ih)'",
            *_ENCODE, "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            str(destination),
        )  # fmt: skip

    async def speech(self, source: Path, destination: Path) -> None:
        await _run(
            self._ffmpeg,
            "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(SPEECH_RATE_HZ), "-c:a", "pcm_s16le",
            "-fflags", "+bitexact", "-f", "wav",
            str(destination),
        )  # fmt: skip

    async def still(self, source: Path, t: float, destination: Path) -> None:
        await _run(
            self._ffmpeg,
            "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "2",
            str(destination),
        )  # fmt: skip


def _quarter_turned(stream: dict[str, Any]) -> bool:
    return any(
        int(side.get("rotation", 0)) % 180 == 90 for side in stream.get("side_data_list", ())
    )


@dataclass(frozen=True, slots=True)
class _Output:
    stdout: str
    stderr: str


async def _run(*command: str) -> _Output:
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as error:
        raise VideoToolsError(f"{command[0]} is not installed or not on PATH") from error
    stdout, stderr = await process.communicate()
    out = _Output(stdout.decode(errors="replace"), stderr.decode(errors="replace"))
    if process.returncode != 0:
        tail = out.stderr.strip().splitlines()[-3:]
        raise VideoToolsError(f"{Path(command[0]).name} failed: {' | '.join(tail)}")
    return out
