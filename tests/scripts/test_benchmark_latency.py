import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_latency.py"


def load_benchmark() -> ModuleType:
    # scripts/ is not a package, so the module is loaded from its file path.
    spec = importlib.util.spec_from_file_location("benchmark_latency", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("benchmark_latency", module)
    spec.loader.exec_module(module)
    return module


bench = load_benchmark()


def scenario(mode: str = "cloud", runs: list[dict[str, object]] | None = None) -> object:
    return bench.ScenarioResult(mode, "en", {"llm": "fake"}, runs=runs or [])


# --- statistics -------------------------------------------------------------


def test_percentile_interpolates_linearly_like_numpy() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert bench.percentile(values, 0.9) == pytest.approx(float(np.percentile(values, 90)))
    assert bench.percentile(values, 0.0) == 10.0
    assert bench.percentile(values, 1.0) == 50.0


def test_percentile_of_one_value_is_that_value() -> None:
    assert bench.percentile([7.0], 0.9) == 7.0


@pytest.mark.parametrize(("values", "fraction"), [([], 0.5), ([1.0], 1.5), ([1.0], -0.1)])
def test_percentile_rejects_bad_input(values: list[float], fraction: float) -> None:
    with pytest.raises(ValueError):
        bench.percentile(values, fraction)


def test_summarise_reports_median_p90_and_range() -> None:
    stats = bench.summarise([400.0, 100.0, 300.0, 200.0])

    assert stats == bench.StageStats(
        n=4, median_ms=250.0, p90_ms=pytest.approx(370.0), min_ms=100.0, max_ms=400.0
    )


def test_summarise_of_nothing_is_none() -> None:
    assert bench.summarise([]) is None


def test_scenario_separates_turns_from_barge_in_runs() -> None:
    result = scenario(
        runs=[
            {"prompt": "a", "transcript": "hi", "tools": ["current_time"], "x": 1.0},
            {"prompt": "b", "transcript": "hello", "tools": []},
            {"prompt": "story", "barge_in_onset_to_stop": 400.0},
        ]
    )

    assert len(result.turns()) == 2
    assert result.tool_turns() == 1
    assert result.values(bench.BARGE_IN_TO_STOP) == [400.0]


# --- rendering ----------------------------------------------------------------


def test_render_table_has_a_row_per_stage_and_a_column_per_scenario() -> None:
    cloud = scenario(
        runs=[
            {"transcript": "a", "tools": [], "speech_end_to_first_audio": 1000.0},
            {"transcript": "b", "tools": ["web_search"], "speech_end_to_first_audio": 3000.0},
        ]
    )
    local = scenario("local")

    lines = bench.render_table([cloud, local]).splitlines()

    assert lines[0] == "| Stage | cloud / en (ms) | local / en (ms) |"
    assert lines[1] == "| --- | ---: | ---: |"
    assert len(lines) == 2 + len(bench.STAGES) + 1
    total = next(line for line in lines if "first audio (total)" in line)
    assert total.endswith("| 2000 / 2800 | - |")
    assert lines[-1].endswith("| 2 (1) | 0 (0) |")


def test_format_cell_rounds_to_whole_milliseconds() -> None:
    stats = bench.StageStats(n=3, median_ms=12.4, p90_ms=19.6, min_ms=1.0, max_ms=20.0)

    assert bench.format_cell(stats) == "12 / 20"
    assert bench.format_cell(None) == "-"


def test_caption_names_machine_runs_and_sink() -> None:
    hardware = {"cpu": "CPU X", "os": "Windows-11", "cores": 4, "threads": 8, "ram_gb": 16}

    text = bench.caption(hardware, 5, "null")

    assert "CPU X (4 cores / 8 threads), 16 GB RAM, no NVIDIA GPU, Windows-11" in text
    assert "5 runs per prompt" in text
    assert text.endswith("playback sink: null.")


# --- JSON ---------------------------------------------------------------------


def test_report_json_shape_is_stable_and_serialisable() -> None:
    result = scenario(runs=[{"prompt": "a", "transcript": "hi", "tools": [], "x": 1}])
    result.errors.append("run 3: TimeoutError()")

    document = bench.report_json([result], hardware={"cpu": "CPU X"}, config={"runs_per_prompt": 5})

    assert set(document) == {"schema", "generated_at", "hardware", "config", "stages", "scenarios"}
    assert document["schema"] == bench.JSON_SCHEMA_VERSION
    assert list(document["stages"]) == [stage.key for stage in bench.STAGES]
    (entry,) = document["scenarios"]
    assert set(entry) == {"mode", "language", "providers", "summary", "runs", "errors"}
    assert entry["summary"]["speech_end_to_first_audio"] is None
    assert entry["errors"] == ["run 3: TimeoutError()"]
    assert json.loads(json.dumps(document)) == document


def test_stage_labels_in_json_carry_no_markdown() -> None:
    document = bench.report_json([], hardware={}, config={})

    assert all("*" not in label for label in document["stages"].values())


# --- prompt audio ---------------------------------------------------------------


def test_speech_bounds_skip_leading_and_trailing_silence() -> None:
    samples = np.zeros(bench.SAMPLE_RATE, dtype=np.float32)
    samples[1600:8000] = 0.5  # speech from 100 ms to 500 ms

    start, end = bench.speech_bounds(samples)

    assert start == pytest.approx(0.1)
    assert end == pytest.approx(0.5)


def test_speech_bounds_of_silence_are_zero() -> None:
    assert bench.speech_bounds(np.zeros(100, dtype=np.float32)) == (0.0, 0.0)


def test_wav_round_trip_keeps_samples(tmp_path: Path) -> None:
    samples = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    path = tmp_path / "prompt.wav"

    bench.write_wav(path, samples)

    assert np.allclose(bench.read_wav(path), samples, atol=1e-4)


def test_prompt_cache_name_changes_with_the_text() -> None:
    first = bench.prompt_path(bench.Prompt("k", "en", "one"))
    second = bench.prompt_path(bench.Prompt("k", "en", "two"))

    assert first != second
    assert first.parent == bench.PROMPT_DIR


def test_local_mode_skips_swedish() -> None:
    options = bench.parse_args(["--modes", "cloud,local", "--languages", "en,sv"])

    specs = [(spec.mode, spec.language) for spec in bench.scenarios_for(options)]

    assert specs == [("cloud", "en"), ("cloud", "sv"), ("local", "en")]


# --- instrumentation --------------------------------------------------------------


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_null_playback_plays_in_real_time_and_counts_stops() -> None:
    clock = ManualClock()
    playback = bench.NullPlayback(1000, clock)

    playback.enqueue(np.zeros(500, dtype=np.float32))
    playback.mark_end_of_stream()
    assert playback.is_playing()
    assert playback.queued_samples() == 500
    clock.now = 0.6
    assert not playback.is_playing()

    playback.enqueue(np.zeros(500, dtype=np.float32))
    playback.stop()
    assert not playback.is_playing()
    assert playback.stop_count() == 1


async def test_recorder_wait_returns_the_first_matching_mark() -> None:
    recorder = bench.Recorder()
    recorder.add("state", ("hotkey", "listening"))
    start = recorder.position()

    async def later() -> None:
        await asyncio.sleep(0)
        recorder.add("state", ("request received", "thinking"))
        recorder.add("state", ("answer finished", "idle"))

    task = asyncio.create_task(later())
    mark = await recorder.wait("state", start, lambda value: value[1] == "idle")
    await task

    assert mark.value == ("answer finished", "idle")
    assert recorder.find("state", start, lambda value: value[1] == "listening") is None


def test_turn_metrics_measure_from_speech_end_to_first_answer_audio() -> None:
    recorder = bench.Recorder()
    start = recorder.position()
    recorder.marks.extend(
        [
            bench.Mark(1.0, "enqueue", 100),  # the earcon, before the user spoke
            bench.Mark(2.0, "audio", ("SPEECH_ENDED", 32000)),
            bench.Mark(2.5, "sample", ("voice.speech_end_to_transcript", 480.0)),
            bench.Mark(2.5, "transcript", "hello"),
            bench.Mark(2.6, "tool", "current_time"),
            bench.Mark(4.0, "enqueue", 2400),
        ]
    )

    metrics = bench.turn_metrics(recorder, start, last_word_at=1.3)

    assert metrics["speech_end_to_transcript"] == 480.0
    assert metrics["speech_end_to_first_audio"] == 2000.0
    assert metrics["last_word_to_first_audio"] == 2700.0
    assert metrics["transcript"] == "hello"
    assert metrics["tools"] == ["current_time"]


def test_barge_in_metrics_map_the_engine_clock_to_wall_time() -> None:
    recorder = bench.Recorder()
    push = bench.Mark(10.0, "push", 16000)
    recorder.marks.extend(
        [
            push,
            bench.Mark(10.6, "audio", ("SPEECH_STARTED", 16000 + 8000)),  # 500 ms in
            bench.Mark(10.52, "state", ("barge-in", "listening")),
        ]
    )

    metrics = bench.barge_in_metrics(recorder, 0, push, speech_start_s=0.1)

    assert metrics["barge_in_onset_to_stop"] == pytest.approx(400.0)
    assert metrics["barge_in_to_listening"] == pytest.approx(20.0)


def test_barge_in_metrics_require_an_observed_interruption() -> None:
    recorder = bench.Recorder()
    push = recorder.add("push", 0)

    with pytest.raises(bench.BenchmarkError):
        bench.barge_in_metrics(recorder, 0, push, speech_start_s=0.0)
