from pathlib import Path
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from add.voice_output import (
    PROMPT_ONLY,
    RECOMMENDED_GIFT_FILE,
    START_RECOGNITION,
    START_RECOMMENDED_GIFT,
    START_RETRY_RECOGNITION,
    VoiceOutput,
    VoicePlaybackResult,
)
from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp
from grasp_core.tools.flowpose_request_ik_app import PipelineStage, RuntimeState
from add.voice import VoiceGraspInput


class _FirstChoice:
    def choice(self, items):
        return items[0]


def test_welcome_selects_greeting_then_first_question(tmp_path: Path) -> None:
    greeting = tmp_path / "01_greeting"
    question = tmp_path / "02_1_first_gift_question"
    greeting.mkdir()
    question.mkdir()
    (greeting / "01_hello.mp3").touch()
    (question / "01_question.mp3").touch()

    output = VoiceOutput(tmp_path, player="/bin/true", rng=_FirstChoice())
    selected = output._select_files(
        ("01_greeting", "02_1_first_gift_question")
    )

    assert selected == (
        greeting / "01_hello.mp3",
        question / "01_question.mp3",
    )


def test_repeat_question_requests_retry_recognition(tmp_path: Path) -> None:
    repeat = tmp_path / "02_2_repeat_question"
    repeat.mkdir()
    prompt = repeat / "01_repeat.mp3"
    prompt.touch()
    output = VoiceOutput(tmp_path, player="/bin/true", rng=_FirstChoice())
    try:
        output.play_repeat_question()
        result = None
        deadline = time.monotonic() + 1.0
        while result is None and time.monotonic() < deadline:
            result = output.poll()
            if result is None:
                time.sleep(0.01)
    finally:
        output.close()

    assert result is not None
    assert result.files == (prompt,)
    assert result.purpose == START_RETRY_RECOGNITION


def test_recommended_gift_plays_the_exact_configured_file(tmp_path: Path) -> None:
    prompt = tmp_path / RECOMMENDED_GIFT_FILE
    prompt.parent.mkdir()
    prompt.touch()
    output = VoiceOutput(tmp_path, player="/bin/true")
    try:
        status = output.play_recommended_gift()
        result = None
        deadline = time.monotonic() + 1.0
        while result is None and time.monotonic() < deadline:
            result = output.poll()
            if result is None:
                time.sleep(0.01)
    finally:
        output.close()

    assert "03_2_task_accepted" in status
    assert result is not None
    assert result.files == (prompt,)
    assert result.purpose == START_RECOMMENDED_GIFT


class _VoiceOutput:
    def __init__(self, result=None) -> None:
        self.result = result
        self.play_calls = 0
        self.accepted_calls = 0
        self.failed_calls = 0
        self.repeat_calls = 0
        self.recommended_calls = 0
        self.running = False
        self.cancel_calls = 0

    def play_welcome(self) -> str:
        self.play_calls += 1
        return "prompt playing"

    def play_task_accepted(self) -> str:
        self.accepted_calls += 1
        return "accepted playing"

    def play_task_failed(self) -> str:
        self.failed_calls += 1
        return "failed playing"

    def play_repeat_question(self) -> str:
        self.repeat_calls += 1
        return "repeat question playing"

    def play_recommended_gift(self) -> str:
        self.recommended_calls += 1
        return "recommendation playing"

    def poll(self):
        result, self.result = self.result, None
        return result

    def cancel(self) -> None:
        self.cancel_calls += 1


class _VoiceInput:
    enabled = True
    task_active = False
    recognition_running = False

    def __init__(self) -> None:
        self.request_calls = 0
        self.cancel_calls = 0
        self.request_durations = []

    def request(self, *, robot_busy: bool, record_seconds=None) -> str:
        assert robot_busy is False
        self.request_calls += 1
        self.request_durations.append(record_seconds)
        duration = "default" if record_seconds is None else f"{record_seconds:g}"
        return f"recognition started ({duration})"

    def cancel(self) -> None:
        self.cancel_calls += 1


def _app(result=None) -> GraspDemoApp:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState()
    app.voice_input = _VoiceInput()
    app.voice_output = _VoiceOutput(result)
    app.grasp_future = None
    app.place_future = None
    app.manual_home_future = None
    app.auto_home_future = None
    app.gripper_future = None
    app.wave_future = None
    app.wave_recovery_required = False
    app.voice_intro_active = False
    app.voice_intro_prompt_done = False
    app.voice_intro_wave_done = False
    app.voice_recognition_attempt = 0
    app.robot_actions = Mock()
    app.ik_publisher = Mock()
    app.action_executor = Mock()
    app.action_executor.submit.return_value = Future()
    app.robot_busy = lambda: False
    return app


def test_v_request_starts_welcome_and_wave_together() -> None:
    app = _app()

    app.request_voice()

    assert app.voice_output.play_calls == 1
    assert app.voice_input.request_calls == 0
    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_wave
    )
    assert app.wave_future is app.action_executor.submit.return_value
    assert app.voice_intro_active
    assert app.state.status == "prompt playing; right-arm wave running"


def test_recognition_starts_after_prompt_while_wave_is_running() -> None:
    result = VoicePlaybackResult(
        files=(Path("greeting.mp3"), Path("question.mp3")),
        purpose=START_RECOGNITION,
    )
    app = _app(result)
    app.request_voice()

    app.collect_voice_output()

    assert app.voice_input.request_calls == 1
    assert app.state.status == "recognition started (default); wave returning HOME"

    app.wave_future.set_result("Right-arm wave completed")
    app.collect_wave_result()

    assert app.voice_input.request_calls == 1
    assert app.state.status == "Wave HOME complete; voice recognition active"


def test_recognition_waits_for_prompt_when_wave_returns_first() -> None:
    result = VoicePlaybackResult(
        files=(Path("greeting.mp3"), Path("question.mp3")),
        purpose=START_RECOGNITION,
    )
    app = _app(result)
    app.request_voice()
    app.wave_future.set_result("Right-arm wave completed")

    app.collect_wave_result()

    assert app.voice_input.request_calls == 0
    assert app.state.status == "Wave HOME complete; waiting for welcome prompt"

    app.collect_voice_output()

    assert app.voice_input.request_calls == 1
    assert app.state.status == "recognition started (default)"


def test_failed_wave_cancels_v_intro_without_starting_recognition() -> None:
    result = VoicePlaybackResult(
        files=(Path("greeting.mp3"), Path("question.mp3")),
        purpose=START_RECOGNITION,
    )
    app = _app(result)
    app.request_voice()
    app.wave_future.set_exception(RuntimeError("wave unavailable"))

    app.collect_wave_result()
    app.collect_voice_output()

    assert app.voice_output.cancel_calls == 1
    assert app.voice_input.cancel_calls == 1
    assert app.voice_input.request_calls == 0
    assert app.state.status == "Wave failed: wave unavailable"


def test_recognized_command_is_held_until_wave_returns_home() -> None:
    result = VoicePlaybackResult(
        files=(Path("greeting.mp3"), Path("question.mp3")),
        purpose=START_RECOGNITION,
    )
    app = _app(result)
    app.request_voice()
    app.collect_voice_output()
    parsed = SimpleNamespace(target="white toy", handover=True, status="recognized")
    app.voice_input.poll = Mock(return_value=parsed)
    app.voice_input.activate = Mock()
    app.start_pipeline = Mock(return_value=True)

    app.collect_voice_result()

    app.voice_input.poll.assert_not_called()
    app.start_pipeline.assert_not_called()

    app.wave_future.set_result("Right-arm wave completed")
    app.collect_wave_result()
    app.collect_voice_result()

    app.voice_input.poll.assert_called_once_with()
    app.voice_input.activate.assert_called_once_with("white toy", handover=True)
    app.start_pipeline.assert_called_once_with(grasp_on_done=True)


def test_e_key_prompt_then_starts_rightmost_toy_handover_workflow() -> None:
    result = VoicePlaybackResult(
        files=(Path("04_let_me_pick_a_gift_for_you.mp3"),),
        purpose=START_RECOMMENDED_GIFT,
    )
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState()
    app.manual_home_future = None
    app.voice_input = Mock()
    app.voice_input.task_active = False
    app.voice_input.recognition_running = False
    app.voice_output = Mock()
    app.voice_output.running = False
    app.voice_output.play_recommended_gift.return_value = "recommendation playing"
    app.voice_output.poll.return_value = result
    app.robot_busy = Mock(return_value=False)
    app.start_pipeline = Mock(return_value=True)

    assert app.handle_key(ord("e"), None)
    assert app.state.status == "recommendation playing"
    app.start_pipeline.assert_not_called()

    app.collect_voice_output()

    app.voice_input.activate.assert_called_once_with("white toy", handover=True)
    app.start_pipeline.assert_called_once_with(grasp_on_done=True)
    app.voice_output.play_task_accepted.assert_not_called()


def test_recognition_does_not_start_if_app_is_paused_during_prompt() -> None:
    result = VoicePlaybackResult(
        files=(Path("greeting.mp3"), Path("question.mp3")),
        purpose=START_RECOGNITION,
    )
    app = _app(result)
    app.state.paused = True

    app.collect_voice_output()

    assert app.voice_input.request_calls == 0
    assert app.state.status == "Voice recognition cancelled: application paused"


def test_other_prompt_completion_does_not_restart_recognition() -> None:
    app = _app(VoicePlaybackResult(files=(Path("accepted.mp3"),)))

    app.collect_voice_output()

    assert app.voice_input.request_calls == 0


def test_valid_recognized_gift_plays_task_accepted_prompt() -> None:
    app = _app()
    app.state.paused = False
    app.voice_input = Mock()
    app.voice_input.poll.return_value = SimpleNamespace(
        target="toy", handover=True, status="recognized"
    )
    app.start_pipeline = Mock(return_value=True)

    app.collect_voice_result()

    assert app.voice_output.accepted_calls == 1
    app.voice_input.activate.assert_called_once_with("toy", handover=True)
    app.start_pipeline.assert_called_once_with(grasp_on_done=True)


def test_valid_voice_with_failed_capture_plays_only_failure() -> None:
    app = _app()
    app.state.paused = False
    app.voice_input = Mock()
    app.voice_input.poll.return_value = SimpleNamespace(
        target="toy", handover=True, status="recognized"
    )
    app.start_pipeline = Mock(return_value=False)

    app.collect_voice_result()

    assert app.voice_output.accepted_calls == 0
    assert app.voice_output.failed_calls == 1
    app.voice_input.finish_task.assert_called_once_with()


def test_unrecognized_voice_plays_task_failed_prompt() -> None:
    app = _app()
    app.state.paused = False
    app.voice_input = Mock()
    app.voice_input.poll.return_value = SimpleNamespace(
        target=None, handover=False, status="not recognized"
    )

    app.collect_voice_result()

    assert app.voice_output.failed_calls == 1
    app.voice_input.finish_task.assert_called_once_with()


def test_first_unrecognized_v_command_asks_again() -> None:
    app = _app()
    app.state.paused = False
    app.voice_recognition_attempt = 1
    app.voice_input = Mock()
    app.voice_input.poll.return_value = SimpleNamespace(
        target=None, handover=False, status="not recognized"
    )

    app.collect_voice_result()

    assert app.voice_output.repeat_calls == 1
    assert app.voice_output.failed_calls == 0
    assert app.voice_output.recommended_calls == 0
    assert app.voice_recognition_attempt == 2
    app.voice_input.finish_task.assert_called_once_with()


def test_repeat_question_starts_three_second_recognition() -> None:
    result = VoicePlaybackResult(
        files=(Path("02_2_repeat_question/01_repeat.mp3"),),
        purpose=START_RETRY_RECOGNITION,
    )
    app = _app(result)

    app.collect_voice_output()

    assert app.voice_input.request_calls == 1
    assert app.voice_input.request_durations == [3.0]
    assert app.state.status == "recognition started (3)"


def test_second_unrecognized_v_command_enters_recommendation_flow() -> None:
    app = _app()
    app.state.paused = False
    app.voice_recognition_attempt = 2
    app.voice_input = Mock()
    app.voice_input.poll.return_value = SimpleNamespace(
        target=None, handover=False, status="not recognized again"
    )

    app.collect_voice_result()

    assert app.voice_output.recommended_calls == 1
    assert app.voice_output.failed_calls == 0
    assert app.voice_recognition_attempt == 0
    app.voice_input.finish_task.assert_called_once_with()


def test_missing_voice_gift_plays_failure_and_clears_target() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState(pipeline_stage=PipelineStage.FLOWPOSE)
    app.voice_input = VoiceGraspInput()
    app.voice_input.activate("toy", handover=True)
    app.voice_output = _VoiceOutput()
    app.home_both_when_no_targets = Mock()

    app.advance_pipeline()

    assert app.voice_output.failed_calls == 1
    assert app.voice_input.task_active is False
    app.home_both_when_no_targets.assert_called_once_with()


def test_followup_prompts_are_queued_in_workflow_order(tmp_path: Path) -> None:
    directories = (
        "03_1_task_failed",
        "03_2_task_accepted",
        "05_1_reply_thanks",
        "06_1_goodbye",
    )
    for directory_name in directories:
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "01_prompt.mp3").touch()

    output = VoiceOutput(tmp_path, player="/bin/true", rng=_FirstChoice())
    try:
        output.play_task_failed()
        output.play_task_accepted()
        output.play_reply_thanks()
        output.play_goodbye()
        results = []
        deadline = time.monotonic() + 1.0
        while len(results) < len(directories) and time.monotonic() < deadline:
            result = output.poll()
            if result is None:
                time.sleep(0.01)
            else:
                results.append(result)
    finally:
        output.close()

    assert [result.files[0].parent.name for result in results] == list(directories)
    assert all(result.purpose == PROMPT_ONLY for result in results)


def test_no_handover_prompt_waits_until_playback_finishes(tmp_path: Path) -> None:
    directory = tmp_path / "05_2_no_handover"
    directory.mkdir()
    (directory / "01_no_handover.mp3").touch()
    output = VoiceOutput(tmp_path, player="/bin/true", rng=_FirstChoice())
    try:
        status = output.play_no_handover_and_wait()
        result = output.poll()
    finally:
        output.close()

    assert "05_2_no_handover" in status
    assert result is not None
    assert result.files == (directory / "01_no_handover.mp3",)
