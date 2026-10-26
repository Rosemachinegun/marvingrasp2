from queue import SimpleQueue
from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import add.web as web_module
from add.settings import VOICE_ENABLED, WEB_ENABLED
from add.voice import (
    VoiceGraspInput,
    VoiceToText,
    parse_voice_target,
    requests_handover,
    voice_hotwords,
)
from add.web import (
    TabletCommand,
    TabletTaskLoopBridge,
    TabletWebService,
    format_base_link_targets,
)
from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp, RuntimeState
from grasp_core.core.types.robot_target_pose import TargetObjectPose


class _VoiceInput:
    def __init__(self) -> None:
        self.busy_values: list[bool] = []

    def request(self, *, robot_busy: bool) -> str:
        self.busy_values.append(robot_busy)
        return "voice requested"


class _TabletBridge:
    def __init__(self, *commands: TabletCommand) -> None:
        self.commands: SimpleQueue[TabletCommand] = SimpleQueue()
        for command in commands:
            self.commands.put(command)

    def drain_commands(self) -> list[TabletCommand]:
        commands = []
        while not self.commands.empty():
            commands.append(self.commands.get_nowait())
        return commands


def _app() -> GraspDemoApp:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = SimpleNamespace(paused=False, status="", base_targets=[])
    app.voice_input = _VoiceInput()
    app.robot_busy = lambda: False
    app.manual_home_future = None
    return app


def test_v_key_requests_voice() -> None:
    app = _app()

    assert app.handle_key(ord("v"), None)

    assert app.voice_input.busy_values == [False]
    assert app.state.status == "voice requested"


def test_tablet_voice_button_uses_same_voice_request() -> None:
    app = _app()
    app.tablet_bridge = _TabletBridge(TabletCommand.VOICE)

    app.handle_tablet_commands(None)

    assert app.voice_input.busy_values == [False]
    assert app.state.status == "voice requested"


def test_web_and_voice_are_enabled_by_default() -> None:
    assert WEB_ENABLED is True
    assert VOICE_ENABLED is True
    assert VoiceGraspInput().enabled is True
    assert VoiceGraspInput().language == "zh"


def test_voice_handover_phrase_is_preserved_as_active_intent() -> None:
    voice = VoiceGraspInput()
    assert requests_handover("把笔盒递给我") is True
    assert voice.submit_transcript("把笔盒递给我") is True

    parsed = voice.poll()

    assert parsed is not None
    assert parsed.target == "black box"
    assert parsed.handover is True
    voice.activate(parsed.target, handover=parsed.handover)
    assert voice.active_handover is True
    voice.finish_task()
    assert voice.active_handover is False


@pytest.mark.parametrize(
    "transcript",
    ("把玩具递给我", "我想要玩具", "给我推荐一个"),
)
def test_chinese_toy_handover_uses_only_white_toy_prompt(transcript: str) -> None:
    voice = VoiceGraspInput(language="zh")
    assert "玩具" in voice_hotwords("zh")
    assert "我想要" in voice_hotwords("zh")
    assert "给我推荐一个" in voice_hotwords("zh")
    assert requests_handover(transcript) is True
    assert voice.submit_transcript(transcript) is True

    parsed = voice.poll()

    assert parsed is not None
    assert parsed.target == "white toy"
    assert parsed.handover is True
    voice.activate(parsed.target, handover=parsed.handover)
    assert voice.sam_prompt("blue stuff,toy", {}) == "white toy"
    assert voice.active_handover is True


def test_explicit_white_toy_request_uses_white_toy_prompt() -> None:
    voice = VoiceGraspInput(language="en")
    assert voice.submit_transcript("give me the white toy") is True

    parsed = voice.poll()

    assert parsed is not None
    assert parsed.target == "white toy"
    voice.activate(parsed.target, handover=parsed.handover)
    assert voice.sam_prompt("toy,white toy", {}) == "white toy"


def _toy_target(label: str, *, camera_x: float, camera_z: float, base_y: float):
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [camera_x, 0.0, camera_z]
    base_pose = np.eye(4)
    base_pose[:3, 3] = [0.4, base_y, 0.7]
    return TargetObjectPose(label, label, camera_pose, base_pose)


def test_voice_toy_policy_keeps_only_rightmost_image_target() -> None:
    left = _toy_target("white toy_0", camera_x=-0.1, camera_z=0.8, base_y=-0.2)
    middle = _toy_target("white toy_1", camera_x=0.1, camera_z=0.5, base_y=-0.1)
    right = _toy_target("white toy_2", camera_x=0.3, camera_z=1.0, base_y=0.2)
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState(base_targets=[middle, right, left], status="FlowPose done")
    app.voice_input = VoiceGraspInput()
    app.voice_input.activate("white toy", handover=True)

    app.apply_voice_target_policy()

    # Image horizontal position follows x/z, so white toy_2 is rightmost.
    assert app.state.base_targets == [right]
    assert "selected rightmost white toy: white toy_2" in app.state.status


@pytest.mark.parametrize("transcript", ("把笔盒递给我", "把盒子递给我"))
def test_chinese_box_handover_uses_black_box_prompt(transcript: str) -> None:
    voice = VoiceGraspInput(language="zh")
    assert voice.submit_transcript(transcript) is True

    parsed = voice.poll()

    assert parsed is not None
    assert parsed.target == "black box"
    assert parsed.handover is True
    voice.activate(parsed.target, handover=parsed.handover)
    assert voice.sam_prompt("white toy,black box", {}) == "black box"


def test_voice_black_box_policy_keeps_only_rightmost_image_target() -> None:
    left = _toy_target("black box_0", camera_x=-0.1, camera_z=0.8, base_y=-0.2)
    right = _toy_target("black box_1", camera_x=0.3, camera_z=1.0, base_y=0.2)
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState(base_targets=[right, left], status="FlowPose done")
    app.voice_input = VoiceGraspInput()
    app.voice_input.activate("black box", handover=True)

    app.apply_voice_target_policy()

    assert app.state.base_targets == [right]
    assert "selected rightmost black box: black box_1" in app.state.status


def test_every_voice_grasp_forces_right_hand() -> None:
    # A positive base Y would normally select the left hand.
    target = _toy_target("black box_0", camera_x=0.2, camera_z=0.8, base_y=0.2)
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(ik_target_index=0, ik_hand="auto")
    app.state = RuntimeState(base_targets=[target])
    app.voice_input = VoiceGraspInput()
    app.voice_input.activate("black box", handover=True)
    app.robot_actions = Mock()
    app.action_executor = Mock()
    app.action_executor.submit.return_value = Future()
    app.grasp_future = None
    app.place_future = None
    app.ik_publisher = None
    app.auto_home_future = None
    app.auto_home_hand = None

    app.publish_grasp()

    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_grasp,
        [target],
        hand_override="right",
    )


@pytest.mark.parametrize(
    "phrase",
    (
        "give me the pencil case",
        "give me the box",
        "give me the black box",
    ),
)
def test_english_box_phrases_request_black_box_handover(phrase: str) -> None:
    voice = VoiceGraspInput(language="en")
    assert requests_handover(phrase) is True
    assert parse_voice_target(phrase) == "black box"
    assert voice.submit_transcript(phrase) is True

    parsed = voice.poll()

    assert parsed is not None
    assert parsed.target == "black box"
    assert parsed.handover is True


def test_voice_language_can_be_switched_without_reloading_model() -> None:
    recognizer = VoiceToText(language="English")
    model_marker = object()
    recognizer._model = model_marker

    assert recognizer.language == "en"
    assert recognizer.switch_language("Chinese") == "zh"
    assert recognizer.language == "zh"
    assert recognizer._model is model_marker

    voice = VoiceGraspInput(language="zh")
    assert voice.switch_language("en-US") == "en"
    assert voice.language == "en"


def test_recognition_exception_is_reported_as_no_speech() -> None:
    voice = VoiceGraspInput()
    voice._recognizer = Mock()
    voice._recognizer.listen_once.side_effect = RuntimeError("microphone failed")

    voice._recognize_once()
    parsed = voice.poll()

    assert parsed is not None
    assert parsed.target is None
    assert parsed.status == "Voice: no speech detected"


def test_voice_handover_uses_dedicated_action_and_clears_grasp_state() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState(
        base_targets=[SimpleNamespace(label="pen")],
        grasp_confirmed=True,
        grasp_confirmed_hand="left",
        grasp_confirmed_label="pen",
    )
    app.voice_input = VoiceGraspInput()
    app.voice_input.activate("pen", handover=True)
    app.voice_output = Mock()
    app.place_future = None
    app.robot_actions = Mock()
    app.action_executor = Mock()
    future = Future()
    app.action_executor.submit.return_value = future

    app.start_place()

    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_handover,
        grasp_confirmed=True,
        hand="left",
        object_label="pen",
        release_requested=app.handover_release_requested,
        ready_event=app.handover_ready,
        fallback_requested=app.handover_fallback_requested,
        timeout_prompt=app.voice_output.play_no_handover_and_wait,
    )
    future.set_result(SimpleNamespace(
        ok=True, status="left handover pose reached", grasp_hand="left",
    ))
    app.collect_place_result()

    app.voice_output.play_goodbye.assert_called_once_with()
    assert app.voice_input.task_active is False
    assert app.state.grasp_confirmed is False
    assert app.state.grasp_confirmed_hand is None
    assert app.state.grasp_confirmed_label is None


def test_toy_handover_routes_to_handover_skill_service() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState(
        base_targets=[SimpleNamespace(label="toy")],
        grasp_confirmed=True,
        grasp_confirmed_hand="right",
        grasp_confirmed_label="toy",
    )
    app.voice_input = VoiceGraspInput(language="zh")
    app.voice_input.activate("toy", handover=True)
    app.place_future = None
    app.robot_actions = Mock()
    app.action_executor = Mock()
    app.action_executor.submit.return_value = Future()

    app.start_place()

    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_handover,
        grasp_confirmed=True,
        hand="right",
        object_label="toy",
        release_requested=app.handover_release_requested,
        ready_event=app.handover_ready,
        fallback_requested=app.handover_fallback_requested,
    )


def test_b_confirms_release_only_after_handover_is_ready() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState()
    app.place_handover_task = True
    app.handover_ready = Event()
    app.handover_release_requested = Event()
    app.handover_fallback_requested = Event()
    app.manual_home_future = None
    app.voice_output = Mock()

    assert app.handle_key(ord("b"), None)
    assert app.handover_release_requested.is_set() is False
    assert app.state.status == "Handover not ready: B release ignored"

    app.handover_ready.set()
    assert app.handle_key(ord("b"), None)
    app.voice_output.play_reply_thanks.assert_called_once_with()
    assert app.handover_release_requested.is_set() is True
    assert app.state.status == "Handover release confirmed by B"

    assert app.handle_key(ord("b"), None)
    app.voice_output.play_reply_thanks.assert_called_once_with()
    assert app.state.status == "Handover release already confirmed by B"


def test_x_requests_immediate_handover_fallback_only_when_ready() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState()
    app.place_handover_task = True
    app.handover_ready = Event()
    app.handover_release_requested = Event()
    app.handover_fallback_requested = Event()
    app.manual_home_future = None

    assert app.handle_key(ord("x"), None)
    assert app.handover_fallback_requested.is_set() is False
    assert app.state.status == "Handover not ready: X fallback ignored"

    app.handover_ready.set()
    assert app.handle_key(ord("x"), None)
    assert app.handover_fallback_requested.is_set() is True
    assert app.handover_release_requested.is_set() is False
    assert app.state.status == "Handover fallback requested by X"

    assert app.handle_key(ord("b"), None)
    assert app.handover_release_requested.is_set() is False
    assert app.state.status == "Handover fallback requested by X: B ignored"

    assert app.handle_key(ord("x"), None)
    assert app.state.status == "Handover fallback already requested by X"


def test_disabled_web_service_does_not_build_interface(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled web service must not build Gradio UI")

    monkeypatch.setattr(web_module, "create_tablet_interface", fail_if_called)
    service = TabletWebService(
        TabletTaskLoopBridge(),
        host="127.0.0.1",
        port=7860,
        enabled=False,
    )

    service.start()

    assert service.demo is None


def test_base_link_targets_use_three_decimal_places() -> None:
    targets = [
        SimpleNamespace(
            label="toy_0",
            base_xyz=[0.573452943007408, -0.11985761159706018, 0.66],
        )
    ]

    assert format_base_link_targets(targets) == (
        "0: toy_0 xyz=[0.573, -0.120, 0.660]"
    )


@pytest.mark.parametrize("voice_task", [True, False])
@pytest.mark.parametrize("target_count", [1, 3])
def test_voice_place_returns_home_without_starting_another_grasp(
    voice_task, target_count,
) -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState(
        base_targets=[SimpleNamespace(label="pen")] * target_count,
        grasp_confirmed=True,
        grasp_confirmed_hand="right",
        grasp_confirmed_label="pen",
        continuous_grasp_pending=True,
    )
    app.voice_input = VoiceGraspInput()
    if voice_task:
        app.voice_input.activate("pen")
    app.place_future = None
    app.robot_actions = Mock()
    app.action_executor = Mock()
    future = Future()
    app.action_executor.submit.return_value = future
    app.restart_grasp_pipeline_after_place = Mock()

    app.start_place()

    # Place uses count <= 1 to release then HOME, and > 1 to park.
    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_place,
        grasp_confirmed=True,
        hand="right",
        object_label="pen",
        target_count=1 if voice_task else target_count,
    )
    app.collect_place_result()
    assert app.voice_input.task_active is voice_task
    app.restart_grasp_pipeline_after_place.assert_not_called()

    future.set_result(SimpleNamespace(
        ok=True, status="place and HOME complete", grasp_hand="right",
    ))
    app.collect_place_result()

    assert app.voice_input.task_active is False
    assert app.state.grasp_confirmed is False
    assert app.place_future is None
    if voice_task:
        app.restart_grasp_pipeline_after_place.assert_not_called()
        assert app.state.continuous_grasp_pending is False
        assert app.state.parked_after_place_hand is None
    else:
        app.restart_grasp_pipeline_after_place.assert_called_once_with()
        assert app.state.parked_after_place_hand == (
            "right" if target_count > 1 else None
        )
