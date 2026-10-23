from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from grasp_core.execution.robot_skill_service import RobotActionService
from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp, RuntimeState


def make_app() -> GraspDemoApp:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState()
    app.grasp_future = None
    app.place_future = None
    app.manual_home_future = None
    app.auto_home_future = None
    app.gripper_future = None
    app.wave_future = None
    app.gripper_interrupted = False
    app.robot_actions = Mock()
    app.ik_publisher = Mock()
    app.action_executor = Mock()
    app.action_executor.submit.return_value = Future()
    return app


def test_w_key_queues_wave_on_robot_action_executor() -> None:
    app = make_app()

    assert app.handle_key(ord("W"), None)

    app.ik_publisher.clear_stop.assert_called_once_with()
    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_wave
    )
    assert app.wave_future is app.action_executor.submit.return_value
    assert app.state.status == "Right-arm wave running"


def test_w_key_does_not_queue_a_second_wave_while_busy() -> None:
    app = make_app()
    app.publish_wave()

    app.publish_wave()

    app.action_executor.submit.assert_called_once()
    assert app.state.status == "Wave ignored: another task is running"


def test_gripper_command_is_blocked_during_wave() -> None:
    app = make_app()
    app.publish_wave()

    app.send_gripper("release", followup=False)

    assert app.state.status == "Gripper command ignored: wave action is running"


def test_s_requests_stop_for_running_wave() -> None:
    app = make_app()
    app.publish_wave()

    app.handle_key(ord("s"), None)

    assert app.state.paused
    app.ik_publisher.request_stop.assert_called_once_with()


def test_completed_wave_updates_status() -> None:
    app = make_app()
    future = Future()
    future.set_result("Right-arm wave completed")
    app.wave_future = future

    app.collect_wave_result()

    assert app.wave_future is None
    assert app.state.status == "Right-arm wave completed"


def test_interrupted_wave_away_from_home_cannot_resume_automatically() -> None:
    class RecoveryRequired(RuntimeError):
        manual_recovery_required = True

    app = make_app()
    app.state.paused = True
    future = Future()
    future.set_exception(RecoveryRequired("mux remains on replay"))
    app.wave_future = future

    app.handle_key(ord("s"), None)

    assert app.state.paused
    assert app.wave_recovery_required
    assert "manual recovery required" in app.state.status
    app.ik_publisher.clear_stop.assert_not_called()


def test_robot_action_service_forwards_existing_stop_callback(monkeypatch) -> None:
    publisher = Mock()
    service = RobotActionService.__new__(RobotActionService)
    service.ik_publisher = publisher
    execute = Mock(return_value="done")
    fake_module = SimpleNamespace(execute_wave=execute)
    monkeypatch.setitem(__import__("sys").modules, "add.wave", fake_module)

    assert service.publish_wave() == "done"
    execute.assert_called_once_with(stop_requested=publisher.stop_requested)
