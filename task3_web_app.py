#!/usr/bin/env python3
"""Local graphical workbench for the Task 3 image-quality agent."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from task3_agent import (
    DEFAULT_LIBRARY_ROOT,
    TOOLS,
    WorkflowError,
    auto_detect_colorchecker_rois,
    auto_detect_sfr_rectangle_edges,
    auto_detect_step_chart_rois,
    execute,
    load_image,
    plan_with_deepseek,
    save_annotated_rois,
    summarize_with_deepseek,
    validate_deepseek_key,
)


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "task3_web"
RUNS_DIR = BASE_DIR / "图形界面结果"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024


def _json_error(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


def _safe_run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id or ""):
        raise WorkflowError("无效的任务编号，请重新选择图片。")
    run_dir = (RUNS_DIR / run_id).resolve()
    if RUNS_DIR.resolve() not in run_dir.parents:
        raise WorkflowError("任务目录不合法。")
    return run_dir


def _save_upload() -> tuple[str, Path, Path]:
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        raise WorkflowError("请先选择一张测试图片。")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise WorkflowError("仅支持 JPG、PNG、BMP、TIF/TIFF 图片。")
    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    safe_name = secure_filename(upload.filename) or f"input{suffix}"
    if Path(safe_name).suffix.lower() != suffix:
        safe_name = f"input{suffix}"
    image_path = run_dir / safe_name
    upload.save(image_path)
    (run_dir / "session.json").write_text(
        json.dumps(
            {"image": image_path.name, "original_name": upload.filename},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_id, run_dir, image_path


def _load_session(run_id: str) -> tuple[Path, Path]:
    run_dir = _safe_run_dir(run_id)
    session_path = run_dir / "session.json"
    if not session_path.is_file():
        raise WorkflowError("找不到这次预览记录，请重新选择图片。")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    image_path = run_dir / session["image"]
    if not image_path.is_file():
        raise WorkflowError("预览图片已不存在，请重新选择图片。")
    return run_dir, image_path


def _preview_detection(
    intent: dict[str, Any],
    image_path: Path,
    annotated_path: Path,
) -> dict[str, Any]:
    image = load_image(image_path)
    tool = intent["tool"]
    coordinates: list[tuple[int, int, int, int]] = []
    metadata: dict[str, Any] = {"method": "whole_image"}

    if tool == "sfr":
        parameters = intent.get("parameters") if isinstance(intent.get("parameters"), dict) else {}
        field_position = str(parameters.get("field_position") or "center")
        coordinates, candidates = auto_detect_sfr_rectangle_edges(
            image,
            field_position=field_position,
        )
        target_polygons = []
        seen_polygons = set()
        for candidate in candidates:
            polygon = candidate["target_polygon"]
            key = tuple(tuple(point) for point in polygon)
            if key in seen_polygons:
                continue
            seen_polygons.add(key)
            target_polygons.append(polygon)
        metadata = {
            "method": "tilted_square_two_orthogonal_edges",
            "candidates": candidates,
            "target_polygon": candidates[0]["target_polygon"],
            "target_polygons": target_polygons,
            "field_position": field_position,
        }
        save_annotated_rois(
            image,
            coordinates,
            annotated_path,
            labels=[
                (
                    f"T{item.get('target_rank', 1)} "
                    + ("MTF-H" if item.get("mtf_direction") == "horizontal" else "MTF-V")
                )
                for item in candidates
            ],
            polygons=target_polygons,
        )
    elif tool == "color_check":
        coordinates, metadata = auto_detect_colorchecker_rois(image)
        save_annotated_rois(
            image,
            coordinates,
            annotated_path,
            labels=[str(index) for index in range(1, 25)],
        )
    elif tool == "step_chart":
        coordinates, metadata = auto_detect_step_chart_rois(image, 20)
        save_annotated_rois(
            image,
            coordinates,
            annotated_path,
            labels=[str(index) for index in range(1, 21)],
            center_markers=True,
            symmetry_axis_x=metadata.get("symmetry_axis_x"),
            symmetric_pairs=metadata.get("symmetric_pairs"),
        )
    elif tool == "white_balance":
        color_coordinates, color_metadata = auto_detect_colorchecker_rois(image)
        coordinates = color_coordinates[-6:]
        metadata = {
            **color_metadata,
            "method": "colorchecker_last_row_gray_patches",
        }
        save_annotated_rois(
            image,
            coordinates,
            annotated_path,
            labels=[f"Gray {index}" for index in range(1, 7)],
        )
    elif tool == "distortion":
        coordinates = [(0, 0, image.shape[1], image.shape[0])]
        save_annotated_rois(
            image,
            coordinates,
            annotated_path,
            labels=["Distortion"],
        )
    else:
        save_annotated_rois(image, [], annotated_path)

    return {
        "success": True,
        "intent": intent,
        "tool": tool,
        "image_shape": list(image.shape),
        "roi_count": len(coordinates),
        "roi_coordinates": [list(item) for item in coordinates],
        "automatic_detection": metadata,
    }


def _quality_messages(payload: dict[str, Any], algorithm_log: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    tool = payload["intent"]["tool"]
    if "WARNING" in algorithm_log:
        messages.append(
            {
                "level": "warning",
                "text": (
                    "SFR算法提示边缘对比度不足20%。这是输入质量警告，不代表计算失败；"
                    "ESF/MTF曲线仍已生成。请结合OpenCV框选图检查ROI，并与MATLAB使用"
                    "完全相同的ROI、Gamma和方向后再比较。"
                ),
            }
        )
    if tool == "sfr":
        detection = payload.get("automatic_detection") or {}
        candidates = detection.get("candidates") or []
        for index, item in enumerate(candidates, start=1):
            contrast = item.get("local_contrast")
            if contrast is not None and contrast < 0.25:
                messages.append(
                    {
                        "level": "warning",
                        "text": f"ROI {index} 的定位对比度仅为 {contrast:.1%}，建议手动画框复核。",
                    }
                )
    elif tool == "color_check":
        mean_delta_e = payload["result"].get("mean_delta_e")
        if mean_delta_e is not None and mean_delta_e > 10:
            messages.append(
                {
                    "level": "warning",
                    "text": "平均 ΔE 较大。这可能反映样片色差、光源/白平衡条件不匹配，不能仅凭该数值判断自动框选错误。",
                }
            )
    elif tool == "step_chart":
        transitions = payload["result"].get("detected_transitions", 0)
        if transitions >= 19:
            messages.append(
                {
                    "level": "success",
                    "text": "20 级灰阶的 19 个相邻过渡均被分辨。",
                }
            )
    if not messages:
        messages.append({"level": "success", "text": "计算完成，未发现明显流程警告。"})
    return messages


EDITABLE_ROI_TOOLS = {"sfr", "color_check", "step_chart"}


def _tool_description(tool: str) -> str:
    return TOOLS.get(tool, {}).get("description", tool)


def _failed_task_messages(task_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for item in task_results:
        if item.get("success"):
            continue
        tool = str(item.get("tool", ""))
        text = (
            f"{_tool_description(tool)} 自动定位或计算失败：{item.get('error', '未知错误')}。"
        )
        if tool in EDITABLE_ROI_TOOLS:
            text += "可切换到该任务的原图标签，点击人工修正 ROI 后重新画框计算。"
        messages.append({"level": "warning", "text": text})
    return messages


def _opencv_outputs_for_tasks(
    task_results: list[dict[str, Any]],
    run_id: str,
    image_path: Path,
) -> list[dict[str, Any]]:
    outputs = []
    for item in task_results:
        tool = str(item.get("tool", ""))
        if item.get("success"):
            outputs.append(
                {
                    "tool": tool,
                    "url": item["annotated_url"],
                    "title": f"{_tool_description(tool)} OpenCV框选",
                    "failed": False,
                }
            )
        else:
            outputs.append(
                {
                    "tool": tool,
                    "url": f"/runs/{run_id}/{image_path.name}",
                    "title": f"{_tool_description(tool)} 原图，等待人工ROI",
                    "failed": True,
                    "error": item.get("error", ""),
                }
            )
    return outputs


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"success": True, "status": "ready"})


@app.post("/api/validate-key")
def validate_key():
    try:
        data = request.get_json(silent=True) or {}
        api_key = str(data.get("api_key", "")).strip()
        validation = validate_deepseek_key(api_key)
        return jsonify({"success": True, **validation})
    except WorkflowError as exc:
        return _json_error(str(exc), 401)
    except Exception as exc:
        return _json_error(f"API Key验证失败：{type(exc).__name__}: {exc}", 500)


@app.get("/api/session/<run_id>")
def session(run_id: str):
    try:
        run_dir, image_path = _load_session(run_id)
        result_path = next(
            (
                path
                for path in (
                    run_dir / "评价结果.json",
                    run_dir / "Agent评价结果.json",
                )
                if path.is_file()
            ),
            run_dir / "评价结果.json",
        )
        preview_path = run_dir / "preview.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            annotated_url = payload.get("annotated_url")
            if not annotated_url and payload.get("opencv_outputs"):
                annotated_url = payload["opencv_outputs"][0]["url"]
            if not annotated_url:
                annotated_candidates = [
                    run_dir / "Agent_OpenCV输出.jpg",
                    run_dir / "手动ROI计算结果.jpg",
                    run_dir / "自动ROI计算结果.jpg",
                    run_dir / "自动框选预览.jpg",
                ]
                annotated_path = next(
                    (path for path in annotated_candidates if path.is_file()),
                    None,
                )
                if annotated_path is None:
                    raise WorkflowError("找不到这次计算的框选图片。")
                annotated_url = f"/runs/{run_id}/{annotated_path.name}"
            return jsonify(
                {
                    **payload,
                    "success": True,
                    "run_id": run_id,
                    "annotated_url": annotated_url,
                    "input_url": f"/runs/{run_id}/{image_path.name}",
                    "result_url": f"/runs/{run_id}/{result_path.name}",
                    "session_stage": "result",
                }
            )
        if preview_path.is_file():
            payload = json.loads(preview_path.read_text(encoding="utf-8"))
            return jsonify({**payload, "session_stage": "preview"})
        raise WorkflowError("这次任务还没有可恢复的预览或结果。")
    except WorkflowError as exc:
        return _json_error(str(exc), 404)


@app.post("/api/agent")
def run_agent():
    """Accept one image and one natural-language request, then run the full agent."""
    try:
        message = request.form.get("message", "").strip()
        if not message:
            raise WorkflowError("请用一句自然语言告诉 Agent 想评价什么。")
        api_key = request.form.get("api_key", "").strip()
        if not api_key:
            raise WorkflowError("请先输入并验证 DeepSeek API Key。")
        plan = plan_with_deepseek(message, api_key)
        run_id, run_dir, image_path = _save_upload()
        task_results = []
        combined_trace = [
            {
                "stage": "ai_planning",
                "title": "DeepSeek理解目标并生成Workflow",
                "status": "success",
                "detail": plan,
            }
        ]
        for task_index, task in enumerate(plan["tasks"], start=1):
            tool = task["tool"]
            annotated_path = run_dir / f"{task_index:02d}_{tool}_OpenCV输出.jpg"
            output_buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(output_buffer):
                    task_payload = execute(
                        task["request"],
                        image_path,
                        [],
                        DEFAULT_LIBRARY_ROOT,
                        annotated_path=annotated_path,
                        forced_tool=tool,
                        task_parameters=task.get("parameters"),
                        intent_method="deepseek",
                    )
                algorithm_log = output_buffer.getvalue()
                task_payload["algorithm_log"] = algorithm_log
                task_payload["quality_messages"] = _quality_messages(
                    task_payload,
                    algorithm_log,
                )
                task_payload["manual_roi_used"] = False
                task_payload["tool"] = tool
                task_payload["task_index"] = task_index
                task_payload["annotated_url"] = (
                    f"/runs/{run_id}/{annotated_path.name}"
                )
                for artifact in task_payload["result"].get("artifacts", []):
                    artifact["url"] = (
                        f"/runs/{run_id}/{Path(artifact['path']).name}"
                    )
                task_results.append(task_payload)
                combined_trace.extend(
                    [
                        {
                            **trace,
                            "task_index": task_index,
                            "task_tool": tool,
                        }
                        for trace in task_payload["agent_trace"]
                    ]
                )
            except Exception as exc:
                error_text = (
                    str(exc)
                    if isinstance(exc, WorkflowError)
                    else f"{type(exc).__name__}: {exc}"
                )
                task_results.append(
                    {
                        "success": False,
                        "tool": tool,
                        "task_index": task_index,
                        "request": task["request"],
                        "error": error_text,
                    }
                )
                combined_trace.append(
                    {
                        "stage": "task_failed",
                        "title": f"{tool}工具执行失败",
                        "status": "failed",
                        "task_index": task_index,
                        "task_tool": tool,
                        "detail": {"error": error_text},
                    }
                )
        successful = [item for item in task_results if item.get("success")]
        if not successful:
            image_shape = list(load_image(image_path).shape)
            opencv_outputs = _opencv_outputs_for_tasks(
                task_results,
                run_id,
                image_path,
            )
            agent_response = summarize_with_deepseek(
                message,
                plan,
                task_results,
                api_key,
            )
            combined_trace.append(
                {
                    "stage": "ai_synthesis",
                    "title": "DeepSeek整理失败原因与人工修正建议",
                    "status": "success",
                    "detail": {"response": agent_response},
                }
            )
            first_tool = plan["tasks"][0]["tool"]
            payload = {
                "success": True,
                "request": message,
                "image": str(image_path.resolve()),
                "image_shape": image_shape,
                "intent": {
                    "tool": first_tool,
                    "tools": [task["tool"] for task in plan["tasks"]],
                    "method": "deepseek",
                    "model": plan["model"],
                    "confidence": 0.9,
                },
                "agent_plan": plan,
                "task_results": task_results,
                "agent_trace": combined_trace,
                "agent_response": agent_response,
                "roi_count": 0,
                "roi_coordinates": [],
                "automatic_detection": None,
                "result": {},
                "quality_messages": [
                    {
                        "level": "warning",
                        "text": (
                            "自动定位没有得到可靠 ROI。页面已保留本次任务，"
                            "可点击人工修正 ROI 后从原图重新画框。"
                        ),
                    }
                ]
                + _failed_task_messages(task_results),
                "opencv_outputs": opencv_outputs,
                "chart_outputs": [],
                "algorithm_log": "\n\n".join(
                    item.get("error", "") for item in task_results
                ),
                "manual_roi_used": False,
            }
            result_path = run_dir / "Agent评价结果.json"
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return jsonify(
                {
                    **payload,
                    "run_id": run_id,
                    "annotated_url": (
                        opencv_outputs[0]["url"]
                        if opencv_outputs
                        else f"/runs/{run_id}/{image_path.name}"
                    ),
                    "input_url": f"/runs/{run_id}/{image_path.name}",
                    "result_url": f"/runs/{run_id}/{result_path.name}",
                    "session_stage": "result",
                }
            )
        agent_response = summarize_with_deepseek(
            message,
            plan,
            task_results,
            api_key,
        )
        first = successful[0]
        opencv_outputs = _opencv_outputs_for_tasks(
            task_results,
            run_id,
            image_path,
        )
        combined_trace.append(
            {
                "stage": "ai_synthesis",
                "title": "DeepSeek整理多工具测量结果",
                "status": "success",
                "detail": {"response": agent_response},
            }
        )
        payload = {
            "success": True,
            "request": message,
            "image": str(image_path.resolve()),
            "image_shape": first["image_shape"],
            "intent": {
                "tool": first["intent"]["tool"],
                "tools": [task["tool"] for task in plan["tasks"]],
                "method": "deepseek",
                "model": plan["model"],
                "confidence": 0.9,
            },
            "agent_plan": plan,
            "task_results": task_results,
            "agent_trace": combined_trace,
            "agent_response": agent_response,
            "roi_count": sum(
                int(item.get("roi_count", 0))
                for item in successful
            ),
            "roi_coordinates": first.get("roi_coordinates", []),
            "automatic_detection": first.get("automatic_detection"),
            "result": first["result"],
            "quality_messages": [
                message
                for item in successful
                for message in item.get("quality_messages", [])
            ]
            + _failed_task_messages(task_results),
            "opencv_outputs": opencv_outputs,
            "chart_outputs": [
                {
                    "tool": item["tool"],
                    **artifact,
                }
                for item in successful
                for artifact in item["result"].get("artifacts", [])
            ],
            "algorithm_log": "\n\n".join(
                item.get("algorithm_log") or item.get("error", "")
                for item in task_results
                if item.get("algorithm_log") or item.get("error")
            ),
            "manual_roi_used": False,
        }
        result_path = run_dir / "Agent评价结果.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return jsonify(
            {
                **payload,
                "run_id": run_id,
                "annotated_url": first["annotated_url"],
                "input_url": f"/runs/{run_id}/{image_path.name}",
                "result_url": f"/runs/{run_id}/{result_path.name}",
                "session_stage": "result",
            }
        )
    except WorkflowError as exc:
        return _json_error(str(exc))
    except Exception as exc:
        return _json_error(f"Agent 执行失败：{type(exc).__name__}: {exc}", 500)


@app.post("/api/preview")
def preview():
    try:
        message = request.form.get("message", "").strip()
        if not message:
            raise WorkflowError("请输入自然语言测试要求。")
        api_key = request.form.get("api_key", "").strip()
        if not api_key:
            raise WorkflowError("请先输入并验证 DeepSeek API Key。")
        plan = plan_with_deepseek(message, api_key)
        first_task = plan["tasks"][0]
        intent = {
            "tool": first_task["tool"],
            "method": "deepseek",
            "model": plan["model"],
            "confidence": 0.9,
            "parameters": first_task.get("parameters") or {},
        }
        run_id, run_dir, image_path = _save_upload()
        annotated_path = run_dir / "自动框选预览.jpg"
        payload = _preview_detection(intent, image_path, annotated_path)
        payload["agent_plan"] = plan
        payload.update(
            {
                "run_id": run_id,
                "annotated_url": f"/runs/{run_id}/{annotated_path.name}",
                "input_url": f"/runs/{run_id}/{image_path.name}",
            }
        )
        (run_dir / "preview.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return jsonify(payload)
    except WorkflowError as exc:
        return _json_error(str(exc))
    except Exception as exc:
        return _json_error(f"自动框选失败：{type(exc).__name__}: {exc}", 500)


@app.post("/api/run")
def run_evaluation():
    try:
        data = request.get_json(silent=True) or {}
        run_id = str(data.get("run_id", ""))
        message = str(data.get("message", "")).strip()
        if not message:
            raise WorkflowError("请输入自然语言测试要求。")
        run_dir, image_path = _load_session(run_id)
        selected_tool = str(data.get("tool", "")).strip()
        manual_rois = data.get("manual_rois")
        if manual_rois is None and data.get("manual_roi"):
            manual_rois = [data["manual_roi"]]
        roi_coordinates: list[tuple[int, int, int, int]] = []
        if not isinstance(manual_rois, list) or not manual_rois:
            raise WorkflowError("请至少保留或绘制一个人工 ROI。")
        for manual_roi in manual_rois:
            if not isinstance(manual_roi, list) or len(manual_roi) != 4:
                raise WorkflowError("手动画框坐标无效，请重新绘制。")
            values = tuple(int(round(float(value))) for value in manual_roi)
            if min(values[:2]) < 0 or min(values[2:]) <= 0:
                raise WorkflowError("手动画框坐标无效，请重新绘制。")
            roi_coordinates.append(values)
        api_key = str(data.get("api_key", "")).strip()
        if not api_key:
            raise WorkflowError("人工修正后重新计算仍需要已验证的 DeepSeek API Key。")
        existing_result_path = run_dir / "Agent评价结果.json"
        existing_payload = (
            json.loads(existing_result_path.read_text(encoding="utf-8"))
            if existing_result_path.is_file()
            else {}
        )
        existing_plan = existing_payload.get("agent_plan") or plan_with_deepseek(
            message,
            api_key,
        )
        selected_task = next(
            (
                task
                for task in existing_plan.get("tasks", [])
                if task.get("tool") == selected_tool
            ),
            None,
        )
        if selected_task is None:
            raise WorkflowError("当前 Agent 计划中没有这个可修正的评价任务。")
        expected_counts = {"color_check": 24, "step_chart": 20}
        if selected_tool not in {"sfr", *expected_counts}:
            raise WorkflowError("当前工具暂不支持在图形页面人工修改 ROI。")
        expected_count = expected_counts.get(selected_tool)
        if expected_count is not None and len(roi_coordinates) != expected_count:
            raise WorkflowError(
                f"{TOOLS[selected_tool]['description']} 必须提供 {expected_count} 个 ROI，"
                f"当前为 {len(roi_coordinates)} 个。"
            )
        annotated_path = run_dir / f"手动_{selected_tool}_ROI计算结果.jpg"
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            payload = execute(
                selected_task["request"],
                image_path,
                roi_coordinates,
                DEFAULT_LIBRARY_ROOT,
                annotated_path=annotated_path,
                forced_tool=selected_tool,
                task_parameters=selected_task.get("parameters"),
                intent_method="deepseek",
            )
        algorithm_log = output_buffer.getvalue()
        payload["algorithm_log"] = algorithm_log
        payload["quality_messages"] = _quality_messages(payload, algorithm_log)
        payload["manual_roi_used"] = bool(roi_coordinates)
        payload["agent_plan"] = existing_plan
        payload["agent_response"] = summarize_with_deepseek(
            message,
            existing_plan,
            [
                {
                    "success": True,
                    "tool": payload["intent"]["tool"],
                    "request": payload["request"],
                    "result": payload["result"],
                    "roi_count": payload["roi_count"],
                }
            ],
            api_key,
        )
        for artifact in payload["result"].get("artifacts", []):
            artifact["url"] = f"/runs/{run_id}/{Path(artifact['path']).name}"
        payload["chart_outputs"] = payload["result"].get("artifacts", [])
        result_path = run_dir / "评价结果.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        response_payload = {
            **payload,
            "run_id": run_id,
            "annotated_url": f"/runs/{run_id}/{annotated_path.name}",
            "input_url": f"/runs/{run_id}/{image_path.name}",
            "opencv_outputs": [
                {
                    "tool": selected_tool,
                    "url": f"/runs/{run_id}/{annotated_path.name}",
                    "title": f"{selected_tool} 人工 ROI",
                }
            ],
            "result_url": f"/runs/{run_id}/{result_path.name}",
        }
        return jsonify(response_payload)
    except WorkflowError as exc:
        return _json_error(str(exc))
    except Exception as exc:
        return _json_error(f"算法执行失败：{type(exc).__name__}: {exc}", 500)


@app.get("/runs/<run_id>/<path:filename>")
def run_file(run_id: str, filename: str):
    try:
        run_dir = _safe_run_dir(run_id)
        return send_from_directory(run_dir, filename, as_attachment=False)
    except WorkflowError as exc:
        return _json_error(str(exc), 404)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="任务三像质评价图形工作台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_browser:
        threading.Timer(
            1.1,
            lambda: webbrowser.open(f"http://{args.host}:{args.port}"),
        ).start()
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=False,
    )


if __name__ == "__main__":
    main()
