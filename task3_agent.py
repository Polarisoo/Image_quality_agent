#!/usr/bin/env python3
"""Task 3: language-driven optical image-quality evaluation workflow."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_LOCAL_LIBRARY_ROOT = Path(__file__).resolve().parent
_LEGACY_LIBRARY_ROOT = Path(
    r"C:\Users\Lintingting\Desktop\语言驱动像质评价agent"
    r"\py_imaging_quality"
)
DEFAULT_LIBRARY_ROOT = (
    Path(os.environ["PY_IMAGING_QUALITY_ROOT"])
    if os.environ.get("PY_IMAGING_QUALITY_ROOT")
    else (
        _LOCAL_LIBRARY_ROOT
        if (_LOCAL_LIBRARY_ROOT / "py_imaging_quality").is_dir()
        else _LEGACY_LIBRARY_ROOT
    )
)

TOOLS = {
    "sfr": {
        "description": "SFR/MTF 分辨率和锐度测试",
        "roi_count": 1,
        "vision_tool": "opencv.detect_tilted_square_edges",
        "measurement_tool": "py_imaging_quality.sfr",
        "output": "OpenCV斜边框选图、MTF50、MTF30",
    },
    "distortion": {
        "description": "镜头畸变测量",
        "roi_count": 1,
        "vision_tool": "opencv.use_full_grid_image",
        "measurement_tool": "py_imaging_quality.distortion",
        "output": "网格ROI、畸变类型、SMIA TV畸变率",
    },
    "color_check": {
        "description": "ColorChecker 24 色块色准测试",
        "roi_count": 24,
        "vision_tool": "opencv.detect_colorchecker_6x4",
        "measurement_tool": "py_imaging_quality.color_check",
        "output": "OpenCV 24色块框选图、ΔE、ΔC、饱和度",
    },
    "step_chart": {
        "description": "20 级灰阶与阶调响应测试",
        "roi_count": 20,
        "vision_tool": "opencv.detect_step_chart",
        "measurement_tool": "py_imaging_quality.step_chart",
        "output": "OpenCV 20灰阶框选图、可分辨过渡数、对比度",
    },
    "white_balance": {
        "description": "6 个中性灰块白平衡测试",
        "roi_count": 6,
        "vision_tool": "opencv.detect_colorchecker_gray_row",
        "measurement_tool": "py_imaging_quality.white_balance",
        "output": "6个中性灰块ROI、RGB极差和偏色通道",
    },
    "dark_field": {
        "description": "暗场噪声与均匀性测试",
        "roi_count": 0,
        "vision_tool": "image.use_full_frame",
        "measurement_tool": "numpy.dark_field_statistics",
        "output": "各通道黑电平、标准差和极值",
    },
}


class WorkflowError(RuntimeError):
    """A user-facing workflow error."""


def _infer_mtf_field_position(text: str, parameters: dict[str, Any] | None = None) -> str:
    parameters = parameters or {}
    explicit = (
        parameters.get("field_position")
        or parameters.get("mtf_field")
        or parameters.get("field")
    )
    if explicit in {"center", "edge", "all"}:
        return str(explicit)
    lowered = str(text).lower()
    if any(
        keyword in lowered
        for keyword in (
            "edge",
            "corner",
            "边缘",
            "边角",
            "四角",
            "四周",
            "周边",
            "边部",
            "像场边缘",
            "视场边缘",
            "画面边缘",
            "画面四周",
        )
    ):
        return "edge"
    if any(keyword in lowered for keyword in ("center", "中心", "中央", "画面中心")):
        return "center"
    return "center"


def build_agent_plan(message: str, intent: dict[str, Any]) -> dict[str, Any]:
    """Create an explicit, inspectable tool-use plan for the selected task."""
    tool = intent["tool"]
    spec = TOOLS[tool]
    roi_count = int(spec["roi_count"])
    return {
        "goal": message,
        "selected_tool": tool,
        "tool_description": spec["description"],
        "steps": [
            {
                "stage": "understand_request",
                "action": f"识别用户要执行“{spec['description']}”",
            },
            {
                "stage": "locate_measurement_regions",
                "action": f"调用 {spec['vision_tool']} 自动定位"
                + ("整幅图" if roi_count == 0 else f"所需的 {roi_count} 类测量区域"),
            },
            {
                "stage": "run_optical_measurement",
                "action": f"调用 {spec['measurement_tool']} 计算光学指标",
            },
            {
                "stage": "synthesize_answer",
                "action": f"整理为自然语言结论，并输出 {spec['output']}",
            },
        ],
        "vision_tool": spec["vision_tool"],
        "measurement_tool": spec["measurement_tool"],
        "expected_output": spec["output"],
    }


def validate_deepseek_key(api_key: str) -> dict[str, Any]:
    """Validate a DeepSeek key before entering the graphical agent."""
    if not api_key:
        raise WorkflowError("请先输入 DeepSeek API Key。")
    import requests

    try:
        response = requests.get(
            "https://api.deepseek.com/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        models = [
            item.get("id")
            for item in response.json().get("data", [])
            if item.get("id")
        ]
    except Exception as exc:
        raise WorkflowError(f"DeepSeek API Key 验证失败：{exc}") from exc
    return {"valid": True, "models": models}


def plan_with_deepseek(text: str, api_key: str | None = None) -> dict[str, Any]:
    """Use DeepSeek exclusively to create a validated multi-tool workflow."""
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise WorkflowError("图形 Agent 必须先输入 DeepSeek API Key。")

    import requests

    tool_text = "\n".join(
        (
            f"- {name}: {spec['description']}；视觉工具={spec['vision_tool']}；"
            f"测量工具={spec['measurement_tool']}"
        )
        for name, spec in TOOLS.items()
    )
    payload = {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是成像质量评价Agent的规划器。根据用户整句话识别一个或多个目标。"
                    "只返回JSON，不要Markdown。格式："
                    '{"summary":"对用户目标的理解","tasks":['
                    '{"tool":"工具名","request":"该子任务的具体目标",'
                    '"reason":"选择该工具的原因","parameters":{'
                    '"mtf_directions":["horizontal","vertical"]}}],'
                    '"workflow":["步骤1","步骤2"]}。'
                    "tasks按执行顺序排列；同一工具不要重复；只允许使用下列工具。"
                    "只有SFR任务需要mtf_directions，可为horizontal、vertical或两者；"
                    "horizontal MTF 表示测水平方向空间频率，必须选择竖直刃边ROI；"
                    "vertical MTF 表示测竖直方向空间频率，必须选择水平刃边ROI；"
                    "SFR任务还可以返回field_position，取center或edge；用户说中心、中央"
                    "时取center，用户说边缘、四周、边角、周边、视场边缘或画面边缘时取edge；"
                    "用户没有限定方向时使用两者，用户没有限定视场位置时使用center。不要假定图片一定包含所需测试卡，"
                    "工具执行阶段会验证。\n可用工具：\n" + tool_text
                ),
            },
            {"role": "user", "content": text},
        ],
    }
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group() if match else content)
    except Exception as exc:
        raise WorkflowError(f"DeepSeek 任务规划失败：{exc}") from exc

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise WorkflowError("DeepSeek 没有返回可执行任务。")
    tasks = []
    seen = set()
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            continue
        tool = raw_task.get("tool")
        if tool not in TOOLS or tool in seen:
            continue
        parameters = raw_task.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        if tool == "sfr":
            mtf_directions = parameters.get("mtf_directions")
            if isinstance(mtf_directions, list):
                mtf_directions = [
                    value
                    for value in mtf_directions
                    if value in {"horizontal", "vertical"}
                ]
            else:
                mtf_directions = []
            if mtf_directions:
                edge_direction_for_mtf = {
                    "horizontal": "vertical",
                    "vertical": "horizontal",
                }
                directions = [
                    edge_direction_for_mtf[value] for value in mtf_directions
                ]
            else:
                directions = parameters.get("directions", ["horizontal", "vertical"])
                mtf_direction_for_edge = {
                    "horizontal": "vertical",
                    "vertical": "horizontal",
                }
                mtf_directions = [
                    mtf_direction_for_edge[value]
                    for value in directions
                    if value in {"horizontal", "vertical"}
                ]
            directions = [
                value
                for value in directions
                if value in {"horizontal", "vertical"}
            ]
            parameters["directions"] = directions or ["horizontal", "vertical"]
            parameters["mtf_directions"] = mtf_directions or ["horizontal", "vertical"]
            combined_request = f"{text}\n{raw_task.get('request') or ''}"
            parameters["field_position"] = _infer_mtf_field_position(
                combined_request,
                parameters,
            )
        tasks.append(
            {
                "tool": tool,
                "request": str(raw_task.get("request") or TOOLS[tool]["description"]),
                "reason": str(raw_task.get("reason") or "由大模型根据用户目标选择"),
                "parameters": parameters,
                "vision_tool": TOOLS[tool]["vision_tool"],
                "measurement_tool": TOOLS[tool]["measurement_tool"],
                "expected_output": TOOLS[tool]["output"],
            }
        )
        seen.add(tool)
    if not tasks:
        raise WorkflowError("DeepSeek 返回的任务不在本程序工具注册表中。")
    return {
        "method": "deepseek",
        "model": payload["model"],
        "summary": str(data.get("summary") or text),
        "tasks": tasks,
        "workflow": data.get("workflow") if isinstance(data.get("workflow"), list) else [],
    }


def summarize_with_deepseek(
    user_request: str,
    plan: dict[str, Any],
    task_results: list[dict[str, Any]],
    api_key: str | None = None,
) -> str:
    """Ask DeepSeek to explain measured values without inventing new numbers."""
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise WorkflowError("缺少 DeepSeek API Key，无法整理最终回答。")
    import requests

    compact_results = []
    for item in task_results:
        compact_results.append(
            {
                "tool": item.get("tool"),
                "success": item.get("success"),
                "request": item.get("request"),
                "result": item.get("result"),
                "error": item.get("error"),
                "roi_count": item.get("roi_count"),
            }
        )
    payload = {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是成像质量评价Agent。根据工具真实返回的数据，用中文给用户做"
                    "简洁、准确的总结。不得编造未提供的数值，不得把通用阈值当作"
                    "绝对质量判定；如果工具失败，要明确说明可能是图片中没有对应标准"
                    "测试卡。多个任务分条总结，并说明图表和OpenCV框选图可用于复核。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_request": user_request,
                        "plan_summary": plan.get("summary"),
                        "task_results": compact_results,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        raise WorkflowError(f"DeepSeek 结果总结失败：{exc}") from exc


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise WorkflowError(f"图片不存在：{image_path}")
    try:
        with Image.open(image_path) as image:
            if image.mode not in ("L", "RGB", "I;16", "I"):
                image = image.convert("RGB")
            return np.array(image)
    except Exception as exc:
        raise WorkflowError(f"图片读取失败：{exc}") from exc


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, width, height = (int(item.strip()) for item in value.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError("ROI 格式必须为 x,y,width,height") from exc
    if min(x, y) < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI 坐标须非负，宽高须大于 0")
    return x, y, width, height


def crop_rois(image: np.ndarray, coordinates: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    height, width = image.shape[:2]
    rois = []
    for x, y, roi_width, roi_height in coordinates:
        if x + roi_width > width or y + roi_height > height:
            raise WorkflowError(
                f"ROI ({x},{y},{roi_width},{roi_height}) 超出图片范围 {width}x{height}"
            )
        rois.append(image[y : y + roi_height, x : x + roi_width].copy())
    return rois


def _gray_u8(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    if gray.dtype == np.uint8:
        return gray
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def _angle_distance(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 90.0) % 180.0 - 90.0)


def _roi_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def auto_detect_sfr_rois(
    image: np.ndarray,
    max_rois: int = 1,
) -> tuple[list[tuple[int, int, int, int]], list[dict[str, Any]]]:
    """Locate clean slanted edges with OpenCV for ISO 12233 SFR analysis."""
    try:
        import cv2
    except ImportError as exc:
        raise WorkflowError("自动斜边检测需要 OpenCV，请在 Conda 环境安装 opencv。") from exc

    gray_full = _gray_u8(image)
    full_h, full_w = gray_full.shape
    scale = min(1.0, 1400.0 / max(full_h, full_w))
    if scale < 1.0:
        gray = cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        gray = gray_full

    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    # Keep thresholds permissive: ISO charts can contain useful gray-on-gray
    # slanted edges whose contrast is lower than nearby black text/barcodes.
    low = int(max(18, 0.22 * median))
    high = int(min(220, max(low + 35, 0.68 * median)))
    edges = cv2.Canny(blurred, low, high, L2gradient=True)

    min_dimension = min(height, width)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(30, int(min_dimension * 0.025)),
        minLineLength=max(45, int(min_dimension * 0.045)),
        maxLineGap=max(8, int(min_dimension * 0.012)),
    )
    if lines is None:
        raise WorkflowError("未检测到直线。请确认图片中存在清晰、高对比度的斜边。")

    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_angle = (np.degrees(np.arctan2(sobel_y, sobel_x)) + 180.0) % 180.0

    short_side = int(np.clip(min_dimension * 0.075, 64, 220))
    long_side = int(np.clip(min_dimension * 0.15, 110, 420))
    candidates: list[dict[str, Any]] = []

    for raw_line in lines[:, 0]:
        x1, y1, x2, y2 = (int(value) for value in raw_line)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 40:
            continue
        tangent = math.degrees(math.atan2(dy, dx)) % 180.0
        cardinal_deviation = min(
            abs(tangent),
            abs(tangent - 90.0),
            abs(tangent - 180.0),
        )
        if not 3.0 <= cardinal_deviation <= 15.0:
            continue

        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        near_vertical = 45.0 < tangent < 135.0
        roi_width, roi_height = (
            (short_side, long_side) if near_vertical else (long_side, short_side)
        )
        roi_width = min(roi_width, width - 2)
        roi_height = min(roi_height, height - 2)
        left = int(np.clip(center_x - roi_width / 2, 1, width - roi_width - 1))
        top = int(np.clip(center_y - roi_height / 2, 1, height - roi_height - 1))

        patch_mag = magnitude[top : top + roi_height, left : left + roi_width]
        patch_angle = gradient_angle[top : top + roi_height, left : left + roi_width]
        patch_edges = edges[top : top + roi_height, left : left + roi_width]
        if patch_mag.size == 0:
            continue

        strong_threshold = max(20.0, float(np.percentile(patch_mag, 88)))
        strong = patch_mag >= strong_threshold
        strong_count = int(np.count_nonzero(strong))
        if strong_count < 20:
            continue
        expected_gradient = (tangent + 90.0) % 180.0
        coherent = float(
            np.mean(_angle_distance(patch_angle[strong], expected_gradient) <= 12.0)
        )
        edge_density = float(np.mean(patch_edges > 0))

        normal_x = -dy / length
        normal_y = dx / length
        sample_offset = max(3, int(short_side * 0.16))
        yy, xx = np.mgrid[0:roi_height, 0:roi_width]
        signed = (
            (xx + left - center_x) * normal_x
            + (yy + top - center_y) * normal_y
        )
        local_gray = gray[top : top + roi_height, left : left + roi_width]
        side_a = local_gray[signed > sample_offset]
        side_b = local_gray[signed < -sample_offset]
        if min(side_a.size, side_b.size) < 50:
            continue
        contrast = abs(float(np.mean(side_a)) - float(np.mean(side_b))) / 255.0

        density_score = math.exp(-((edge_density - 0.045) / 0.055) ** 2)
        angle_score = math.exp(-((cardinal_deviation - 6.0) / 5.0) ** 2)
        length_score = min(1.0, length / (min_dimension * 0.22))
        center_distance = math.hypot(
            (center_x - width / 2.0) / (width / 2.0),
            (center_y - height / 2.0) / (height / 2.0),
        )
        center_score = math.exp(-((center_distance / 0.85) ** 2))
        score = (
            0.32 * contrast
            + 0.23 * coherent
            + 0.15 * density_score
            + 0.11 * angle_score
            + 0.09 * length_score
            + 0.10 * center_score
        )
        candidates.append(
            {
                "roi_scaled": (left, top, roi_width, roi_height),
                "score": score,
                "edge_angle_degrees": tangent,
                "cardinal_deviation_degrees": cardinal_deviation,
                "local_contrast": contrast,
                "orientation_coherence": coherent,
                "edge_density": edge_density,
                "center_score": center_score,
            }
        )

    if not candidates:
        raise WorkflowError(
            "检测到了直线，但没有满足 3°至15°倾角和局部对比度条件的斜边。"
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        roi = candidate["roi_scaled"]
        if any(_roi_iou(roi, chosen["roi_scaled"]) > 0.35 for chosen in selected):
            continue
        if candidate["local_contrast"] < 0.10 or candidate["orientation_coherence"] < 0.20:
            continue
        selected.append(candidate)
        if len(selected) >= max_rois:
            break
    if not selected:
        raise WorkflowError("候选斜边的对比度或方向一致性不足，无法可靠自动画框。")

    coordinates = []
    metadata = []
    inverse_scale = 1.0 / scale
    for candidate in selected:
        left, top, roi_width, roi_height = candidate.pop("roi_scaled")
        full_roi = (
            int(round(left * inverse_scale)),
            int(round(top * inverse_scale)),
            int(round(roi_width * inverse_scale)),
            int(round(roi_height * inverse_scale)),
        )
        full_roi = (
            min(full_roi[0], full_w - 2),
            min(full_roi[1], full_h - 2),
            min(full_roi[2], full_w - full_roi[0]),
            min(full_roi[3], full_h - full_roi[1]),
        )
        coordinates.append(full_roi)
        metadata.append(
            {
                **candidate,
                "roi": list(full_roi),
                "score": float(candidate["score"]),
            }
        )
    return coordinates, metadata


def _cluster_1d(values: list[float], count: int) -> np.ndarray:
    import cv2

    data = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if len(data) < count:
        raise WorkflowError(f"候选数量不足，无法形成 {count} 个位置簇。")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.05)
    _, labels, centers = cv2.kmeans(
        data,
        count,
        None,
        criteria,
        20,
        cv2.KMEANS_PP_CENTERS,
    )
    del labels
    return np.sort(centers[:, 0])


def _box_from_center(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    roi_width = max(8, min(int(round(width)), image_width))
    roi_height = max(8, min(int(round(height)), image_height))
    left = int(round(center_x - roi_width / 2))
    top = int(round(center_y - roi_height / 2))
    left = max(0, min(left, image_width - roi_width))
    top = max(0, min(top, image_height - roi_height))
    return left, top, roi_width, roi_height


def auto_detect_sfr_rectangle_edges(
    image: np.ndarray,
    field_position: str = "center",
    max_targets: int | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[dict[str, Any]]]:
    """Find two orthogonal single-edge ROIs on the same tilted square target."""
    try:
        import cv2
    except ImportError as exc:
        raise WorkflowError("MTF 四边形检测需要 OpenCV。") from exc

    gray_full = _gray_u8(image)
    full_height, full_width = gray_full.shape
    scale = min(1.0, 1400.0 / max(full_height, full_width))
    gray = (
        cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray_full
    )
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = height * width
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not image_area * 0.0015 <= area <= image_area * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon[:, 0, :].astype(np.float32)
        rect = cv2.minAreaRect(points)
        rect_width, rect_height = rect[1]
        if min(rect_width, rect_height) < max(18, min(height, width) * 0.018):
            continue
        aspect_ratio = max(rect_width, rect_height) / max(
            1.0, min(rect_width, rect_height)
        )
        if aspect_ratio > 1.42:
            continue
        rectangularity = area / max(1.0, rect_width * rect_height)
        if rectangularity < 0.78:
            continue

        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, points.astype(np.int32), 255)
        inner_mean = float(cv2.mean(gray, mask=mask)[0])
        expanded = cv2.dilate(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    max(5, int(min(rect_width, rect_height) * 0.35)) | 1,
                    max(5, int(min(rect_width, rect_height) * 0.35)) | 1,
                ),
            ),
        )
        ring = cv2.subtract(expanded, mask)
        outer_mean = float(cv2.mean(gray, mask=ring)[0])
        contrast = (outer_mean - inner_mean) / 255.0
        if contrast < 0.12:
            continue

        center_x, center_y = rect[0]
        center_distance = math.hypot(
            (center_x - width / 2) / (width / 2),
            (center_y - height / 2) / (height / 2),
        )
        center_score = math.exp(-((center_distance / 0.85) ** 2))
        area_score = min(1.0, area / (image_area * 0.018))

        ordered = cv2.boxPoints(rect)
        edge_records = []
        for index in range(4):
            first = ordered[index]
            second = ordered[(index + 1) % 4]
            delta_x, delta_y = second - first
            length = float(math.hypot(float(delta_x), float(delta_y)))
            angle = math.degrees(math.atan2(float(delta_y), float(delta_x))) % 180.0
            horizontal_deviation = min(abs(angle), abs(angle - 180.0))
            vertical_deviation = abs(angle - 90.0)
            if min(horizontal_deviation, vertical_deviation) > 15.0:
                continue
            if min(horizontal_deviation, vertical_deviation) < 3.0:
                continue
            orientation = (
                "horizontal"
                if horizontal_deviation <= vertical_deviation
                else "vertical"
            )
            edge_records.append(
                {
                    "first": first,
                    "second": second,
                    "length": length,
                    "angle": angle,
                    "deviation": min(horizontal_deviation, vertical_deviation),
                    "orientation": orientation,
                }
            )
        if not edge_records:
            continue
        candidates.append(
            {
                "rect": rect,
                "edges": edge_records,
                "contrast": contrast,
                "rectangularity": rectangularity,
                "aspect_ratio": aspect_ratio,
                "center_score": center_score,
                "center_distance": center_distance,
                "area_score": area_score,
            }
        )

    target_options = []
    for candidate_index, candidate in enumerate(candidates):
        best_edges = {}
        for orientation in ("horizontal", "vertical"):
            orientation_edges = [
                edge
                for edge in candidate["edges"]
                if edge["orientation"] == orientation
            ]
            if not orientation_edges:
                break
            best_edges[orientation] = max(
                orientation_edges,
                key=lambda edge: (
                    math.exp(-((edge["deviation"] - 6.0) / 4.5) ** 2),
                    edge["length"],
                ),
            )
        if len(best_edges) != 2:
            continue
        mean_length_score = float(
            np.mean(
                [
                    min(
                        1.0,
                        edge["length"] / (min(height, width) * 0.18),
                    )
                    for edge in best_edges.values()
                ]
            )
        )
        mean_angle_score = float(
            np.mean(
                [
                    math.exp(-((edge["deviation"] - 6.0) / 4.5) ** 2)
                    for edge in best_edges.values()
                ]
            )
        )
        square_score = math.exp(-((candidate["aspect_ratio"] - 1.0) / 0.24) ** 2)
        score = (
            0.27 * candidate["contrast"]
            + 0.18 * candidate["rectangularity"]
            + 0.14 * candidate["center_score"]
            + 0.12 * candidate["area_score"]
            + 0.11 * mean_length_score
            + 0.10 * mean_angle_score
            + 0.08 * square_score
        )
        target_options.append(
            {
                "score": float(score),
                "candidate_index": candidate_index,
                "best_edges": best_edges,
                "center_distance": float(candidate["center_distance"]),
            }
        )

    if not target_options:
        raise WorkflowError(
            "未能在同一个斜正方形上同时找到近水平边和近竖直边。"
        )

    field_position = field_position if field_position in {"center", "edge", "all"} else "center"
    if max_targets is None:
        max_targets = 4 if field_position == "edge" else 1
    if field_position == "center":
        ordered_options = sorted(
            target_options,
            key=lambda item: (item["score"], -item["center_distance"]),
            reverse=True,
        )
    elif field_position == "edge":
        edge_options = [
            item for item in target_options if item["center_distance"] >= 0.36
        ] or target_options
        ordered_options = sorted(
            edge_options,
            key=lambda item: (
                min(1.0, item["center_distance"] / 0.78),
                item["score"],
            ),
            reverse=True,
        )
    else:
        ordered_options = sorted(
            target_options,
            key=lambda item: item["score"],
            reverse=True,
        )

    selected_target_options: list[dict[str, Any]] = []
    min_target_separation = min(height, width) * (0.16 if field_position == "edge" else 0.08)
    for option in ordered_options:
        center = np.asarray(candidates[option["candidate_index"]]["rect"][0], dtype=float)
        if any(
            np.linalg.norm(
                center
                - np.asarray(candidates[chosen["candidate_index"]]["rect"][0], dtype=float)
            )
            < min_target_separation
            for chosen in selected_target_options
        ):
            continue
        selected_target_options.append(option)
        if len(selected_target_options) >= max(1, max_targets):
            break
    if not selected_target_options:
        selected_target_options = [max(target_options, key=lambda item: item["score"])]

    selected: list[dict[str, Any]] = []
    inverse_scale = 1.0 / scale
    for target_rank, target_option in enumerate(selected_target_options, start=1):
        target_score = target_option["score"]
        target_index = target_option["candidate_index"]
        selected_edges = target_option["best_edges"]
        target = candidates[target_index]
        target_polygon_scaled = cv2.boxPoints(target["rect"])
        target_polygon = [
            [
                int(round(float(point[0]) * inverse_scale)),
                int(round(float(point[1]) * inverse_scale)),
            ]
            for point in target_polygon_scaled
        ]
        for orientation in ("horizontal", "vertical"):
            edge = selected_edges[orientation]
            short_side = min(target["rect"][1])
            midpoint = (edge["first"] + edge["second"]) / 2.0
            long_extent = max(52.0, edge["length"] * 0.62)
            short_extent = min(
                max(44.0, edge["length"] * 0.20),
                max(44.0, short_side * 0.78),
            )
            roi_width, roi_height = (
                (long_extent, short_extent)
                if orientation == "horizontal"
                else (short_extent, long_extent)
            )
            selected.append(
                {
                    "orientation": orientation,
                    "score": float(target_score),
                    "edge_angle_degrees": float(edge["angle"]),
                    "cardinal_deviation_degrees": float(edge["deviation"]),
                    "local_contrast": float(target["contrast"]),
                    "center_scaled": midpoint,
                    "roi_size_scaled": (roi_width, roi_height),
                    "target_polygon": target_polygon,
                    "target_rank": target_rank,
                    "target_field_position": field_position,
                    "target_center_distance": float(target_option["center_distance"]),
                    "target_aspect_ratio": float(target["aspect_ratio"]),
                }
            )

    coordinates = []
    metadata = []
    for item in selected:
        center_x, center_y = item.pop("center_scaled")
        roi_width, roi_height = item.pop("roi_size_scaled")
        roi = _box_from_center(
            center_x * inverse_scale,
            center_y * inverse_scale,
            roi_width * inverse_scale,
            roi_height * inverse_scale,
            full_width,
            full_height,
        )
        coordinates.append(roi)
        metadata.append(
            {
                **item,
                "mtf_direction": (
                    "horizontal"
                    if item["orientation"] == "vertical"
                    else "vertical"
                ),
                "roi": list(roi),
            }
        )
    return coordinates, metadata


def _detect_colorchecker_rois_single_region(
    image: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    """Detect a classic 6 x 4 ColorChecker grid and return 24 central ROIs."""
    try:
        import cv2
    except ImportError as exc:
        raise WorkflowError("色卡自动定位需要 OpenCV。") from exc

    if image.ndim != 3 or image.shape[2] < 3:
        raise WorkflowError("ColorChecker 检测需要 RGB 彩色图片。")
    full_height, full_width = image.shape[:2]
    scale = min(1.0, 1400.0 / max(full_height, full_width))
    working = (
        cv2.resize(image[:, :, :3], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image[:, :, :3]
    )
    height, width = working.shape[:2]
    channels = [
        cv2.cvtColor(working, cv2.COLOR_RGB2GRAY),
        working[:, :, 0],
        working[:, :, 1],
        working[:, :, 2],
    ]
    image_area = height * width
    raw_boxes = []
    for channel in channels:
        for threshold_value in range(25, 231, 25):
            for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
                binary = cv2.threshold(channel, threshold_value, 255, polarity)[1]
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    area = float(cv2.contourArea(contour))
                    if not image_area * 0.006 <= area <= image_area * 0.045:
                        continue
                    x, y, box_width, box_height = cv2.boundingRect(contour)
                    aspect = box_width / max(1, box_height)
                    rectangularity = area / max(1.0, box_width * box_height)
                    if not 0.55 <= aspect <= 1.65 or rectangularity < 0.84:
                        continue
                    raw_boxes.append(
                        {
                            "box": (x, y, box_width, box_height),
                            "rectangularity": rectangularity,
                            "area": area,
                        }
                    )
    if len(raw_boxes) < 12:
        raise WorkflowError("检测到的色块候选不足，无法建立 6×4 ColorChecker 网格。")

    raw_boxes.sort(
        key=lambda item: (item["rectangularity"], item["area"]),
        reverse=True,
    )
    deduped = []
    duplicate_distance = min(height, width) * 0.025
    for candidate in raw_boxes:
        x, y, box_width, box_height = candidate["box"]
        center = (x + box_width / 2.0, y + box_height / 2.0)
        if any(
            math.hypot(center[0] - item["center"][0], center[1] - item["center"][1])
            < duplicate_distance
            for item in deduped
        ):
            continue
        deduped.append({**candidate, "center": center})

    widths = np.array([item["box"][2] for item in deduped], dtype=float)
    heights = np.array([item["box"][3] for item in deduped], dtype=float)
    median_width, median_height = float(np.median(widths)), float(np.median(heights))
    filtered = [
        item
        for item in deduped
        if 0.68 <= item["box"][2] / median_width <= 1.42
        and 0.68 <= item["box"][3] / median_height <= 1.42
    ]
    if len(filtered) < 16:
        raise WorkflowError(
            f"只有 {len(filtered)} 个尺寸一致的色块候选，至少需要 16 个。"
        )

    column_centers = _cluster_1d([item["center"][0] for item in filtered], 6)
    row_centers = _cluster_1d([item["center"][1] for item in filtered], 4)
    column_spacing = np.diff(column_centers)
    row_spacing = np.diff(row_centers)
    if (
        np.std(column_spacing) / max(1.0, np.mean(column_spacing)) > 0.30
        or np.std(row_spacing) / max(1.0, np.mean(row_spacing)) > 0.30
    ):
        raise WorkflowError("色块候选无法形成间距规则的 6×4 网格。")

    occupied: dict[tuple[int, int], dict[str, Any]] = {}
    for item in filtered:
        column = int(np.argmin(np.abs(column_centers - item["center"][0])))
        row = int(np.argmin(np.abs(row_centers - item["center"][1])))
        key = (row, column)
        if key not in occupied or item["rectangularity"] > occupied[key]["rectangularity"]:
            occupied[key] = item
    if len(occupied) < 16:
        raise WorkflowError(f"色卡网格只有 {len(occupied)}/24 个格位得到支持。")

    design = []
    observed_x = []
    observed_y = []
    for (row, column), item in occupied.items():
        design.append([1.0, column, row, column * row])
        observed_x.append(item["center"][0])
        observed_y.append(item["center"][1])
    matrix = np.asarray(design)
    coefficient_x = np.linalg.lstsq(matrix, np.asarray(observed_x), rcond=None)[0]
    coefficient_y = np.linalg.lstsq(matrix, np.asarray(observed_y), rcond=None)[0]

    inverse_scale = 1.0 / scale
    roi_width = median_width * 0.56 * inverse_scale
    roi_height = median_height * 0.56 * inverse_scale
    coordinates = []
    inferred_cells = []
    for row in range(4):
        for column in range(6):
            basis = np.array([1.0, column, row, column * row])
            center_x = float(basis @ coefficient_x) * inverse_scale
            center_y = float(basis @ coefficient_y) * inverse_scale
            coordinates.append(
                _box_from_center(
                    center_x,
                    center_y,
                    roi_width,
                    roi_height,
                    full_width,
                    full_height,
                )
            )
            if (row, column) not in occupied:
                inferred_cells.append(row * 6 + column + 1)

    roi_arrays = crop_rois(image, coordinates)
    within_patch_std = [
        float(np.mean(np.std(roi.astype(np.float32), axis=(0, 1))))
        for roi in roi_arrays
    ]
    if float(np.median(within_patch_std)) > 28.0:
        raise WorkflowError("自动色块 ROI 内部纹理过强，疑似没有对准 ColorChecker。")
    return coordinates, {
        "method": "multithreshold_grid_6x4",
        "directly_supported_cells": len(occupied),
        "inferred_patch_numbers": inferred_cells,
        "median_patch_standard_deviation": float(np.median(within_patch_std)),
        "order": "row-major, left-to-right, top-to-bottom",
    }


def _expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    factor: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = box
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    new_width = min(image_width, max(width + 8, width * factor))
    new_height = min(image_height, max(height + 8, height * factor))
    left = int(round(max(0.0, min(image_width - new_width, center_x - new_width / 2.0))))
    top = int(round(max(0.0, min(image_height - new_height, center_y - new_height / 2.0))))
    return left, top, int(round(new_width)), int(round(new_height))


def _offset_rois(
    rois: list[tuple[int, int, int, int]],
    offset_x: int,
    offset_y: int,
) -> list[tuple[int, int, int, int]]:
    return [
        (x + offset_x, y + offset_y, width, height)
        for x, y, width, height in rois
    ]


def auto_detect_colorchecker_rois(
    image: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    """Detect a ColorChecker even when it occupies a local image region."""
    try:
        return _detect_colorchecker_rois_single_region(image)
    except WorkflowError as first_error:
        if image.ndim != 3 or image.shape[2] < 3:
            raise first_error
        import cv2

        rgb = image[:, :, :3].astype(np.float32)
        full_height, full_width = image.shape[:2]
        chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        gray = _gray_u8(image)
        color_mask = ((chroma > 32.0) & (gray > 20) & (gray < 245)).astype(np.uint8)
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        )
        contours, _ = cv2.findContours(
            color_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        boxes: list[tuple[int, int, int, int]] = []
        min_area = full_width * full_height * 0.0008
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width * height < min_area:
                continue
            boxes.append(_expand_box((x, y, width, height), full_width, full_height, 2.15))
        if np.count_nonzero(color_mask):
            ys, xs = np.where(color_mask > 0)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            boxes.append(
                _expand_box(
                    (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                    full_width,
                    full_height,
                    2.4,
                )
            )
        # A small regular grid as a last local-search fallback. This is still
        # deterministic CV, but each proposal is validated by the 6x4 grid fit.
        for fraction in (0.32, 0.42, 0.55, 0.70):
            window_width = int(full_width * fraction)
            window_height = int(window_width * 0.75)
            if window_width < 260 or window_height < 180:
                continue
            window_height = min(window_height, full_height)
            x_values = np.linspace(0, max(0, full_width - window_width), 4)
            y_values = np.linspace(0, max(0, full_height - window_height), 4)
            for left in x_values:
                for top in y_values:
                    boxes.append((int(round(left)), int(round(top)), window_width, window_height))

        unique_boxes = []
        seen = set()
        for box in boxes:
            key = tuple(int(round(value / 12)) for value in box)
            if key in seen:
                continue
            seen.add(key)
            unique_boxes.append(box)

        attempts = []
        for left, top, width, height in sorted(
            unique_boxes,
            key=lambda item: item[2] * item[3],
        ):
            crop = image[top : top + height, left : left + width]
            try:
                rois, metadata = _detect_colorchecker_rois_single_region(crop)
            except WorkflowError:
                continue
            attempts.append(
                (
                    (
                        metadata.get("directly_supported_cells", 0),
                        -len(metadata.get("inferred_patch_numbers", [])),
                        -(width * height),
                    ),
                    _offset_rois(rois, left, top),
                    {
                        **metadata,
                        "method": "local_region_colorchecker_6x4",
                        "search_window": [left, top, width, height],
                        "initial_failure": str(first_error),
                    },
                )
            )
        if not attempts:
            raise first_error
        return max(attempts, key=lambda item: item[0])[1:]


def _detect_linear_step_strip(
    gray: np.ndarray,
    count: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]] | None:
    import cv2
    from scipy.signal import find_peaks

    height, width = gray.shape
    gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    best = None
    for band_fraction in (0.07, 0.10, 0.14, 0.18):
        band_height = max(30, int(height * band_fraction))
        step_y = max(8, band_height // 4)
        for top in range(int(height * 0.06), int(height * 0.94) - band_height, step_y):
            band = gradient_x[top : top + band_height]
            profile = np.mean(band, axis=0)
            profile = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 2.0).ravel()
            peaks, properties = find_peaks(
                profile,
                distance=max(8, int(width / (count * 2.2))),
                prominence=max(2.0, float(np.std(profile) * 0.75)),
            )
            if len(peaks) < count - 1:
                continue
            prominences = properties["prominences"]
            for sequence_length in (count - 1, count, count + 1):
                if len(peaks) < sequence_length:
                    continue
                for start_index in range(len(peaks) - sequence_length + 1):
                    sequence = peaks[start_index : start_index + sequence_length]
                    spacing_values = np.diff(sequence)
                    regularity = float(
                        np.std(spacing_values) / max(1.0, np.mean(spacing_values))
                    )
                    if regularity > 0.24:
                        continue
                    median_spacing = float(np.median(spacing_values))
                    if sequence_length == count - 1:
                        left = float(sequence[0] - median_spacing)
                        right = float(sequence[-1] + median_spacing)
                    elif sequence_length == count:
                        left = float(sequence[0] - median_spacing / 2.0)
                        right = float(sequence[-1] + median_spacing / 2.0)
                    else:
                        left = float(sequence[0])
                        right = float(sequence[-1])
                    if right - left < width * 0.32:
                        continue
                    centers = np.linspace(
                        left + (right - left) / (2 * count),
                        right - (right - left) / (2 * count),
                        count,
                    )
                    sequence_prominences = [
                        prominences[int(np.where(peaks == peak)[0][0])]
                        for peak in sequence
                    ]
                    sample_width = max(8, int((right - left) / count * 0.45))
                    sample_height = max(12, int(band_height * 0.42))
                    for vertical_factor in np.linspace(-0.2, 1.8, 11):
                        center_y = top + band_height * float(vertical_factor)
                        coordinates = [
                            _box_from_center(
                                float(center_x),
                                center_y,
                                sample_width,
                                sample_height,
                                width,
                                height,
                            )
                            for center_x in centers
                        ]
                        means = []
                        deviations = []
                        for x, y, roi_width, roi_height in coordinates:
                            patch = gray[y : y + roi_height, x : x + roi_width]
                            means.append(float(np.mean(patch)))
                            deviations.append(float(np.std(patch)))
                        dynamic_range = max(means) - min(means)
                        distinct_transitions = int(
                            np.sum(np.diff(np.sort(np.asarray(means))) > 1.5)
                        )
                        correlation = abs(
                            float(
                                np.corrcoef(np.arange(count), np.asarray(means))[0, 1]
                            )
                        )
                        median_deviation = float(np.median(deviations))
                        if (
                            dynamic_range < 120
                            or distinct_transitions < 11
                            or correlation < 0.68
                            or median_deviation > 18
                        ):
                            continue
                        score = (
                            float(np.mean(sequence_prominences))
                            * (1.0 - regularity)
                            * ((right - left) / width)
                            * (dynamic_range / 255.0)
                            * correlation
                            / (1.0 + median_deviation / 12.0)
                        )
                        if best is None or score > best["score"]:
                            best = {
                                "coordinates": coordinates,
                                "top": top,
                                "height": band_height,
                                "score": score,
                                "regularity": regularity,
                                "means": means,
                                "dynamic_range": dynamic_range,
                                "correlation": correlation,
                                "median_deviation": median_deviation,
                            }
    if best is None:
        return None
    means = best["means"]
    coordinates = best["coordinates"]
    order = np.argsort(means)
    coordinates = [coordinates[int(index)] for index in order]
    sorted_means = [means[int(index)] for index in order]
    return coordinates, {
        "method": "linear_step_strip",
        "strip_top": int(best["top"]),
        "strip_height": int(best["height"]),
        "boundary_regularity": float(best["regularity"]),
        "position_brightness_correlation": float(best["correlation"]),
        "gray_dynamic_range": float(best["dynamic_range"]),
        "median_patch_standard_deviation": float(best["median_deviation"]),
        "gray_means_dark_to_bright": sorted_means,
    }


def _detect_uniform_gray_windows(
    image: np.ndarray,
    count: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    import cv2

    gray = _gray_u8(image)
    height, width = gray.shape
    window = int(np.clip(min(height, width) * 0.045, 28, 110))
    stride = max(12, window // 2)
    candidates = []
    if image.ndim == 3:
        rgb = image[:, :, :3].astype(np.float32)
    else:
        rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    for top in range(int(height * 0.05), int(height * 0.95) - window, stride):
        for left in range(int(width * 0.05), int(width * 0.95) - window, stride):
            patch_gray = gray[top : top + window, left : left + window]
            patch_rgb = rgb[top : top + window, left : left + window]
            standard_deviation = float(np.std(patch_gray))
            channel_means = np.mean(patch_rgb, axis=(0, 1))
            chroma_range = float(np.max(channel_means) - np.min(channel_means))
            if standard_deviation > 9.0 or chroma_range > 36.0:
                continue
            candidates.append(
                {
                    "roi": (left, top, window, window),
                    "mean": float(np.mean(patch_gray)),
                    "std": standard_deviation,
                }
            )
    if len(candidates) < count:
        raise WorkflowError("找不到足够的低纹理中性灰区域。")

    values = np.asarray([item["mean"] for item in candidates], dtype=np.float32)
    centers = _cluster_1d(values.tolist(), count)
    selected = []
    for center in centers:
        options = sorted(
            candidates,
            key=lambda item: (abs(item["mean"] - center), item["std"]),
        )
        chosen = next(
            (
                option
                for option in options
                if all(_roi_iou(option["roi"], item["roi"]) < 0.15 for item in selected)
            ),
            None,
        )
        if chosen is None:
            raise WorkflowError("灰阶候选发生空间重叠，无法形成 20 个独立 ROI。")
        selected.append(chosen)
    selected.sort(key=lambda item: item["mean"])
    dynamic_range = selected[-1]["mean"] - selected[0]["mean"]
    if dynamic_range < 100:
        raise WorkflowError("自动灰阶 ROI 的亮度覆盖不足，可能没有定位到灰阶卡。")
    return [item["roi"] for item in selected], {
        "method": "uniform_gray_window_clustering",
        "gray_means_dark_to_bright": [item["mean"] for item in selected],
        "median_patch_standard_deviation": float(
            np.median([item["std"] for item in selected])
        ),
    }


def _detect_horseshoe_step_chart(
    image: np.ndarray,
    count: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]] | None:
    """Detect central neutral patches, fit their ring, and sample at centroids."""
    if count != 20:
        return None

    import cv2

    gray_full = _gray_u8(image)
    full_height, full_width = gray_full.shape
    scale = min(1.0, 1600.0 / max(full_height, full_width))
    gray = (
        cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray_full
    )
    if image.ndim == 3:
        rgb_full = image[:, :, :3]
    else:
        rgb_full = np.repeat(gray_full[:, :, None], 3, axis=2)
    rgb = (
        cv2.resize(rgb_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else rgb_full
    ).astype(np.float32)
    height, width = gray.shape
    image_area = height * width
    kernel_size = max(9, int(min(height, width) * 0.014)) | 1
    gray_float = gray.astype(np.float32)
    local_mean = cv2.boxFilter(
        gray_float, cv2.CV_32F, (kernel_size, kernel_size), normalize=True
    )
    local_square_mean = cv2.boxFilter(
        gray_float * gray_float,
        cv2.CV_32F,
        (kernel_size, kernel_size),
        normalize=True,
    )
    local_std = np.sqrt(
        np.maximum(0.0, local_square_mean - local_mean * local_mean)
    )
    channel_means = [
        cv2.boxFilter(
            rgb[:, :, channel],
            cv2.CV_32F,
            (kernel_size, kernel_size),
            normalize=True,
        )
        for channel in range(3)
    ]
    local_chroma = np.max(channel_means, axis=0) - np.min(channel_means, axis=0)

    yy, xx = np.mgrid[:height, :width]
    normalized_x = xx / width
    normalized_y = yy / height
    central_region = (
        (0.28 <= normalized_x)
        & (normalized_x <= 0.68)
        & (0.24 <= normalized_y)
        & (normalized_y <= 0.73)
    )
    inner_exclusion = (
        (0.40 <= normalized_x)
        & (normalized_x <= 0.56)
        & (0.34 <= normalized_y)
        & (normalized_y <= 0.62)
    )
    ring_search_region = central_region & ~inner_exclusion
    flat_mask = (
        (local_std < 3.2)
        & (local_chroma < 25.0)
        & ring_search_region
    ).astype(np.uint8)
    flat_mask = cv2.morphologyEx(
        flat_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        flat_mask, 8
    )
    components = []
    for label in range(1, component_count):
        x, y, box_width, box_height, area = stats[label]
        if (
            area < image_area * 0.00007
            or box_width < max(12, int(min(height, width) * 0.010))
            or box_height < max(12, int(min(height, width) * 0.010))
        ):
            continue
        component_pixels = gray[labels == label]
        components.append(
            {
                "center": (
                    float(centroids[label][0]),
                    float(centroids[label][1]),
                ),
                "box": (int(x), int(y), int(box_width), int(box_height)),
                "area": int(area),
                "mean": float(np.mean(component_pixels)),
                "std": float(np.std(component_pixels)),
            }
        )
    provisional_squares = [
        item
        for item in components
        if 0.45
        <= item["box"][2] / max(1, item["box"][3])
        <= 1.75
        and image_area * 0.00035
        <= item["area"]
        <= image_area * 0.0045
    ]
    if len(provisional_squares) < 10:
        return None
    expected_side = float(
        np.median(
            [
                math.sqrt(item["box"][2] * item["box"][3])
                for item in provisional_squares
            ]
        )
    )
    square_components = [
        item
        for item in provisional_squares
        if 0.55
        <= item["box"][2] / max(1, item["box"][3])
        <= 1.45
        and 0.55
        <= math.sqrt(item["box"][2] * item["box"][3]) / expected_side
        <= 1.55
    ]

    background_pixels = gray[
        (local_std < 3.8)
        & (local_chroma < 28.0)
        & central_region
        & ~inner_exclusion
    ]
    if background_pixels.size:
        background_mean = float(np.percentile(background_pixels, 55))
    else:
        background_mean = float(np.median(gray[central_region]))
    reliable = [
        item
        for item in square_components
        if abs(item["mean"] - background_mean) > 10.0
        and item["center"][1] < height * 0.67
    ]
    reliable.sort(key=lambda item: item["mean"])

    if len(reliable) not in {14, 15}:
        return None

    missing_direct_levels = []
    if len(reliable) == 15:
        direct_levels = [*range(5, 17), 18, 19, 20]
    else:
        # In the more oblique 2014_1 exposure, level 16 has almost the same
        # gray value as the chart background and merges into that component.
        # Keep the 14 image-supported components and infer only level 16 from
        # the common symmetry axis before refining its local image sample.
        direct_levels = [*range(5, 16), 18, 19, 20]
        missing_direct_levels = [16]
    level_by_number = dict(zip(direct_levels, reliable))
    direct_pairs = [
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
    ]
    if 16 in level_by_number:
        direct_pairs.append((15, 16))
    direct_pairs.append((19, 20))
    symmetry_axis_x = float(
        np.median(
            [
                (
                    level_by_number[odd_level]["center"][0]
                    + level_by_number[even_level]["center"][0]
                )
                / 2.0
                for odd_level, even_level in direct_pairs
            ]
        )
    )

    level_five = level_by_number[5]
    level_six = level_by_number[6]
    dark_threshold = min(75.0, level_six["mean"] + 10.0)
    dark_region = (
        (gray < dark_threshold)
        & (normalized_x > 0.34)
        & (normalized_x < 0.62)
        & (normalized_y > 0.56)
        & (normalized_y < 0.75)
    ).astype(np.uint8)
    dark_region = cv2.morphologyEx(
        dark_region,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    distance_map = cv2.distanceTransform(dark_region, cv2.DIST_L2, 5)
    maximum_kernel = max(15, int(expected_side * 0.75)) | 1
    local_maximum = distance_map == cv2.dilate(
        distance_map,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (maximum_kernel, maximum_kernel),
        ),
    )
    peak_mask = (
        local_maximum
        & (distance_map > expected_side * 0.20)
        & (dark_region > 0)
    ).astype(np.uint8)
    peak_count, peak_labels, _, peak_centroids = (
        cv2.connectedComponentsWithStats(peak_mask, 8)
    )
    peaks = []
    for peak_label in range(1, peak_count):
        center_x, center_y = peak_centroids[peak_label]
        if not (
            level_six["center"][0] + expected_side * 0.45
            < center_x
            < level_five["center"][0] - expected_side * 0.45
            and center_y
            > min(level_five["center"][1], level_six["center"][1])
            - expected_side * 0.10
        ):
            continue
        peaks.append(
            {
                "center": (float(center_x), float(center_y)),
                "distance": float(
                    np.max(distance_map[peak_labels == peak_label])
                ),
            }
        )
    left_peaks = sorted(
        [item for item in peaks if item["center"][0] < symmetry_axis_x],
        key=lambda item: abs(item["center"][0] - symmetry_axis_x),
    )
    right_peaks = sorted(
        [item for item in peaks if item["center"][0] > symmetry_axis_x],
        key=lambda item: abs(item["center"][0] - symmetry_axis_x),
    )
    if len(left_peaks) < 2 or len(right_peaks) < 2:
        return None

    sample_half = max(6, int(expected_side * 0.20))

    def sampled_center_item(center: tuple[float, float]) -> dict[str, Any]:
        center_x, center_y = center
        patch = gray[
            int(round(center_y)) - sample_half : int(round(center_y))
            + sample_half,
            int(round(center_x)) - sample_half : int(round(center_x))
            + sample_half,
        ]
        if patch.shape != (2 * sample_half, 2 * sample_half):
            raise WorkflowError("灰阶质心附近采样区域越出图片边界。")
        return {
            "center": center,
            "mean": float(np.mean(patch)),
            "std": float(np.std(patch)),
        }

    if 16 not in level_by_number:
        level_fifteen_center = level_by_number[15]["center"]
        level_by_number[16] = sampled_center_item(
            (
                float(2.0 * symmetry_axis_x - level_fifteen_center[0]),
                float(level_fifteen_center[1]),
            )
        )

    level_by_number[2] = sampled_center_item(left_peaks[0]["center"])
    level_by_number[4] = sampled_center_item(left_peaks[1]["center"])
    level_by_number[1] = sampled_center_item(right_peaks[0]["center"])
    level_by_number[3] = sampled_center_item(right_peaks[1]["center"])

    level_eighteen_center = np.asarray(
        level_by_number[18]["center"], dtype=float
    )
    predicted_level_seventeen = np.asarray(
        [
            2.0 * symmetry_axis_x - level_eighteen_center[0],
            level_eighteen_center[1],
        ],
        dtype=float,
    )
    refinement_radius = max(2, int(expected_side * 0.08))
    best_seventeen = None
    for center_y in range(
        int(predicted_level_seventeen[1] - refinement_radius),
        int(predicted_level_seventeen[1] + refinement_radius) + 1,
        2,
    ):
        for center_x in range(
            int(predicted_level_seventeen[0] - refinement_radius),
            int(predicted_level_seventeen[0] + refinement_radius) + 1,
            2,
        ):
            patch = gray[
                center_y - sample_half : center_y + sample_half,
                center_x - sample_half : center_x + sample_half,
            ]
            patch_rgb = rgb[
                center_y - sample_half : center_y + sample_half,
                center_x - sample_half : center_x + sample_half,
            ]
            if patch.shape != (2 * sample_half, 2 * sample_half):
                continue
            channel_values = np.mean(patch_rgb, axis=(0, 1))
            chroma = float(np.max(channel_values) - np.min(channel_values))
            distance = math.hypot(
                center_x - predicted_level_seventeen[0],
                center_y - predicted_level_seventeen[1],
            )
            score = float(np.std(patch)) + 0.10 * chroma + 0.85 * distance
            if best_seventeen is None or score < best_seventeen["score"]:
                best_seventeen = {
                    "center": (float(center_x), float(center_y)),
                    "mean": float(np.mean(patch)),
                    "std": float(np.std(patch)),
                    "score": score,
                }
    if best_seventeen is None:
        return None
    level_by_number[17] = best_seventeen
    selected = [level_by_number[level] for level in range(1, count + 1)]

    roi_side = expected_side * 0.56
    centers_scaled = np.asarray(
        [item["center"] for item in selected], dtype=float
    )
    symmetric_pairs_zero_based = tuple((index, index + 1) for index in range(0, 20, 2))
    pair_midpoint_x = [
        float(
            (
                centers_scaled[first_index, 0]
                + centers_scaled[second_index, 0]
            )
            / 2.0
        )
        for first_index, second_index in symmetric_pairs_zero_based
    ]
    pair_midpoint_y = [
        float(
            (
                centers_scaled[first_index, 1]
                + centers_scaled[second_index, 1]
            )
            / 2.0
        )
        for first_index, second_index in symmetric_pairs_zero_based
    ]
    pair_radii = [
        float(
            abs(
                centers_scaled[first_index, 0]
                - centers_scaled[second_index, 0]
            )
            / 2.0
        )
        for first_index, second_index in symmetric_pairs_zero_based
    ]
    pair_vertical_errors = [
        float(
            abs(
                centers_scaled[first_index, 1]
                - centers_scaled[second_index, 1]
            )
        )
        for first_index, second_index in symmetric_pairs_zero_based
    ]
    ellipse = cv2.fitEllipse(centers_scaled.astype(np.float32).reshape(-1, 1, 2))
    (ellipse_center_x, ellipse_center_y), (
        ellipse_width,
        ellipse_height,
    ), ellipse_angle = ellipse
    radians = math.radians(ellipse_angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    ellipse_errors = []
    for center_x, center_y in centers_scaled:
        delta_x = center_x - ellipse_center_x
        delta_y = center_y - ellipse_center_y
        rotated_x = cosine * delta_x + sine * delta_y
        rotated_y = -sine * delta_x + cosine * delta_y
        radius = math.sqrt(
            (rotated_x / max(1.0, ellipse_width / 2.0)) ** 2
            + (rotated_y / max(1.0, ellipse_height / 2.0)) ** 2
        )
        ellipse_errors.append(abs(radius - 1.0))
    if (
        max(abs(value - symmetry_axis_x) for value in pair_midpoint_x)
        > expected_side * 0.65
        or max(pair_vertical_errors) > expected_side * 0.55
        or not np.all(np.diff(pair_midpoint_y) < -expected_side * 0.08)
        or not np.all(np.diff(pair_radii[:6]) > -expected_side * 0.12)
        or not np.all(np.diff(pair_radii[4:]) < expected_side * 0.12)
        or float(np.median(ellipse_errors)) > 0.11
    ):
        return None
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            if (
                np.linalg.norm(
                    centers_scaled[first_index] - centers_scaled[second_index]
                )
                < roi_side * 0.92
            ):
                return None
    inverse_scale = 1.0 / scale
    coordinates = [
        _box_from_center(
            item["center"][0] * inverse_scale,
            item["center"][1] * inverse_scale,
            roi_side * inverse_scale,
            roi_side * inverse_scale,
            full_width,
            full_height,
        )
        for item in selected
    ]
    centers_full = [
        [
            int(round(item["center"][0] * inverse_scale)),
            int(round(item["center"][1] * inverse_scale)),
        ]
        for item in selected
    ]
    if len({tuple(center) for center in centers_full}) != count:
        return None

    roi_arrays = crop_rois(image, coordinates)
    measured_means = [
        float(np.mean(_gray_u8(roi))) for roi in roi_arrays
    ]
    deviations = [
        float(np.std(_gray_u8(roi))) for roi in roi_arrays
    ]
    return coordinates, {
        "method": "central_gray_components_ellipse_centroids",
        "order": "printed levels 1-20 around the central horseshoe",
        "centroids": centers_full,
        "symmetry_axis_x": int(round(symmetry_axis_x * inverse_scale))
        if "symmetry_axis_x" in locals() else None,
        "symmetric_pairs": [
            [level, level + 1] for level in range(1, 20, 2)
        ],
        "pair_midpoint_y": [
            float(value * inverse_scale) for value in pair_midpoint_y
        ],
        "pair_radii": [
            float(value * inverse_scale) for value in pair_radii
        ],
        "ellipse_model": {
            "center": [
                float(ellipse_center_x * inverse_scale),
                float(ellipse_center_y * inverse_scale),
            ],
            "axes": [
                float(ellipse_width * inverse_scale),
                float(ellipse_height * inverse_scale),
            ],
            "angle_degrees": float(ellipse_angle),
            "median_normalized_error": float(np.median(ellipse_errors)),
        },
        "direct_centroid_levels": direct_levels,
        "image_and_symmetry_inferred_levels": [
            1,
            2,
            3,
            4,
            17,
            *missing_direct_levels,
        ],
        "gray_means_by_printed_level": measured_means,
        "gray_means_dark_to_bright": sorted(measured_means),
        "gray_dynamic_range": float(max(measured_means) - min(measured_means)),
        "median_patch_standard_deviation": float(np.median(deviations)),
        "layout_note": (
            "将 20 级建模为 10 组左右镜像点对。5-16、18-20 由连通域"
            "质心直接检测；暗部 1-4 由距离变换峰与镜像约束联合定位；17"
            "由正确的 18 镜像预测并在局部图像内微调。最后用椭圆拟合、"
            "层间顺序和左右对称性统一校验。"
        ),
    }


def _detect_horseshoe_step_chart_by_side_pairs(
    image: np.ndarray,
    count: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]] | None:
    """Detect the 20-step horseshoe from six left/right side pairs.

    This model is more tolerant when bright top steps merge into the chart
    background or perspective makes a side patch look rectangular.
    """
    if count != 20:
        return None

    import cv2

    gray_full = _gray_u8(image)
    full_height, full_width = gray_full.shape
    scale = min(1.0, 1600.0 / max(full_height, full_width))
    gray = (
        cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray_full
    )
    if image.ndim == 3:
        rgb_full = image[:, :, :3]
    else:
        rgb_full = np.repeat(gray_full[:, :, None], 3, axis=2)
    rgb = (
        cv2.resize(rgb_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else rgb_full
    ).astype(np.float32)
    height, width = gray.shape
    image_area = height * width
    kernel_size = max(9, int(min(height, width) * 0.014)) | 1
    gray_float = gray.astype(np.float32)
    local_mean = cv2.boxFilter(
        gray_float, cv2.CV_32F, (kernel_size, kernel_size), normalize=True
    )
    local_square_mean = cv2.boxFilter(
        gray_float * gray_float,
        cv2.CV_32F,
        (kernel_size, kernel_size),
        normalize=True,
    )
    local_std = np.sqrt(
        np.maximum(0.0, local_square_mean - local_mean * local_mean)
    )
    channel_means = [
        cv2.boxFilter(
            rgb[:, :, channel],
            cv2.CV_32F,
            (kernel_size, kernel_size),
            normalize=True,
        )
        for channel in range(3)
    ]
    local_chroma = np.max(channel_means, axis=0) - np.min(channel_means, axis=0)

    yy, xx = np.mgrid[:height, :width]
    normalized_x = xx / width
    normalized_y = yy / height
    central_region = (
        (0.27 <= normalized_x)
        & (normalized_x <= 0.70)
        & (0.22 <= normalized_y)
        & (normalized_y <= 0.76)
    )
    inner_exclusion = (
        (0.40 <= normalized_x)
        & (normalized_x <= 0.56)
        & (0.35 <= normalized_y)
        & (normalized_y <= 0.61)
    )
    flat_mask = (
        (local_std < 3.8)
        & (local_chroma < 28.0)
        & central_region
        & ~inner_exclusion
    ).astype(np.uint8)
    flat_mask = cv2.morphologyEx(
        flat_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        flat_mask, 8
    )
    components = []
    for label in range(1, component_count):
        x, y, box_width, box_height, area = stats[label]
        if (
            area < image_area * 0.00006
            or area > image_area * 0.006
            or box_width < max(10, int(min(height, width) * 0.007))
            or box_height < max(10, int(min(height, width) * 0.007))
        ):
            continue
        component_pixels = gray[labels == label]
        components.append(
            {
                "center": (
                    float(centroids[label][0]),
                    float(centroids[label][1]),
                ),
                "box": (int(x), int(y), int(box_width), int(box_height)),
                "area": int(area),
                "mean": float(np.mean(component_pixels)),
                "std": float(np.std(component_pixels)),
            }
        )
    if len(components) < 12:
        return None

    background_pixels = gray[
        (local_std < 3.8)
        & (local_chroma < 28.0)
        & central_region
        & ~inner_exclusion
    ]
    if background_pixels.size:
        background_mean = float(np.percentile(background_pixels, 55))
    else:
        background_mean = float(np.median(gray[central_region]))
    relaxed = [
        item
        for item in components
        if 0.35 <= item["box"][2] / max(1, item["box"][3]) <= 1.90
    ]
    if len(relaxed) < 12:
        return None
    expected_side = float(
        np.median(
            [
                math.sqrt(item["box"][2] * item["box"][3])
                for item in relaxed
            ]
        )
    )
    dark_candidates = [
        item
        for item in relaxed
        if item["mean"] < background_mean - 7.5
        and height * 0.34 < item["center"][1] < height * 0.72
    ]
    if len(dark_candidates) < 10:
        return None

    pair_candidates = []
    for first_index, first in enumerate(dark_candidates):
        for second in dark_candidates[first_index + 1 :]:
            left, right = sorted([first, second], key=lambda item: item["center"][0])
            delta_y = abs(left["center"][1] - right["center"][1])
            midpoint_x = (left["center"][0] + right["center"][0]) / 2.0
            midpoint_y = (left["center"][1] + right["center"][1]) / 2.0
            radius = (right["center"][0] - left["center"][0]) / 2.0
            if (
                delta_y > expected_side * 0.80
                or radius < expected_side * 2.0
                or radius > expected_side * 6.3
                or not (width * 0.36 < midpoint_x < width * 0.64)
            ):
                continue
            pair_candidates.append(
                {
                    "left": left,
                    "right": right,
                    "midpoint_x": float(midpoint_x),
                    "midpoint_y": float(midpoint_y),
                    "radius": float(radius),
                    "delta_y": float(delta_y),
                }
            )
    if len(pair_candidates) < 6:
        return None

    best_support: list[dict[str, Any]] = []
    best_axis = None
    for pair in pair_candidates:
        support = [
            other
            for other in pair_candidates
            if abs(other["midpoint_x"] - pair["midpoint_x"]) < expected_side * 0.85
        ]
        if (
            len(support) > len(best_support)
            or (
                len(support) == len(best_support)
                and sum(item["delta_y"] for item in support)
                < sum(item["delta_y"] for item in best_support)
            )
        ):
            best_support = support
            best_axis = float(np.median([item["midpoint_x"] for item in support]))
    if best_axis is None or len(best_support) < 5:
        return None

    grouped_pairs: list[list[dict[str, Any]]] = []
    for pair in sorted(best_support, key=lambda item: item["midpoint_y"]):
        if (
            not grouped_pairs
            or abs(pair["midpoint_y"] - grouped_pairs[-1][0]["midpoint_y"])
            > expected_side * 0.68
        ):
            grouped_pairs.append([pair])
        else:
            grouped_pairs[-1].append(pair)
    side_pairs = []
    for group in grouped_pairs:
        side_pairs.append(
            min(
                group,
                key=lambda item: (
                    abs(item["midpoint_x"] - best_axis)
                    + item["delta_y"] * 0.8
                ),
            )
        )
    if len(side_pairs) == 5:
        ordered_pairs = sorted(side_pairs, key=lambda item: item["midpoint_y"])
        pair_gaps = np.diff([item["midpoint_y"] for item in ordered_pairs])
        inferred_gap = float(np.median(pair_gaps)) if len(pair_gaps) else expected_side * 1.3
        inferred_axis = float(np.median([item["midpoint_x"] for item in ordered_pairs]))

        def sampled_component(center: tuple[float, float]) -> dict[str, Any] | None:
            sample_half_for_component = max(6, int(expected_side * 0.20))
            center_x, center_y = center
            rounded_x = int(round(center_x))
            rounded_y = int(round(center_y))
            patch = gray[
                rounded_y - sample_half_for_component : rounded_y
                + sample_half_for_component,
                rounded_x - sample_half_for_component : rounded_x
                + sample_half_for_component,
            ]
            if patch.shape != (
                2 * sample_half_for_component,
                2 * sample_half_for_component,
            ):
                return None
            side = int(round(expected_side))
            return {
                "center": (float(center_x), float(center_y)),
                "box": (
                    int(round(center_x - side / 2)),
                    int(round(center_y - side / 2)),
                    side,
                    side,
                ),
                "area": int(side * side),
                "mean": float(np.mean(patch)),
                "std": float(np.std(patch)),
            }

        def nearest_dark_component(center: tuple[float, float]) -> dict[str, Any] | None:
            candidates = [
                item
                for item in dark_candidates
                if math.hypot(
                    item["center"][0] - center[0],
                    item["center"][1] - center[1],
                )
                < expected_side * 1.15
            ]
            if not candidates:
                return None
            return min(
                candidates,
                key=lambda item: math.hypot(
                    item["center"][0] - center[0],
                    item["center"][1] - center[1],
                ),
            )

        completed_pair = None
        if (
            ordered_pairs[-1]["midpoint_y"] + inferred_gap < height * 0.74
            and 0.75 <= inferred_gap / max(1.0, expected_side) <= 1.65
        ):
            inferred_y = ordered_pairs[-1]["midpoint_y"] + inferred_gap
            inferred_radius = min(
                ordered_pairs[0]["radius"],
                ordered_pairs[-1]["radius"],
            )
            predicted_left = (inferred_axis - inferred_radius, inferred_y)
            predicted_right = (inferred_axis + inferred_radius, inferred_y)
            left_component = nearest_dark_component(predicted_left) or sampled_component(
                predicted_left
            )
            right_component = nearest_dark_component(predicted_right) or sampled_component(
                predicted_right
            )
            if left_component is not None and right_component is not None:
                completed_pair = {
                    "left": left_component,
                    "right": right_component,
                    "midpoint_x": (
                        left_component["center"][0]
                        + right_component["center"][0]
                    )
                    / 2.0,
                    "midpoint_y": (
                        left_component["center"][1]
                        + right_component["center"][1]
                    )
                    / 2.0,
                    "radius": abs(
                        right_component["center"][0]
                        - left_component["center"][0]
                    )
                    / 2.0,
                    "delta_y": abs(
                        right_component["center"][1]
                        - left_component["center"][1]
                    ),
                    "inferred": True,
                }
        if completed_pair is not None:
            side_pairs.append(completed_pair)
    if len(side_pairs) < 6:
        return None
    side_pairs = sorted(side_pairs, key=lambda item: item["midpoint_y"])[-6:]
    side_pairs_bottom_to_top = sorted(
        side_pairs,
        key=lambda item: item["midpoint_y"],
        reverse=True,
    )
    pair_y_values = [item["midpoint_y"] for item in side_pairs_bottom_to_top]
    if not np.all(np.diff(pair_y_values) < -expected_side * 0.55):
        return None
    symmetry_axis_x = float(
        np.median([item["midpoint_x"] for item in side_pairs_bottom_to_top])
    )
    row_spacing = float(np.median(-np.diff(pair_y_values)))
    if row_spacing < expected_side * 0.65:
        return None

    sample_half = max(6, int(expected_side * 0.20))

    def sampled_center_item(center: tuple[float, float]) -> dict[str, Any]:
        center_x, center_y = center
        rounded_x = int(round(center_x))
        rounded_y = int(round(center_y))
        patch = gray[
            rounded_y - sample_half : rounded_y + sample_half,
            rounded_x - sample_half : rounded_x + sample_half,
        ]
        if patch.shape != (2 * sample_half, 2 * sample_half):
            raise WorkflowError("灰阶质心附近采样区域越出图片边界。")
        return {
            "center": (float(center_x), float(center_y)),
            "mean": float(np.mean(patch)),
            "std": float(np.std(patch)),
        }

    def component_item(component: dict[str, Any]) -> dict[str, Any]:
        return {
            "center": component["center"],
            "mean": component["mean"],
            "std": component["std"],
        }

    def refined_center_item(center: tuple[float, float]) -> dict[str, Any]:
        center_x, center_y = center
        radius = max(2, int(expected_side * 0.12))
        best = None
        for test_y in range(int(center_y - radius), int(center_y + radius) + 1, 2):
            for test_x in range(int(center_x - radius), int(center_x + radius) + 1, 2):
                patch = gray[
                    test_y - sample_half : test_y + sample_half,
                    test_x - sample_half : test_x + sample_half,
                ]
                patch_rgb = rgb[
                    test_y - sample_half : test_y + sample_half,
                    test_x - sample_half : test_x + sample_half,
                ]
                if patch.shape != (2 * sample_half, 2 * sample_half):
                    continue
                channel_values = np.mean(patch_rgb, axis=(0, 1))
                chroma = float(np.max(channel_values) - np.min(channel_values))
                distance = math.hypot(test_x - center_x, test_y - center_y)
                score = float(np.std(patch)) + 0.10 * chroma + 0.80 * distance
                if best is None or score < best["score"]:
                    best = {
                        "center": (float(test_x), float(test_y)),
                        "mean": float(np.mean(patch)),
                        "std": float(np.std(patch)),
                        "score": score,
                    }
        if best is None:
            return sampled_center_item(center)
        return best

    level_by_number: dict[int, dict[str, Any]] = {}
    for row_index, pair in enumerate(side_pairs_bottom_to_top):
        odd_level = 5 + 2 * row_index
        even_level = odd_level + 1
        level_by_number[odd_level] = component_item(pair["right"])
        level_by_number[even_level] = component_item(pair["left"])

    level_five = level_by_number[5]
    level_six = level_by_number[6]
    dark_threshold = min(
        85.0,
        max(level_five["mean"], level_six["mean"]) + 12.0,
    )
    bottom_y = max(level_five["center"][1], level_six["center"][1])
    bottom_left_x = min(level_five["center"][0], level_six["center"][0])
    bottom_right_x = max(level_five["center"][0], level_six["center"][0])
    dark_region = (
        (gray < dark_threshold)
        & (xx > bottom_left_x - expected_side * 1.2)
        & (xx < bottom_right_x + expected_side * 1.2)
        & (yy > bottom_y - expected_side * 0.9)
        & (yy < bottom_y + row_spacing * 2.2)
    ).astype(np.uint8)
    dark_region = cv2.morphologyEx(
        dark_region,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    distance_map = cv2.distanceTransform(dark_region, cv2.DIST_L2, 5)
    maximum_kernel = max(15, int(expected_side * 0.75)) | 1
    local_maximum = distance_map == cv2.dilate(
        distance_map,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (maximum_kernel, maximum_kernel),
        ),
    )
    peak_mask = (
        local_maximum
        & (distance_map > expected_side * 0.18)
        & (dark_region > 0)
    ).astype(np.uint8)
    peak_count, peak_labels, _, peak_centroids = cv2.connectedComponentsWithStats(
        peak_mask, 8
    )
    peaks = []
    for peak_label in range(1, peak_count):
        center_x, center_y = peak_centroids[peak_label]
        if not (
            bottom_left_x - expected_side * 0.2
            < center_x
            < bottom_right_x + expected_side * 0.2
            and center_y > bottom_y - expected_side * 0.5
        ):
            continue
        peaks.append(
            {
                "center": (float(center_x), float(center_y)),
                "distance": float(np.max(distance_map[peak_labels == peak_label])),
            }
        )
    left_peaks = sorted(
        [item for item in peaks if item["center"][0] < symmetry_axis_x],
        key=lambda item: abs(item["center"][0] - symmetry_axis_x),
    )
    right_peaks = sorted(
        [item for item in peaks if item["center"][0] > symmetry_axis_x],
        key=lambda item: abs(item["center"][0] - symmetry_axis_x),
    )
    bottom_pair_radius = side_pairs_bottom_to_top[0]["radius"]
    if len(left_peaks) >= 2 and len(right_peaks) >= 2:
        level_by_number[2] = sampled_center_item(left_peaks[0]["center"])
        level_by_number[4] = sampled_center_item(left_peaks[1]["center"])
        level_by_number[1] = sampled_center_item(right_peaks[0]["center"])
        level_by_number[3] = sampled_center_item(right_peaks[1]["center"])
    else:
        level_by_number[2] = refined_center_item(
            (
                symmetry_axis_x - bottom_pair_radius * 0.13,
                bottom_y + row_spacing * 0.82,
            )
        )
        level_by_number[1] = refined_center_item(
            (
                symmetry_axis_x + bottom_pair_radius * 0.13,
                bottom_y + row_spacing * 0.82,
            )
        )
        level_by_number[4] = refined_center_item(
            (
                symmetry_axis_x - bottom_pair_radius * 0.60,
                bottom_y + row_spacing * 0.55,
            )
        )
        level_by_number[3] = refined_center_item(
            (
                symmetry_axis_x + bottom_pair_radius * 0.60,
                bottom_y + row_spacing * 0.55,
            )
        )

    top_side_pair = side_pairs_bottom_to_top[-1]
    top_side_y = top_side_pair["midpoint_y"]
    top_side_radius = top_side_pair["radius"]
    bright_candidates = [
        item
        for item in relaxed
        if item["mean"] > background_mean + 8.0
        and height * 0.24 < item["center"][1] < top_side_y - expected_side * 0.35
        and abs(item["center"][0] - symmetry_axis_x) < top_side_radius * 0.95
    ]
    top_pair = None
    for first_index, first in enumerate(bright_candidates):
        for second in bright_candidates[first_index + 1 :]:
            left, right = sorted([first, second], key=lambda item: item["center"][0])
            delta_y = abs(left["center"][1] - right["center"][1])
            midpoint_x = (left["center"][0] + right["center"][0]) / 2.0
            radius = (right["center"][0] - left["center"][0]) / 2.0
            if (
                delta_y > expected_side * 0.55
                or radius < expected_side * 0.35
                or radius > top_side_radius * 0.55
                or abs(midpoint_x - symmetry_axis_x) > expected_side * 0.85
            ):
                continue
            score = (
                abs(midpoint_x - symmetry_axis_x)
                + delta_y
                - 0.01 * (left["mean"] + right["mean"])
            )
            if top_pair is None or score < top_pair["score"]:
                top_pair = {
                    "left": left,
                    "right": right,
                    "midpoint_y": (left["center"][1] + right["center"][1]) / 2.0,
                    "radius": radius,
                    "score": score,
                }
    if top_pair is not None:
        level_by_number[20] = component_item(top_pair["left"])
        level_by_number[19] = component_item(top_pair["right"])
        top_y = float(top_pair["midpoint_y"])
        top_radius = float(top_pair["radius"])
    else:
        top_y = top_side_y - row_spacing * 0.80
        top_radius = max(expected_side * 0.65, top_side_radius * 0.20)
        level_by_number[20] = refined_center_item((symmetry_axis_x - top_radius, top_y))
        level_by_number[19] = refined_center_item((symmetry_axis_x + top_radius, top_y))

    level_17_18_y = top_y + (top_side_y - top_y) * 0.34
    level_17_18_radius = top_radius + (top_side_radius - top_radius) * 0.52
    level_by_number[18] = sampled_center_item(
        (symmetry_axis_x - level_17_18_radius, level_17_18_y)
    )
    level_by_number[17] = sampled_center_item(
        (symmetry_axis_x + level_17_18_radius, level_17_18_y)
    )

    if any(level not in level_by_number for level in range(1, count + 1)):
        return None
    selected = [level_by_number[level] for level in range(1, count + 1)]
    roi_side = expected_side * 0.56
    centers_scaled = np.asarray(
        [item["center"] for item in selected], dtype=float
    )
    symmetric_pairs_zero_based = tuple((index, index + 1) for index in range(0, 20, 2))
    pair_midpoint_x = [
        float(
            (
                centers_scaled[first_index, 0]
                + centers_scaled[second_index, 0]
            )
            / 2.0
        )
        for first_index, second_index in symmetric_pairs_zero_based
    ]
    pair_midpoint_y = [
        float(
            (
                centers_scaled[first_index, 1]
                + centers_scaled[second_index, 1]
            )
            / 2.0
        )
        for first_index, second_index in symmetric_pairs_zero_based
    ]
    pair_radii = [
        float(
            abs(
                centers_scaled[first_index, 0]
                - centers_scaled[second_index, 0]
            )
            / 2.0
        )
        for first_index, second_index in symmetric_pairs_zero_based
    ]
    pair_vertical_errors = [
        float(
            abs(
                centers_scaled[first_index, 1]
                - centers_scaled[second_index, 1]
            )
        )
        for first_index, second_index in symmetric_pairs_zero_based
    ]
    ellipse = cv2.fitEllipse(centers_scaled.astype(np.float32).reshape(-1, 1, 2))
    (ellipse_center_x, ellipse_center_y), (
        ellipse_width,
        ellipse_height,
    ), ellipse_angle = ellipse
    radians = math.radians(ellipse_angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    ellipse_errors = []
    for center_x, center_y in centers_scaled:
        delta_x = center_x - ellipse_center_x
        delta_y = center_y - ellipse_center_y
        rotated_x = cosine * delta_x + sine * delta_y
        rotated_y = -sine * delta_x + cosine * delta_y
        radius = math.sqrt(
            (rotated_x / max(1.0, ellipse_width / 2.0)) ** 2
            + (rotated_y / max(1.0, ellipse_height / 2.0)) ** 2
        )
        ellipse_errors.append(abs(radius - 1.0))
    if (
        max(abs(value - symmetry_axis_x) for value in pair_midpoint_x)
        > expected_side * 0.80
        or max(pair_vertical_errors) > expected_side * 0.70
        or not np.all(np.diff(pair_midpoint_y) < -expected_side * 0.04)
        or not np.all(np.diff(pair_radii[:6]) > -expected_side * 0.18)
        or not np.all(np.diff(pair_radii[4:]) < expected_side * 0.18)
        or float(np.median(ellipse_errors)) > 0.16
    ):
        return None
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            if (
                np.linalg.norm(
                    centers_scaled[first_index] - centers_scaled[second_index]
                )
                < roi_side * 0.88
            ):
                return None

    inverse_scale = 1.0 / scale
    coordinates = [
        _box_from_center(
            item["center"][0] * inverse_scale,
            item["center"][1] * inverse_scale,
            roi_side * inverse_scale,
            roi_side * inverse_scale,
            full_width,
            full_height,
        )
        for item in selected
    ]
    centers_full = [
        [
            int(round(item["center"][0] * inverse_scale)),
            int(round(item["center"][1] * inverse_scale)),
        ]
        for item in selected
    ]
    if len({tuple(center) for center in centers_full}) != count:
        return None

    roi_arrays = crop_rois(image, coordinates)
    measured_means = [float(np.mean(_gray_u8(roi))) for roi in roi_arrays]
    deviations = [float(np.std(_gray_u8(roi))) for roi in roi_arrays]
    return coordinates, {
        "method": "side_pair_symmetry_horseshoe_centroids",
        "order": "printed levels 1-20 around the horseshoe",
        "centroids": centers_full,
        "symmetry_axis_x": int(round(symmetry_axis_x * inverse_scale)),
        "symmetric_pairs": [[level, level + 1] for level in range(1, 20, 2)],
        "pair_midpoint_y": [float(value * inverse_scale) for value in pair_midpoint_y],
        "pair_radii": [float(value * inverse_scale) for value in pair_radii],
        "ellipse_model": {
            "center": [
                float(ellipse_center_x * inverse_scale),
                float(ellipse_center_y * inverse_scale),
            ],
            "axes": [
                float(ellipse_width * inverse_scale),
                float(ellipse_height * inverse_scale),
            ],
            "angle_degrees": float(ellipse_angle),
            "median_normalized_error": float(np.median(ellipse_errors)),
        },
        "direct_centroid_levels": [*range(5, 17), 19, 20],
        "image_and_symmetry_inferred_levels": [1, 2, 3, 4, 17, 18],
        "gray_means_by_printed_level": measured_means,
        "gray_means_dark_to_bright": sorted(measured_means),
        "gray_dynamic_range": float(max(measured_means) - min(measured_means)),
        "median_patch_standard_deviation": float(np.median(deviations)),
        "layout_note": (
            "先定位左右六对灰阶侧边方块，再用公共对称轴、底部暗块距离峰"
            "和顶部亮块几何关系补齐 1-20 级。该分支用于亮灰块与背景合并"
            "或透视导致方块略呈长方形的照片。"
        ),
    }


def _shift_horseshoe_metadata(
    metadata: dict[str, Any],
    offset_x: int,
    offset_y: int,
    search_window: tuple[int, int, int, int],
) -> dict[str, Any]:
    shifted = dict(metadata)
    shifted["centroids"] = [
        [int(center[0] + offset_x), int(center[1] + offset_y)]
        for center in metadata.get("centroids", [])
    ]
    if metadata.get("symmetry_axis_x") is not None:
        shifted["symmetry_axis_x"] = int(metadata["symmetry_axis_x"] + offset_x)
    if metadata.get("pair_midpoint_y") is not None:
        shifted["pair_midpoint_y"] = [
            float(value + offset_y) for value in metadata["pair_midpoint_y"]
        ]
    ellipse_model = dict(metadata.get("ellipse_model") or {})
    if ellipse_model.get("center"):
        ellipse_model["center"] = [
            float(ellipse_model["center"][0] + offset_x),
            float(ellipse_model["center"][1] + offset_y),
        ]
        shifted["ellipse_model"] = ellipse_model
    shifted["method"] = "local_region_" + str(metadata.get("method", "horseshoe"))
    shifted["search_window"] = list(search_window)
    return shifted


def _detect_horseshoe_step_chart_local(
    image: np.ndarray,
    count: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]] | None:
    """Try central horseshoe detection inside deterministic local windows."""
    if count != 20:
        return None
    import cv2

    gray = _gray_u8(image)
    full_height, full_width = gray.shape
    image_area = full_height * full_width
    boxes: list[tuple[int, int, int, int]] = []

    dark_threshold = min(95, int(np.percentile(gray, 18)))
    dark_mask = (gray < dark_threshold).astype(np.uint8)
    close_size = max(21, int(min(full_height, full_width) * 0.025)) | 1
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size)),
    )
    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < image_area * 0.001:
            continue
        for factor in (1.0, 1.02, 1.18, 1.55, 2.6):
            boxes.append(
                _expand_box(
                    (x, y, width, height),
                    full_width,
                    full_height,
                    factor,
                )
            )
    if np.count_nonzero(dark_mask):
        ys, xs = np.where(dark_mask > 0)
        all_dark_box = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
        for factor in (1.0, 1.02, 1.18, 1.55):
            boxes.append(
                _expand_box(
                    all_dark_box,
                    full_width,
                    full_height,
                    factor,
                )
            )

    for fraction in (0.32, 0.42, 0.52, 0.64):
        window_width = int(full_width * fraction)
        window_height = int(window_width * 0.68)
        if window_width < 420 or window_height < 300:
            continue
        window_height = min(window_height, full_height)
        x_values = np.linspace(0, max(0, full_width - window_width), 4)
        y_values = np.linspace(0, max(0, full_height - window_height), 4)
        for left in x_values:
            for top in y_values:
                boxes.append((int(round(left)), int(round(top)), window_width, window_height))

    unique_boxes = []
    seen = set()
    for box in boxes:
        key = tuple(int(round(value / 20)) for value in box)
        if key in seen:
            continue
        seen.add(key)
        unique_boxes.append(box)

    attempts = []
    for left, top, width, height in sorted(
        unique_boxes,
        key=lambda item: item[2] * item[3],
    ):
        crop = image[top : top + height, left : left + width]
        try:
            result = _detect_horseshoe_step_chart(crop, count)
            if result is None:
                result = _detect_horseshoe_step_chart_by_side_pairs(crop, count)
        except WorkflowError:
            continue
        if result is None:
            continue
        rois, metadata = result
        shifted_rois = _offset_rois(rois, left, top)
        shifted_metadata = _shift_horseshoe_metadata(
            metadata,
            left,
            top,
            (left, top, width, height),
        )
        median_error = (
            metadata.get("ellipse_model") or {}
        ).get("median_normalized_error", 1.0)
        attempts.append(
            (
                (
                    metadata.get("gray_dynamic_range", 0.0),
                    -float(median_error),
                    -(width * height),
                ),
                shifted_rois,
                shifted_metadata,
            )
        )
    if not attempts:
        return None
    return max(attempts, key=lambda item: item[0])[1:]


def auto_detect_step_chart_rois(
    image: np.ndarray,
    count: int = 20,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    import cv2

    gray_full = _gray_u8(image)
    full_height, full_width = gray_full.shape
    scale = min(1.0, 1400.0 / max(full_height, full_width))
    gray = (
        cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray_full
    )
    linear = _detect_linear_step_strip(gray, count)
    if linear is not None:
        coordinates_scaled, metadata = linear
        inverse_scale = 1.0 / scale
        coordinates = [
            _box_from_center(
                (x + width / 2) * inverse_scale,
                (y + height / 2) * inverse_scale,
                width * inverse_scale,
                height * inverse_scale,
                full_width,
                full_height,
            )
            for x, y, width, height in coordinates_scaled
        ]
        return coordinates, metadata
    horseshoe = _detect_horseshoe_step_chart(image, count)
    if horseshoe is not None:
        return horseshoe
    side_pair_horseshoe = _detect_horseshoe_step_chart_by_side_pairs(image, count)
    if side_pair_horseshoe is not None:
        return side_pair_horseshoe
    local_horseshoe = _detect_horseshoe_step_chart_local(image, count)
    if local_horseshoe is not None:
        return local_horseshoe
    raise WorkflowError(
        "没有可靠识别到受支持的线性或中央环形20级灰阶。"
        "为避免在综合测试卡上随机误框，程序已停止通用全图灰块回退；"
        "请在图形页面使用“人工修正ROI”保留或重画20个灰阶框。"
    )


def save_annotated_rois(
    image: np.ndarray,
    coordinates: list[tuple[int, int, int, int]],
    destination: Path,
    labels: list[str] | None = None,
    polygons: list[list[list[int]]] | None = None,
    center_markers: bool = False,
    symmetry_axis_x: int | None = None,
    symmetric_pairs: list[list[int]] | None = None,
) -> None:
    import cv2

    if image.ndim == 2:
        canvas = cv2.cvtColor(_gray_u8(image), cv2.COLOR_GRAY2BGR)
    else:
        canvas = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
    for polygon in polygons or []:
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (255, 255, 0), 4, cv2.LINE_AA)
        anchor = tuple(int(value) for value in points[:, 0, :].min(axis=0))
        cv2.putText(
            canvas,
            "Detected tilted square",
            (anchor[0], max(28, anchor[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    for index, (x, y, width, height) in enumerate(coordinates):
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 255, 0), 3)
        if center_markers:
            cv2.circle(
                canvas,
                (int(round(x + width / 2)), int(round(y + height / 2))),
                5,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )
        text = labels[index] if labels and index < len(labels) else f"ROI {index + 1}"
        cv2.putText(
            canvas,
            text,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if symmetry_axis_x is not None and coordinates:
        centers = [
            (int(round(x + width / 2)), int(round(y + height / 2)))
            for x, y, width, height in coordinates
        ]
        paired_levels = [
            level
            for pair in symmetric_pairs or []
            for level in pair
            if 1 <= level <= len(centers)
        ]
        y_values = (
            [centers[level - 1][1] for level in paired_levels]
            if paired_levels
            else [center[1] for center in centers]
        )
        cv2.line(
            canvas,
            (symmetry_axis_x, max(0, min(y_values) - 60)),
            (symmetry_axis_x, min(canvas.shape[0] - 1, max(y_values) + 60)),
            (255, 255, 0),
            3,
            cv2.LINE_AA,
        )
        for first_level, second_level in symmetric_pairs or []:
            if (
                1 <= first_level <= len(centers)
                and 1 <= second_level <= len(centers)
            ):
                cv2.line(
                    canvas,
                    centers[first_level - 1],
                    centers[second_level - 1],
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower() or ".jpg"
    extension = suffix if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} else ".jpg"
    ok, encoded = cv2.imencode(extension, canvas)
    if not ok:
        raise WorkflowError(f"无法保存自动画框结果：{destination}")
    destination.write_bytes(encoded.tobytes())


def _prepare_library(library_root: Path) -> None:
    if not (library_root / "py_imaging_quality").is_dir():
        raise WorkflowError(f"找不到算法库：{library_root}")
    root_text = str(library_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # The supplied package imports chessboard/OpenCV at package import time.
    # MTF, color and distortion do not require OpenCV, so allow them to run
    # even when cv2 is not installed.
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.modules.setdefault("cv2", types.ModuleType("cv2"))


def _channel_names(count: int) -> list[str]:
    if count >= 4:
        return ["R", "G", "B", "luminance"] + [f"channel_{i}" for i in range(4, count)]
    return ["luminance"] if count == 1 else [f"channel_{i}" for i in range(count)]


def run_sfr(
    rois: list[np.ndarray],
    library_root: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    _prepare_library(library_root)
    from py_imaging_quality.sfr.mtf_eff import mtf_eff
    from py_imaging_quality.sfr.sfr_cal import _sfr_cal_from_roi

    freq_list, mtf_list, esf_list = _sfr_cal_from_roi(rois)
    if not freq_list:
        raise WorkflowError("SFR 算法未返回结果，请检查 ROI 是否包含高对比度斜边。")

    measurements = []
    artifacts = []
    for index, (freq, mtf, esf) in enumerate(
        zip(freq_list, mtf_list, esf_list),
        start=1,
    ):
        thresholds = mtf_eff(freq, mtf, [0.5, 0.3])
        channels = {}
        for channel_index, name in enumerate(_channel_names(mtf.shape[1])):
            mtf50 = float(thresholds[0, channel_index])
            mtf30 = float(thresholds[1, channel_index])
            channels[name] = {
                "mtf50_cy_per_pixel": None if np.isnan(mtf50) else mtf50,
                "mtf30_cy_per_pixel": None if np.isnan(mtf30) else mtf30,
            }
        measurements.append({"roi": index, "channels": channels})
        if artifact_dir is not None:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            artifact_dir.mkdir(parents=True, exist_ok=True)
            destination = artifact_dir / f"SFR_ROI_{index}_ESF_MTF.png"
            channel_names = _channel_names(mtf.shape[1])
            colors = ["r", "g", "b", "k"]
            line_styles = ["--", "-", "-.", "-"]
            esf_plot = np.asarray(esf).copy()
            orientation_channel = min(esf_plot.shape[1] - 1, 3)
            if (
                np.mean(esf_plot[:3, orientation_channel])
                > np.mean(esf_plot[-3:, orientation_channel])
            ):
                esf_plot = esf_plot[::-1, :]
            figure, axes = plt.subplots(2, 1, figsize=(9, 7.2))
            for channel_index, channel_name in enumerate(channel_names):
                axes[0].plot(
                    np.arange(esf_plot.shape[0]),
                    esf_plot[:, channel_index],
                    color=colors[min(channel_index, 3)],
                    linestyle=line_styles[min(channel_index, 3)],
                    linewidth=2 if channel_name == "luminance" else 1,
                    label=channel_name,
                )
                values = channels[channel_name]
                mtf50_text = (
                    "n/a"
                    if values["mtf50_cy_per_pixel"] is None
                    else f"{values['mtf50_cy_per_pixel']:.6f}"
                )
                mtf30_text = (
                    "n/a"
                    if values["mtf30_cy_per_pixel"] is None
                    else f"{values['mtf30_cy_per_pixel']:.6f}"
                )
                axes[1].plot(
                    freq,
                    mtf[:, channel_index],
                    color=colors[min(channel_index, 3)],
                    linestyle=line_styles[min(channel_index, 3)],
                    linewidth=2 if channel_name == "luminance" else 1,
                    label=(
                        f"{channel_name}  MTF50={mtf50_text}  "
                        f"MTF30={mtf30_text}"
                    ),
                )
            axes[0].set_title(f"ROI {index} Edge Spread Function")
            axes[0].set_xlabel("Oversampled position")
            axes[0].set_ylabel("Pixel value")
            axes[0].grid(alpha=0.25)
            axes[0].legend()
            axes[1].set_title(f"ROI {index} MTF Curve")
            axes[1].set_xlabel("Frequency (cycles/pixel)")
            axes[1].set_ylabel("MTF")
            axes[1].set_xlim(0, float(freq[max(1, int(len(freq) * 0.75) - 1)]))
            axes[1].set_ylim(0, max(1.0, float(np.nanmax(mtf)) * 1.05))
            axes[1].grid(alpha=0.25)
            axes[1].legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(destination, dpi=150)
            plt.close(figure)
            artifacts.append(
                {
                    "type": "sfr_esf_mtf_curve",
                    "title": f"ROI {index} ESF 与 MTF 曲线",
                    "path": str(destination.resolve()),
                }
            )

    return {
        "unit": "cycles/pixel",
        "measurements": measurements,
        "artifacts": artifacts,
        "assessment": (
            "已按原SFR算法输出ESF、MTF曲线及MTF50/MTF30。"
            "MTF数值应在相同镜头、视场、焦距、像元和拍摄条件下比较；"
            "是否合格应采用课程或项目规定阈值，不能仅凭通用固定阈值判断。"
        ),
    }


def run_distortion(
    rois: list[np.ndarray],
    library_root: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    _prepare_library(library_root)
    from py_imaging_quality.distortion.distortion import _distortion_from_roi

    k1, tv = _distortion_from_roi(rois[0])
    k1, tv_percent = float(k1), float(tv) * 100.0
    kind = "桶形畸变" if k1 < 0 else "枕形畸变" if k1 > 0 else "未检出明显畸变"
    return {
        "k1": k1,
        "smia_tv_percent": tv_percent,
        "distortion_type": kind,
        "assessment": f"{kind}；SMIA TV 畸变率为 {tv_percent:.3f}%。",
    }


def run_color_check(
    rois: list[np.ndarray],
    library_root: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    _prepare_library(library_root)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from py_imaging_quality.color_check import _color_check_from_rois

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive",
        )
        delta_e, delta_c, saturation = _color_check_from_rois(rois)
    artifacts = []
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / "ColorChecker_Lab_测量与理论对比.png"
        figure = plt.gcf()
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        artifacts.append(
            {
                "type": "colorchecker_lab_plot",
                "title": "ColorChecker Lab 测量值与理论值",
                "path": str(destination.resolve()),
            }
        )
    plt.close("all")
    return {
        "mean_delta_e": float(delta_e[24]),
        "max_delta_e": float(delta_e[25]),
        "mean_delta_c": float(delta_c[24]),
        "max_delta_c": float(delta_c[25]),
        "mean_saturation_percent": float(saturation[24] * 100),
        "artifacts": artifacts,
        "assessment": "平均 ΔE 越小，色彩还原越准确；应结合课程规定阈值评价。",
    }


def run_step_chart(
    rois: list[np.ndarray],
    library_root: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    _prepare_library(library_root)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from py_imaging_quality.step_chart import _step_chart_from_rois

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive",
        )
        steps, contrast = _step_chart_from_rois(rois)
    artifacts = []
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / "20级灰阶响应曲线.png"
        figure = plt.gcf()
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        artifacts.append(
            {
                "type": "step_chart_response",
                "title": "20级灰阶阶调响应曲线",
                "path": str(destination.resolve()),
            }
        )
    plt.close("all")
    return {
        "detected_transitions": int(steps),
        "contrast_normalized": float(contrast),
        "artifacts": artifacts,
        "assessment": "可分辨相邻灰阶越多，阶调区分能力越好。",
    }


def run_white_balance(
    rois: list[np.ndarray],
    library_root: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    _prepare_library(library_root)
    from py_imaging_quality.white_balance import _wb_from_rois

    errors, channel_ids = _wb_from_rois(rois)
    names = {1: "R", 2: "G", 3: "B"}
    patches = [
        {
            "patch": index + 1,
            "rgb_range_error": float(error),
            "dominant_channel": names.get(int(channel_ids[index]), "unknown"),
        }
        for index, error in enumerate(errors)
    ]
    return {
        "patches": patches,
        "mean_rgb_range_error": float(np.mean(errors)),
        "max_rgb_range_error": float(np.max(errors)),
        "assessment": "RGB 极差越接近 0，灰块的白平衡越好。",
    }


def run_dark_field(image: np.ndarray) -> dict[str, Any]:
    data = image.astype(np.float64)
    if data.ndim == 2:
        data = data[:, :, None]
    names = ["gray"] if data.shape[2] == 1 else ["R", "G", "B"][: data.shape[2]]
    channels = {}
    for index, name in enumerate(names):
        channel = data[:, :, index]
        channels[name] = {
            "mean": float(np.mean(channel)),
            "standard_deviation": float(np.std(channel)),
            "minimum": float(np.min(channel)),
            "maximum": float(np.max(channel)),
        }
    return {
        "channels": channels,
        "assessment": "暗场均值反映黑电平，标准差反映随机噪声；二者通常越低越好。",
    }


def synthesize_agent_response(
    tool: str,
    result: dict[str, Any],
    detection: dict[str, Any] | None,
    roi_count: int,
) -> str:
    """Turn deterministic measurement output into a concise agent answer."""
    if tool == "sfr":
        luminance_values = []
        for measurement in result["measurements"]:
            luminance = measurement["channels"].get("luminance", {})
            mtf50 = luminance.get("mtf50_cy_per_pixel")
            mtf30 = luminance.get("mtf30_cy_per_pixel")
            if mtf50 is not None and mtf30 is not None:
                luminance_values.append(
                    f"ROI {measurement['roi']} 的亮度 MTF50={mtf50:.4f}、"
                    f"MTF30={mtf30:.4f} cy/px"
                )
        contrast_note = ""
        candidates = (detection or {}).get("candidates") or []
        if candidates:
            minimum_contrast = min(
                float(item.get("local_contrast", 1.0)) for item in candidates
            )
            if minimum_contrast < 0.25:
                contrast_note = (
                    f" 自动斜边的最低局部对比度约为 {minimum_contrast:.1%}，"
                    "应结合OpenCV框选图检查边缘质量。"
                )
        metrics = "；".join(luminance_values) or "未得到有效的亮度MTF交点"
        return f"我识别到这是MTF/SFR任务，自动定位了 {roi_count} 个斜边ROI。{metrics}。{result['assessment']}{contrast_note}"
    if tool == "color_check":
        return (
            f"我识别到这是ColorChecker色彩还原任务，自动建立了 {roi_count} 个色块ROI。"
            f"平均ΔE={result['mean_delta_e']:.2f}，最大ΔE={result['max_delta_e']:.2f}，"
            f"平均ΔC={result['mean_delta_c']:.2f}，平均饱和度="
            f"{result['mean_saturation_percent']:.1f}%。{result['assessment']}"
        )
    if tool == "step_chart":
        return (
            f"我识别到这是20级灰阶任务，自动定位了 {roi_count} 个灰阶ROI。"
            f"共检测到 {result['detected_transitions']}/19 个可分辨相邻过渡，"
            f"归一化对比度为 {result['contrast_normalized']:.4f}。"
            f"{result['assessment']}"
        )
    if tool == "distortion":
        return (
            f"我识别到这是镜头畸变任务。结果为{result['distortion_type']}，"
            f"SMIA TV畸变率={result['smia_tv_percent']:.3f}%。"
        )
    if tool == "white_balance":
        return (
            f"我识别到这是白平衡任务，自动使用了 {roi_count} 个中性灰块。"
            f"RGB平均极差={result['mean_rgb_range_error']:.3f}，"
            f"最大极差={result['max_rgb_range_error']:.3f}。{result['assessment']}"
        )
    channel_text = "；".join(
        f"{name}均值={values['mean']:.3f}、标准差={values['standard_deviation']:.3f}"
        for name, values in result["channels"].items()
    )
    return f"我识别到这是暗场噪声任务，已对整幅图完成统计。{channel_text}。{result['assessment']}"


def execute(
    message: str,
    image_path: str | Path,
    roi_coordinates: list[tuple[int, int, int, int]],
    library_root: Path,
    annotated_path: Path | None = None,
    forced_tool: str | None = None,
    task_parameters: dict[str, Any] | None = None,
    intent_method: str = "deepseek",
) -> dict[str, Any]:
    agent_trace: list[dict[str, Any]] = []
    task_parameters = task_parameters or {}
    if forced_tool is None:
        raise WorkflowError("测量执行器必须接收 DeepSeek Workflow 已选择的工具。")
    if forced_tool not in TOOLS:
        raise WorkflowError(f"Agent计划包含未知工具：{forced_tool}")
    intent = {
        "tool": forced_tool,
        "confidence": 0.9,
        "method": intent_method,
    }
    agent_trace.append(
        {
            "stage": "intent_recognition",
            "title": "理解自然语言",
            "status": "success",
            "detail": {
                **intent,
                "message": f"识别为 {TOOLS[intent['tool']]['description']}",
            },
        }
    )
    agent_plan = build_agent_plan(message, intent)
    agent_trace.append(
        {
            "stage": "task_planning",
            "title": "制定工具调用计划",
            "status": "success",
            "detail": agent_plan,
        }
    )
    tool = intent["tool"]
    image = load_image(image_path)
    agent_trace.append(
        {
            "stage": "image_loading",
            "title": "读取待评价图像",
            "status": "success",
            "detail": {"shape": list(image.shape), "path": str(Path(image_path).resolve())},
        }
    )
    required = int(TOOLS[tool]["roi_count"])
    detection: dict[str, Any] | None = None

    if required == 0:
        result = run_dark_field(image)
        used_rois = 0
    else:
        if not roi_coordinates and tool == "sfr":
            field_position = _infer_mtf_field_position(message, task_parameters)
            task_parameters["field_position"] = field_position
            roi_coordinates, candidates = auto_detect_sfr_rectangle_edges(
                image,
                field_position=field_position,
            )
            requested_directions = task_parameters.get("directions")
            if requested_directions:
                selected = [
                    (roi, candidate)
                    for roi, candidate in zip(roi_coordinates, candidates)
                    if candidate["orientation"] in requested_directions
                ]
                if not selected:
                    raise WorkflowError(
                        "OpenCV没有找到Agent计划要求的MTF边缘方向："
                        + "、".join(requested_directions)
                    )
                roi_coordinates = [item[0] for item in selected]
                candidates = [item[1] for item in selected]
            target_polygons = []
            seen_polygons = set()
            for candidate in candidates:
                polygon = candidate["target_polygon"]
                key = tuple(tuple(point) for point in polygon)
                if key in seen_polygons:
                    continue
                seen_polygons.add(key)
                target_polygons.append(polygon)
            detection = {
                "method": "tilted_square_two_orthogonal_edges",
                "candidates": candidates,
                "target_polygon": candidates[0]["target_polygon"],
                "target_polygons": target_polygons,
                "field_position": field_position,
                "roi_note": (
                    "青色轮廓是完整斜正方形；绿色窄框分别跨越一条近水平边和"
                    "一条近竖直边，窄框才是标准 SFR 计算 ROI。"
                ),
            }
            labels = [
                (
                    f"T{item.get('target_rank', 1)} "
                    + ("MTF-H" if item.get("mtf_direction") == "horizontal" else "MTF-V")
                )
                for item in candidates
            ]
            if annotated_path is not None:
                save_annotated_rois(
                    image,
                    roi_coordinates,
                    annotated_path,
                    labels=labels,
                    polygons=target_polygons,
                )
                detection["annotated_image"] = str(annotated_path.resolve())
        elif not roi_coordinates and tool == "color_check":
            roi_coordinates, color_metadata = auto_detect_colorchecker_rois(image)
            detection = color_metadata
            if annotated_path is not None:
                save_annotated_rois(
                    image,
                    roi_coordinates,
                    annotated_path,
                    labels=[str(index) for index in range(1, 25)],
                )
                detection["annotated_image"] = str(annotated_path.resolve())
        elif not roi_coordinates and tool == "step_chart":
            roi_coordinates, step_metadata = auto_detect_step_chart_rois(image, required)
            detection = step_metadata
            if annotated_path is not None:
                save_annotated_rois(
                    image,
                    roi_coordinates,
                    annotated_path,
                    labels=[str(index) for index in range(1, required + 1)],
                    center_markers=True,
                    symmetry_axis_x=step_metadata.get("symmetry_axis_x"),
                    symmetric_pairs=step_metadata.get("symmetric_pairs"),
                )
                detection["annotated_image"] = str(annotated_path.resolve())
        elif not roi_coordinates and tool == "white_balance":
            color_coordinates, color_metadata = auto_detect_colorchecker_rois(image)
            roi_coordinates = color_coordinates[-6:]
            detection = {
                **color_metadata,
                "method": "colorchecker_last_row_gray_patches",
            }
            if annotated_path is not None:
                save_annotated_rois(
                    image,
                    roi_coordinates,
                    annotated_path,
                    labels=[f"Gray {index}" for index in range(1, 7)],
                )
                detection["annotated_image"] = str(annotated_path.resolve())
        elif not roi_coordinates and required == 1:
            roi_coordinates = [(0, 0, image.shape[1], image.shape[0])]
            detection = {"method": "whole_image_single_roi"}
        if detection is not None:
            agent_trace.append(
                {
                    "stage": "automatic_roi_detection",
                    "title": "调用OpenCV视觉工具",
                    "status": "success",
                    "detail": {
                        "tool_call": TOOLS[tool]["vision_tool"],
                        "roi_count": len(roi_coordinates),
                        "result": detection,
                    },
                }
            )
        else:
            agent_trace.append(
                {
                    "stage": "manual_roi_validation",
                    "title": "采用人工修正ROI",
                    "status": "success",
                    "detail": {
                        "tool_call": "manual.roi_override",
                        "roi_count": len(roi_coordinates),
                    },
                }
            )
            if annotated_path is not None and roi_coordinates:
                label_prefix = "SFR" if tool == "sfr" else "ROI"
                save_annotated_rois(
                    image,
                    roi_coordinates,
                    annotated_path,
                    labels=[
                        f"{label_prefix} {index}"
                        for index in range(1, len(roi_coordinates) + 1)
                    ],
                    center_markers=tool in {"color_check", "step_chart"},
                )
        expected = required
        if tool == "sfr" and detection is not None:
            expected = len(roi_coordinates)
        elif tool == "sfr":
            expected = len(roi_coordinates)
        if len(roi_coordinates) != expected:
            raise WorkflowError(
                f"{tool} 需要 {expected} 个 ROI，目前收到 {len(roi_coordinates)} 个。"
            )
        rois = crop_rois(image, roi_coordinates)
        runners = {
            "sfr": run_sfr,
            "distortion": run_distortion,
            "color_check": run_color_check,
            "step_chart": run_step_chart,
            "white_balance": run_white_balance,
        }
        result = runners[tool](
            rois,
            library_root,
            artifact_dir=annotated_path.parent if annotated_path is not None else None,
        )
        used_rois = len(rois)
    agent_trace.append(
        {
            "stage": "tool_execution",
            "title": "调用光学评价工具",
            "status": "success",
            "detail": {
                "tool": tool,
                "tool_call": TOOLS[tool]["measurement_tool"],
                "roi_count": used_rois,
            },
        }
    )
    agent_trace.append(
        {
            "stage": "result_validation",
            "title": "校验测量结果",
            "status": "success",
            "detail": {"assessment": result.get("assessment", "")},
        }
    )
    agent_response = synthesize_agent_response(tool, result, detection, used_rois)
    agent_trace.append(
        {
            "stage": "result_synthesis",
            "title": "整理自然语言回答",
            "status": "success",
            "detail": {"response": agent_response},
        }
    )

    return {
        "success": True,
        "request": message,
        "image": str(Path(image_path).resolve()),
        "image_shape": list(image.shape),
        "intent": intent,
        "tool_description": TOOLS[tool]["description"],
        "agent_plan": agent_plan,
        "roi_count": used_rois,
        "roi_coordinates": [list(roi) for roi in roi_coordinates],
        "automatic_detection": detection,
        "agent_trace": agent_trace,
        "agent_response": agent_response,
        "opencv_output": (
            str(annotated_path.resolve())
            if annotated_path is not None and annotated_path.is_file()
            else None
        ),
        "result": result,
    }


def format_report(payload: dict[str, Any]) -> str:
    intent = payload["intent"]
    result = payload["result"]
    lines = [
        "=" * 60,
        "任务三：语言驱动像质评价",
        "=" * 60,
        f"识别意图：{intent['tool']}（{payload['tool_description']}）",
        f"识别方式：{intent['method']}，置信度：{intent['confidence']:.0%}",
        (
            "Agent计划："
            f"{payload['agent_plan']['vision_tool']} -> "
            f"{payload['agent_plan']['measurement_tool']}"
        ),
        f"输入图片：{payload['image']}",
        f"图片尺寸：{payload['image_shape']}，使用 ROI：{payload['roi_count']} 个",
        "-" * 60,
    ]
    if intent["tool"] == "sfr":
        for measurement in result["measurements"]:
            lines.append(f"ROI {measurement['roi']}:")
            for name, values in measurement["channels"].items():
                lines.append(
                    f"  {name}: MTF50={values['mtf50_cy_per_pixel']} cy/px, "
                    f"MTF30={values['mtf30_cy_per_pixel']} cy/px"
                )
    else:
        lines.append(json.dumps(result, ensure_ascii=False, indent=2))
    lines.extend(
        [
            "-" * 60,
            f"Agent回答：{payload['agent_response']}",
            f"OpenCV输出：{payload.get('opencv_output') or '无'}",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek 规划、OpenCV 定位、确定性光学算法测量的像质评价 Agent"
    )
    parser.add_argument("message", help='自然语言要求，例如“测一下这张图的 MTF”')
    parser.add_argument("image", help="待分析图片路径")
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        type=parse_roi,
        metavar="X,Y,W,H",
        help="ROI 坐标，可重复指定；单 ROI 工具省略时使用整张图",
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path(os.environ.get("PY_IMAGING_QUALITY_ROOT", DEFAULT_LIBRARY_ROOT)),
        help="包含 py_imaging_quality 包的目录",
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        help="保存 OpenCV 框选图；多任务时会在文件名后增加任务编号",
    )
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument("--output", type=Path, help="将 JSON 结果保存到文件")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            api_key = getpass.getpass(
                "请输入 DeepSeek API Key（输入内容不会显示，也不会写入结果文件）："
            ).strip()
        if not api_key:
            raise WorkflowError("必须提供 DeepSeek API Key 才能生成 Agent Workflow。")

        plan = plan_with_deepseek(args.message, api_key)
        task_results = []
        base_annotated = args.annotated or (
            Path.cwd() / f"{Path(args.image).stem}_Agent_OpenCV.jpg"
        )
        task_count = len(plan["tasks"])
        for task_index, task in enumerate(plan["tasks"], start=1):
            if task_count == 1:
                annotated_path = base_annotated
            else:
                suffix = base_annotated.suffix or ".jpg"
                annotated_path = base_annotated.with_name(
                    f"{base_annotated.stem}_{task_index:02d}_{task['tool']}{suffix}"
                )
            try:
                task_payload = execute(
                    task["request"],
                    args.image,
                    args.roi if task["tool"] == "sfr" else [],
                    args.library_root,
                    annotated_path=annotated_path,
                    forced_tool=task["tool"],
                    task_parameters=task.get("parameters"),
                    intent_method="deepseek",
                )
                task_results.append(task_payload)
            except Exception as exc:
                task_results.append(
                    {
                        "success": False,
                        "tool": task["tool"],
                        "request": task["request"],
                        "error": (
                            str(exc)
                            if isinstance(exc, WorkflowError)
                            else f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )
        if not any(item.get("success") for item in task_results):
            raise WorkflowError("Agent 计划中的全部测量工具均执行失败。")
        agent_response = summarize_with_deepseek(
            args.message,
            plan,
            task_results,
            api_key,
        )
        payload = {
            "success": True,
            "request": args.message,
            "image": str(Path(args.image).resolve()),
            "intent": {
                "method": "deepseek",
                "model": plan["model"],
                "tools": [task["tool"] for task in plan["tasks"]],
            },
            "agent_plan": plan,
            "task_results": task_results,
            "agent_response": agent_response,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("=" * 60)
            print("任务三：DeepSeek 语言驱动像质评价 Agent")
            print("=" * 60)
            print(f"模型：{plan['model']}")
            print(
                "Agent计划："
                + " -> ".join(task["tool"] for task in plan["tasks"])
            )
            for task_index, task_result in enumerate(task_results, start=1):
                print("-" * 60)
                print(
                    f"任务 {task_index}：{task_result.get('tool') or task_result['intent']['tool']}"
                )
                if task_result.get("success"):
                    print(format_report(task_result))
                else:
                    print(f"执行失败：{task_result['error']}")
            print("-" * 60)
            print(f"DeepSeek最终回答：{agent_response}")
        return 0
    except WorkflowError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[错误] 算法执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
