from pathlib import Path

from task3_agent import (
    DEFAULT_LIBRARY_ROOT,
    auto_detect_colorchecker_rois,
    auto_detect_sfr_rectangle_edges,
    auto_detect_step_chart_rois,
    load_image,
)


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "sample_images"


def main() -> None:
    assert (DEFAULT_LIBRARY_ROOT / "py_imaging_quality").is_dir(), (
        "py_imaging_quality package was not found next to task3_agent.py"
    )

    composite = load_image(SAMPLES / "2014_3_composite.JPG")
    gray_rois, gray_meta = auto_detect_step_chart_rois(composite, 20)
    assert len(gray_rois) == 20, f"expected 20 gray ROIs, got {len(gray_rois)}"
    print("[OK] 20-step gray chart:", gray_meta.get("method"))

    edge_rois, edge_meta = auto_detect_sfr_rectangle_edges(
        composite,
        field_position="edge",
    )
    assert len(edge_rois) >= 4, f"expected edge MTF ROIs, got {len(edge_rois)}"
    print(
        "[OK] edge MTF ROIs:",
        len(edge_rois),
        "targets=",
        sorted({item.get("target_rank") for item in edge_meta}),
    )

    color_image = load_image(SAMPLES / "colorD65.bmp")
    color_rois, color_meta = auto_detect_colorchecker_rois(color_image)
    assert len(color_rois) == 24, f"expected 24 color ROIs, got {len(color_rois)}"
    print("[OK] ColorChecker:", color_meta.get("method"))

    print("\nSmoke test passed. You can run start_web.bat now.")


if __name__ == "__main__":
    main()
