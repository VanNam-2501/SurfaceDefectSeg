from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER = ["U-Net", "SegFormer", "VMamba"]
MODEL_COLORS = {
    "U-Net": "#3B6FB6",
    "SegFormer": "#E07A2D",
    "VMamba": "#2D8B73",
}
FAMILY_COLORS = {
    "Ngưỡng pixel gốc": "#8795A1",
    "Adaptive": "#3B6FB6",
    "Learned": "#9C6ADE",
    "Hybrid": "#2D8B73",
}
OUTCOME_COLORS = {
    "TN": "#86BFA6",
    "TP": "#356E5C",
    "FP": "#E7A54A",
    "FN": "#C95A5A",
}
GRID_COLOR = "#D9E1E8"
TEXT_COLOR = "#20303C"
MUTED_COLOR = "#60717D"
BACKGROUND = "#F7F9FB"


def clean_model(value: object) -> str:
    text_value = str(value).strip()
    lowered = text_value.lower()
    if "unet" in lowered or "u-net" in lowered:
        return "U-Net"
    if "segformer" in lowered:
        return "SegFormer"
    if "mamba" in lowered:
        return "VMamba"
    return text_value


def pct(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def pct_or_identity(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.nanmax(np.abs(arr)) <= 1.000001:
        return arr * 100.0
    return arr


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": "white",
            "savefig.facecolor": BACKGROUND,
            "axes.edgecolor": "#BAC7D1",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.titlesize": 17,
            "figure.titleweight": "bold",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    if y_grid:
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
    ax.spines["left"].set_color("#BAC7D1")
    ax.spines["bottom"].set_color("#BAC7D1")


def percent_axis(ax: plt.Axes, upper: float = 100.0) -> None:
    ax.set_ylim(0, upper)
    ax.set_ylabel("Tỷ lệ (%)")
    style_axis(ax)


def annotate_bars(ax: plt.Axes, *, decimals: int = 1, suffix: str = "", fontsize: int = 8) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            if not np.isfinite(height):
                labels.append("")
            else:
                labels.append(f"{height:.{decimals}f}{suffix}")
        ax.bar_label(container, labels=labels, padding=2, fontsize=fontsize, color=TEXT_COLOR)


def grouped_bars(
    ax: plt.Axes,
    x_labels: list[str],
    series: list[tuple[str, np.ndarray, str]],
    *,
    ylabel: str = "Tỷ lệ (%)",
    ylim: tuple[float, float] | None = None,
    rotate: int = 0,
    annotate: bool = True,
) -> None:
    x = np.arange(len(x_labels), dtype=float)
    width = 0.78 / max(1, len(series))
    for idx, (name, values, color) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width
        ax.bar(x + offset, values, width * 0.9, label=name, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)
    if annotate:
        annotate_bars(ax)


@dataclass
class ChartEntry:
    section: str
    slug: str
    title: str
    description: str
    sources: str
    png: str
    svg: str | None


class Exporter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.png_dir = out_dir / "figures" / "png"
        self.svg_dir = out_dir / "figures" / "svg"
        self.table_dir = out_dir / "tables"
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.svg_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[ChartEntry] = []

    def save(
        self,
        fig: plt.Figure,
        *,
        section: str,
        slug: str,
        title: str,
        description: str,
        sources: str,
        svg: bool = True,
    ) -> None:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        png_path = self.png_dir / f"{slug}.png"
        fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.16)
        svg_path: Path | None = None
        if svg:
            svg_path = self.svg_dir / f"{slug}.svg"
            fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.16)
        plt.close(fig)
        self.entries.append(
            ChartEntry(
                section=section,
                slug=slug,
                title=title,
                description=description,
                sources=sources,
                png=png_path.relative_to(self.out_dir).as_posix(),
                svg=svg_path.relative_to(self.out_dir).as_posix() if svg_path else None,
            )
        )

    def register_raster(
        self,
        *,
        section: str,
        slug: str,
        title: str,
        description: str,
        sources: str,
        png_path: Path,
    ) -> None:
        self.entries.append(
            ChartEntry(
                section=section,
                slug=slug,
                title=title,
                description=description,
                sources=sources,
                png=png_path.relative_to(self.out_dir).as_posix(),
                svg=None,
            )
        )


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    thesis_tables = root / "thesis_evaluation_report" / "tables"
    decision_root = root / "decision_and_test_audit"
    data_root = root / "full_dataset_data_audit"
    paths = {
        "e2": thesis_tables / "01_e2_architecture_comparison.csv",
        "e3": thesis_tables / "02_e3_defect_size.csv",
        "e4": thesis_tables / "03_e4_defect_group.csv",
        "e5": thesis_tables / "04_e5_multi_region.csv",
        "e7": thesis_tables / "05_e7_thresholds_from_validation.csv",
        "e8": thesis_tables / "06_e8_qualitative_manifest.csv",
        "base": decision_root / "tables" / "00_base_model_segmentation_test.csv",
        "automatic": decision_root / "tables" / "02_fully_automatic_comparison.csv",
        "outcomes": decision_root / "tables" / "03_test_case_outcome_summary.csv",
        "adaptive_group": decision_root / "adaptive_single" / "adaptive_defect_group_test.csv",
        "learned_group": decision_root / "learned_all3" / "defect_group_comparison.csv",
        "hybrid_um_group": decision_root / "hybrid_pairs" / "unet_vmamba" / "defect_group_comparison.csv",
        "all_images": data_root / "01_all_images_frozen_policy.csv",
        "audit_reasons": data_root / "03_summary_by_split_and_reason.csv",
        "integrity": data_root / "04_mask_integrity_issues.csv",
        "split_outcomes": data_root / "05_frozen_policy_model_outcomes_by_split.csv",
    }
    for model in ("unet", "segformer", "vmamba"):
        paths[f"scan_{model}"] = decision_root / "adaptive_single" / f"{model}_adaptive_scan.csv"

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Thiếu file đầu vào:\n" + "\n".join(missing))

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    frames["_paths"] = pd.DataFrame({"key": list(paths), "path": [str(p) for p in paths.values()]})
    return frames


def prepare_models(frame: pd.DataFrame, column: str = "model") -> pd.DataFrame:
    result = frame.copy()
    result[column] = result[column].map(clean_model)
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    return result.sort_values(column, key=lambda values: values.map(lambda item: order.get(item, 999)))


def plot_e2_overall(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e2"])
    metrics = [
        ("AUROC ảnh", "image_auroc"),
        ("AUPRC ảnh", "image_auprc"),
        ("F1 ảnh", "image_f1"),
        ("Dice lỗi", "positive_dice"),
        ("IoU lỗi", "positive_iou"),
        ("Recall vùng", "region_recall_any_overlap"),
    ]
    series = []
    for _, row in frame.iterrows():
        model = row["model"]
        series.append((model, np.array([row[col] for _, col in metrics]) * 100, MODEL_COLORS[model]))
    fig, ax = plt.subplots(figsize=(14, 6.2))
    grouped_bars(ax, [label for label, _ in metrics], series, ylim=(0, 104), annotate=True)
    ax.legend(ncol=3, loc="upper center")
    ax.set_title("E2 · So sánh hiệu năng tổng thể trên tập Test")
    fig.suptitle("Ba kiến trúc segmentation", y=1.015)
    exporter.save(
        fig,
        section="E2 · So sánh kiến trúc",
        slug="01_e2_overall_metrics",
        title="Hiệu năng tổng thể của ba model",
        description="So sánh đồng thời phân loại ảnh, chất lượng mask và khả năng bắt vùng lỗi.",
        sources="01_e2_architecture_comparison.csv",
    )


def plot_e2_error_rates(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e2"])
    series = [
        ("FNR · bỏ sót", pct(frame["image_fnr"]), "#C95A5A"),
        ("FPR · báo động giả", pct(frame["image_fpr"]), "#E7A54A"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    grouped_bars(ax, frame["model"].tolist(), series, ylim=(0, 70), annotate=True)
    ax.legend(ncol=2, loc="upper left")
    ax.set_title("Sai số ở mức ảnh: bỏ sót và báo động giả")
    ax.text(
        0.99,
        0.97,
        "Càng thấp càng tốt",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=MUTED_COLOR,
    )
    exporter.save(
        fig,
        section="E2 · So sánh kiến trúc",
        slug="02_e2_error_rates",
        title="FNR và FPR của từng model",
        description="VMamba có FPR thấp nhất trong ba model gốc; SegFormer có FNR thấp nhất nhưng FPR cao.",
        sources="01_e2_architecture_comparison.csv",
    )


def plot_e2_confusions(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e2"])
    cmap = LinearSegmentedColormap.from_list("confusion", ["#EEF4F2", "#2D8B73"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (_, row) in zip(axes, frame.iterrows()):
        matrix = np.array([[row["image_tn"], row["image_fp"]], [row["image_fn"], row["image_tp"]]], dtype=float)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
        for y in range(2):
            for x in range(2):
                color = "white" if normalized[y, x] > 0.58 else TEXT_COLOR
                ax.text(
                    x,
                    y,
                    f"{int(matrix[y, x])}\n{normalized[y, x] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1], ["Dự báo Good", "Dự báo Defect"])
        ax.set_yticks([0, 1], ["GT Good", "GT Defect"])
        ax.set_title(row["model"])
        ax.set_xlabel("Dự báo")
        if ax is axes[0]:
            ax.set_ylabel("Ground truth")
    fig.suptitle("E2 · Ma trận nhầm lẫn trên tập Test", y=1.02)
    exporter.save(
        fig,
        section="E2 · So sánh kiến trúc",
        slug="03_e2_confusion_matrices",
        title="Ma trận nhầm lẫn",
        description="Mỗi ô thể hiện số ảnh và tỷ lệ theo từng nhãn thật.",
        sources="01_e2_architecture_comparison.csv",
    )


def plot_e2_pixel_metrics(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e2"])
    metrics = [
        ("Pixel AUPRC", "pixel_auprc_hist"),
        ("Pixel Recall", "pixel_recall"),
        ("Pixel Precision", "pixel_precision"),
    ]
    series = []
    for _, row in frame.iterrows():
        model = row["model"]
        series.append((model, np.array([row[col] for _, col in metrics]) * 100, MODEL_COLORS[model]))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    grouped_bars(ax, [label for label, _ in metrics], series, ylim=(0, 100), annotate=True)
    ax.legend(ncol=3, loc="upper center")
    ax.set_title("E2 · Chất lượng dự báo ở mức pixel")
    exporter.save(
        fig,
        section="E2 · So sánh kiến trúc",
        slug="04_e2_pixel_metrics",
        title="Precision, Recall và AUPRC ở mức pixel",
        description="VMamba dẫn đầu đồng thời cả ba chỉ số pixel trong kết quả hiện tại.",
        sources="01_e2_architecture_comparison.csv",
    )


def plot_e2_efficiency(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e2"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    colors = [MODEL_COLORS[m] for m in frame["model"]]
    axes[0].bar(frame["model"], frame["full_image_latency_mean_seconds"] * 1000, color=colors)
    axes[0].set_ylabel("Thời gian trung bình (ms/ảnh)")
    axes[0].set_title("Độ trễ toàn ảnh")
    style_axis(axes[0])
    annotate_bars(axes[0], decimals=0)
    axes[1].bar(frame["model"], frame["tiles_per_second"], color=colors)
    axes[1].set_ylabel("Số tile / giây")
    axes[1].set_title("Thông lượng inference")
    style_axis(axes[1])
    annotate_bars(axes[1], decimals=1)
    fig.suptitle("Phụ lục · Hiệu năng suy luận trong cùng quy trình đánh giá", y=1.02)
    exporter.save(
        fig,
        section="Phụ lục hiệu năng",
        slug="05_inference_efficiency",
        title="Độ trễ và thông lượng inference",
        description="Số liệu được lấy trực tiếp từ cùng pipeline đánh giá; dùng làm thông tin bổ sung, không thay cho benchmark triển khai độc lập.",
        sources="01_e2_architecture_comparison.csv",
    )


def line_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_order: list[str],
    title: str,
    ylabel: str = "Tỷ lệ (%)",
) -> None:
    for model in MODEL_ORDER:
        subset = frame[frame["model"] == model].copy()
        subset["_order"] = subset[x_col].map({name: idx for idx, name in enumerate(x_order)})
        subset = subset.sort_values("_order")
        ax.plot(
            subset[x_col],
            pct(subset[y_col]),
            marker="o",
            linewidth=2.3,
            markersize=6,
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(45, 102)
    style_axis(ax)


def plot_e3_size(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e3"])
    size_order = ["Tiny", "Small", "Medium", "Large"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    line_panel(axes[0], frame, "size_bin", "region_recall", size_order, "Recall theo từng vùng lỗi")
    line_panel(axes[1], frame, "size_bin", "image_recall", size_order, "Recall ảnh theo lỗi nhỏ nhất")
    axes[0].legend(ncol=3, loc="lower right")
    fig.suptitle("E3 · Khả năng phát hiện theo kích thước bất thường", y=1.02)
    exporter.save(
        fig,
        section="E3 · Kích thước lỗi",
        slug="06_e3_size_recall",
        title="Recall theo kích thước lỗi",
        description="Tách rõ khả năng bắt từng vùng lỗi và khả năng kết luận đúng ở mức ảnh.",
        sources="02_e3_defect_size.csv",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    line_panel(axes[0], frame, "size_bin", "positive_dice", size_order, "Positive Dice")
    line_panel(axes[1], frame, "size_bin", "positive_iou", size_order, "Positive IoU")
    axes[0].legend(ncol=3, loc="lower right")
    fig.suptitle("E3 · Chất lượng mask theo kích thước bất thường", y=1.02)
    exporter.save(
        fig,
        section="E3 · Kích thước lỗi",
        slug="07_e3_size_mask_quality",
        title="Dice và IoU theo kích thước lỗi",
        description="Cho thấy model nào giữ chất lượng mask tốt hơn với lỗi Tiny, Small, Medium và Large.",
        sources="02_e3_defect_size.csv",
    )


def plot_e4_group(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e4"])
    groups = frame["group"].drop_duplicates().tolist()
    metric_specs = [("image_recall", "Image Recall"), ("positive_dice", "Positive Dice"), ("positive_iou", "Positive IoU")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    for ax, (metric, title) in zip(axes, metric_specs):
        series = []
        for model in MODEL_ORDER:
            subset = frame[frame["model"] == model].set_index("group").reindex(groups)
            series.append((model, pct(subset[metric]), MODEL_COLORS[model]))
        grouped_bars(ax, groups, series, ylim=(0, 108), rotate=28, annotate=False)
        ax.set_title(title)
    axes[0].legend(ncol=3, loc="lower left")
    fig.suptitle("E4 · Hiệu năng theo nhóm khuyết tật", y=1.02)
    exporter.save(
        fig,
        section="E4 · Nhóm khuyết tật",
        slug="08_e4_defect_group_performance",
        title="Recall, Dice và IoU theo nhóm lỗi",
        description="Nhóm knife mark chỉ có 2 ảnh nên cần diễn giải thận trọng dù chỉ số cao.",
        sources="03_e4_defect_group.csv",
    )

    counts = frame.groupby("group", as_index=False).agg(n_images=("n_images", "max"), warning=("small_sample_warning", "max"))
    counts = counts.sort_values("n_images")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bar_colors = ["#C95A5A" if bool(value) else "#3B6FB6" for value in counts["warning"]]
    ax.barh(counts["group"], counts["n_images"], color=bar_colors)
    ax.set_xlabel("Số ảnh Defect")
    ax.set_title("E4 · Quy mô mẫu của từng nhóm khuyết tật")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for y, value in enumerate(counts["n_images"]):
        ax.text(value + max(counts["n_images"]) * 0.012, y, str(int(value)), va="center", fontsize=9)
    ax.legend(
        handles=[Patch(color="#3B6FB6", label="Đủ mẫu hơn"), Patch(color="#C95A5A", label="Cảnh báo ít mẫu")],
        loc="lower right",
    )
    exporter.save(
        fig,
        section="E4 · Nhóm khuyết tật",
        slug="09_e4_group_sample_sizes",
        title="Phân bố số mẫu theo nhóm lỗi",
        description="Biểu đồ này phải đi kèm biểu đồ metric để tránh kết luận mạnh từ nhóm có quá ít mẫu.",
        sources="03_e4_defect_group.csv",
    )


def plot_e5_multi_region(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e5"])
    order = ["single", "few", "many"]
    display = {"single": "1 vùng", "few": "2–3 vùng", "many": "≥4 vùng"}
    metrics = [("image_recall", "Image Recall"), ("region_recall", "Region Recall"), ("positive_dice", "Positive Dice")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for model in MODEL_ORDER:
            subset = frame[frame["model"] == model].set_index("multi_region_bin").reindex(order)
            ax.plot(
                [display[item] for item in order],
                pct(subset[metric]),
                marker="o",
                linewidth=2.3,
                color=MODEL_COLORS[model],
                label=model,
            )
        ax.set_title(title)
        ax.set_ylim(60, 102)
        ax.set_ylabel("Tỷ lệ (%)")
        style_axis(ax)
    axes[0].legend(ncol=3, loc="lower left")
    fig.suptitle("E5 · Ảnh có một hoặc nhiều vùng bất thường", y=1.02)
    exporter.save(
        fig,
        section="E5 · Nhiều vùng lỗi",
        slug="10_e5_multi_region_performance",
        title="Hiệu năng theo số vùng lỗi trong ảnh",
        description="So sánh khả năng kết luận ảnh, bắt đủ vùng và chất lượng mask khi độ phức tạp tăng.",
        sources="04_e5_multi_region.csv",
    )


def plot_e7_thresholds(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = prepare_models(data["e7"])
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = [MODEL_COLORS[m] for m in frame["model"]]
    ax.bar(frame["model"], frame["threshold"], color=colors, width=0.62)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ngưỡng probability")
    ax.set_title("E7 · Ngưỡng segmentation được chọn trên Validation")
    style_axis(ax)
    annotate_bars(ax, decimals=2)
    ax.text(
        0.99,
        0.95,
        "Chỉ chọn bằng Validation\nTest chỉ dùng để báo cáo cuối",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED_COLOR,
    )
    exporter.save(
        fig,
        section="E7 · Chọn ngưỡng",
        slug="11_e7_selected_thresholds",
        title="Ngưỡng riêng của từng kiến trúc",
        description="Mỗi model có calibration khác nhau nên không dùng chung threshold 0.5.",
        sources="05_e7_thresholds_from_validation.csv",
    )


def experiment_label(row: pd.Series) -> str:
    raw = str(row.get("experiment_id", row.get("experiment", row.get("name", ""))))
    policy = str(row.get("policy", ""))
    branch = str(row.get("branch", ""))
    model_set = str(row.get("model_set", row.get("models_used", "")))
    joined = " ".join([raw, policy, branch, model_set]).lower()
    model_tokens = []
    for token, display in [("unet", "U"), ("segformer", "S"), ("vmamba", "M")]:
        if token in joined:
            model_tokens.append(display)
    combo = "+".join(model_tokens)
    if "adaptive" in joined:
        return f"Adaptive {combo}".strip()
    if "hybrid" in joined:
        return f"Hybrid {combo}".strip()
    if "learned_all3__fusion" in joined or ("fusion" in joined and len(model_tokens) >= 3):
        return "Learned fusion U+S+M"
    if "learned" in joined or "fully_automatic" in joined:
        return f"Learned {combo}".strip()
    return raw.replace("__", " · ").replace("_", " ")


def experiment_family(row: pd.Series) -> str:
    declared = str(row.get("family", "")).lower()
    joined = " ".join(str(value).lower() for value in row.values)
    if "raw_segmentation" in declared or "base__" in joined:
        return "Ngưỡng pixel gốc"
    if "adaptive" in joined:
        return "Adaptive"
    if "hybrid" in joined:
        return "Hybrid"
    if "learned" in joined or "fusion" in joined:
        return "Learned"
    return "Ngưỡng pixel gốc"


def automatic_with_base(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    automatic = data["automatic"].copy()
    automatic["label"] = automatic.apply(experiment_label, axis=1)
    automatic["family"] = automatic.apply(experiment_family, axis=1)
    automatic["fnr_pct"] = pct_or_identity(automatic["test_auto_defect_fnr_pct"])
    automatic["fpr_pct"] = pct_or_identity(automatic["test_auto_defect_fpr_pct"])
    automatic["accuracy_pct"] = pct_or_identity(automatic["test_accuracy_pct"])

    base = data["base"].copy()
    base["model"] = base["model"].map(clean_model)
    base_frame = pd.DataFrame(
        {
            "label": "Gốc " + base["model"],
            "family": "Ngưỡng pixel gốc",
            "fnr_pct": pct_or_identity(base["fnr_pct"]),
            "fpr_pct": pct_or_identity(base["fpr_pct"]),
            "accuracy_pct": ((394 - base["false_negatives"]) + (323 - base["false_positives"])) / 717 * 100,
        }
    )
    return pd.concat([base_frame, automatic[["label", "family", "fnr_pct", "fpr_pct", "accuracy_pct"]]], ignore_index=True)


def plot_decision_tradeoff(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = automatic_with_base(data)
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    markers = {"Ngưỡng pixel gốc": "o", "Adaptive": "s", "Learned": "D", "Hybrid": "^"}
    label_offsets = {
        "Learned M": (7, -16),
        "Hybrid S+M": (7, 12),
        "Learned fusion U+S+M": (7, -18),
        "Hybrid U+M": (7, 10),
        "Learned U": (7, 8),
        "Adaptive U": (7, -15),
    }
    for family, group in frame.groupby("family", sort=False):
        ax.scatter(
            group["fpr_pct"],
            group["fnr_pct"],
            s=92,
            marker=markers[family],
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.8,
            label=family,
            zorder=3,
        )
        for _, row in group.iterrows():
            offset = label_offsets.get(row["label"], (6, 5))
            ax.annotate(
                row["label"],
                (row["fpr_pct"], row["fnr_pct"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.2,
                color=TEXT_COLOR,
            )
    ax.set_xlabel("FPR · Báo động giả (%)")
    ax.set_ylabel("FNR · Bỏ sót bất thường (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 5.8)
    style_axis(ax)
    ax.legend(ncol=2, loc="upper right")
    ax.set_title("Logic tự động · Đánh đổi giữa báo động giả và bỏ sót trên Test")
    ax.text(
        0.015,
        0.97,
        "Vùng góc trái dưới tốt hơn",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED_COLOR,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID_COLOR),
    )
    exporter.save(
        fig,
        section="Logic quyết định tự động",
        slug="12_decision_fpr_fnr_tradeoff",
        title="Trade-off FPR–FNR của toàn bộ logic tự động",
        description="Biểu đồ trung tâm để chọn cấu hình: điểm càng gần góc trái dưới càng cân bằng.",
        sources="00_base_model_segmentation_test.csv; 02_fully_automatic_comparison.csv",
    )


def plot_strategy_by_model(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    full = automatic_with_base(data)
    rows: list[dict[str, object]] = []
    for model, short in [("U-Net", "U"), ("SegFormer", "S"), ("VMamba", "M")]:
        selectors = {
            "Gốc": full["label"] == f"Gốc {model}",
            "Adaptive": full["label"] == f"Adaptive {short}",
            "Learned": full["label"] == f"Learned {short}",
        }
        for strategy, mask in selectors.items():
            subset = full[mask]
            if not subset.empty:
                row = subset.iloc[0]
                rows.append(
                    {
                        "model": model,
                        "strategy": strategy,
                        "FNR": row["fnr_pct"],
                        "FPR": row["fpr_pct"],
                    }
                )
    frame = pd.DataFrame(rows)
    strategy_colors = {"Gốc": "#8795A1", "Adaptive": "#3B6FB6", "Learned": "#9C6ADE"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        subset = frame[frame["model"] == model]
        x = np.arange(len(subset))
        width = 0.34
        ax.bar(x - width / 2, subset["FPR"], width, label="FPR", color="#E7A54A")
        ax.bar(x + width / 2, subset["FNR"], width, label="FNR", color="#C95A5A")
        ax.set_xticks(x, subset["strategy"])
        ax.set_title(model, color=MODEL_COLORS[model])
        ax.set_ylabel("Tỷ lệ (%)")
        ax.set_ylim(0, 68)
        style_axis(ax)
        annotate_bars(ax, decimals=1, fontsize=8)
        for label in ax.get_xticklabels():
            label.set_color(strategy_colors.get(label.get_text(), TEXT_COLOR))
    axes[0].legend(ncol=2, loc="upper left")
    fig.suptitle("Model riêng · Logic hậu xử lý giảm FPR như thế nào?", y=1.02)
    exporter.save(
        fig,
        section="Logic quyết định tự động",
        slug="13_single_model_strategy_comparison",
        title="So sánh ngưỡng gốc, adaptive và learned cho từng model riêng",
        description="Thể hiện trực tiếp mức giảm báo động giả và phần đánh đổi bằng bỏ sót.",
        sources="00_base_model_segmentation_test.csv; 02_fully_automatic_comparison.csv",
    )


def plot_outcomes(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = data["outcomes"].copy()
    frame["label"] = frame.apply(experiment_label, axis=1)
    frame["family"] = frame.apply(experiment_family, axis=1)
    frame["errors"] = frame["false_alarm"] + frame["missed_defect"] if "false_alarm" in frame else frame["fp"] + frame["fn"]
    count_columns = {
        "TN": "correct_pass" if "correct_pass" in frame else "tn",
        "TP": "correct_defect" if "correct_defect" in frame else "tp",
        "FP": "false_alarm" if "false_alarm" in frame else "fp",
        "FN": "missed_defect" if "missed_defect" in frame else "fn",
    }
    frame = frame.sort_values("errors", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 8.2))
    y = np.arange(len(frame))
    left = np.zeros(len(frame))
    for outcome in ("TN", "TP", "FP", "FN"):
        values = frame[count_columns[outcome]].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=OUTCOME_COLORS[outcome], label=outcome, height=0.72)
        left += values
    ax.set_yticks(y, frame["label"])
    ax.set_xlabel("Số ảnh Test")
    ax.set_title("Kết quả từng ảnh: đúng, báo động giả và bỏ sót")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="lower right")
    exporter.save(
        fig,
        section="Logic quyết định tự động",
        slug="14_test_case_outcomes",
        title="Phân rã TP, TN, FP, FN của từng thí nghiệm",
        description="Cho biết thay đổi FPR/FNR tương ứng chính xác bao nhiêu ảnh trong 717 ảnh Test.",
        sources="03_test_case_outcome_summary.csv",
    )


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    points = frame[["fpr", "fnr"]].dropna().sort_values(["fpr", "fnr"])
    best_fnr = math.inf
    keep: list[int] = []
    for idx, row in points.iterrows():
        if row["fnr"] < best_fnr - 1e-12:
            keep.append(idx)
            best_fnr = row["fnr"]
    return frame.loc[keep].sort_values("fpr")


def plot_validation_pareto(data: dict[str, pd.DataFrame], root: Path, exporter: Exporter) -> None:
    policy_path = root / "decision_and_test_audit" / "adaptive_single" / "adaptive_component_policy.json"
    with policy_path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True, sharey=True)
    for ax, model, display in zip(axes, ("unet", "segformer", "vmamba"), MODEL_ORDER):
        scan = data[f"scan_{model}"].copy()
        sample = scan.sample(min(2600, len(scan)), random_state=42)
        ax.scatter(pct(sample["fpr"]), pct(sample["fnr"]), s=10, alpha=0.16, color=MODEL_COLORS[display], linewidth=0)
        frontier = pareto_front(scan)
        ax.plot(pct(frontier["fpr"]), pct(frontier["fnr"]), color="#1E2C36", linewidth=2, label="Pareto frontier")
        chosen = policy["models"][model]["validation_metrics"]
        ax.scatter(chosen["fpr"] * 100, chosen["fnr"] * 100, s=130, marker="*", color="#E7A54A", edgecolor="white", linewidth=0.8, zorder=5, label="Cấu hình đã chọn")
        ax.axhline(policy["validation_target_max_fnr"] * 100, color="#C95A5A", linestyle="--", linewidth=1.2, label="Giới hạn FNR")
        ax.set_title(display)
        ax.set_xlabel("FPR Validation (%)")
        ax.set_ylabel("FNR Validation (%)")
        ax.set_xlim(0, 88)
        ax.set_ylim(0, 32)
        style_axis(ax)
    axes[0].legend(loc="upper right")
    fig.suptitle("Validation · Quét 10.720 cấu hình adaptive cho mỗi model", y=1.02)
    exporter.save(
        fig,
        section="Logic quyết định tự động",
        slug="15_validation_adaptive_pareto",
        title="Không gian FPR–FNR và cấu hình được chọn trên Validation",
        description="Ngôi sao là cấu hình được khóa trước khi đánh giá Test; đường đen là biên Pareto.",
        sources="*_adaptive_scan.csv; adaptive_component_policy.json",
    )


def plot_automatic_group_recall(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    adaptive = data["adaptive_group"].copy()
    adaptive["model"] = adaptive["model"].map(clean_model)
    adaptive_vmamba = adaptive[adaptive["model"] == "VMamba"].copy()
    adaptive_vmamba["method"] = "Adaptive VMamba"
    adaptive_vmamba = adaptive_vmamba.rename(columns={"defect_group": "group", "recall": "automatic_recall"})

    learned = data["learned_group"]
    learned = learned[learned["branch"].astype(str).str.lower() == "fusion"].copy()
    learned["method"] = "Learned fusion U+S+M"
    learned = learned.rename(columns={"defect_group": "group"})

    hybrid = data["hybrid_um_group"]
    hybrid = hybrid[hybrid["branch"].astype(str).str.lower() == "hybrid_fusion"].copy()
    hybrid["method"] = "Hybrid U+M"
    hybrid = hybrid.rename(columns={"defect_group": "group"})

    frame = pd.concat(
        [
            adaptive_vmamba[["method", "group", "automatic_recall", "positive_images"]],
            learned[["method", "group", "automatic_recall", "positive_images"]],
            hybrid[["method", "group", "automatic_recall", "positive_images"]],
        ],
        ignore_index=True,
    )
    groups = frame["group"].drop_duplicates().tolist()
    methods = ["Adaptive VMamba", "Learned fusion U+S+M", "Hybrid U+M"]
    colors = ["#3B6FB6", "#9C6ADE", "#2D8B73"]
    series = []
    for method, color in zip(methods, colors):
        subset = frame[frame["method"] == method].set_index("group").reindex(groups)
        series.append((method, pct(subset["automatic_recall"]), color))
    fig, ax = plt.subplots(figsize=(14, 6.2))
    grouped_bars(ax, groups, series, ylim=(0, 104), rotate=22, annotate=True)
    ax.legend(ncol=3, loc="lower left")
    ax.set_title("Logic tự động · Recall theo từng nhóm khuyết tật")
    exporter.save(
        fig,
        section="Logic quyết định tự động",
        slug="16_automatic_defect_group_recall",
        title="Độ nhạy theo loại lỗi của các logic nổi bật",
        description="Giúp kiểm tra việc giảm FPR có làm yếu riêng một nhóm khuyết tật hay không.",
        sources="adaptive_defect_group_test.csv; learned_all3/defect_group_comparison.csv; unet_vmamba/defect_group_comparison.csv",
    )


def plot_audit_reasons(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = data["audit_reasons"].copy()
    split_order = ["train", "val", "test"]
    reason_map = {
        "good_label_three_model_defect": "Good nhưng 3 model báo Defect",
        "good_partial_false_alarm": "Good nhưng một phần model báo Defect",
        "defect_label_three_model_pass": "Defect nhưng 3 model đều Pass",
        "defect_partial_miss": "Defect nhưng một phần model bỏ sót",
        "agree_defect": "Đồng thuận đúng Defect",
        "agree_pass": "Đồng thuận đúng Good",
    }
    frame["reason_label"] = frame["review_reason"].map(reason_map).fillna(frame["review_reason"])
    issue_order = [
        "Good nhưng 3 model báo Defect",
        "Good nhưng một phần model báo Defect",
        "Defect nhưng 3 model đều Pass",
        "Defect nhưng một phần model bỏ sót",
    ]
    issue = frame[frame["reason_label"].isin(issue_order)]
    pivot = issue.pivot_table(index="split", columns="reason_label", values="images", aggfunc="sum", fill_value=0).reindex(split_order).fillna(0)
    fig, ax = plt.subplots(figsize=(12.5, 6))
    bottom = np.zeros(len(pivot))
    issue_colors = ["#C95A5A", "#E7A54A", "#7E57C2", "#4E79A7"]
    for segment_idx, (label, color) in enumerate(zip(issue_order, issue_colors)):
        values = pivot[label].to_numpy() if label in pivot else np.zeros(len(pivot))
        ax.bar(pivot.index, values, bottom=bottom, label=label, color=color)
        for x, (base, value) in enumerate(zip(bottom, values)):
            if value >= 10:
                ax.text(x, base + value / 2, str(int(value)), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += values
    for x, total in enumerate(bottom):
        ax.text(x, total + max(bottom) * 0.012, f"Tổng {int(total)}", ha="center", va="bottom", fontsize=9, color=MUTED_COLOR)
    ax.set_ylabel("Số ảnh cần xem lại")
    ax.set_title("Audit toàn bộ dữ liệu · Các mẫu nghi vấn theo split")
    style_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    exporter.save(
        fig,
        section="Audit toàn bộ dữ liệu",
        slug="17_data_audit_suspicious_reasons",
        title="Nguồn gốc các mẫu nghi vấn trong train/val/test",
        description="Chỉ hiển thị nhóm bất đồng với nhãn; các ảnh model và nhãn đồng thuận không nằm trong cột nghi vấn.",
        sources="03_summary_by_split_and_reason.csv",
    )


def plot_audit_split_errors(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = data["split_outcomes"].copy()
    frame["model"] = frame["model"].map(clean_model)
    frame["fpr_pct"] = frame["false_alarm"] / (frame["false_alarm"] + frame["correct_pass"]) * 100
    frame["fnr_pct"] = frame["missed_defect"] / (frame["missed_defect"] + frame["correct_defect"]) * 100
    split_order = ["train", "val", "test"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharex=True)
    for ax, metric, title, color in [
        (axes[0], "fpr_pct", "FPR · Báo động giả", "#E7A54A"),
        (axes[1], "fnr_pct", "FNR · Bỏ sót", "#C95A5A"),
    ]:
        series = []
        for model in MODEL_ORDER:
            subset = frame[frame["model"] == model].set_index("split").reindex(split_order)
            series.append((model, subset[metric].to_numpy(), MODEL_COLORS[model]))
        grouped_bars(ax, [item.capitalize() for item in split_order], series, ylim=(0, 35 if metric == "fpr_pct" else 6), annotate=True)
        ax.set_title(title)
        ax.legend(ncol=3, loc="upper center")
    fig.suptitle("Frozen policy · Sai số trên toàn bộ Train / Validation / Test", y=1.02)
    exporter.save(
        fig,
        section="Audit toàn bộ dữ liệu",
        slug="18_frozen_policy_errors_by_split",
        title="Độ ổn định của policy trên ba split",
        description="Dùng cùng policy đã khóa để phát hiện split nào có phân bố lỗi khác thường.",
        sources="05_frozen_policy_model_outcomes_by_split.csv",
    )


def plot_vote_patterns(data: dict[str, pd.DataFrame], exporter: Exporter) -> None:
    frame = data["all_images"].copy()
    frame["vote_pattern"] = (
        frame["unet_decision"].map({"defect": "D", "pass": "P"}).fillna("?")
        + frame["segformer_decision"].map({"defect": "D", "pass": "P"}).fillna("?")
        + frame["vmamba_decision"].map({"defect": "D", "pass": "P"}).fillna("?")
    )
    counts = frame.groupby(["vote_pattern", "label_name"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.7))
    y = np.arange(len(counts))
    good = counts["Good"].to_numpy() if "Good" in counts else np.zeros(len(counts))
    defect = counts["Defect"].to_numpy() if "Defect" in counts else np.zeros(len(counts))
    ax.barh(y, good, color="#86BFA6", label="GT Good")
    ax.barh(y, defect, left=good, color="#C95A5A", label="GT Defect")
    readable = [pattern.replace("D", "Defect ").replace("P", "Pass ").strip() for pattern in counts.index]
    readable = [f"U/S/M = {value}" for value in readable]
    ax.set_yticks(y, readable)
    ax.set_xlabel("Số ảnh trong toàn bộ dữ liệu")
    ax.set_title("Mẫu đồng thuận của U-Net / SegFormer / VMamba")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower right")
    for y_pos, total in enumerate(counts["total"]):
        ax.text(total + counts["total"].max() * 0.008, y_pos, str(int(total)), va="center", fontsize=8)
    exporter.save(
        fig,
        section="Audit toàn bộ dữ liệu",
        slug="19_three_model_vote_patterns",
        title="Tám kiểu bỏ phiếu của ba model",
        description="Cho biết dữ liệu tập trung ở đồng thuận hay bất đồng, đồng thời tách theo nhãn Good/Defect.",
        sources="01_all_images_frozen_policy.csv",
    )


def make_error_contact_sheet(root: Path, exporter: Exporter) -> None:
    base = root / "decision_and_test_audit" / "test_case_audit" / "visual_gallery" / "errors_only" / "hybrid_unet_vmamba__hybrid_fusion__fully_automatic"
    categories = [("false_alarm", "Báo động giả"), ("missed_defect", "Bỏ sót bất thường")]
    selected: list[tuple[Path, str]] = []
    for folder, label in categories:
        paths = sorted((base / folder).glob("*.jpg"))[:4]
        selected.extend((path, label) for path in paths)
    if not selected:
        return

    thumb_w, thumb_h = 640, 270
    header_h = 72
    margin = 18
    columns = 2
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (columns * thumb_w + (columns + 1) * margin, rows * (thumb_h + header_h) + (rows + 1) * margin), "#F7F9FB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = font
    for idx, (path, category_label) in enumerate(selected):
        row, col = divmod(idx, columns)
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + header_h + margin)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_w, thumb_h), "white")
            paste_x = (thumb_w - rgb.width) // 2
            paste_y = (thumb_h - rgb.height) // 2
            tile.paste(rgb, (paste_x, paste_y))
            canvas.paste(tile, (x, y + header_h))
        draw.rounded_rectangle((x, y, x + thumb_w, y + header_h - 4), radius=8, fill="#E9EFF4")
        draw.text((x + 12, y + 5), category_label, fill="#20303C", font=font)
        display_name = path.stem.replace("good_good_", "good_").replace("defect_", "")
        draw.text((x + 12, y + 40), display_name[:64], fill="#60717D", font=small_font)
    png_path = exporter.png_dir / "20_qualitative_hybrid_um_errors.png"
    canvas.save(png_path, quality=96)
    exporter.register_raster(
        section="E8 · Định tính",
        slug="20_qualitative_hybrid_um_errors",
        title="Ví dụ lỗi còn lại của Hybrid U-Net + VMamba",
        description="Mẫu định tính lấy từ gallery đã xuất: bốn báo động giả và bốn bất thường bị bỏ sót đầu tiên theo thứ tự file.",
        sources="test_case_audit/visual_gallery/errors_only/hybrid_unet_vmamba__hybrid_fusion__fully_automatic",
        png_path=png_path,
    )


def write_tables(data: dict[str, pd.DataFrame], exporter: Exporter) -> dict[str, pd.DataFrame]:
    e2 = prepare_models(data["e2"])[
        ["model", "threshold", "image_auroc", "image_auprc", "image_fnr", "image_fpr", "positive_dice", "positive_iou", "region_recall_any_overlap"]
    ].copy()
    for column in ["image_auroc", "image_auprc", "image_fnr", "image_fpr", "positive_dice", "positive_iou", "region_recall_any_overlap"]:
        e2[column] = e2[column] * 100
    e2.columns = ["Model", "Threshold", "AUROC (%)", "AUPRC (%)", "FNR (%)", "FPR (%)", "Dice (%)", "IoU (%)", "Region recall (%)"]

    automatic = automatic_with_base(data).sort_values(["fpr_pct", "fnr_pct"])
    automatic = automatic.rename(columns={"label": "Thí nghiệm", "family": "Nhóm", "fnr_pct": "FNR (%)", "fpr_pct": "FPR (%)", "accuracy_pct": "Accuracy (%)"})

    all_images = data["all_images"]
    integrity = data["integrity"]
    audit_summary = pd.DataFrame(
        [
            {"Chỉ số": "Tổng ảnh đã audit", "Giá trị": len(all_images)},
            {"Chỉ số": "Train", "Giá trị": int((all_images["split"] == "train").sum())},
            {"Chỉ số": "Validation", "Giá trị": int((all_images["split"] == "val").sum())},
            {"Chỉ số": "Test", "Giá trị": int((all_images["split"] == "test").sum())},
            {"Chỉ số": "Vấn đề toàn vẹn mask", "Giá trị": len(integrity)},
            {"Chỉ số": "Ảnh ba model cùng Defect", "Giá trị": int((all_images["model_agreement"] == "all_defect").sum())},
            {"Chỉ số": "Ảnh ba model cùng Pass", "Giá trị": int((all_images["model_agreement"] == "all_pass").sum())},
        ]
    )

    tables = {"architecture": e2, "automatic": automatic, "audit": audit_summary}
    for name, frame in tables.items():
        frame.to_csv(exporter.table_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    return tables


def table_html(frame: pd.DataFrame, percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    display = frame.copy()
    for column in display.columns:
        if column in percent_columns:
            display[column] = display[column].map(lambda value: f"{float(value):.2f}%")
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.3f}")
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def write_dashboard(root: Path, exporter: Exporter, tables: dict[str, pd.DataFrame]) -> None:
    sections: dict[str, list[ChartEntry]] = {}
    for entry in exporter.entries:
        sections.setdefault(entry.section, []).append(entry)

    e2 = tables["architecture"]
    best_dice = e2.loc[e2["Dice (%)"].idxmax()]
    automatic = tables["automatic"]
    eligible = automatic[automatic["FNR (%)"] <= 4.0]
    best_auto = eligible.loc[eligible["FPR (%)"].idxmin()] if not eligible.empty else automatic.loc[automatic["FPR (%)"].idxmin()]
    audit = tables["audit"].set_index("Chỉ số")["Giá trị"]

    cards = [
        ("Model segmentation tốt nhất theo Dice", str(best_dice["Model"]), f"Dice {best_dice['Dice (%)']:.2f}%"),
        ("Logic tự động cân bằng trong FNR ≤ 4%", str(best_auto["Thí nghiệm"]), f"FPR {best_auto['FPR (%)']:.2f}% · FNR {best_auto['FNR (%)']:.2f}%"),
        ("Toàn bộ dữ liệu đã audit", f"{int(audit['Tổng ảnh đã audit']):,} ảnh", "Train + Validation + Test"),
        ("Vấn đề toàn vẹn mask", f"{int(audit['Vấn đề toàn vẹn mask'])}", "Theo file kiểm tra hiện có"),
    ]

    card_html = "".join(
        f'<article class="stat"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small></article>'
        for label, value, note in cards
    )
    section_html_parts: list[str] = []
    for section, entries in sections.items():
        figures = []
        for entry in entries:
            links = [f'<a href="{html.escape(entry.png)}" download>PNG</a>']
            if entry.svg:
                links.append(f'<a href="{html.escape(entry.svg)}" download>SVG</a>')
            figures.append(
                "".join(
                    [
                        '<article class="figure-card">',
                        f'<a href="{html.escape(entry.png)}"><img src="{html.escape(entry.png)}" alt="{html.escape(entry.title)}" loading="lazy"></a>',
                        '<div class="figure-copy">',
                        f'<h3>{html.escape(entry.title)}</h3>',
                        f'<p>{html.escape(entry.description)}</p>',
                        f'<small>Nguồn: {html.escape(entry.sources)}</small>',
                        f'<div class="downloads">{"".join(links)}</div>',
                        "</div></article>",
                    ]
                )
            )
        section_id = "sec-" + str(len(section_html_parts) + 1)
        section_html_parts.append(
            f'<section id="{section_id}"><h2>{html.escape(section)}</h2><div class="figure-grid">{"".join(figures)}</div></section>'
        )

    nav_html = "".join(
        f'<a href="#sec-{idx + 1}">{html.escape(section)}</a>' for idx, section in enumerate(sections)
    )
    architecture_table = table_html(
        tables["architecture"],
        {"AUROC (%)", "AUPRC (%)", "FNR (%)", "FPR (%)", "Dice (%)", "IoU (%)", "Region recall (%)"},
    )
    automatic_table = table_html(tables["automatic"], {"FNR (%)", "FPR (%)", "Accuracy (%)"})
    audit_table = table_html(tables["audit"])

    document = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trực quan hóa kết quả đồ án TTTN</title>
  <style>
    :root {{ color-scheme: light; --ink:#20303c; --muted:#60717d; --line:#d9e1e8; --paper:#f7f9fb; --card:#ffffff; --accent:#2d8b73; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:var(--paper); line-height:1.55; }}
    header {{ background:#173342; color:white; padding:42px clamp(20px,5vw,72px) 34px; }}
    header h1 {{ margin:0 0 8px; font-size:clamp(28px,4vw,46px); letter-spacing:-.02em; }}
    header p {{ margin:0; max-width:850px; color:#d9e7ed; }}
    nav {{ position:sticky; top:0; z-index:2; display:flex; gap:8px; overflow:auto; padding:10px clamp(14px,4vw,56px); background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); }}
    nav a {{ white-space:nowrap; color:#31505f; text-decoration:none; padding:7px 10px; border-radius:6px; font-size:13px; }}
    nav a:hover {{ background:#e9f2ef; color:#176b57; }}
    main {{ width:min(1540px,96vw); margin:0 auto; padding:28px 0 60px; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:34px; }}
    .stat {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:10px; padding:16px; min-height:130px; }}
    .stat span,.stat small {{ display:block; color:var(--muted); font-size:13px; }}
    .stat strong {{ display:block; margin:12px 0 5px; font-size:23px; }}
    section {{ scroll-margin-top:64px; margin:38px 0 54px; }}
    h2 {{ font-size:25px; margin:0 0 17px; padding-bottom:8px; border-bottom:2px solid #cbd8de; }}
    h3 {{ margin:0 0 7px; font-size:18px; }}
    .figure-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; align-items:start; }}
    .figure-card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; box-shadow:0 5px 18px rgba(32,48,60,.06); }}
    .figure-card > a {{ display:block; background:#eef2f5; }}
    .figure-card img {{ width:100%; height:auto; display:block; }}
    .figure-copy {{ padding:16px 18px 18px; }}
    .figure-copy p {{ margin:0 0 8px; color:#425765; }}
    .figure-copy small {{ color:var(--muted); }}
    .downloads {{ display:flex; gap:8px; margin-top:12px; }}
    .downloads a {{ color:#176b57; border:1px solid #a9cfc3; border-radius:6px; padding:4px 9px; text-decoration:none; font-size:13px; font-weight:600; }}
    .table-wrap {{ overflow:auto; background:white; border:1px solid var(--line); border-radius:10px; margin:16px 0 28px; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .data-table th {{ position:sticky; top:0; background:#e8eff2; text-align:left; }}
    .data-table th,.data-table td {{ padding:10px 12px; border-bottom:1px solid #e5ebef; white-space:nowrap; }}
    .note {{ background:#fff8e9; border:1px solid #efd59c; border-radius:8px; padding:13px 15px; color:#694f20; }}
    footer {{ color:var(--muted); text-align:center; padding:18px; border-top:1px solid var(--line); }}
    @media (max-width:980px) {{ .stats,.figure-grid {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:650px) {{ .stats,.figure-grid {{ grid-template-columns:1fr; }} header {{ padding:28px 20px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Trực quan hóa kết quả đồ án TTTN</h1>
    <p>U-Net/ResNet18 · SegFormer-B0 · VMamba-T · E2–E5 · E7–E8 · logic quyết định tự động · hybrid từng cặp · audit toàn bộ dữ liệu.</p>
  </header>
  <nav>{nav_html}<a href="#tables">Bảng tóm tắt</a></nav>
  <main>
    <div class="stats">{card_html}</div>
    <p class="note">Ngưỡng và policy được chọn trên Validation; các chỉ số cuối cùng được báo cáo trên Test. Các biểu đồ audit toàn bộ dữ liệu chỉ dùng để tìm case nghi vấn/label noise, không dùng để chọn lại hyperparameter theo Test.</p>
    {''.join(section_html_parts)}
    <section id="tables">
      <h2>Bảng tóm tắt dùng trong báo cáo</h2>
      <h3>Kiến trúc segmentation</h3><div class="table-wrap">{architecture_table}</div>
      <h3>Logic quyết định hoàn toàn tự động</h3><div class="table-wrap">{automatic_table}</div>
      <h3>Audit dữ liệu</h3><div class="table-wrap">{audit_table}</div>
    </section>
  </main>
  <footer>Được sinh tự động từ artifacts/reports/final · Không sửa số liệu nguồn.</footer>
</body>
</html>"""
    (exporter.out_dir / "index.html").write_text(document, encoding="utf-8")


def write_readme(exporter: Exporter) -> None:
    lines = [
        "# Bộ trực quan hóa kết quả đồ án",
        "",
        "Mở `index.html` để xem toàn bộ biểu đồ và bảng tóm tắt.",
        "",
        "- `figures/png`: ảnh độ phân giải cao để xem hoặc chèn nhanh.",
        "- `figures/svg`: hình vector để chèn luận văn, phóng to không vỡ.",
        "- `tables`: bảng rút gọn đã định dạng lại để dùng trong báo cáo.",
        "- `chart_manifest.csv`: danh mục biểu đồ và nguồn dữ liệu tương ứng.",
        "",
        "## Danh mục",
        "",
    ]
    for entry in exporter.entries:
        lines.append(f"- **{entry.title}** — `{entry.png}`")
    lines.extend(
        [
            "",
            "## Tái tạo",
            "",
            "Từ thư mục `E:\\Project\\TTTN` chạy:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe .\\scripts\\reporting\\generate_thesis_visualizations.py --root .\\artifacts\\reports\\final",
            "```",
            "",
        ]
    )
    (exporter.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(exporter: Exporter) -> None:
    pd.DataFrame([entry.__dict__ for entry in exporter.entries]).to_csv(
        exporter.out_dir / "chart_manifest.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh toàn bộ biểu đồ trực quan từ artifacts/reports/final.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "reports" / "final",
        help="Thư mục artifacts/reports/final.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Thư mục xuất; mặc định <root>/visualizations.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.output or (root / "visualizations")).resolve()
    apply_style()
    data = load_inputs(root)
    exporter = Exporter(out_dir)

    plot_e2_overall(data, exporter)
    plot_e2_error_rates(data, exporter)
    plot_e2_confusions(data, exporter)
    plot_e2_pixel_metrics(data, exporter)
    plot_e2_efficiency(data, exporter)
    plot_e3_size(data, exporter)
    plot_e4_group(data, exporter)
    plot_e5_multi_region(data, exporter)
    plot_e7_thresholds(data, exporter)
    plot_decision_tradeoff(data, exporter)
    plot_strategy_by_model(data, exporter)
    plot_outcomes(data, exporter)
    plot_validation_pareto(data, root, exporter)
    plot_automatic_group_recall(data, exporter)
    plot_audit_reasons(data, exporter)
    plot_audit_split_errors(data, exporter)
    plot_vote_patterns(data, exporter)
    make_error_contact_sheet(root, exporter)

    tables = write_tables(data, exporter)
    write_manifest(exporter)
    write_dashboard(root, exporter, tables)
    write_readme(exporter)
    print(f"Created {len(exporter.entries)} charts at: {out_dir}")
    print(f"Open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
