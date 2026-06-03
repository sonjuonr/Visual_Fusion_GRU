"""Generate simple vector PDF figures for the final report.

The script intentionally avoids external plotting dependencies so the report
package can be rebuilt in the Isaac Sim workspace with only Python 3.
"""

from __future__ import annotations

from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "figures"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfCanvas:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def text(self, x: float, y: float, text: str, size: int = 12, bold: bool = False) -> None:
        font = "F2" if bold else "F1"
        self.ops.append(f"0 0 0 rg BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(text)}) Tj ET\n")

    def text_center(self, x: float, y: float, text: str, size: int = 12, bold: bool = False) -> None:
        approx_width = len(text) * size * 0.27
        self.text(x - approx_width, y, text, size=size, bold=bold)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        lines: list[str],
        fill: tuple[float, float, float],
        border: tuple[float, float, float],
    ) -> None:
        self.rect(x, y, w, h, fill)
        self.stroke_rect(x, y, w, h, color=border)
        self.text(x + 12, y + h - 22, title, 12, bold=True)
        for idx, line in enumerate(lines):
            self.text(x + 12, y + h - 42 - idx * 16, line, 9)

    def rect(self, x: float, y: float, w: float, h: float, fill: tuple[float, float, float]) -> None:
        r, g, b = fill
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f\n")

    def stroke_rect(self, x: float, y: float, w: float, h: float, color: tuple[float, float, float] = (0, 0, 0)) -> None:
        r, g, b = color
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG {x:.2f} {y:.2f} {w:.2f} {h:.2f} re S\n")

    def line(self, x1: float, y1: float, x2: float, y2: float, color: tuple[float, float, float] = (0, 0, 0), width: float = 1.2) -> None:
        r, g, b = color
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n")

    def arrow(self, x1: float, y1: float, x2: float, y2: float, color: tuple[float, float, float] = (0.1, 0.1, 0.1)) -> None:
        self.line(x1, y1, x2, y2, color=color, width=1.4)
        if x2 >= x1:
            points = [(x2, y2), (x2 - 9, y2 + 5), (x2 - 9, y2 - 5)]
        else:
            points = [(x2, y2), (x2 + 9, y2 + 5), (x2 + 9, y2 - 5)]
        r, g, b = color
        path = f"{r:.3f} {g:.3f} {b:.3f} rg {points[0][0]:.2f} {points[0][1]:.2f} m "
        path += " ".join(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
        path += " h f\n"
        self.ops.append(path)

    def save(self) -> None:
        content = "".join(self.ops).encode("latin-1")
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("latin-1")
        objects.append(page)
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        objects.append(b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream")

        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{idx} 0 obj\n".encode("ascii"))
            out.extend(obj)
            out.extend(b"\nendobj\n")
        xref_start = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        out.extend(
            (
                f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_start}\n%%EOF\n"
            ).encode("ascii")
        )
        self.path.write_bytes(out)


def make_success_ladder() -> None:
    data = [
        ("BC", 0.150),
        ("MLP DAgger", 0.660),
        ("GRU 1000eps", 0.883),
        ("Hard-case", 0.946),
        ("Radius-1", 0.957),
    ]
    canvas = PdfCanvas(OUT_DIR / "success_rate_ladder.pdf", 720, 420)
    canvas.text(32, 382, "Evaluation Success-Rate Ladder", 18, bold=True)
    canvas.text(32, 360, "Pure CLIP student evaluations unless noted otherwise", 10)
    x0, y0 = 84, 82
    bar_w, gap, max_h = 84, 34, 250
    for i in range(6):
        y = y0 + i * 50
        canvas.line(62, y, 660, y, color=(0.82, 0.84, 0.86), width=0.5)
        canvas.text(26, y - 4, f"{i * 20}%", 8)
    colors = [
        (0.77, 0.27, 0.30),
        (0.90, 0.58, 0.20),
        (0.25, 0.48, 0.72),
        (0.32, 0.62, 0.42),
        (0.14, 0.54, 0.58),
    ]
    for idx, (label, value) in enumerate(data):
        x = x0 + idx * (bar_w + gap)
        h = value * max_h
        canvas.rect(x, y0, bar_w, h, colors[idx])
        canvas.stroke_rect(x, y0, bar_w, h, color=(0.2, 0.2, 0.2))
        canvas.text(x + 10, y0 + h + 12, f"{value * 100:.1f}%", 11, bold=True)
        canvas.text(x + 2, 46, label, 9)
    canvas.line(62, y0, 660, y0, color=(0.1, 0.1, 0.1), width=1.0)
    canvas.line(62, y0, 62, y0 + max_h, color=(0.1, 0.1, 0.1), width=1.0)
    canvas.save()


def make_pipeline() -> None:
    canvas = PdfCanvas(OUT_DIR / "training_pipeline.pdf", 840, 420)
    canvas.text(34, 382, "Research Pipeline: From RL Diagnosis to Pure-CLIP GRU Student", 17, bold=True)
    canvas.text(34, 360, "Top row: original RL migration. Bottom row: final imitation-learning route.", 10)
    blue_fill = (0.91, 0.95, 0.98)
    green_fill = (0.91, 0.97, 0.92)
    red_fill = (0.98, 0.91, 0.90)
    border = (0.12, 0.32, 0.46)
    boxes = [
        (35, 245, 135, 74, "Isaac Sim", ["robotic fish", "red target"]),
        (205, 245, 145, 74, "Fallback RL", ["color heatmap", "stable teacher"]),
        (385, 245, 150, 74, "Alpha Fusion", ["alpha * CLIP", "+ fallback"]),
        (570, 245, 210, 74, "RL Bottleneck", ["alpha 0.993", "0.007 fallback still matters"]),
        (205, 95, 145, 74, "Teacher Labels", ["fallback actions", "student states"]),
        (385, 95, 150, 74, "GRU DAgger", ["CLIP heatmap", "temporal memory"]),
        (570, 95, 210, 74, "Hard-Case Tuning", ["radius-1 failed seeds", "0.957 over 1000 eps"]),
    ]
    for idx, (x, y, w, h, title, lines) in enumerate(boxes):
        fill = red_fill if title == "RL Bottleneck" else green_fill if y < 200 else blue_fill
        canvas.box(x, y, w, h, title, lines, fill, border)
    canvas.arrow(170, 282, 205, 282)
    canvas.arrow(350, 282, 385, 282)
    canvas.arrow(535, 282, 570, 282)
    canvas.arrow(277, 245, 277, 169)
    canvas.arrow(350, 132, 385, 132)
    canvas.arrow(535, 132, 570, 132)
    canvas.text(35, 46, "Final deployment path: pure CLIP observation -> GRU student -> forward / left / right action.", 10, bold=True)
    canvas.save()


def make_imitation_structure() -> None:
    canvas = PdfCanvas(OUT_DIR / "imitation_learning_structure.pdf", 860, 430)
    canvas.text(32, 392, "Imitation-Learning Structure: BC to DAgger to GRU DAgger", 17, bold=True)
    canvas.text(32, 370, "The fallback policy labels student observations; each stage fixes a specific failure mode.", 10)
    border = (0.11, 0.31, 0.48)
    fill_seed = (0.93, 0.95, 0.98)
    fill_stage = (0.91, 0.97, 0.92)
    fill_best = (0.89, 0.95, 0.96)
    fill_warn = (0.98, 0.94, 0.86)

    canvas.box(35, 245, 155, 82, "Seed Dataset", ["360 episodes", "11720 labels"], fill_seed, border)
    canvas.box(230, 245, 155, 82, "BC Student", ["MLP baseline", "0.15 success"], fill_warn, border)
    canvas.box(425, 245, 155, 82, "MLP DAgger", ["off-policy states", "0.66 success"], fill_stage, border)
    canvas.box(620, 245, 180, 82, "GRU DAgger", ["sequence length 16", "0.883 over 1000 eps"], fill_stage, border)

    canvas.box(230, 95, 155, 82, "Failed Seeds", ["119 hard seeds", "teacher relabel"], fill_seed, border)
    canvas.box(425, 95, 155, 82, "Hard-Case GRU", ["0.942-0.946", "1000-episode sweeps"], fill_stage, border)
    canvas.box(620, 95, 180, 82, "Radius-1 GRU", ["299 targeted eps", "0.957 final result"], fill_best, border)

    canvas.arrow(190, 286, 230, 286)
    canvas.arrow(385, 286, 425, 286)
    canvas.arrow(580, 286, 620, 286)
    canvas.arrow(710, 245, 308, 177)
    canvas.arrow(385, 136, 425, 136)
    canvas.arrow(580, 136, 620, 136)

    canvas.text(58, 207, "BC learns teacher states.", 9)
    canvas.text(238, 207, "DAgger labels states visited by the student.", 9)
    canvas.text(615, 207, "GRU adds memory for target-loss recovery.", 9)
    canvas.text(34, 48, "Final training summary: 73737 labeled records from seed, balanced DAgger, and radius-1 hard-case datasets.", 10, bold=True)
    canvas.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_success_ladder()
    make_pipeline()
    make_imitation_structure()


if __name__ == "__main__":
    main()
