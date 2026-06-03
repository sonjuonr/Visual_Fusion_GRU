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
        self.ops.append(f"BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(text)}) Tj ET\n")

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
    canvas = PdfCanvas(OUT_DIR / "training_pipeline.pdf", 760, 340)
    canvas.text(32, 300, "Training Pipeline Used in Visual-Fusion-GRU", 17, bold=True)
    boxes = [
        (35, 195, 120, 66, "Isaac Sim", "fish + target"),
        (190, 195, 130, 66, "Fallback RL", "color heatmap"),
        (360, 195, 130, 66, "Alpha Fusion", "CLIP + fallback"),
        (535, 195, 170, 66, "RL Choke Point", "alpha 0.993"),
        (190, 75, 130, 66, "Teacher Labels", "fallback policy"),
        (360, 75, 130, 66, "GRU DAgger", "CLIP sequence"),
        (535, 75, 170, 66, "Hard-case Tuning", "95.7% success"),
    ]
    fill = (0.92, 0.95, 0.97)
    accent = (0.13, 0.38, 0.55)
    for x, y, w, h, title, subtitle in boxes:
        canvas.rect(x, y, w, h, fill)
        canvas.stroke_rect(x, y, w, h, color=accent)
        canvas.text(x + 12, y + 39, title, 12, bold=True)
        canvas.text(x + 12, y + 20, subtitle, 10)
    canvas.arrow(155, 228, 190, 228)
    canvas.arrow(320, 228, 360, 228)
    canvas.arrow(490, 228, 535, 228)
    canvas.arrow(255, 195, 255, 141)
    canvas.arrow(320, 108, 360, 108)
    canvas.arrow(490, 108, 535, 108)
    canvas.text(34, 36, "The deployed student uses pure CLIP observations; fallback is retained as teacher and diagnostic signal.", 10)
    canvas.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_success_ladder()
    make_pipeline()


if __name__ == "__main__":
    main()
