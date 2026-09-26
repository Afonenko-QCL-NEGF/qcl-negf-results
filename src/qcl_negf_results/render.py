"""Materialize annotated native HDF5 records into immutable browser assets.

The renderer only plots published quantities. It never computes a scientific
observable, changes acceptance, or feeds sampled data back to the solver.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import struct
import textwrap
import zlib
from pathlib import Path
from typing import Any

from qcl_negf_contracts.artifacts import CONTRACT_SET

from .presentation import atomic_json, inside, read_contract_json

RENDERER_REVISION = "qcl-negf-materialized-v4.0"
CONVERGENCE_REFERENCES = ((1e-4, "10⁻⁴"), (1e-6, "10⁻⁶"), (1e-8, "10⁻⁸"))
COLORS = ("#176d52", "#a43b31", "#365b9e", "#8a651d", "#7b488c")
MAX_CURVE_POINTS = 4096


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _axis_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exponent = f"{value:.2e}".split("e")
        power = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{mantissa.rstrip('0').rstrip('.')}×10{power}"
    return f"{value:.4g}"


def _curve_limits(curves: list[tuple[str, Any, Any]], logarithmic: bool) -> tuple[float, float, float, float]:
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    for _, xs, ys in curves:
        for x, y in zip(xs, ys, strict=True):
            if math.isfinite(float(x)) and math.isfinite(float(y)) and (not logarithmic or y > 0):
                value = math.log10(float(y)) if logarithmic else float(y)
                xmin, xmax = min(xmin, float(x)), max(xmax, float(x))
                ymin, ymax = min(ymin, value), max(ymax, value)
    if not math.isfinite(xmin):
        xmin, xmax, ymin, ymax = 0, 1, 0, 1
    if logarithmic:
        # Display guides participate in limits; raw values remain untouched.
        ymin, ymax = min(ymin, -8), max(ymax, -4)
        padding = (ymax - ymin) * .035
        ymin, ymax = ymin - padding, ymax + padding
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        ymax = ymin + max(abs(ymin) * .05, 1e-12)
    return xmin, xmax, ymin, ymax


def _convergence_guides(left: int, right: int, top: int, bottom: int,
                        ymin: float, ymax: float, logarithmic: bool) -> list[str]:
    if not logarithmic:
        return []
    parts = []
    for value, label in CONVERGENCE_REFERENCES:
        y = bottom - (bottom - top) * (math.log10(value) - ymin) / (ymax - ymin)
        parts.append(f'<g class="convergence-reference" data-reference-level="{value:g}"><title>Референс {label}; не критерий принятия</title>'
                     f'<path d="M{left} {y:.3f}H{right}" fill="none" stroke="#7b8580" stroke-width="1.2" stroke-dasharray="6 5"/>'
                     f'<text x="{right - 6}" y="{y - 5:.3f}" text-anchor="end" font-size="16" fill="#53645b" stroke="white" stroke-width="4" paint-order="stroke">{label}</text></g>')
    return parts


def _reference_caption(left: int, height: int, logarithmic: bool) -> str:
    if not logarithmic:
        return ""
    return f'<text x="{left}" y="{height - 35}" font-size="15" fill="#53645b">Пунктир: референсные уровни; критерии принятия заданы отдельно</text>'


def _svg(title: str, x_label: str, y_label: str, curves: list[tuple[str, Any, Any]], quality: str, *, logarithmic: bool = False, monitor: bool = False, boundaries: list[tuple[float, str]] | None = None) -> str:
    import numpy as np

    xmin, xmax, ymin, ymax = _curve_limits(curves, logarithmic)
    width = 760 if monitor else 840
    left, right, top, bottom = (130, 735, 50, 500) if monitor else (146, 808, 86, 386)
    legend_y, legend_step = (600, 24) if monitor else (496, 26)
    font_size = 18 if monitor else 22
    legend_positions: list[tuple[float, int, list[str]]] = []
    if monitor:
        height = max(720, legend_y + math.ceil(len(curves) / 3) * legend_step + 64)
    else:
        longest = max((len(label) for label, _, _ in curves), default=1)
        columns = max(1, min(3, int((right - left) / (longest * font_size * .57 + 20))))
        cell_width = (right - left) / columns
        wrapped = [textwrap.wrap(label, width=max(8, int((cell_width - 16) / (font_size * .57))),
                                  break_long_words=True, break_on_hyphens=False) or [""] for label, _, _ in curves]
        baseline = legend_y
        for row in range(0, len(wrapped), columns):
            for column, lines in enumerate(wrapped[row:row + columns]):
                legend_positions.append((left + column * cell_width, baseline, lines))
            baseline += max(len(lines) for lines in wrapped[row:row + columns]) * legend_step + 10
        height = max(580, baseline + 59)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"><title>{html.escape(title)}</title>',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<g font-family="sans-serif" font-size="{font_size}" fill="#17211d">',
             f'<text x="{left}" y="28" font-size="{22 if monitor else 24}">{html.escape(title)}</text>']
    if not monitor:
        quality_label = {"not_evaluated": "Промежуточные данные", "unconverged": "Сходимость не подтверждена",
                         "strict": "Критерии выполнены", "approximate": "Приближённое решение", "invalid": "Некорректный результат"}.get(quality, "Опубликованные значения")
        parts.append(f'<text x="{left}" y="56" font-size="18">{html.escape(quality_label)}</text>')
    parts.extend([f'<path d="M{left} {top}V{bottom}H{right}" fill="none" stroke="#53685d"/>',
                  f'<text x="{(left + right) / 2}" y="{bottom + 58 if monitor else 450}" text-anchor="middle">{html.escape(x_label)}</text>',
                  f'<text transform="translate({22 if monitor else 25} {(top + bottom) / 2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'])
    for tick in range(6):
        fraction = tick / 5
        x = left + (right - left) * fraction
        y = bottom - (bottom - top) * fraction
        xv = xmin + (xmax - xmin) * fraction
        yv = ymin + (ymax - ymin) * fraction
        yl = f"10^{yv:.1f}" if logarithmic and monitor else f"10^{yv:.2f}" if logarithmic else _axis_number(yv)
        parts.extend([f'<text x="{x:.1f}" y="{bottom + 26 if monitor else 419}" font-size="{18 if monitor else 20}" text-anchor="middle">{_axis_number(xv)}</text>',
                      f'<text x="{left - 8}" y="{y + 5:.1f}" font-size="{18 if monitor else 20}" text-anchor="end">{yl}</text>'])
    parts.extend(_convergence_guides(left, right, top, bottom, ymin, ymax, logarithmic))
    previous_label_x = -math.inf
    for value, label in boundaries or []:
        if xmin <= value <= xmax:
            x = left + (right - left) * (value - xmin) / (xmax - xmin)
            parts.append(f'<path d="M{x:.3f} {top}V{bottom}" fill="none" stroke="#9daaa4" stroke-dasharray="4 5"><title>{html.escape(label)}</title></path>')
            if x - previous_label_x >= (95 if monitor else 145):
                parts.append(f'<text x="{x + 4:.1f}" y="{top + (14 if monitor else 20)}" font-size="{11 if monitor else 16}">{html.escape(label)}</text>')
                previous_label_x = x
    sampled = False
    invalid_count = 0
    for number, (label, xs, ys) in enumerate(curves):
        if not len(xs):
            continue
        stride = max(1, math.ceil(len(xs) / MAX_CURVE_POINTS))
        sampled |= stride > 1
        # Preserve validity boundaries: sampled paths must not bridge an invalid interval.
        indices = np.unique(np.r_[np.arange(0, len(xs), stride), max(0, len(xs) - 1)])
        segment: list[str] = []
        color = COLORS[number % len(COLORS)]
        def publish_segment() -> None:
            if len(segment) == 1:
                cx, cy = segment[0].split(",")
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{color}"/>')
            elif segment:
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.7" points="{" ".join(segment)}"/>')
            segment.clear()
        previous = -1
        for index in indices:
            x, y = float(xs[index]), float(ys[index])
            gap = any(not math.isfinite(float(a)) or not math.isfinite(float(b)) or (logarithmic and b <= 0)
                      for a, b in zip(xs[previous + 1:index + 1], ys[previous + 1:index + 1], strict=True))
            if gap:
                publish_segment()
                invalid_count += 1
            if not math.isfinite(x) or not math.isfinite(y) or (logarithmic and y <= 0):
                previous = int(index)
                continue
            value = math.log10(y) if logarithmic else y
            segment.append(f"{left + (right - left) * (x - xmin) / (xmax - xmin):.3f},{bottom - (bottom - top) * (value - ymin) / (ymax - ymin):.3f}")
            previous = int(index)
        publish_segment()
        if monitor:
            parts.append(f'<text x="{left + (number % 3) * ((right - left) / 3)}" y="{legend_y + (number // 3) * legend_step}" fill="{color}">{html.escape(label)}</text>')
        else:
            legend_x, legend_baseline, lines = legend_positions[number]
            parts.append(f'<text x="{legend_x}" y="{legend_baseline}" fill="{color}">')
            for line_index, line in enumerate(lines):
                parts.append(f'<tspan x="{legend_x}" dy="{0 if line_index == 0 else legend_step}">{html.escape(line)}</tspan>')
            parts.append('</text>')
    parts.append(_reference_caption(left, height, logarithmic))
    if monitor:
        label = {"not_evaluated": "Промежуточные данные", "unconverged": "Сходимость не подтверждена",
                 "strict": "Критерии выполнены", "approximate": "Приближённое решение", "invalid": "Некорректный результат"}.get(quality, quality)
        parts.append(f'<text x="{left}" y="{height - 12}" font-size="15">{html.escape(label)} · пропуски не соединены</text>')
    elif sampled or invalid_count:
        parts.append(f'<text x="{left}" y="{height - 12}" font-size="16">Полные данные сохранены; разрывы/неположительные значения: {invalid_count}</text>')
    return "".join(parts) + "</g></svg>"


def _png(values: Any, *, color_limit: float | None = None) -> tuple[bytes, dict[str, Any]]:
    import numpy as np

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("heatmap must have exactly two annotated axes")
    # One PNG pixel per native cell; CSS controls the viewport, not the stored
    # raster resolution. In particular, narrow negative/invalid witnesses are
    # not discarded by a stride before a user opens the figure at full size.
    stride_y = stride_x = 1
    display = matrix
    valid = np.isfinite(display)
    finite = np.isfinite(matrix)
    observed_limit = float(np.max(np.abs(matrix[finite]))) if finite.any() else 0.0
    limit = observed_limit if color_limit is None else color_limit
    if not math.isfinite(limit) or limit < observed_limit:
        raise ValueError("heatmap color range cannot clip a finite native value")
    # A signed diverging scale preserves negative witnesses; missing is gray.
    scaled = np.clip(np.nan_to_num(display / (limit or 1.0)), -1, 1)
    pixels = np.empty((*display.shape, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255 * (1 - np.maximum(0, -scaled))
    pixels[:, :, 1] = 255 * (1 - np.abs(scaled))
    pixels[:, :, 2] = 255 * (1 - np.maximum(0, scaled))
    pixels[~valid] = [155, 155, 155]
    pixels = pixels[::-1]  # lower energy is at the bottom, axes documented in index
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    height, width = display.shape
    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    return payload, {"native_shape": list(matrix.shape), "display_shape": list(display.shape), "sampling": [stride_y, stride_x], "color_range": [-limit, limit], "native_range": [float(matrix[finite].min()), float(matrix[finite].max())] if finite.any() else None, "missing_count": int((~finite).sum()), "color_normalization": "linear, symmetric about zero; range uses all finite native values, without clipping", "missing_color": "#9b9b9b", "orientation": "energy increasing upward; position increasing rightward"}


def _heatmap_svg(values: Any, z: Any, energy: Any, units: str, quality: str) -> tuple[str, bytes, dict[str, Any]]:
    """Render a physical E,z plot; raster dimensions never dictate card shape.

    The original HDF5 stays exact. On a nonuniform physical grid only the display
    raster selects the nearest native cell; it never interpolates physical data.
    """
    import numpy as np

    matrix = np.asarray(values, dtype=float)
    z = np.asarray(z, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if matrix.shape != (len(energy), len(z)) or min(matrix.shape) < 2:
        raise ValueError("heatmap requires at least two samples on each physical axis")
    for axis in (z, energy):
        if axis.ndim != 1 or not np.isfinite(axis).all() or not (np.diff(axis) > 0).all():
            raise ValueError("heatmap physical axes must be finite and strictly increasing")
    raw, annotation = _png(matrix)
    regular = all(np.allclose(np.diff(axis), (axis[-1] - axis[0]) / (len(axis) - 1),
                              rtol=1e-7, atol=max(abs(axis[0]), abs(axis[-1]), 1e-12) * 1e-12) for axis in (z, energy))
    raster = raw
    if not regular:
        # A stretched index raster would misplace nonuniform physical samples.
        # Uniform screen pixels instead inherit the nearest original E,z cell.
        xx = np.linspace(z[0], z[-1], min(1600, max(760, len(z))))
        yy = np.linspace(energy[0], energy[-1], min(1024, max(370, len(energy))))
        zi = np.searchsorted((z[:-1] + z[1:]) / 2, xx)
        ei = np.searchsorted((energy[:-1] + energy[1:]) / 2, yy)
        raster, display = _png(matrix[np.ix_(ei, zi)], color_limit=annotation["color_range"][1])
        annotation["display_shape"] = display["display_shape"]
    annotation.update(axes_extent=[float(z[0]), float(z[-1]), float(energy[0]), float(energy[-1])],
        display_projection="native regular grid" if regular else "nearest native cell on physical axes; no interpolation",
        display_aspect_ratio="3:2", raw_raster_axes="native row/column ordinal")
    width, height = 840, 560
    left, right, top, bottom = 118, 654, 88, 420
    bar_x, bar_width = 686, 20
    unit_label = units.replace("m^-3", "m⁻³").replace("m^3", "m³")
    title = "Энергетическая и пространственная плотность"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"><title>{title}</title>',
             f'<desc>Опубликованная плотность на физических осях. Линейная цветовая шкала, без отсечения значений. {html.escape(annotation["display_projection"])}</desc>',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             '<defs><linearGradient id="density-scale" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#0000ff"/><stop offset=".5" stop-color="#ffffff"/><stop offset="1" stop-color="#ff0000"/></linearGradient></defs>',
             '<g font-family="sans-serif" font-size="18" fill="#17211d">',
             f'<text x="{left}" y="30" font-size="23">{title}</text>',
             f'<text x="{left}" y="60">n(E,z), {html.escape(unit_label)} · линейная шкала</text>',
             f'<image x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" preserveAspectRatio="none" href="data:image/png;base64,{base64.b64encode(raster).decode()}"/>',
             f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#53685d"/>',
             f'<text x="{(left+right)/2}" y="480" text-anchor="middle">z, nm</text>',
             f'<text transform="translate(24 {(top+bottom)/2}) rotate(-90)" text-anchor="middle">E, eV</text>']
    for tick in range(6):
        fraction = tick / 5
        x = left + (right - left) * fraction
        y = bottom - (bottom - top) * fraction
        parts.extend([f'<path d="M{x:.1f} {bottom}v6M{left} {y:.1f}h-6" stroke="#53685d"/>',
            f'<text x="{x:.1f}" y="{bottom+29}" text-anchor="middle">{_axis_number(float(z[0] + (z[-1]-z[0])*fraction))}</text>',
            f'<text x="{left-10}" y="{y+6:.1f}" text-anchor="end">{_axis_number(float(energy[0] + (energy[-1]-energy[0])*fraction))}</text>'])
    limit = annotation["color_range"][1]
    fill = "url(#density-scale)" if limit else "#fff" if annotation["native_range"] is not None else "#9b9b9b"
    parts.append(f'<rect x="{bar_x}" y="{top}" width="{bar_width}" height="{bottom-top}" fill="{fill}" stroke="#53685d"/>')
    for fraction in (0., .25, .5, .75, 1.):
        value = (2 * fraction - 1) * limit
        y = bottom - (bottom - top) * fraction
        parts.append(f'<text x="{bar_x+bar_width+9}" y="{y+6:.1f}" font-size="16">{_axis_number(value)}</text>')
    parts.extend([f'<rect x="{left}" y="500" width="18" height="18" fill="#9b9b9b"/><text x="{left+27}" y="515" font-size="16">Нет конечного значения · {annotation["missing_count"]}</text>',
                  f'<text x="{left}" y="543" font-size="15">Полная сетка {len(energy)} × {len(z)} в HDF5 · отображение не меняет данные</text>',
                  '</g></svg>'])
    return "".join(parts), raw, annotation


def materialize(generation: Path, commit: dict[str, Any], commit_hash: str, destination: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    from qcl_negf_contracts.artifacts import validate_commit
    validate_commit(commit)
    key = hashlib.sha256((commit_hash + CONTRACT_SET + RENDERER_REVISION).encode()).hexdigest()
    target = destination / key
    index_path = target / "index.json"
    if index_path.is_file():
        return read_contract_json(index_path, "qcl-negf.materialized-point.v3")
    artifacts = [a for a in commit.get("artifacts", []) if a.get("role") == "physics.analysis"]
    if not artifacts and any(a.get("schema") == "qcl-negf-operator-diagnostics-v4" for a in commit.get("artifacts", [])):
        return {"schema": "qcl-negf.materialized-point.v3", "contract_set": CONTRACT_SET, "source_commit": commit_hash,
                "renderer_revision": RENDERER_REVISION, "generation": commit.get("generation"),
                "identity": commit.get("identity", {}), "quality": commit.get("quality"),
                "scientific_accepted": commit.get("scientific_accepted") is True,
                "physics_ready": False, "checkpoint_ready": False, "presentation_ready": True,
                "figures": [], "validation": None, "positivity": None}
    if len(artifacts) != 1:
        raise ValueError("commit must declare one native analytical HDF5 record")
    artifact = artifacts[0]
    source = inside(generation, artifact["path"])
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != artifact["sha256"]:
        raise ValueError("HDF5 hash differs from its committed record")
    target.mkdir(parents=True, exist_ok=True)
    quality = str(commit.get("quality", "not_evaluated"))
    figures: list[dict[str, Any]] = []
    def add_curve(identifier: str, title: str, x_label: str, y_label: str, curves: list[tuple[str, Any, Any]], datasets: list[str], log: bool = False) -> None:
        curves = [curve for curve in curves if len(curve[1])]
        if not curves:
            return
        filename = f"{identifier}.svg"
        (target / filename).write_text(_svg(title, x_label, y_label, curves, quality, logarithmic=log), encoding="utf-8")
        figures.append({"id": identifier, "title": title, "path": f"{key}/{filename}", "media_type": "image/svg+xml", "source_datasets": datasets, "units": {"x": x_label, "y": y_label}, "sampling": "display-only; at most 4096 samples per curve", "quality": quality, "source_commit": commit_hash})
    validation, positivity = None, None
    with h5py.File(source, "r") as record:
        from .native import validate_native_handle
        validate_native_handle(record, "physics.analysis", "qcl-negf-physics-analysis-v4")
        def array(name: str):
            if name not in record:
                return None
            item = record[name]
            if not isinstance(item, h5py.Dataset):
                raise ValueError(f"{name} must be a native dataset")
            expected_axes = {"axes/z_nm": "z", "axes/energy_eV": "E", "axes/k_per_nm": "k",
                             "observables/spatial_energy_density": "E,z", "observables/spectral": "E,k",
                             "observables/occupied_spectral": "E,k", "basis/effective_wavefunctions/real": "z,state"}.get(name)
            if name.startswith("observables/potential_") or name == "observables/density_per_m3":
                expected_axes = "z"
            if expected_axes and (item.ndim != len(expected_axes.split(",")) or _text(item.attrs.get("logical_axis_order", "")) != expected_axes):
                raise ValueError(f"{name} has missing or inconsistent logical axis annotation")
            shape = _text(item.attrs.get("logical_shape", ""))
            if shape and tuple(int(size) for size in shape.split(",")) != item.shape:
                raise ValueError(f"{name} logical shape disagrees with its stored shape")
            return np.asarray(item[...])
        for field in ("validation", "positivity"):
            key_path = f"diagnostics/{field}_json"
            if key_path in record:
                encoded = _text(record[key_path][()])
                if field == "validation":
                    validation = json.loads(encoded)
                else:
                    positivity = json.loads(encoded)
        z, energy = array("axes/z_nm"), array("axes/energy_eV")
        if z is not None:
            curves, datasets = [], []
            for field, label in (("structure", "Структура"), ("external", "Внешний"), ("hartree", "Хартри"), ("total", "Полный")):
                path = f"observables/potential_{field}_eV"
                values = array(path)
                if values is not None and values.ndim == 1 and len(values) == len(z):
                    curves.append((label, z, values))
                    datasets.append(path)
            add_curve("potentials", "Потенциалы", "z, nm", "U, eV", curves, datasets)
            density = array("observables/density_per_m3")
            if density is not None and density.ndim == 1 and len(density) == len(z):
                add_curve("density", "Плотность электронов", "z, nm", "n, m⁻³", [("n(z)", z, density)], ["observables/density_per_m3"])
            wave = array("basis/effective_wavefunctions/real")
            if wave is not None and wave.ndim == 2:
                if wave.shape[0] == len(z):
                    add_curve("wavefunctions", "Волновые функции", "z, nm", "Re ψ, опубликованная нормировка", [(f"Re ψ {i + 1}", z, wave[:, i]) for i in range(min(12, wave.shape[1]))], ["basis/effective_wavefunctions/real"])
        if energy is not None:
            curves, datasets = [], []
            for name, label in (("spectral", "Спектральная"), ("occupied_spectral", "Занятые состояния")):
                path = f"observables/{name}"
                values = array(path)
                if values is not None and values.ndim == 1 and len(values) == len(energy):
                    curves.append((label, energy, values))
                    datasets.append(path)
                elif values is not None and values.ndim == 2 and values.shape[0] == len(energy):
                    momentum = array("axes/k_per_nm")
                    for index in sorted({0, values.shape[1] // 2, values.shape[1] - 1}):
                        selected = float(momentum[index]) if momentum is not None else index
                        curves.append((f"{label}, k={selected:.4g} nm⁻¹", energy, values[:, index]))
                        datasets.append(path)
            add_curve("spectra", "Опубликованные энергетические спектры", "E, eV", "trace A, trace (−iG˂), eV⁻¹", curves, datasets)
            heat = array("observables/spatial_energy_density")
            if heat is not None and heat.ndim == 2 and z is not None:
                if heat.shape != (len(energy), len(z)):
                    raise ValueError("spatial energy density does not match native axes")
                units = _text(record["observables/spatial_energy_density"].attrs.get("units", "see HDF5"))
                figure, payload, annotation = _heatmap_svg(heat, z, energy, units, quality)
                (target / "energy-density.svg").write_text(figure, encoding="utf-8")
                (target / "energy-density.png").write_bytes(payload)
                figures.append({"id": "energy-density", "title": "Энергетическая и пространственная плотность", "path": f"{key}/energy-density.svg", "raw_path": f"{key}/energy-density.png", "media_type": "image/svg+xml", "source_datasets": ["observables/spatial_energy_density"], "source_commit": commit_hash, "quality": quality, "units": {"x": "z, nm", "y": "E, eV", "value": units}, **annotation})
        for phase, counter in (("scba", "ν"), ("outer", "μ")):
            curves, datasets = [], []
            if f"diagnostics/{phase}" in record and isinstance(record[f"diagnostics/{phase}"], h5py.Group):
                history = record[f"diagnostics/{phase}"]
                for name, dataset in history.items():
                    if name.startswith("r_") and isinstance(dataset, h5py.Dataset):
                        values = np.asarray(dataset[...])
                        if values.ndim == 1:
                            curves.append((name, np.arange(1, len(values) + 1), values))
                            datasets.append(f"diagnostics/{phase}/{name}")
            add_curve(f"convergence-{phase}", f"История сходимости: {phase}", "Номер сохранённого шага", "невязка, log₁₀", curves, datasets, True)
    result = {"schema": "qcl-negf.materialized-point.v3", "contract_set": CONTRACT_SET, "source_commit": commit_hash, "renderer_revision": RENDERER_REVISION, "generation": commit.get("generation"), "identity": commit.get("identity", {}), "quality": quality, "scientific_accepted": commit.get("scientific_accepted") is True, "physics_ready": commit.get("physics_ready") is True, "checkpoint_ready": commit.get("checkpoint_ready") is True, "presentation_ready": True, "figures": figures, "validation": validation, "positivity": positivity}
    atomic_json(index_path, result)
    return result


def materialize_series(points: list[dict[str, Any]], destination: Path) -> list[dict[str, Any]]:
    """Render already-published scalar currents; invalid points break the line."""
    groups: dict[tuple[str, Any, Any], list[dict[str, Any]]] = {}
    for point in points:
        coordinates = point.get("coordinates") or {}
        if point.get("observables", {}).get("current_density_A_per_m2") is None:
            continue
        key = (point.get("execution_id", "unknown"), coordinates.get("temperature_K"), coordinates.get("branch"))
        groups.setdefault(key, []).append(point)
    curves = []
    for key, rows in groups.items():
        rows.sort(key=lambda p: p["coordinates"].get("order", 0))
        xs = [p["coordinates"]["voltage_per_period_V"] for p in rows]
        ys = [p["observables"]["current_density_A_per_m2"] if p.get("scientific_accepted") else float("nan") for p in rows]
        curves.append((f"{key[0]} · {key[1]} K · {key[2]}", xs, ys))
        for point in rows:
            if not point.get("scientific_accepted"):
                curves.append((f"{point['id']} · непринята", [point["coordinates"]["voltage_per_period_V"]], [point["observables"]["current_density_A_per_m2"]]))
    if not curves:
        return []
    render_key = hashlib.sha256(json.dumps(points, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + CONTRACT_SET.encode() + RENDERER_REVISION.encode()).hexdigest()
    target = destination / render_key
    target.mkdir(parents=True, exist_ok=True)
    path = target / "current-voltage.svg"
    if not path.is_file():
        path.write_text(_svg("Ток и напряжение на период", "V периода, V", "J, A/m²", curves, "научно принятые точки соединены; непринятые показаны отдельно"), encoding="utf-8")
    return [{"id": "current-voltage", "title": "Ток и напряжение на период", "path": f"{render_key}/current-voltage.svg", "media_type": "image/svg+xml", "source_commit": render_key, "units": {"x": "V", "y": "A/m²"}}]
