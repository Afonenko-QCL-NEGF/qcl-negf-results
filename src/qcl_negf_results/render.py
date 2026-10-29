"""Materialize annotated native HDF5 records into immutable browser assets.

The renderer only plots published quantities. It never computes a scientific
observable, changes acceptance, or feeds sampled data back to the solver.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import html
import json
import math
import os
import struct
import sys
import textwrap
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from qcl_negf_contracts.artifacts import CONTRACT_SET

from .presentation import atomic_json, inside
from .state import Document
from .saved_views import SavedAnalysis
from .saved_views import _open_regular
from .diagnostic_views import DiagnosticViewBatch

RENDERER_REVISION = "qcl-negf-materialized-native5-v1.0"
CONVERGENCE_REFERENCES = ((1e-4, "10⁻⁴"), (1e-6, "10⁻⁶"), (1e-8, "10⁻⁸"))
COLORS = ("#176d52", "#a43b31", "#365b9e", "#8a651d", "#7b488c")


def _json_tree_weight(value):
    """Serialization reserve: repeated aliases expand again in JSON."""
    weight=sys.getsizeof(value)
    if isinstance(value,dict):
        return weight+sum(_json_tree_weight(k)+_json_tree_weight(v) for k,v in value.items())
    if isinstance(value,(tuple,list)):
        return weight+sum(_json_tree_weight(v) for v in value)
    return weight


@contextmanager
def _cache_file(path, *, maximum_size, expected_size=None):
    """Qualify this one nofollow regular FD and check its path after reading."""
    from qcl_negf_contracts.messages import ContractError
    def signature(value):
        return (value.st_dev,value.st_ino,value.st_mode,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    with _open_regular(path) as stream:
        opened=os.fstat(stream.fileno());identity=signature(opened)
        if expected_size is not None and opened.st_size!=expected_size:
            raise ContractError('cached metadata length differs from qualified source/contract memo','corrupt_result')
        if opened.st_size>maximum_size:
            raise ContractError('opened cached file exceeds workspace allowance','budget_exceeded')
        yield stream,opened.st_size
        if (signature(os.fstat(stream.fileno()))!=identity
                or signature(path.stat(follow_symlinks=False))!=identity):
            raise ContractError('cached file FD/path changed during bounded read','corrupt_result')
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


def _compress_bound(size: int) -> int:
    # zlib compressBound formula for default zlib.compress parameters.
    return size + (size >> 12) + (size >> 14) + (size >> 25) + 13


def _raster_forecast(shape, display_shape=None, *, retained_bytes=0):
    """Conservative explicit buffers, never allocator/RSS/native scratch bound.

    N=H*W, R=3N+H, C=compressBound(R). Codec keeps matrix8N,
    row<=23W, raw construction<=2R, chunk/CRC/result joins<=4C+4096.
    Nonuniform display retains native PNG(C+57), source coordinates,
    source midpoint temporaries, screen coordinates/indices and display8M.
    SVG base64 bytes+text+joined copies charged at 16*ceil(P/3)+32768.
    Caller containers and prior output strings are supplied separately.
    """
    h,w=shape; n=h*w; r=3*n+h; c=_compress_bound(r)
    native=8*n+23*w+2*r+4*c+4096
    peak=native+retained_bytes
    if display_shape is not None:
        dh,dw=display_shape; m=dh*dw; dr=3*m+dh; dc=_compress_bound(dr)
        nearest=(c+57)+8*n+32*(w+h)+16*(dw+dh)+8*m
        codec=23*dw+2*dr+4*dc+4096
        # Native PNG and display PNG both retained while embedding SVG.
        svg=(c+57)+(dc+57)+32*(w+h)+16*(dw+dh)+16*((dc+59)//3)+32768
        peak=max(peak,retained_bytes+nearest+codec,retained_bytes+8*n+svg)
    else:
        peak=max(peak,retained_bytes+8*n+(c+57)+16*((c+59)//3)+32768)
    return native,peak


def _png(values: Any, *, color_limit: float | None = None) -> tuple[bytes, dict[str, Any]]:
    import numpy as np
    matrix=np.asarray(values,dtype=np.float64)
    if matrix.ndim != 2 or not all(matrix.shape):
        raise ValueError('heatmap must have exactly two nonempty annotated axes')
    minimum=math.inf; maximum=-math.inf; missing=0
    for row in matrix:
        for cell in row:
            value=float(cell)
            if math.isfinite(value): minimum=min(minimum,value); maximum=max(maximum,value)
            else: missing+=1
    observed=max(abs(minimum),abs(maximum)) if missing < matrix.size else 0.
    limit=observed if color_limit is None else color_limit
    if not math.isfinite(limit) or limit < observed:
        raise ValueError('heatmap color range cannot clip a finite native value')
    height,width=matrix.shape; raw=bytearray()
    # One row and Python scalar arithmetic replace whole-image float/mask/RGB
    # temporaries. Exact original uint8 truncation and signed scale are retained.
    for row in matrix[::-1]:
        pixels=bytearray(3*width)
        for index,cell in enumerate(row):
            value=float(cell); offset=3*index
            if not math.isfinite(value): pixels[offset:offset+3]=b'\x9b\x9b\x9b'
            else:
                scaled=max(-1.,min(1.,value/(limit or 1.)))
                pixels[offset]=int(255*(1-max(0.,-scaled)))
                pixels[offset+1]=int(255*(1-abs(scaled)))
                pixels[offset+2]=int(255*(1-max(0.,scaled)))
        raw.append(0); raw.extend(pixels)
    del pixels
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    compressed=zlib.compress(raw,6)
    payload=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',compressed)+chunk(b'IEND',b'')
    return payload,dict(native_shape=list(matrix.shape),display_shape=list(matrix.shape),sampling=[1,1],
        color_range=[-limit,limit],native_range=None if missing==matrix.size else [minimum,maximum],
        missing_count=missing,color_normalization='linear, symmetric about zero; range uses all finite native values, without clipping',
        missing_color='#9b9b9b',orientation='energy increasing upward; position increasing rightward')


def _heatmap_svg(values: Any, z: Any, energy: Any, units: str, quality: str, *, maximum_workspace_bytes=None, retained_bytes=0) -> tuple[str, bytes, dict[str, Any]]:
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
    display_height,display_width=matrix.shape
    if maximum_workspace_bytes is not None:
        native,_=_raster_forecast(matrix.shape,retained_bytes=retained_bytes)
        if native+retained_bytes > maximum_workspace_bytes:
            raise ValueError('native raster exceeds explicit workspace allowance')
        while _raster_forecast(matrix.shape,(display_height,display_width),retained_bytes=retained_bytes)[1] > maximum_workspace_bytes:
            if display_height==2 and display_width==2:
                raise ValueError('raster/output buffers exceed explicit workspace allowance')
            if display_height >= display_width and display_height>2: display_height=max(2,display_height//2)
            elif display_width>2: display_width=max(2,display_width//2)
            else: display_height=max(2,display_height//2)
    raw, annotation = _png(matrix)
    regular = all(np.allclose(np.diff(axis), (axis[-1] - axis[0]) / (len(axis) - 1),
                              rtol=1e-7, atol=max(abs(axis[0]), abs(axis[-1]), 1e-12) * 1e-12) for axis in (z, energy))
    raster = raw
    if not regular:
        # A stretched index raster would misplace nonuniform physical samples.
        # Uniform screen pixels instead inherit the nearest original E,z cell.
        xx = np.linspace(z[0], z[-1], display_width)
        yy = np.linspace(energy[0], energy[-1], display_height)
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


def materialize(analysis: SavedAnalysis, destination: Path, *, selections: Mapping[str, Sequence[int | slice | Document]], history: tuple[DiagnosticViewBatch, ...] = ()) -> Document:
    """Render explicit semantic blocks from a borrowed native5 reader only."""
    import numpy as np
    from .saved_views import AnalysisReader, _bounded
    from .diagnostic_views import DiagnosticViewBatch
    from .state import _metadata_weight
    from qcl_negf_contracts.messages import ContractError
    if not isinstance(analysis,AnalysisReader): raise TypeError('explicit SavedAnalysis required')
    if not isinstance(selections,Mapping) or not selections: raise ValueError('explicit nonempty semantic selections required')
    if not isinstance(history,tuple) or any(not isinstance(v,DiagnosticViewBatch) for v in history):
        raise TypeError('explicit DiagnosticViewBatch tuple required')
    with analysis._operation():
        description=analysis.describe()
        if any(v.source != description['source_state'] for v in history):
            raise ContractError('history belongs to another selected StateRef','identity_mismatch')
        normalized={}
        for name,selection in selections.items():
            if name not in description['views']: raise ValueError('unknown explicit semantic view')
            view=description['views'][name]
            if view['availability']!='available': raise ContractError(str(view['issue']),'capability_unavailable')
            shape=view['shape'][:1] if name.endswith('_populations') else view['shape']
            _,_,normalized[name]=analysis._normalize(selection,shape)
        history_key=[dict(source=v.source,records=v.records,rows=v.rows,witness_updates=v.witness_updates,
            cursor=v.cursor,at_cutoff=v.at_cutoff) for v in history]
        retained=_metadata_weight((description,history_key,normalized))
        if retained>analysis.limits.maximum_metadata_bytes: raise ContractError('retained render metadata exceeds allowance','budget_exceeded')
        _bounded([dict(source=v.source,rows=v.rows,cursor=v.cursor) for v in history],analysis.limits.maximum_json_bytes)
        effective=min(analysis.limits.maximum_selected_bytes,analysis.reader.limits.maximum_array_bytes)
        workspace=16*effective+8*analysis.limits.maximum_cells
        binding=dict(source_state=description['source_state'],identity=description['identity'],invocation=description['invocation'],
            artifact_ref=description['artifact_ref'],checkpoint_ref=description['checkpoint_ref'],diagnostics_ref=description['diagnostics_ref'],
            conventions=description['conventions'],recorded_statuses=description['recorded_statuses'],
            limits=description['limits'],selections=normalized,history=history_key,renderer_revision=RENDERER_REVISION)
        memo=getattr(analysis.reader,'_materialized_point_memo',None)
        memo_weight=_metadata_weight(memo) if memo is not None else 0
        if retained+memo_weight+13*_json_tree_weight(binding)+4096>workspace:
            context='native raster' if 'spatial_energy_density' in selections else 'render'
            raise ContractError(context+' binding and memo exceed workspace allowance','budget_exceeded')
        key=hashlib.sha256(json.dumps(binding,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        # This one-entry, reader-lifetime memo comes ONLY from a qualified cold
        # publication. Bytes detach it from returned dictionaries/caller aliases.
        # A new key drops the previous entry before any new payload allocation.
        if memo is not None and memo[0]!=key:
            del analysis.reader._materialized_point_memo
            memo=None
        memo_weight=_metadata_weight(memo) if memo is not None else 0
        retained+=_metadata_weight(binding)
        # Forecast selected heatmap, finite response lists, coordinate arrays and
        # output buffers from authenticated metadata BEFORE any numeric read.
        for name,selection in selections.items():
            view=description['views'][name]
            items,counts,_=analysis._normalize(selection,view['shape'][:1] if name.endswith('_populations') else view['shape'])
            if name=='spatial_energy_density':
                if len(counts)!=2 or any(type(v) is int for v in items) or min(counts)<2:
                    raise ValueError('heatmap requires explicit E,z ranges of at least two samples')
                forecast=retained+memo_weight+128*math.prod(counts)+256*sum(counts)+4096
                native,_=_raster_forecast(tuple(counts),retained_bytes=forecast)
                if native+forecast>workspace: raise ContractError('native raster exceeds workspace allowance','budget_exceeded')
        if retained+memo_weight>workspace:
            raise ContractError('render metadata and memo exceed workspace allowance','budget_exceeded')
        target=inside(Path(destination),key); index_path=inside(target,'index.json')
        if memo is not None and index_path.exists():
            # Exact bytes are compared before decoding untrusted JSON. No digest
            # stored beside the untrusted metadata supplies its own authority.
            # The 64L reserve covers decoded JSON containers; 4L covers a Unicode
            # decode temporary. This is engineering workspace, not allocator RSS.
            length=len(memo[1])
            if retained+memo_weight+69*length+8192>workspace:
                raise ContractError('cached metadata and memo exceed workspace allowance','budget_exceeded')
            with _cache_file(index_path,maximum_size=length,expected_size=length) as (stream,_):
                payload=stream.read(length+1)
            if payload!=memo[1]:
                raise ContractError('cached metadata differs from qualified source/contract reader memo','corrupt_result')
            del payload
            cached=json.loads(memo[1])
            for figure in cached['figures']:
                for path_key,hash_key in (('path','sha256'),('raw_path','raw_sha256'),('monitor_path','monitor_sha256')):
                    if path_key not in figure: continue
                    asset=inside(Path(destination),figure[path_key])
                    available=workspace-retained-memo_weight-_metadata_weight(cached)-4096
                    with _cache_file(asset,maximum_size=max(0,available//2)) as (stream,size):
                        if available<0:
                            raise ContractError('cached asset containers exceed workspace allowance','budget_exceeded')
                        digest=hashlib.sha256();remaining=size
                        while remaining:
                            chunk=stream.read(min(65536,remaining))
                            if not chunk:raise ContractError('cached asset shortened during bounded read','corrupt_result')
                            digest.update(chunk);remaining-=len(chunk)
                            del chunk
                        if stream.read(1):raise ContractError('cached asset grew during bounded read','corrupt_result')
                        if digest.hexdigest()!=figure[hash_key]:raise ValueError('cached asset hash differs')
            analysis._publication_guard(description['source_state'],(description['artifact_ref'],description['diagnostics_ref']))
            return cached
        # An index left by another reader lifetime has no qualified memo here;
        # recreate it from explicit selected source data instead of trusting it.
        if memo is not None:
            del analysis.reader._materialized_point_memo
            memo=None
            memo_weight=0
        target.mkdir(parents=True,exist_ok=True)
        summary=description['recorded_statuses']['current'];quality=summary['quality'];figures=[]
        def publish(identifier,title,text,source,units,annotation=None,payload=None):
            if len(text.encode('utf-8'))>workspace: raise ContractError('SVG exceeds workspace/output allowance','budget_exceeded')
            filename=identifier+'.svg'; data=text.encode('utf-8');(target/filename).write_bytes(data)
            figure=dict(id=identifier,title=title,path=f'{key}/{filename}',sha256=hashlib.sha256(data).hexdigest(),
                media_type='image/svg+xml',source_state=description['source_state'],raw_selections=source,
                units=units,quality=quality)
            if annotation: figure.update(annotation)
            if payload is not None:
                (target/(identifier+'.png')).write_bytes(payload)
                figure.update(raw_path=f'{key}/{identifier}.png',raw_sha256=hashlib.sha256(payload).hexdigest())
            figures.append(figure)
        for name,selection in selections.items():
            result=analysis.read(name,selection)
            values=np.asarray(result['values'],dtype=float); axes=result['axes']; raw=[result['raw_selection']]
            if name=='spatial_energy_density':
                forecast=retained+_metadata_weight(result)+values.nbytes+sum(8*len(v['values']) for v in axes)
                text,payload,annotation=_heatmap_svg(values,axes[1]['values'],axes[0]['values'],result['units'],quality,
                    maximum_workspace_bytes=workspace,retained_bytes=forecast)
                publish('energy-density','Энергетическая и пространственная плотность',text,raw,
                    dict(x='z, nm',y='E, eV',value=result['units']),annotation,payload)
                del text,payload,annotation
            elif values.ndim in (1,2):
                coordinate=next(v for v in axes if v['name']==result['logical_axes'][0])
                xs=np.asarray(coordinate['values'],dtype=float)
                curves=[(name,xs,values)] if values.ndim==1 else [(name+f' [{i}]',xs,values[:,i]) for i in range(values.shape[1])]
                text=_svg(name,coordinate['name']+', '+coordinate['units'],result['units'],curves,quality)
                identifier=('potentials' if name=='potential_total' else 'wavefunctions' if name=='effective_wavefunctions_real'
                    else 'spectra' if name=='spectral_trace' else name.replace('_','-'))
                publish(identifier,name,text,raw,dict(x=coordinate['units'],y=result['units']))
                del text,curves,xs,coordinate
            else: raise ValueError('renderer requires a line or two dimensional semantic selection')
            del result,values,axes
        for stage in ('scba','outer'):
            entries=[];names=set();previous_sequence={};seen_rows=set()
            for batch in history:
                def row_id(value):
                    return json.dumps([value['identity'],value['invocation']['source_epoch'],value['row_key'],value['stream_ordinal']],sort_keys=True)
                by_key={row_id(v):v for v in batch.records}
                for row in batch.rows:
                    row_lookup=json.dumps([row['source_identity'],row['source_epoch'],row['row_key'],row['stream_ordinal']],sort_keys=True)
                    record=by_key.get(row_lookup)
                    if record is None: raise ValueError('history row has no explicit raw record')
                    core=record['core']['fields']
                    if core['stage']['value']!=stage: continue
                    native_source=json.dumps([record['identity'],record['invocation']],sort_keys=True)
                    sequence=int(core['sequence']['decimal'])
                    if row_lookup in seen_rows or sequence<=previous_sequence.get(native_source,-1) or row['sequence']!=core['sequence']:
                        raise ValueError('explicit native history append sequence is duplicated or unordered')
                    seen_rows.add(row_lookup);previous_sequence[native_source]=sequence
                    from qcl_negf_contracts.results import validate_scientific_record
                    from .diagnostic_views import _pairs
                    validate_scientific_record(record)
                    units=_pairs(core['units']);missing=_pairs(core['missing_reasons'])
                    for field,cell in row['cells'].items():
                        tag=core['row']['fields'][field]
                        value=struct.unpack('>d' if tag['type']=='f64' else '>f',bytes.fromhex(tag['bits']))[0]
                        if cell['tagged']!=tag or (cell['value']!=value if math.isfinite(value) else cell['value'] is not None):
                            raise ValueError('history finite projection differs from explicit raw record')
                        if cell['units']!=units.get(field) or cell['missing_reason']!=missing.get(field):
                            raise ContractError('history units/missing projection differs from explicit raw pairs','corrupt_result')
                    cycle=(json.dumps(record['identity'],sort_keys=True),record['invocation']['source_epoch'],int(core['outer_iteration']['decimal']),
                        json.dumps([core[k] for k in ('domain_revision','discretization_id','basis_id')],sort_keys=True))
                    entries.append((record,row,cycle));names.update(row['cells'])
            if not entries: continue
            if len(entries)>analysis.limits.maximum_line_points: raise ContractError('explicit history row allowance exceeded','budget_exceeded')
            curves=[];boundaries=[];cycles=[]
            for index,(record,row,cycle) in enumerate(entries):
                if not cycles or cycles[-1]!=cycle:
                    cycles.append(cycle)
                    if index: boundaries.append((float(row['sequence']['decimal']),f"μ={cycle[2]} · a={record['identity']['attempt']}"))
            for field in sorted(names):
                xs=[];ys=[];previous_units=None
                for record,row,cycle in entries:
                    cell=row['cells'].get(field);x=float(row['sequence']['decimal'])
                    units=cell['units'] if cell else None
                    if (row['break_before'] or units!=previous_units) and xs: xs.append(float('nan'));ys.append(float('nan'))
                    xs.append(x);ys.append(float(cell['value']) if cell and cell['value'] is not None and cell['missing_reason'] is None else float('nan'))
                    previous_units=units
                curves.append((field,xs,ys))
            # Residuals use log display; J and other direct stored scalars
            # remain signed linear values with their recorded units.
            for field,xs,ys in curves:
                if field.startswith('r_'):continue
                units={row['cells'][field]['units'] for _,row,_ in entries if field in row['cells']}
                unit=next(iter(units)) if len(units)==1 else 'stored units vary'
                unit=unit if unit is not None else 'units not recorded'
                text=_svg(field+': '+stage,'Сохранённая последовательность событий',unit,[(field,xs,ys)],quality)
                publish('history-'+field+'-'+stage,field+': '+stage,text,[v[1]['row_key'] for v in entries],
                    dict(x='stored event sequence',y=unit),dict(history_rows=len(entries),history_cycle_count=len(cycles),
                        history_scope='explicit bounded batches; no all-history completeness inference'))
                del text
            curves=[curve for curve in curves if curve[0].startswith('r_')]
            names={curve[0] for curve in curves}
            if not curves:continue
            latest=entries[-1][2];monitor=[v for v in entries if v[2]==latest]
            native_iteration='inner_iteration' if stage=='scba' else 'outer_iteration'
            monitor_curves=[]
            for field in sorted(names):
                xs=[];ys=[]
                for record,row,cycle in monitor:
                    cell=row['cells'].get(field)
                    xs.append(float(record['core']['fields'][native_iteration]['decimal']))
                    ys.append(float(cell['value']) if cell and cell['value'] is not None and cell['missing_reason'] is None else float('nan'))
                monitor_curves.append((field,xs,ys))
            first,last=monitor[0],monitor[-1]
            context=dict(phase=stage,attempt=last[0]['identity']['attempt'],outer_iteration=latest[2],
                first_iteration=int(first[0]['core']['fields'][native_iteration]['decimal']),last_iteration=int(last[0]['core']['fields'][native_iteration]['decimal']),
                first_sequence=int(first[1]['sequence']['decimal']),last_sequence=int(last[1]['sequence']['decimal']),rows=len(monitor))
            text=_svg('История сходимости: '+stage,'Сохранённая последовательность событий','невязка, log₁₀',curves,quality,logarithmic=True,boundaries=boundaries)
            monitor_text=_svg('История сходимости: '+stage,'Итерация SCBA, ν' if stage=='scba' else 'Итерация внешнего цикла, μ',
                'невязка, log₁₀',monitor_curves,quality,logarithmic=True,monitor=True)
            monitor_path='convergence-'+stage+'-monitor.svg';monitor_data=monitor_text.encode('utf-8')
            if len(monitor_data)>workspace:raise ContractError('monitor output exceeds allowance','budget_exceeded')
            (target/monitor_path).write_bytes(monitor_data)
            annotation=dict(history_rows=len(entries),history_cycle_count=len(cycles),monitor_context=context,
                monitor_path=f'{key}/{monitor_path}',monitor_sha256=hashlib.sha256(monitor_data).hexdigest(),
                history_scope='explicit bounded batches; no all-history completeness inference')
            publish('convergence-'+stage,'История сходимости: '+stage,text,[v[1]['row_key'] for v in entries],
                dict(x='stored event sequence',y='stored scalar; logarithmic display'),annotation)
            del text,monitor_text,monitor_data,curves,monitor_curves
        # No D1 numerical integral scan: only explicit stored tagged diagnostics.
        diagnostic=analysis.raw._reader
        from .final_diagnostics import FinalDiagnosticsReader
        stored=FinalDiagnosticsReader(diagnostic,artifact_ref=description['diagnostics_ref']).describe().declaration
        result=dict(schema='qcl-negf.materialized-point.v4',contract_set='qcl-negf.results.v2',cache_key=key,
            source_binding=binding,source_state=description['source_state'],identity=description['identity'],
            renderer_revision=RENDERER_REVISION,quality=quality,scientific_accepted=summary['scientific_accepted'],
            presentation_ready=True,figures=figures,validation=stored['solution_metrics'],positivity=stored['positivity'])
        weight=_metadata_weight(result)
        if retained+weight+12*_json_tree_weight(result)+4096>workspace:
            raise ContractError('new render memo/index exceeds workspace allowance','budget_exceeded')
        _bounded(result,analysis.limits.maximum_json_bytes)
        # Shared reader checks every retained owner before publishing the index.
        analysis.describe()
        atomic_json(index_path,result)
        analysis._publication_guard(description['source_state'],(description['artifact_ref'],description['diagnostics_ref']))
        # JSON string + UTF-8 buffers are conservatively reserved at 12 times
        # retained object weight (escape expansion/Unicode storage), before
        # encoding. The memo is immutable and contains no caller-owned objects.
        payload=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
        if len(payload)>analysis.limits.maximum_json_bytes or retained+weight+len(payload)+256>workspace:
            raise ContractError('new render memo exceeds output allowance','budget_exceeded')
        analysis._publication_guard(description['source_state'],(description['artifact_ref'],description['diagnostics_ref']))
        analysis.reader._materialized_point_memo=(key,payload)
        return result


def materialize_series(points: list[dict[str, Any]], destination: Path) -> list[dict[str, Any]]:
    """Render already-published scalar currents; invalid points break the line."""
    from .presentation import accepted_result
    groups: dict[tuple[str, Any, Any], list[dict[str, Any]]] = {}
    for point in points:
        coordinates = point.get("coordinates") or {}
        key = (point.get("execution_id", "unknown"), coordinates.get("temperature_K"), coordinates.get("branch"))
        groups.setdefault(key, []).append(point)
    curves = []
    for key, rows in groups.items():
        rows.sort(key=lambda p: p["coordinates"].get("order", 0))
        xs = [p["coordinates"]["voltage_per_period_V"] for p in rows]
        ys = [p.get("observables",{}).get("current_density_A_per_m2") if accepted_result(p) and type(p.get("observables",{}).get("current_density_A_per_m2")) in (float,int) else float("nan") for p in rows]
        curves.append((f"{key[0]} · {key[1]} K · {key[2]}", xs, ys))
        for point in rows:
            if not accepted_result(point) and type(point.get("observables",{}).get("current_density_A_per_m2")) in (float,int):
                curves.append((f"{point['id']} · непринята", [point["coordinates"]["voltage_per_period_V"]], [point["observables"]["current_density_A_per_m2"]]))
    if not curves:
        return []
    def typed_key(value):
        if type(value) is float:return ['f64',struct.pack('>d',value).hex()]
        if isinstance(value,dict):return ['map',[[k,typed_key(v)] for k,v in sorted(value.items())]]
        if isinstance(value,list):return ['list',[typed_key(v) for v in value]]
        return [type(value).__name__,value]
    render_key = hashlib.sha256(json.dumps(typed_key(points), sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + CONTRACT_SET.encode() + RENDERER_REVISION.encode()).hexdigest()
    target = destination / render_key
    target.mkdir(parents=True, exist_ok=True)
    path = target / "current-voltage.svg"
    if not path.is_file():
        path.write_text(_svg("Ток и напряжение на период", "V периода, V", "J, A/m²", curves, "научно принятые точки соединены; непринятые показаны отдельно"), encoding="utf-8")
    return [{"id": "current-voltage", "title": "Ток и напряжение на период", "path": f"{render_key}/current-voltage.svg", "media_type": "image/svg+xml", "source_commit": render_key, "units": {"x": "V", "y": "A/m²"}}]
