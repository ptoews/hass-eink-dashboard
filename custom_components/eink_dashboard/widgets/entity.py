# Copyright 2026 Andreas Schneider
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entity widget context builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import markupsafe

from ..conditions import check_conditions
from ..const import (
    COLOR_GRAY,
    DEFAULT_CARD_STYLE,
    DEFAULT_ROW_H,
    PADDING,
    DisplayConfig,
    Widget,
    color_to_hex,
)
from ._helpers import (
    _card_insets,
    _color_context,
    _fmt,
    _metrics_context,
    _resolve_icon_style,
    _resolve_icon_svg,
    _widget_dim,
)

# ── Entity type-scale design tokens ──────────────────────────
# Roboto's cap height as a fraction of em.  The type scale is
# solved against cap heights rather than nominal em boxes so the
# rendered ink fills the card instead of the font's invisible
# line box.
CAP = 0.711
# Name font as a fraction of the value font, and the baseline gap
# between the two lines as a fraction of the name font.
NAME_RATIO = 0.28
GAP_RATIO = 0.34
# Unit font and its gap from the value, both relative to the
# value font so the pairing holds at every widget size.
UNIT_RATIO = 0.46
UNIT_GAP_RATIO = 0.10
# Inline arrangement: icon diameter relative to the text block
# height, its gap to the text, and a hard cap on the share of the
# card width the icon column may take on narrow cards.
INLINE_ICON_RATIO = 0.92
INLINE_ICON_GAP = 0.34
INLINE_ICON_MAX_W = 0.24
# Stacked arrangement: the icon sits on the name's line, so it is
# sized against the name font instead of the whole text block.
STACK_ICON_RATIO = 1.6
STACK_ICON_GAP = 0.42
# How far the name may shrink below its ideal size to avoid being
# truncated.
NAME_MIN_SHRINK = 0.75
# The stacked arrangement only wins when it buys a materially
# bigger value; below this margin the inline arrangement (the
# widget's default visual identity) is kept.
STACK_MARGIN = 1.15
MIN_FONT = 10


def _text_w(size: int, text: str, *, medium: bool, bold: bool) -> float:
    """Measure the rendered width of a string.

    Widths are measured with PIL against the same Roboto faces resvg
    draws with, so the type scale can be fitted to the card before
    any SVG is emitted.

    Args:
        size: Font size in pixels.
        text: The string to measure.
        medium: Measure against Roboto Medium (weight 500).
        bold: Measure against Roboto Bold; takes precedence over
            ``medium``.

    Returns:
        Width in pixels.
    """
    from ..render import _load_font

    return _load_font(size, medium=medium, bold=bold).getlength(text)


def _value_block_w(
    size: int, value_text: str, unit_text: str, value_bold: bool
) -> float:
    """Measure the value and its trailing unit as one block.

    The unit rides along at ``UNIT_RATIO`` of the value size, so it
    has to be part of the width budget the value is fitted against —
    fitting the value alone would push the unit off the card.

    Args:
        size: Value font size in pixels.
        value_text: The value string.
        unit_text: The unit string, or ``""`` when there is no unit.
        value_bold: Whether the value renders in bold.

    Returns:
        Combined width in pixels, including the gap before the unit.
    """
    w = _text_w(size, value_text, medium=not value_bold, bold=value_bold)
    if unit_text:
        w += max(2, round(size * UNIT_GAP_RATIO)) + _text_w(
            max(MIN_FONT, round(size * UNIT_RATIO)),
            unit_text,
            medium=False,
            bold=False,
        )
    return w


def _fit_value_size(
    start: int,
    value_text: str,
    unit_text: str,
    value_bold: bool,
    avail_w: int,
) -> int:
    """Find the largest value size whose value+unit fits a width.

    Text width is monotonic in font size, so a binary search over
    the size range settles it in a handful of measurements instead
    of stepping down one pixel at a time.

    Args:
        start: Upper bound, the height-driven size.
        value_text: The value string.
        unit_text: The unit string, or ``""`` for no unit.
        value_bold: Whether the value renders bold.
        avail_w: Width budget in pixels.

    Returns:
        A font size in ``[MIN_FONT, start]``.
    """
    if _value_block_w(start, value_text, unit_text, value_bold) <= avail_w:
        return start
    lo, hi = MIN_FONT, start
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _value_block_w(mid, value_text, unit_text, value_bold) <= avail_w:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _cap_h(size: int, *, medium: bool = False, bold: bool = False) -> int:
    """Measure Roboto's cap height — digit ink above the baseline.

    The layout is built on cap heights rather than nominal font
    sizes because an em box carries invisible leading above and
    below the ink; centring the em box leaves the visible text
    looking small and off-centre in the card.

    Args:
        size: Font size in pixels.
        medium: Measure against Roboto Medium (weight 500).
        bold: Measure against Roboto Bold; takes precedence over
            ``medium``.

    Returns:
        Cap height in pixels, at least 1.
    """
    from ..render import _load_font

    font = _load_font(size, medium=medium, bold=bold)
    # Digits sit flat on the baseline with nothing below it, so the
    # ink box of "0" spans exactly cap-top to baseline.  Measuring
    # it directly also avoids getmetrics(), which the bitmap font
    # _load_font() falls back to does not implement.
    _, top, _, bottom = font.getbbox("0")
    return max(1, round(bottom - top))


def _fit_name(text: str, size: int, avail_w: int) -> tuple[str, int]:
    """Fit a name to a width, shrinking first and truncating last.

    A name a little too wide reads better one step smaller than
    chopped, so the size walks down to ``NAME_MIN_SHRINK`` of its
    place in the type scale before an ellipsis is spent.  The value
    is never involved: only the name gives way.

    Args:
        text: The name string.
        size: The ideal name font size from the type scale.
        avail_w: Width budget in pixels.

    Returns:
        ``(text, size)`` — the string to draw and the size to draw
        it at.
    """
    floor = max(MIN_FONT, round(size * NAME_MIN_SHRINK))
    fitted = size
    while (
        fitted > floor
        and _text_w(fitted, text, medium=True, bold=False) > avail_w
    ):
        fitted -= 1
    if _text_w(fitted, text, medium=True, bold=False) <= avail_w:
        return text, fitted
    out = text
    while (
        out
        and _text_w(fitted, out + "\u2026", medium=True, bold=False) > avail_w
    ):
        out = out[:-1]
    return (out + "\u2026") if out else "", fitted


def _build_entity_context(
    widget: Widget,
    config: DisplayConfig,
) -> dict[str, object]:
    """Build Jinja2 template context for the entity widget.

    The state value and its unit are the black, dominant element;
    the entity name is smaller and gray, and the icon is secondary
    to both, so its non-filled states render gray rather than black
    to match the name's weight.

    The type scale is not a fixed fraction of the widget height: it
    is solved so the ink block — measured cap heights plus the
    baseline gap between the lines — fills the card's content box,
    then clamped down so the value and unit fit the width available
    to them.  Text is placed by baseline against those same cap
    heights, so what gets centred in the card is the visible ink
    rather than the font's line box.

    Two arrangements are costed against that box and the one giving
    the larger value wins, since the value is what the widget
    exists to show:

    - ``"inline"`` — the icon takes its own column at the left
      content edge, vertically centred on the card, with the
      value/unit and name stacked beside it.
    - ``"stacked"`` — the icon moves onto the name's line so the
      value spans the full content width.  ``name_position``
      decides whether that name row sits above or below the value.

    A tall or narrow card chokes the inline value against the icon
    column, and that is where stacked wins; a wide, short card has
    no room for a separate name row and stays inline.  ``layout``
    pins the choice when a dashboard needs its cards to agree
    rather than each fitting itself.

    Icon style controls circle rendering, with automatic resolution
    based on entity state when ``icon_style`` is omitted:

    - ``"filled"`` — gray-filled circle, white glyph (default for
      active states when ``display_levels > 2``).
    - ``"outlined"`` — white circle with gray stroke, gray glyph
      (default for inactive states and all 2-level displays).
    - ``"none"`` — no circle; icon glyph rendered in gray.

    When ``invert_condition`` inverts the widget, the icon renders
    white on the solid black card regardless of style, matching the
    value/unit/name.

    Args:
        widget: Widget config dict.  Recognised keys:
            ``entity`` (HA entity ID, required),
            ``name`` (display name override),
            ``icon`` (MDI icon name, e.g. ``"mdi:thermometer"``),
            ``hide_icon`` (suppress the icon; default ``False``),
            ``hide_name`` (suppress the entity name text; default
            ``False``),
            ``attribute`` (attribute key to show as value instead
            of state),
            ``unit`` (unit string override),
            ``icon_style`` (``"filled"`` / ``"outlined"`` /
            ``"none"``),
            ``bold_value`` (render the state value in bold;
            default ``False``),
            ``name_position`` (``"top"`` / ``"bottom"``; default
            ``"bottom"`` — position of the name relative to the
            value+unit line),
            ``name_align`` (``"left"`` / ``"right"``; default
            ``"left"``),
            ``layout`` (``"auto"`` / ``"inline"`` / ``"stacked"``;
            default ``"auto"`` — pins the arrangement instead of
            letting it be chosen per card),
            ``invert_condition`` (list of Lovelace condition dicts,
            same format as ``visibility``; when non-empty and all
            conditions are met the widget renders inverted — solid
            black card, white text/icon — as an e-ink "needs
            attention" signal),
            ``card_style``, ``x``, ``w``, ``h``.
        config: Display config with ``width``, ``states``, and
            ``display_levels``.

    Returns:
        Template context dict consumed by ``entity.svg.j2``.
        Returns ``{"w": …, "h": …, "has_entity": False,
        **_color_context()}`` when the entity is missing.
        Full context includes widget dimensions, card style,
        metrics, colors, icon geometry, value/unit/name text and
        geometry, the ``invert`` flag, and ``stacked`` — which
        arrangement was chosen.
    """
    from ..render import _compute_metrics

    x = widget.get("x", PADDING)
    svg_w = _widget_dim(widget, "w", config["width"] - x)
    svg_h = _widget_dim(widget, "h", 2 * DEFAULT_ROW_H)
    entity_id: str = widget.get("entity", "")
    name_override = widget.get("name")
    icon_override = widget.get("icon")
    unit_override = widget.get("unit")
    attribute: str | None = widget.get("attribute")
    hide_icon: bool = widget.get("hide_icon", False)
    hide_name: bool = widget.get("hide_name", False)
    icon_style = widget.get("icon_style")
    card_style = widget.get("card_style", DEFAULT_CARD_STYLE)
    value_bold: bool = widget.get("bold_value", False)
    name_position = widget.get("name_position", "bottom")
    name_align = widget.get("name_align", "left")
    layout = widget.get("layout", "auto")
    states = config.get("states", {})
    display_levels = config.get("display_levels", 16)

    state = states.get(entity_id) if entity_id else None
    if state is None:
        return {
            "w": svg_w,
            "h": svg_h,
            "has_entity": False,
            "invert": False,
            **_color_context(),
        }

    colors = _color_context()

    # ── Layout tokens ────────────────────────────────────────
    # Card chrome (padding, radius, border, accent bar) scales off
    # the card's smaller dimension halved — a row-equivalent
    # reference.  Deriving it from the height alone gives a tall
    # card padding wider than its own content column, and deriving
    # it from the full height (rather than one row of it) is what
    # inflated the insets and accent bar in the first place.
    row_ref = max(1, min(svg_w, svg_h) // 2)
    m = _compute_metrics(row_ref)
    x_off, r_inset, bar_width = _card_insets(m, card_style, display_levels)
    lpad = m.padding if x_off == 0 else 0
    rpad = m.padding if r_inset == 0 else 0

    attrs = state.get("attributes", {})
    domain = entity_id.split(".")[0]
    state_val: str = state.get("state", "")

    name_text: str = (
        str(name_override)
        if name_override is not None
        else attrs.get("friendly_name", entity_id)
    )

    if attribute is not None:
        raw_val = attrs.get(attribute)
        value_text = (
            _fmt(str(raw_val), config)
            if raw_val is not None and raw_val != ""
            else "unknown"
        )
        auto_unit = ""
    else:
        value_text = _fmt(state_val, config)
        auto_unit = attrs.get("unit_of_measurement", "")
    unit_text: str = (
        str(unit_override) if unit_override is not None else auto_unit
    )

    invert_condition = widget.get("invert_condition")
    invert = bool(invert_condition) and check_conditions(
        invert_condition, states
    )

    # ── Content box ──────────────────────────────────────────
    content_l = x_off + lpad
    content_r = svg_w - r_inset - rpad
    content_w = max(1, content_r - content_l)
    content_h = max(8, svg_h - 2 * m.padding)
    show_name = not hide_name and bool(name_text)

    # ── Arrangement ──────────────────────────────────────────
    # Two arrangements are costed against the same box and the one
    # that yields the larger value wins, because the value is the
    # element the widget exists to show.  "inline" puts the icon in
    # its own column left of a value/name stack; "stacked" moves the
    # icon onto the name's line so the value spans the full card
    # width.  Tall or narrow cards choke the inline value against
    # the icon column, and that is exactly where stacked wins.
    inline_fv = _solve_inline(
        content_w,
        content_h,
        value_text,
        unit_text,
        value_bold,
        hide_icon,
        show_name,
    )
    stacked_fv = _solve_stacked(
        content_w,
        content_h,
        value_text,
        unit_text,
        value_bold,
        hide_icon,
        show_name,
    )
    if layout == "inline":
        stacked = False
    elif layout == "stacked":
        stacked = show_name
    else:
        stacked = show_name and stacked_fv > inline_fv * STACK_MARGIN

    value_font_sz = stacked_fv if stacked else inline_fv
    name_font_sz = max(MIN_FONT, round(value_font_sz * NAME_RATIO))
    unit_font_sz = max(MIN_FONT, round(value_font_sz * UNIT_RATIO))
    line_gap = max(1, round(name_font_sz * GAP_RATIO))
    cap_v = _cap_h(value_font_sz, medium=not value_bold, bold=value_bold)
    cap_n = _cap_h(name_font_sz, medium=True) if show_name else 0

    # ── Icon size ────────────────────────────────────────────
    if hide_icon:
        icon_dia = 0
        icon_gap = 0
    elif stacked:
        icon_dia = round(name_font_sz * STACK_ICON_RATIO)
        icon_gap = round(icon_dia * STACK_ICON_GAP)
    else:
        block_h = cap_v + (line_gap + cap_n if show_name else 0)
        icon_dia = min(
            round(block_h * INLINE_ICON_RATIO),
            round(content_w * INLINE_ICON_MAX_W),
        )
        icon_gap = round(icon_dia * INLINE_ICON_GAP)
    icon_inner = icon_dia * 60 // 100

    # ── Icon resolution ──────────────────────────────────────
    if hide_icon:
        icon_svg: markupsafe.Markup | str = ""
        letter = ""
        icon_outline = False
        icon_no_circle = True
    else:
        icon_svg, letter = _resolve_icon_svg(
            icon_override, attrs, state_val, domain, icon_inner, entity_id
        )
        icon_outline, icon_no_circle = _resolve_icon_style(
            icon_style, state_val, display_levels
        )
    # The inverted card forces the flat glyph so it draws cleanly
    # on the solid black background.
    if invert:
        icon_no_circle = True
        icon_outline = False
    # Ring weight tracks the circle, not the card's row metrics, so
    # the outline stays optically consistent at any icon size.
    icon_border = max(2, round(icon_dia * 0.0625)) if icon_dia else m.border
    icon_stroke_w = icon_border * 3 if display_levels <= 2 else icon_border
    icon_fill = color_to_hex(COLOR_GRAY)
    icon_r = icon_dia // 2

    # ── Placement ────────────────────────────────────────────
    # Both arrangements position text by baseline against measured
    # cap heights, so the ink block — not the font's line box — is
    # what gets centred in the card.
    if stacked:
        name_row_h = max(icon_dia, cap_n)
        block_h = name_row_h + line_gap + cap_v
        top = (svg_h - block_h) // 2
        if name_position == "top":
            row_top, value_y = top, top + name_row_h + line_gap + cap_v
        else:
            row_top = top + cap_v + line_gap
            value_y = top + cap_v
        icon_cy = row_top + name_row_h // 2
        icon_cx = content_l + icon_r
        name_baseline = row_top + (name_row_h + cap_n) // 2
        text_x0 = content_l
        name_left = content_l + (icon_dia + icon_gap if not hide_icon else 0)
    else:
        block_h = cap_v + (line_gap + cap_n if show_name else 0)
        top = (svg_h - block_h) // 2
        if not show_name:
            value_y = (svg_h + cap_v) // 2
            name_baseline = value_y
        elif name_position == "top":
            name_baseline = top + cap_n
            value_y = top + cap_n + line_gap + cap_v
        else:
            value_y = top + cap_v
            name_baseline = top + cap_v + line_gap + cap_n
        icon_cx = content_l + icon_r
        icon_cy = svg_h // 2
        text_x0 = content_l + (icon_dia + icon_gap if not hide_icon else 0)
        name_left = text_x0

    icon_glyph_x = icon_cx - icon_inner // 2
    icon_glyph_y = icon_cy - icon_inner // 2
    text_x1 = content_r

    value_x = text_x0
    unit_x = value_x
    if unit_text:
        unit_gap = max(2, round(value_font_sz * UNIT_GAP_RATIO))
        unit_x = (
            value_x
            + round(
                _text_w(
                    value_font_sz,
                    value_text,
                    medium=not value_bold,
                    bold=value_bold,
                )
            )
            + unit_gap
        )
    unit_y = value_y

    # Long names truncate rather than overflowing the card or
    # dragging the value down with them.
    if show_name:
        name_text, name_font_sz = _fit_name(
            name_text, name_font_sz, max(1, text_x1 - name_left)
        )
    if name_align == "right":
        name_x = text_x1
        name_anchor = "end"
    else:
        name_x = name_left
        name_anchor = "start"
    name_y = name_baseline

    return {
        "w": svg_w,
        "h": svg_h,
        "has_entity": True,
        "card_style": card_style,
        "bar_width": bar_width,
        "invert": invert,
        "stacked": stacked,
        **_metrics_context(m),
        **colors,
        # Icon geometry.
        "icon_svg": icon_svg,
        "icon_cx": icon_cx,
        "icon_cy": icon_cy,
        "icon_r": icon_r,
        "icon_stroke_w": icon_stroke_w,
        "icon_fill": icon_fill,
        "icon_outline": icon_outline,
        "icon_no_circle": icon_no_circle,
        "icon_glyph_x": icon_glyph_x,
        "icon_glyph_y": icon_glyph_y,
        "letter": letter,
        "letter_font_sz": icon_dia * 5 // 10,
        # Value + unit.
        "value_text": value_text,
        "value_x": value_x,
        "value_y": value_y,
        "value_font_sz": value_font_sz,
        "value_bold": value_bold,
        "unit_text": unit_text,
        "unit_x": unit_x,
        "unit_y": unit_y,
        "unit_font_sz": unit_font_sz,
        # Name.
        "name_text": name_text,
        "name_x": name_x,
        "name_y": name_y,
        "name_font_sz": name_font_sz,
        "name_anchor": name_anchor,
        "hide_name": not show_name,
    }


def _solve_inline(
    content_w: int,
    content_h: int,
    value_text: str,
    unit_text: str,
    value_bold: bool,
    hide_icon: bool,
    show_name: bool,
) -> int:
    """Value font size for the icon-left arrangement.

    The icon column is sized from the text block it sits beside, so
    icon width and value size depend on each other; the loop settles
    that in a few passes and exits as soon as it stops moving.

    Args:
        content_w: Width of the card's content box.
        content_h: Height of the card's content box.
        value_text: The value string.
        unit_text: The unit string, or ``""``.
        value_bold: Whether the value renders bold.
        hide_icon: Whether the icon column is suppressed.
        show_name: Whether the name line is rendered.

    Returns:
        The value font size in pixels.
    """
    factor = CAP + (NAME_RATIO * (GAP_RATIO + CAP) if show_name else 0.0)
    fv_height = max(MIN_FONT, int(content_h / factor))
    fv = fv_height
    for _ in range(4):
        if hide_icon:
            column = 0
        else:
            block_h = max(1, int(fv * factor))
            dia = min(
                round(block_h * INLINE_ICON_RATIO),
                round(content_w * INLINE_ICON_MAX_W),
            )
            column = dia + round(dia * INLINE_ICON_GAP)
        nxt = _fit_value_size(
            fv_height,
            value_text,
            unit_text,
            value_bold,
            max(1, content_w - column),
        )
        if nxt == fv:
            break
        fv = nxt
    return fv


def _solve_stacked(
    content_w: int,
    content_h: int,
    value_text: str,
    unit_text: str,
    value_bold: bool,
    hide_icon: bool,
    show_name: bool,
) -> int:
    """Value font size for the icon-on-the-name-line arrangement.

    The value spans the full content width here, so the width fit is
    a single pass; only the height budget has to account for the
    name row the icon shares.

    Args:
        content_w: Width of the card's content box.
        content_h: Height of the card's content box.
        value_text: The value string.
        unit_text: The unit string, or ``""``.
        value_bold: Whether the value renders bold.
        hide_icon: Whether the icon is suppressed.
        show_name: Whether the name line is rendered.

    Returns:
        The value font size in pixels.
    """
    if not show_name:
        return MIN_FONT
    row = CAP if hide_icon else max(CAP, STACK_ICON_RATIO)
    factor = CAP + NAME_RATIO * (row + GAP_RATIO)
    fv_height = max(MIN_FONT, int(content_h / factor))
    return _fit_value_size(
        fv_height, value_text, unit_text, value_bold, content_w
    )
