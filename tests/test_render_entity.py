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

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from custom_components.eink_dashboard.const import (
    COLOR_GRAY,
    DEFAULT_ROW_H,
)
from custom_components.eink_dashboard.render import (
    WidgetMetrics,
    _compute_metrics,
    render_dashboard,
)
from custom_components.eink_dashboard.svg_render import render_widget_svg
from custom_components.eink_dashboard.widgets._helpers import _card_insets
from custom_components.eink_dashboard.widgets.entity import (
    NAME_MIN_SHRINK,
    NAME_RATIO,
    _build_entity_context,
)
from tests.helpers import (
    assert_all_white,
    assert_card_border,
    assert_has_dark_pixels,
    assert_has_gray_pixels,
    assert_no_gray_pixels,
    assert_scales_proportionally,
    content_bbox,
    make_config,
    render_to_image,
)

if TYPE_CHECKING:
    from PIL import Image

MOCK_ENTITY_STATES = {
    "sensor.temperature": {
        "state": "22.5",
        "attributes": {
            "friendly_name": "Living Room",
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            # Extra attribute for attribute= display test.
            "humidity": 58,
        },
    },
    "binary_sensor.motion": {
        "state": "on",
        "attributes": {
            "friendly_name": "Motion",
            "device_class": "motion",
        },
    },
    "binary_sensor.front_door": {
        "state": "off",
        "attributes": {
            "friendly_name": "Front Door",
            "device_class": "door",
        },
    },
    "sensor.no_class": {
        "state": "99",
        "attributes": {
            "friendly_name": "Plain",
        },
    },
    "sensor.pressure": {
        "state": "1013",
        "attributes": {
            "friendly_name": "Barometric pressure outdoors north side",
            "device_class": "pressure",
            "unit_of_measurement": "hPa",
        },
    },
    # For invert_condition numeric_state tests.
    "sensor.count": {
        "state": "2",
        "attributes": {"friendly_name": "Count"},
    },
    # For invert_condition state_not tests.
    "sensor.status": {
        "state": "washing",
        "attributes": {"friendly_name": "Status"},
    },
}


def _band_bbox(
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    low: int,
    high: int,
    *,
    min_pixels: int = 20,
) -> tuple[int, int, int, int] | None:
    """Return the bbox of pixels within [low, high] in a region.

    Unlike ``content_bbox`` (any non-white pixel), this isolates a
    specific tone band so gray name text can be distinguished from
    black value/unit text even when both appear in the same region.
    A minimum pixel count is required before trusting a match —
    anti-aliased edges of black text blend through every gray tone
    for a pixel or two, so a handful of stray hits within the band
    are noise, not genuine gray content.

    Args:
        img: A grayscale ("L" mode) PIL image.
        x1: Left edge of the region.
        y1: Top edge of the region.
        x2: Right edge of the region.
        y2: Bottom edge of the region.
        low: Lower bound (exclusive) of the tone band.
        high: Upper bound (exclusive) of the tone band.
        min_pixels: Minimum number of matching pixels required for
            the match to count as real content rather than
            anti-aliasing noise.

    Returns:
        (left, top, right, bottom) of matching content in absolute
        image coordinates, or None if fewer than ``min_pixels`` pixels
        in the region fall within the band.
    """
    crop = img.crop((x1, y1, x2, y2))
    mask = crop.point(lambda p: 255 if low < p < high else 0)
    if sum(1 for v in mask.get_flattened_data() if v == 255) < min_pixels:
        return None
    bbox = mask.getbbox()
    if bbox is None:
        return None
    return (x1 + bbox[0], y1 + bbox[1], x1 + bbox[2], y1 + bbox[3])


def _entity_config(**overrides: object) -> dict[str, object]:
    """Display config for the Entity tests."""
    return make_config(
        {"width": 400, "height": 300, "states": MOCK_ENTITY_STATES},
        **overrides,
    )


def _ctx(widget: dict[str, object], config: dict[str, object]) -> dict:
    """Build the Entity template context for *widget*.

    Geometry assertions read the context rather than re-deriving the
    solver's arithmetic: the redesigned layout fits the type scale to
    the card, so positions cannot be predicted from a formula without
    duplicating the solver in the tests.

    Args:
        widget: Entity widget config dict.
        config: Display config dict.

    Returns:
        The context dict ``entity.svg.j2`` is rendered with.
    """
    return _build_entity_context(widget, config)


def _chrome(w: int, h: int) -> WidgetMetrics:
    """Card chrome metrics: the card's smaller side, halved.

    Padding, corner radius, border stroke and accent-bar width all
    derive from this reference, so a tall card no longer gets insets
    scaled to its height (the regression behind issue #103).

    Args:
        w: Widget width in pixels.
        h: Widget height in pixels.

    Returns:
        The ``WidgetMetrics`` the card frame is drawn from.
    """
    return _compute_metrics(max(1, min(w, h) // 2))


def _content_box(
    w: int, h: int, card_style: str = "none", display_levels: int = 16
) -> tuple[int, int, int]:
    """Left, right and height of a card's content box.

    Args:
        w: Widget width in pixels.
        h: Widget height in pixels.
        card_style: ``"border"``, ``"left_bar"`` or ``"none"``.
        display_levels: Display grayscale depth.

    Returns:
        ``(left, right, height)`` in widget-local pixels.
    """
    m = _chrome(w, h)
    x_off, r_inset, _ = _card_insets(m, card_style, display_levels)
    lpad = m.padding if x_off == 0 else 0
    rpad = m.padding if r_inset == 0 else 0
    return x_off + lpad, w - r_inset - rpad, h - 2 * m.padding


def _content_x_range(w: int, h: int) -> tuple[int, int]:
    """Left/right x-bounds of the inline text column.

    Args:
        w: Widget width in pixels.
        h: Widget height in pixels.

    Returns:
        ``(text_x0, text_x1)`` — where value/unit/name text starts in
        the inline arrangement, and the right content edge.
    """
    widget = {
        "type": "entity",
        "x": 0,
        "y": 0,
        "w": w,
        "h": h,
        "entity": "sensor.temperature",
        "layout": "inline",
    }
    ctx = _ctx(widget, _entity_config())
    _, right, _ = _content_box(w, h)
    return int(ctx["value_x"]), right


def _icon_ring(ctx: dict) -> tuple[int, int, int, int]:
    """Region inside the icon circle, above the glyph.

    Checks a window extending +-icon_r//2 from the circle's
    horizontal center.  The circle's own curve dips measurably below
    the top by the edge of that window — a geometric property of the
    circle, independent of stroke width — so the vertical inset is
    extended by that dip plus half the stroke.

    Args:
        ctx: A context dict from ``_build_entity_context``.

    Returns:
        ``(x1, y1, x2, y2)`` of the ring region.
    """
    icon_cx = int(ctx["icon_cx"])
    icon_cy = int(ctx["icon_cy"])
    icon_r = int(ctx["icon_r"])
    stroke_w = int(ctx["icon_stroke_w"])
    dx_max = icon_r // 2
    dip = icon_r - round((icon_r**2 - dx_max**2) ** 0.5)
    y1 = icon_cy - icon_r + dip + stroke_w // 2 + 2
    y2 = int(ctx["icon_glyph_y"]) - 1
    return icon_cx - dx_max + 3, y1, icon_cx + dx_max - 3, y2


class TestRenderEntity:
    # Verify rendering of the redesigned Entity widget: icon on the
    # left (vertically centered against the full widget height),
    # value+unit (both black) to the right of the icon, and the
    # entity name (gray, smaller) positioned above or below the
    # value+unit line per name_position/name_align.
    _DEFAULTS: ClassVar[dict[str, object]] = {
        "width": 400,
        "height": 300,
        "states": MOCK_ENTITY_STATES,
    }

    def _config(self, **overrides: object) -> dict[str, object]:
        return make_config(self._DEFAULTS, **overrides)

    # ── Structural tests ──────────────────────────────

    def test_entity_card_border(self) -> None:
        # Border style draws dark pixels on all four edges, with
        # the frame scaled to the card's smaller dimension.
        h = 112
        m = _chrome(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
                "card_style": "border",
            }
        ]
        img = render_to_image(widgets, self._config())
        assert_card_border(img, 400, h, m)

    def test_entity_card_left_bar(self) -> None:
        # Left_bar style draws gray pixels on the left edge;
        # the right edge should be white.
        h = 112
        m = _chrome(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
                "card_style": "left_bar",
            }
        ]
        img = render_to_image(widgets, self._config())
        assert_has_gray_pixels(
            img,
            0,
            2,
            m.left_bar,
            h - 2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )
        assert_all_white(img, 395, 0, 400, 1)

    def test_entity_card_none(self) -> None:
        # No-decoration style has white edges — only content
        # (name, icon, value) draws pixels inside the card.
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": 112,
                "entity": "sensor.temperature",
                "card_style": "none",
            }
        ]
        img = render_to_image(widgets, self._config())
        assert_all_white(img, 0, 0, 3, 3)
        assert_all_white(img, 397, 0, 400, 3)

    def test_entity_card_style_none_is_default(self) -> None:
        # Omitting card_style must produce byte-identical output to
        # card_style="none" (no card decoration drawn).
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
        }
        with_none = render_dashboard(
            [{**base, "card_style": "none"}], self._config()
        )
        without = render_dashboard([base], self._config())
        assert with_none == without

    # ── Icon style tests ──────────────────────────────
    # Use h=224 for a large enough icon circle to measure the ring
    # region reliably. The icon is now left-aligned and vertically
    # centered against the full widget height.

    def test_entity_icon_circle_gray_fill_active(self) -> None:
        # An active entity (state "on") without explicit icon_style
        # draws a filled gray circle on the left.
        h = 224
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "binary_sensor.motion",
            # hide_name isolates the icon ring from the name text,
            # which would otherwise confound the gray check.
            "hide_name": True,
        }
        config = self._config()
        rx1, ry1, rx2, ry2 = _icon_ring(_ctx(widget, config))
        img = render_to_image([widget], config)
        assert_has_gray_pixels(
            img,
            rx1,
            ry1,
            rx2,
            ry2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )

    def test_entity_icon_circle_outlined_inactive(self) -> None:
        # An inactive entity (state "off") without explicit
        # icon_style draws an outlined circle: white interior.
        h = 224
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "binary_sensor.front_door",
            # hide_name isolates the icon ring from the name text,
            # which would otherwise confound the gray check.
            "hide_name": True,
        }
        config = self._config()
        rx1, ry1, rx2, ry2 = _icon_ring(_ctx(widget, config))
        img = render_to_image([widget], config)
        assert_no_gray_pixels(
            img,
            rx1,
            ry1,
            rx2,
            ry2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )

    def test_entity_icon_style_filled_explicit(self) -> None:
        # icon_style="filled" forces a gray-filled circle even for
        # an inactive entity.
        h = 224
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "binary_sensor.front_door",
            "icon_style": "filled",
            # hide_name isolates the icon ring from the name text,
            # which would otherwise confound the gray check.
            "hide_name": True,
        }
        config = self._config()
        rx1, ry1, rx2, ry2 = _icon_ring(_ctx(widget, config))
        img = render_to_image([widget], config)
        assert_has_gray_pixels(
            img,
            rx1,
            ry1,
            rx2,
            ry2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )

    def test_entity_icon_style_outlined_explicit(self) -> None:
        # icon_style="outlined" forces an outlined circle even for
        # an active entity: no gray in the ring.
        h = 224
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "binary_sensor.motion",
            "icon_style": "outlined",
            # hide_name isolates the icon ring from the name text,
            # which would otherwise confound the gray check.
            "hide_name": True,
        }
        config = self._config()
        rx1, ry1, rx2, ry2 = _icon_ring(_ctx(widget, config))
        img = render_to_image([widget], config)
        assert_no_gray_pixels(
            img,
            rx1,
            ry1,
            rx2,
            ry2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )

    def test_entity_icon_style_none_no_circle(self) -> None:
        # icon_style="none" suppresses the circle entirely.
        h = 224
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "binary_sensor.motion",
            "icon_style": "none",
            # hide_name isolates the icon ring from the name text,
            # which would otherwise confound the gray check.
            "hide_name": True,
        }
        config = self._config()
        rx1, ry1, rx2, ry2 = _icon_ring(_ctx(widget, config))
        img = render_to_image([widget], config)
        assert_no_gray_pixels(
            img,
            rx1,
            ry1,
            rx2,
            ry2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )

    def test_entity_2level_always_outlined(self) -> None:
        # On a 2-level display the auto-switch forces "outlined"
        # even for an active entity (state "on").
        h = 224
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "binary_sensor.motion",
            # hide_name isolates the icon ring from the name text,
            # which would otherwise confound the gray check.
            "hide_name": True,
        }
        config = self._config(display_levels=2)
        rx1, ry1, rx2, ry2 = _icon_ring(_ctx(widget, config))
        img = render_to_image([widget], config)
        assert_no_gray_pixels(
            img,
            rx1,
            ry1,
            rx2,
            ry2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )

    def test_entity_hide_icon_suppresses_icon(self) -> None:
        # hide_icon=True must emit no circle, glyph or letter
        # fallback at all.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "sensor.no_class",
            "hide_icon": True,
        }
        config = self._config()
        ctx = _ctx(widget, config)
        assert ctx["icon_svg"] == ""
        assert ctx["letter"] == ""
        assert "<circle" not in render_widget_svg(widget, config)

    def test_entity_hide_icon_with_icon_style(self) -> None:
        # hide_icon=True must suppress the icon even when icon_style
        # is set explicitly — the style flag must not override the
        # hide decision.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "sensor.no_class",
            "hide_icon": True,
            "icon_style": "filled",
        }
        config = self._config()
        assert "<circle" not in render_widget_svg(widget, config)

    def test_entity_hide_icon_collapses_column(self) -> None:
        # hide_icon=True must shift the value to the left content
        # edge, reclaiming the space reserved for the icon column.
        h = 224
        config = self._config()
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "sensor.temperature",
            "layout": "inline",
        }
        left, _, _ = _content_box(400, h)
        shown = _ctx(base, config)
        hidden = _ctx({**base, "hide_icon": True}, config)
        assert hidden["value_x"] < shown["value_x"], (
            "value must start further left when the icon is hidden"
        )
        assert hidden["value_x"] == left, (
            "value must start at the left content edge when the "
            "icon column is collapsed"
        )

    def test_entity_hide_name_omits_name_entirely(self) -> None:
        # hide_name=True must omit the name text everywhere in the
        # text column, while the value keeps rendering.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
                "hide_name": True,
            }
        ]
        img = render_to_image(widgets, self._config())
        # min_pixels is raised well above the default: black value
        # text anti-aliases through the entire 0-255 range at its
        # edges, so a handful of those edge pixels can land inside
        # the gray band by chance. A high threshold ensures only
        # genuine gray (name) content trips this assertion.
        assert (
            _band_bbox(img, text_x0, 0, text_x1, h, 100, 140, min_pixels=200)
            is None
        ), "no gray (name) content should render when hide_name=True"
        assert _band_bbox(img, text_x0, 0, text_x1, h, 0, 60) is not None, (
            "value must still render when hide_name=True"
        )

    def test_entity_hide_name_icon_still_visible(self) -> None:
        # hide_name=True must not affect icon rendering — the icon
        # circle keeps drawing on the left.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "binary_sensor.motion",
            "hide_name": True,
        }
        config = self._config()
        rx1, ry1, rx2, ry2 = _icon_ring(_ctx(widget, config))
        img = render_to_image([widget], config)
        assert_has_gray_pixels(
            img,
            rx1,
            ry1,
            rx2,
            ry2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )

    def test_entity_draws_name_and_value(self) -> None:
        # Value (black) and name (gray) both render in the text
        # column right of the icon.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
            }
        ]
        img = render_to_image(widgets, self._config())
        assert _band_bbox(img, text_x0, 0, text_x1, h, 0, 60) is not None, (
            "value should render in black"
        )
        assert _band_bbox(img, text_x0, 0, text_x1, h, 100, 140) is not None, (
            "name should render in gray"
        )

    def test_entity_value_font_larger_than_name(self) -> None:
        # The state value is the element users scan for at a
        # glance, so it must render in a larger font than the
        # entity name -- compare rendered glyph heights.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
            }
        ]
        img = render_to_image(widgets, self._config())
        value_bbox = _band_bbox(img, text_x0, 0, text_x1, h, 0, 60)
        assert value_bbox is not None
        # Search for the name band strictly below the value's own
        # bounding box, skipping one extra row. Black text
        # anti-aliases through every gray tone along its own outline
        # (not just its edges), so searching starting exactly at the
        # value's bottom edge picks up stray gray-band hits from the
        # value glyph itself; the +1 margin clears that row and
        # isolates genuine name content instead.
        name_bbox = _band_bbox(
            img, text_x0, value_bbox[3] + 1, text_x1, h, 100, 140
        )
        assert name_bbox is not None
        value_h = value_bbox[3] - value_bbox[1]
        name_h = name_bbox[3] - name_bbox[1]
        assert value_h > name_h

    def test_entity_bold_value_renders_bold_weight(self) -> None:
        # bold_value=True renders the state value with a bold
        # font-weight attribute in the SVG.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
            "bold_value": True,
        }
        svg = render_widget_svg(widget, self._config())
        assert 'font-weight="bold"' in svg

    def test_entity_default_value_not_bold(self) -> None:
        # Without bold_value, the state value has no bold
        # font-weight attribute.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
        }
        svg = render_widget_svg(widget, self._config())
        assert 'font-weight="bold"' not in svg

    def test_entity_name_font_floor_at_compact_h(self) -> None:
        # At compact widget heights the name font size must not
        # drop below the 10px legibility floor.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 60,
            "entity": "sensor.temperature",
        }
        ctx = _build_entity_context(widget, self._config())
        assert ctx["name_font_sz"] >= 10

    def test_entity_name_override(self) -> None:
        # name= overrides the entity friendly_name; renders differ.
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
        }
        default_render = render_dashboard([base], self._config())
        named_render = render_dashboard(
            [{**base, "name": "Custom Name"}], self._config()
        )
        assert default_render != named_render, (
            "name= override should change rendered output"
        )

    def test_entity_icon_override(self) -> None:
        # icon= overrides the MDI icon resolved from device_class.
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
        }
        override_render = render_dashboard(
            [{**base, "icon": "mdi:star"}], self._config()
        )
        default_render = render_dashboard([base], self._config())
        assert override_render != default_render, (
            "icon= override should change rendered output"
        )

    def test_entity_shows_unit(self) -> None:
        # Entities with unit_of_measurement show the unit alongside the
        # value; renders with and without unit differ.
        base: dict[str, object] = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
        }
        states_no_unit = {
            **MOCK_ENTITY_STATES,
            "sensor.temperature": {
                "state": "22.5",
                "attributes": {
                    "friendly_name": "Living Room",
                    "device_class": "temperature",
                    # No unit_of_measurement.
                },
            },
        }
        with_unit = render_dashboard([base], self._config())
        without_unit = render_dashboard(
            [base], self._config(states=states_no_unit)
        )
        assert with_unit != without_unit, (
            "unit_of_measurement should change rendered output"
        )

    def test_entity_unit_override(self) -> None:
        # unit= overrides the automatically detected unit.
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
        }
        default_render = render_dashboard([base], self._config())
        unit_render = render_dashboard([{**base, "unit": "F"}], self._config())
        assert default_render != unit_render, (
            "unit= override should change rendered output"
        )

    def test_entity_unit_positioned_right_of_value(self) -> None:
        # The unit extends the black value+unit block further right
        # than the value renders alone — proving the unit is placed
        # immediately after the value, not elsewhere.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "sensor.temperature",
        }
        states_no_unit = {
            **MOCK_ENTITY_STATES,
            "sensor.temperature": {
                "state": "22.5",
                "attributes": {
                    "friendly_name": "Living Room",
                    "device_class": "temperature",
                },
            },
        }
        img_with = render_to_image([base], self._config())
        img_without = render_to_image(
            [base], self._config(states=states_no_unit)
        )
        bbox_with = _band_bbox(img_with, text_x0, 0, text_x1, h, 0, 60)
        bbox_without = _band_bbox(img_without, text_x0, 0, text_x1, h, 0, 60)
        assert bbox_with is not None
        assert bbox_without is not None
        assert bbox_with[2] > bbox_without[2], (
            "unit text should extend the value+unit block to the right"
        )
        # ±2 tolerates sub-pixel rounding differences between the
        # two independently rendered images.
        assert abs(bbox_with[0] - bbox_without[0]) <= 2, (
            "value should start at the same x with or without a unit"
        )

    def test_entity_unit_renders_black_not_gray(self) -> None:
        # Unit must render black (like the value), not gray. Push the
        # name to the top so the lower half of the text column
        # contains only value+unit, then confirm no gray pixels
        # appear there.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
                "name_position": "top",
            }
        ]
        img = render_to_image(widgets, self._config())
        lower = (text_x0, h // 2, text_x1, h)
        # min_pixels is raised well above the default: black
        # value+unit text anti-aliases through the entire 0-255
        # range at its edges, so a handful of those edge pixels can
        # land inside the gray band by chance. A high threshold
        # ensures only genuine gray content trips this assertion.
        assert _band_bbox(img, *lower, 100, 140, min_pixels=200) is None, (
            "value+unit row must contain no gray pixels"
        )
        assert _band_bbox(img, *lower, 0, 60) is not None, (
            "value+unit row must contain black pixels"
        )

    def test_entity_attribute_display(self) -> None:
        # attribute= shows the specified attribute value instead of the
        # entity state; renders differ from the default.
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
        }
        default_render = render_dashboard([base], self._config())
        attr_render = render_dashboard(
            [{**base, "attribute": "humidity"}], self._config()
        )
        assert default_render != attr_render, (
            "attribute= should change rendered output"
        )

    def test_entity_attribute_suppresses_unit(self) -> None:
        # When attribute= is set, the automatic unit_of_measurement
        # from the entity state is suppressed.  Only an explicit
        # unit= override would cause a unit to appear.
        base: dict[str, object] = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 112,
            "entity": "sensor.temperature",
        }
        # sensor.temperature has unit_of_measurement="°C".
        # With attribute="humidity" the unit must be suppressed.
        with_attr = render_dashboard(
            [{**base, "attribute": "humidity"}], self._config()
        )
        # Same attribute query against a state dict with no unit.
        states_no_unit = {
            **MOCK_ENTITY_STATES,
            "sensor.temperature": {
                "state": "22.5",
                "attributes": {
                    "friendly_name": "Living Room",
                    "device_class": "temperature",
                    "humidity": 58,
                },
            },
        }
        without_unit = render_dashboard(
            [{**base, "attribute": "humidity"}],
            self._config(states=states_no_unit),
        )
        assert with_attr == without_unit, (
            "attribute= should suppress automatic unit_of_measurement"
        )

    def test_entity_attribute_unknown_no_crash(self) -> None:
        # attribute= with a nonexistent attribute key renders without
        # crashing; value text still appears in the text column.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
                "attribute": "nonexistent_attr",
            }
        ]
        img = render_to_image(widgets, self._config())
        assert_has_dark_pixels(img, text_x0, 0, text_x1, h, threshold=140)

    def test_entity_no_device_class_letter_fallback(self) -> None:
        # An entity without device_class renders a letter fallback in
        # the icon area on the left side of the widget.
        h = 224
        m = _compute_metrics(h)
        x1 = m.padding
        x2 = m.padding + m.icon_dia
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.no_class",
                "hide_name": True,
            }
        ]
        img = render_to_image(widgets, self._config())
        assert_has_dark_pixels(img, x1, 0, x2, h, threshold=200)

    # ── Data edge cases ───────────────────────────────

    def test_entity_missing_entity_white_canvas(self) -> None:
        # A missing entity produces a white canvas without crashing.
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": 112,
                "entity": "sensor.nonexistent",
            }
        ]
        img = render_to_image(widgets, self._config())
        assert_all_white(img, 0, 0, 400, 300)

    def test_entity_no_entity_field_white_canvas(self) -> None:
        # Omitting entity entirely produces a white canvas.
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": 112,
            }
        ]
        img = render_to_image(widgets, self._config())
        assert_all_white(img, 0, 0, 400, 300)

    # ── Alignment tests ───────────────────────────────

    def test_entity_icon_vertically_centered_in_widget(self) -> None:
        # In the inline arrangement the icon is centred against the
        # full widget height, not a header band.
        h = 224
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h,
            "entity": "sensor.temperature",
            "layout": "inline",
        }
        ctx = _ctx(widget, self._config())
        assert ctx["icon_cy"] == h // 2

    def test_entity_value_right_of_icon(self) -> None:
        # The value renders in the text column to the right of the
        # icon column.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
            }
        ]
        img = render_to_image(widgets, self._config())
        assert _band_bbox(img, text_x0, 0, text_x1, h, 0, 60) is not None, (
            "value must render right of the icon column"
        )

    def test_entity_name_align_left_default(self) -> None:
        # Without name_align, the name is left-aligned near the
        # start of the text column.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
            }
        ]
        img = render_to_image(widgets, self._config())
        name_bbox = _band_bbox(img, text_x0, 0, text_x1, h, 100, 140)
        assert name_bbox is not None
        # +5 tolerates anti-aliased glyph edge slop at the start of
        # the name text.
        assert name_bbox[0] <= text_x0 + 5, (
            "name should be left-aligned by default"
        )

    def test_entity_name_align_right(self) -> None:
        # name_align="right" right-aligns the name near the widget's
        # right content edge.
        h = 224
        text_x0, text_x1 = _content_x_range(400, h)
        widgets = [
            {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
                "name_align": "right",
            }
        ]
        img = render_to_image(widgets, self._config())
        name_bbox = _band_bbox(img, text_x0, 0, text_x1, h, 100, 140)
        assert name_bbox is not None
        # -5 tolerates anti-aliased glyph edge slop at the end of
        # the name text.
        assert name_bbox[2] >= text_x1 - 5, (
            "name_align='right' should right-align the name"
        )

    def test_entity_name_position_bottom_is_default(self) -> None:
        # Omitting name_position must produce byte-identical output
        # to name_position="bottom".
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "sensor.temperature",
        }
        default_render = render_dashboard([base], self._config())
        explicit_render = render_dashboard(
            [{**base, "name_position": "bottom"}], self._config()
        )
        assert default_render == explicit_render

    def test_entity_name_position_top_moves_name_above_value(
        self,
    ) -> None:
        # name_position="top" puts the name baseline above the value
        # baseline in both arrangements; "bottom" puts it below.
        h = 224
        config = self._config()
        for layout in ("inline", "stacked"):
            base = {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
                "layout": layout,
            }
            top = _ctx({**base, "name_position": "top"}, config)
            bottom = _ctx({**base, "name_position": "bottom"}, config)
            assert top["name_y"] < top["value_y"], (
                f"{layout}: name_position='top' must sit above the value"
            )
            assert bottom["name_y"] > bottom["value_y"], (
                f"{layout}: name_position='bottom' must sit below the value"
            )

    def test_entity_invert_condition_met(self) -> None:
        # A state condition that matches the entity's current state
        # inverts the widget: context has invert=True and the SVG
        # gains a full-size black background rect plus white text.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "binary_sensor.motion",
            "invert_condition": [
                {
                    "condition": "state",
                    "entity": "binary_sensor.motion",
                    "state": "on",
                }
            ],
        }
        ctx = _build_entity_context(widget, self._config())
        assert ctx["invert"] is True
        svg = render_widget_svg(widget, self._config())
        assert re.search(
            r'<rect x="0" y="0" width="400" height="224"\s*'
            r'rx="\d+" ry="\d+"\s*fill="#000000"/>',
            svg,
        ), "inverted entity must draw a full-size black background rect"
        assert 'fill="#ffffff"' in svg, (
            "inverted entity must render text/icon in white"
        )

    def test_entity_invert_condition_not_met(self) -> None:
        # A state condition that does not match leaves the widget
        # un-inverted: no black background, white canvas outside
        # content.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "binary_sensor.front_door",
            "invert_condition": [
                {
                    "condition": "state",
                    "entity": "binary_sensor.front_door",
                    "state": "on",
                }
            ],
        }
        ctx = _build_entity_context(widget, self._config())
        assert ctx["invert"] is False
        img = render_to_image([widget], self._config())
        assert_all_white(img, 0, 0, 3, 3)

    def test_entity_invert_condition_absent(self) -> None:
        # Omitting invert_condition entirely never inverts the widget.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "binary_sensor.motion",
        }
        ctx = _build_entity_context(widget, self._config())
        assert ctx["invert"] is False

    def test_entity_invert_condition_empty_list(self) -> None:
        # invert_condition=[] must never invert, even though
        # check_conditions([]) alone would return True — the widget
        # must special-case emptiness.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "binary_sensor.motion",
            "invert_condition": [],
        }
        ctx = _build_entity_context(widget, self._config())
        assert ctx["invert"] is False

    def test_entity_invert_forces_no_circle_icon(self) -> None:
        # An active entity normally draws a filled icon circle
        # (<circle> element).  When inverted, the icon style is
        # forced to no-circle so the glyph draws directly on the
        # black background.
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "binary_sensor.motion",
        }
        normal_svg = render_widget_svg(base, self._config())
        assert "<circle" in normal_svg, (
            "sanity check: active entity normally draws an icon circle"
        )
        inverted = {
            **base,
            "invert_condition": [
                {
                    "condition": "state",
                    "entity": "binary_sensor.motion",
                    "state": "on",
                }
            ],
        }
        ctx = _build_entity_context(inverted, self._config())
        assert ctx["icon_no_circle"] is True
        assert ctx["icon_outline"] is False
        inverted_svg = render_widget_svg(inverted, self._config())
        assert "<circle" not in inverted_svg, (
            "inverted entity must suppress the icon circle entirely"
        )

    def test_entity_invert_with_border_uses_white_stroke(self) -> None:
        # When inverted, the card border stroke must switch to white
        # so it stays visible against the solid black card
        # background — a black stroke would vanish against the
        # black fill.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "binary_sensor.motion",
            "card_style": "border",
            "invert_condition": [
                {
                    "condition": "state",
                    "entity": "binary_sensor.motion",
                    "state": "on",
                }
            ],
        }
        svg = render_widget_svg(widget, self._config())
        assert 'stroke="#ffffff"' in svg, (
            "inverted entity with card_style=border must render a "
            "white border stroke, not a black one that vanishes "
            "against the black card background"
        )

    def test_entity_invert_numeric_state_condition(self) -> None:
        # A numeric_state condition (above: 0) inverts when the
        # entity's state is a positive number.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "sensor.count",
            "invert_condition": [
                {
                    "condition": "numeric_state",
                    "entity": "sensor.count",
                    "above": 0,
                }
            ],
        }
        ctx = _build_entity_context(widget, self._config())
        assert ctx["invert"] is True

    def test_entity_invert_numeric_state_condition_zero(self) -> None:
        # numeric_state above=0 does not invert when the state is 0
        # (the exclusive lower bound is not satisfied).
        states = {
            **MOCK_ENTITY_STATES,
            "sensor.count": {
                "state": "0",
                "attributes": {"friendly_name": "Count"},
            },
        }
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "sensor.count",
            "invert_condition": [
                {
                    "condition": "numeric_state",
                    "entity": "sensor.count",
                    "above": 0,
                }
            ],
        }
        ctx = _build_entity_context(widget, self._config(states=states))
        assert ctx["invert"] is False

    def test_entity_invert_numeric_state_condition_non_numeric(
        self,
    ) -> None:
        # numeric_state above=0 does not invert on a non-numeric
        # state such as "unknown".
        states = {
            **MOCK_ENTITY_STATES,
            "sensor.count": {
                "state": "unknown",
                "attributes": {"friendly_name": "Count"},
            },
        }
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "sensor.count",
            "invert_condition": [
                {
                    "condition": "numeric_state",
                    "entity": "sensor.count",
                    "above": 0,
                }
            ],
        }
        ctx = _build_entity_context(widget, self._config(states=states))
        assert ctx["invert"] is False

    def test_entity_invert_state_not_condition(self) -> None:
        # state_not inverts when the entity holds a real value, not
        # one of the excluded placeholder states.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "sensor.status",
            "invert_condition": [
                {
                    "condition": "state",
                    "entity": "sensor.status",
                    "state_not": ["", "unknown", "unavailable"],
                }
            ],
        }
        ctx = _build_entity_context(widget, self._config())
        assert ctx["invert"] is True

    def test_entity_invert_state_not_condition_excluded(self) -> None:
        # state_not does not invert for excluded placeholder states:
        # empty string, "unknown", and "unavailable".
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 224,
            "entity": "sensor.status",
            "invert_condition": [
                {
                    "condition": "state",
                    "entity": "sensor.status",
                    "state_not": ["", "unknown", "unavailable"],
                }
            ],
        }
        for excluded_state in ("", "unknown", "unavailable"):
            states = {
                **MOCK_ENTITY_STATES,
                "sensor.status": {
                    "state": excluded_state,
                    "attributes": {"friendly_name": "Status"},
                },
            }
            ctx = _build_entity_context(widget, self._config(states=states))
            assert ctx["invert"] is False, (
                f"state {excluded_state!r} must not invert"
            )

    # ── Layout tests ──────────────────────────────────

    def test_entity_chrome_scales_with_smaller_dimension(self) -> None:
        # Regression guard for the 0.7 inset blow-up: card padding
        # and the accent bar derive from the card's smaller side
        # halved, so a tall card no longer gets insets scaled to its
        # own height.
        w, h = 352, 200
        m = _chrome(w, h)
        assert m.padding < _compute_metrics(h).padding
        assert m.left_bar < _compute_metrics(h).left_bar
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": w,
            "h": h,
            "entity": "sensor.temperature",
            "card_style": "left_bar",
        }
        img = render_to_image([widget], self._config())
        assert_has_gray_pixels(
            img,
            0,
            2,
            m.left_bar,
            h - 2,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )
        # The strip just right of the bar is inside the padding and
        # must stay clear — it was covered by the inflated bar before.
        assert_all_white(img, m.left_bar + 2, 2, m.left_bar + 6, h - 2)

    def test_entity_value_font_dominates_card(self) -> None:
        # Regression guard for issue #103: the value was pinned at
        # 0.21x the widget height regardless of the space available.
        for h in (112, 160, 224):
            widget = {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 400,
                "h": h,
                "entity": "sensor.temperature",
            }
            ctx = _ctx(widget, self._config())
            assert int(ctx["value_font_sz"]) > round(h * 0.30), (
                f"h={h}: value font must claim the space it is given"
            )

    def test_entity_ink_fills_content_height(self) -> None:
        # The rendered ink fills most of the content box vertically —
        # the space efficiency the redesign exists for.
        for w, h in ((352, 112), (352, 200), (230, 120)):
            widget = {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": w,
                "h": h,
                "entity": "sensor.temperature",
            }
            img = render_to_image([widget], self._config(width=w, height=h))
            bbox = content_bbox(img, 0, 0, w, h)
            assert bbox is not None
            _, _, content_h = _content_box(w, h)
            filled = (bbox[3] - bbox[1]) / content_h
            assert filled >= 0.7, (
                f"{w}x{h}: ink fills only {filled:.0%} of the content box"
            )

    def test_entity_layout_inline_forced(self) -> None:
        # layout="inline" keeps the icon in its own column, left of
        # the value, whatever the card's proportions.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 352,
            "h": 200,
            "entity": "sensor.temperature",
            "layout": "inline",
        }
        ctx = _ctx(widget, self._config())
        assert ctx["stacked"] is False
        assert int(ctx["value_x"]) >= int(ctx["icon_cx"]) + int(ctx["icon_r"])

    def test_entity_layout_stacked_forced(self) -> None:
        # layout="stacked" moves the icon onto the name's line so the
        # value spans the full content width.
        w, h = 352, 112
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": w,
            "h": h,
            "entity": "sensor.temperature",
            "layout": "stacked",
        }
        ctx = _ctx(widget, self._config())
        left, _, _ = _content_box(w, h)
        assert ctx["stacked"] is True
        assert ctx["value_x"] == left

    def test_entity_layout_auto_prefers_stacked_on_tall_card(self) -> None:
        # A card tall enough that the icon column would choke the
        # value switches to the stacked arrangement.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 352,
            "h": 200,
            "entity": "sensor.temperature",
        }
        ctx = _ctx(widget, self._config())
        assert ctx["stacked"] is True
        inline = _ctx({**widget, "layout": "inline"}, self._config())
        assert int(ctx["value_font_sz"]) > int(inline["value_font_sz"]), (
            "stacked must only be chosen when it buys a bigger value"
        )

    def test_entity_layout_auto_keeps_inline_on_short_card(self) -> None:
        # A wide, short card has no vertical room to spend on a
        # separate name row, so the inline arrangement is kept.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 352,
            "h": 112,
            "entity": "sensor.temperature",
        }
        assert _ctx(widget, self._config())["stacked"] is False

    def test_entity_layout_default_is_auto(self) -> None:
        # Omitting layout must render identically to layout="auto".
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 352,
            "h": 200,
            "entity": "sensor.temperature",
        }
        auto = render_dashboard([{**base, "layout": "auto"}], self._config())
        assert render_dashboard([base], self._config()) == auto

    def test_entity_hide_name_forces_inline(self) -> None:
        # With no name there is no second line to stack, so the
        # inline arrangement is always used.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 352,
            "h": 200,
            "entity": "sensor.temperature",
            "hide_name": True,
        }
        assert _ctx(widget, self._config())["stacked"] is False

    def test_entity_stacked_icon_sits_on_the_name_row(self) -> None:
        # In the stacked arrangement the icon is centred on the name
        # row, not on the card.
        h = 200
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 352,
            "h": h,
            "entity": "sensor.temperature",
            "layout": "stacked",
        }
        ctx = _ctx(widget, self._config())
        assert ctx["icon_cy"] != h // 2
        # The name's baseline sits within the icon circle's band.
        assert (
            int(ctx["icon_cy"]) - int(ctx["icon_r"])
            <= int(ctx["name_y"])
            <= int(ctx["icon_cy"]) + int(ctx["icon_r"])
        )

    def test_entity_long_name_truncates_with_ellipsis(self) -> None:
        # A name too wide for the content box is truncated rather
        # than overflowing the card.
        widget = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 260,
            "h": 130,
            "entity": "sensor.pressure",
        }
        ctx = _ctx(widget, self._config())
        assert str(ctx["name_text"]).endswith("\u2026")
        assert (
            str(ctx["name_text"])
            != (
                MOCK_ENTITY_STATES["sensor.pressure"]["attributes"][
                    "friendly_name"
                ]
            )
        )

    def test_entity_long_name_does_not_shrink_value(self) -> None:
        # The name never drags the value's size down with it; only
        # the name gives way.
        base = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 260,
            "h": 130,
            "entity": "sensor.pressure",
        }
        long_name = _ctx(base, self._config())
        short_name = _ctx({**base, "name": "Bar"}, self._config())
        assert long_name["value_font_sz"] == short_name["value_font_sz"]

    def test_entity_name_size_stays_within_shrink_range(self) -> None:
        # The name may shrink to fit, but only down to
        # NAME_MIN_SHRINK of its place in the type scale.
        for name in ("Bar", "Barometric pressure outdoors north side"):
            widget = {
                "type": "entity",
                "x": 0,
                "y": 0,
                "w": 260,
                "h": 130,
                "entity": "sensor.pressure",
                "name": name,
            }
            ctx = _ctx(widget, self._config())
            ideal = round(int(ctx["value_font_sz"]) * NAME_RATIO)
            assert (
                round(ideal * NAME_MIN_SHRINK)
                <= int(ctx["name_font_sz"])
                <= ideal
            )

    # ── Scaling tests ─────────────────────────────────

    def test_entity_scales_with_h(self) -> None:
        # Doubling h roughly doubles the bounding box of rendered content.
        h_small = 112
        h_large = 224
        widget_small = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h_small,
            "entity": "sensor.temperature",
        }
        widget_large = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": h_large,
            "entity": "sensor.temperature",
        }
        img_s = render_to_image([widget_small], self._config())
        img_l = render_to_image([widget_large], self._config())
        assert_scales_proportionally(
            img_s,
            img_l,
            region_small=(0, 0, 400, h_small),
            region_large=(0, 0, 400, h_large),
            expected_ratio=2.0,
        )

    # ── Auto-sizing tests ─────────────────────────────

    def test_entity_auto_height(self) -> None:
        # Without explicit h, the widget height equals 2 * DEFAULT_ROW_H
        # (entity card is inherently a 2-row-tall widget).
        w = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "entity": "sensor.temperature",
        }
        svg = render_widget_svg(w, self._config())
        m = re.search(r'height="(\d+)"', svg)
        assert m is not None
        assert int(m.group(1)) == 2 * DEFAULT_ROW_H

    def test_entity_explicit_h_preserved(self) -> None:
        # An explicit h overrides the auto-sized default.
        w = {
            "type": "entity",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 200,
            "entity": "sensor.temperature",
        }
        svg = render_widget_svg(w, self._config())
        m = re.search(r'height="(\d+)"', svg)
        assert m is not None
        assert int(m.group(1)) == 200
