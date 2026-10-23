# 0028. HUD colours: presets plus one accent, derived in OKLCH

Date: 2026-10-22
Status: Accepted

## Context

The HUD (ADR-0022/0023/0024) had two hand-tuned palettes, cyan and amber,
each a `Theme` of ten colours: four state colours, text, muted text, the
wake flash, the ring accent, the glass plate and the clear colour. Users
want to recolour it: a few named looks (the "Mark" red, green, violet,
white, pink) and, beyond that, *any* colour they like, from the settings
page or by saying "byt färg till röd".

Hand-tuning ten colours per preset does not scale to "any colour", and a
naive derivation breaks in three predictable ways:

- **Legibility.** Text and secondary text sit on a translucent plate over an
  unknown desktop. A yellow or white accent with offsets tuned for cyan
  gives text with no contrast.
- **Distinct states.** Idle, listening, thinking and speaking are told apart
  by brightness and a small hue shift (ADR-0023). For a grey or white
  accent there is no hue to shift, so the states collapse into one colour.
- **Bloom.** The bloom pass thresholds on the brightest channel. A white or
  pale accent carries light in all three channels and blooms into a white
  smear where a saturated colour of the same brightness glows.

HSL offsets do not solve the first two: "lightness 60 %" is a pale yellow
but a deep blue, so one set of offsets cannot be right for every hue.

## Decision

Every palette is a function of one accent colour, computed natively in
OKLCH (`native/overlay/include/jarvis/overlay/palette.hpp`, `deriveTheme`). The named presets
(`red`, `green`, `violet`, `ice`, `pink`) are `deriveTheme` of a fixed seed.
`custom` derives from the user's `overlay_accent`. Cyan and amber stay
hand-tuned and unchanged, so nobody's HUD changes on upgrade.

The derivation:

1. The accent's OKLCH lightness is clamped to [0.64, 0.86], bright enough
   to glow and leaving room above it for the listening state. Chroma is
   clamped to [0.07, 0.19]. A grey input (chroma below 0.02) gets hue 230,
   a faint ice tint, so a white or black pick still has a hue for the
   states to swing.
2. The states are offsets from the accent. Idle is darker and less
   saturated; listening is brighter. Thinking and speaking swing the hue
   by +30 and -30 degrees.
3. Text, muted text, wake flash, plate and background sit at a fixed
   OKLCH lightness. Their chroma is a capped share of the accent's, so
   text stays near neutral and the plate stays a dark tint whatever the
   hue.
4. The warning light (quit control, muted microphone, the quit question)
   was a hard-coded red. It is now a `Theme` role: red-orange, or yellow
   when the accent is within 45 degrees of red.
5. `Theme::bloom` multiplies the bloom strength. It rises from 0.6 for
   greys to 1.0 at chroma 0.14, so pale accents bloom less.
6. Colours outside the sRGB gamut keep their lightness and hue and lose
   chroma (a binary search) until they fit. Clipping channel by channel
   would shift the hue instead.

The invariants are unit-tested (`native/tests/overlay/palette_test.cpp`)
for every preset, a 15-degree hue sweep and the awkward inputs (white,
black, grey, pure primaries):

- text on the plate reaches at least 7:1 contrast (WCAG AAA);
- muted text and the warning light reach at least 4.5:1;
- the four states are pairwise at least 0.05 apart in OKLab (about 2.5
  just noticeable differences);
- the warning light is at least 0.1 from the accent.

**Settings.** `overlay_theme` (`JARVIS_OVERLAY_THEME`) takes `cyan`,
`amber`, `red`, `green`, `violet`, `ice`, `pink` or `custom`.
`overlay_accent` (`JARVIS_OVERLAY_ACCENT`) is a hex colour; any spelling of
six hex digits is accepted and stored as `#rrggbb`. The validator lives in
the type (`utils.colour.HexColour`), so the HUD's per-field check validates
it as well. The settings page gets a new `colour` row kind: a swatch that
opens Dear ImGui's hue bar with a saturation/value square. Picking a colour
also switches the theme to `custom` (the row's `implies`); choosing a preset
keeps the accent for later. All of it applies live:
`set_appearance(theme, intensity, size, accent)` re-derives the palette on
the UI thread. The orb shares the renderer's theme, so it follows too.

**Voice.** The `set_hud_colour` tool maps English and Swedish colour words
to a preset, or for colours without one (orange, blue, teal ...) to a fixed
accent. It also takes a hex code. It hands the result to
`ToolContext.hud`, which is the settings controller, so a spoken change is
validated, saved and applied exactly like a click. The change is cosmetic
and reversible, so the tool needs no confirmation. In text-only mode
`ToolContext.hud` is `None` and the tool says there is no HUD.

## Alternatives considered

- **Hand-tuned palettes only.** This looks best for the presets it covers
  but cannot offer "any colour". It would also leave the contrast and
  distinctness rules to taste rather than to tests.
- **Derive in HSL or HSV.** This is simpler, and Dear ImGui already has the
  conversion. But equal HSL steps are unequal perceived steps, so the
  contrast floor would need per-hue fixes.
- **Derive in Python and send ten colours across the boundary.** This
  would keep C++ smaller. But the palette would then exist in two places,
  and the native tests, the demo theme sheet and a future LOCAL-only
  build could not compute it without Python. The native side already owns
  every other visual decision (ADR-0023).
- **Preview the colour natively while the picker is dragged.** It feels
  more immediate. But the window would need to undo the preview when
  Python rejects the value or never answers. Applying on release costs one
  bridge poll, about 30 ms.

## Consequences

- Any colour works. Legibility and state distinctness are properties the
  tests enforce, not a hope.
- Very dark or very pale picks come out clamped: a black pick gives a
  steel blue and a white pick an ice tint. The picker's swatch shows the
  colour that was picked, not the clamped one. The README says so.
- Cyan and amber are exceptions to the derivation. They meet the same
  invariants and are tested against them, but tuning them is still done
  by hand.
- `scripts/demo_overlay.py --theme-sheet` renders every palette to a PNG
  (`docs/media/overlay-themes.png`) without opening a window.
