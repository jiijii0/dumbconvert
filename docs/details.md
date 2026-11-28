# DumbConvert in detail

How the app works, every key, what each control does in NegPy, and why the defaults are what
they are. For installing and running it, see the [README](../README.md).

## Rolls and single frames

When you add scans you choose how they are analysed:

- **A whole roll** (a folder, or drop on the left half): every frame is measured, the
  measurements are pooled with NegPy's `pool_frame_bounds` (median, unusual frames such as
  a red wall or a sunset left out), and the roll gets **one colour baseline** and one
  grey axis for cast removal (a frame with no grey in it is balanced like its neighbours).
  Each frame keeps its own brightness range, and an unusual frame keeps its own measurement. This is what lab scanners do: the Frontier reads the
  whole order before correcting colour. Choose the **film stock** first (type to search NegPy's
  catalogue of negative stocks, pick "Other colour negative" / "Other B&W negative", or "No film
  stock"): the roll is analysed once one is chosen, and until then the field and the roll ask for it.
  - B&W stocks convert in B&W automatically;
  - colour stocks use their own dye-unmix (crosstalk) matrix when NegPy ships one, and their own
    Lab print colours (see below); respools of Kodak Vision3 (CineStill, Flic Film, Silbersalz35)
    use Vision3's;
  - changing the stock while the roll is being analysed starts a new analysis at once;
  - film, camera, roll name and frame number are written into the exported file (EXIF/XMP).
    The scan's own EXIF is not copied, because it names the scanning camera.
- **Single frames** (drop on the right half): each picture is analysed on its own.

Each roll is a group in the left column; click it to see its frames. Exports go into one
subfolder per roll.

## Keyboard workflow (like a lab scanner)

Rolls are on the left, one card each (name, film stock, frame count; the chevron opens the roll's details);
the frames of the selected roll sit in a **filmstrip** at the bottom, numbered on the thumbnail. Once a roll is loaded you can work from the keyboard alone.

**Every key can be changed** in the **Keys** tab at the top of the window (or press F1). It shows a
keyboard with each key's job on it: click a key, or press it, and choose what it does from the list. Changes apply
at once and are saved (`keys.py` stores them with Qt's `QSettings`). Keys that aren't on the drawing, such as
Home or keys of a non-US layout, can be set by pressing them.

The defaults:

| Keys | Action |
|---|---|
| ← / → | Previous / next frame |
| Space or Return | Next frame, continuing into the next roll |
| ↑ / ↓ | Previous / next roll |
| **Hold Q, W or E, then ↑ / ↓** | Density (darker / lighter) · Contrast · Saturation |
| **Hold A, S or D, then ↑ / ↓** | Cyan ↔ Red · Magenta ↔ Green · Yellow ↔ Blue |
| Shift + ↑ / ↓ while holding | 4× step |
| Z / X / C / V | Zoom · Crop · Auto crop · Auto straighten |
| B (hold) / N / M | Before · Pick a neutral spot (then click it) · Mirror |
| J / K / L | Rotate left · Auto WB (also 0) · Rotate right |
| U / I | Heal dust (click specks, Alt+scroll for brush size) · Clean an area (draw around it) |
| F / P | Full screen: only the photo (F or Esc to leave) · Contact sheet of the roll |
| Tab | Hide / show the panels: the photo takes the whole window (also while healing, cleaning or cropping) |
| , / . | Straighten 0.1° (Shift: 1°) |
| Backspace | Reset the frame |
| Cmd + Backspace | Remove the selected frames from the list (right-click a frame or a roll does the same; files on disk are never touched) |
| Crop (X) | A frame appears on the photo. Pull its corners, drag inside to move it, pick a film format under the photo (3:2, 4:3, 5:4, 6:7, 1:1, XPan; Turn swaps upright and landscape). Return applies, Esc cancels |
| F1 | Keys tab |

The six corrections sit under the left hand (Q W E above A S D), so it never has to move: hold one and
press ↑ or ↓ with the right hand, which is on the arrows anyway to move through the frames. Each colour is one control with two directions: less
cyan is more red, which is why the on-screen C column has a cyan + and a red −. Unassigned actions you can
put on a key: one-step + / − keys for every correction, copy, paste, undo, redo, export.

Always the same (Ctrl instead of Cmd on Windows and Linux): Cmd+Z / Cmd+Shift+Z undo/redo, Cmd+C / Cmd+V copy/paste settings, Cmd+A select every frame
(then any correction changes them all), Cmd+E export, Esc to leave a text field, Tab to hide the panels. Letters
typed in text fields never change the picture, and Tab there still moves to the next field.

## Zoom

Check grain, focus and dust like in Lightroom: **Z** or a click toggles between Fit and 100% (one scan pixel per
screen pixel) around the pointer; **pinch** or **Cmd + scroll** zooms smoothly; **drag** or **scroll** pans. Fit is
the furthest out. Past the preview's resolution DumbConvert renders the full-resolution picture in the background
("Loading detail…", a few seconds for a 24 MP raw) and swaps it in; edits while zoomed update it. Healing works
at any zoom (Alt + scroll sizes the brush); **Tab** hides the panels so dust at the picture's edge is not under
one, and a zoomed photo stays where it is while they go. A new frame, a turn, a crop or straightening goes back to Fit.

## Exif tab

The **Exif** tab keeps the rolls, the photo and the filmstrip, and swaps the right column for the
metadata written into exported files: **date**, **location**, **camera**, **lens** and **film stock**.

- The roll details on the left are every frame's defaults; they show greyed out ("· from roll").
  Typing in the Exif tab gives the selected frames their own value; clearing a field goes back to the roll's.
- **Date** is the day you shot the film, written as EXIF *DateTimeOriginal*: what the iPhone's Photos and
  Lightroom show as "taken" (without it they fall back to the scan or export date). It is a date picker:
  click it for a calendar, or type with the keyboard; an impossible date cannot be entered. Delete clears it.
  When exporting again from Lightroom, keep its metadata setting on "All Metadata" (or "All Except Camera Raw
  Info"): "Copyright Only" drops the date.
- **Location** (magnifying glass) works like the Weather app: type a place, pick it from the list (↑ ↓ Return or a click), and its GPS coordinates
  are attached. You can also paste coordinates (`35.0116, 135.7681`, a Google Maps link, `40.7128° N, 74.0060° W`).
  Only coordinates go into the file, as the iPhone does: Photos and Lightroom show the place on their map
  and look up its name themselves. The roll's Location field on the left works the same way.
  The search uses Photon (photon.komoot.io, OpenStreetMap data); only the typed words are sent.
- Under the photo: **Copy EXIF**, **Paste to Selected**, **Paste to Whole Roll**, **Set as Roll Defaults**
  (this frame's values become the roll's details) and **Clear Selected** (back to the roll's details).
  Select several frames (Cmd/Shift-click, Cmd+A) and a field edit changes all of them.
- **Camera and lens presets**, in the Exif tab and in the roll details alike: typing suggests matching presets;
  the ▾ inside the field lists them, saves the current value as a preset, and opens **Edit Presets…**.
  One shared list, saved between sessions.
- **Saved as you go**: a field lights up when its value is stored, and "✓ Saved · 3 frames" shows how many
  frames it reached. Roll details say so in the status bar.
- **Roll details** fold away (ROLL DETAILS), so a session with several rolls keeps the list in view.

## Editing tools

- **Undo / Redo** (Cmd+Z / Cmd+Shift+Z), per frame. An edit becomes one step once you pause.
- **Hold for Before**: shows the automatic conversion without your adjustments (same crop).
- **Batch editing**: select several frames (Cmd/Shift-click) and any adjustment changes all of
  them. **Copy Settings / Paste to Selected** does the same between any frames. Crop, rotation
  and straightening always stay per frame.
- **Straighten**: ±15° slider, or **Auto**, which levels the picture using the film frame's edges.
  The empty corners are cropped away.
- **Roll details**: film stock, camera, lens, date shot (2024 / 2024-06 / 2024-06-21), location
  and push/pull, all written into exported files.

## Export formats

**Export…** opens one dialog with every option: what to export (everything, this group, or the selected frames), format, size (full, long edge in pixels, or print size in cm at a dpi), colour space (sRGB, Display P3, Adobe RGB), a file-name pattern (`{name}`, `{roll}`, `{film}`, `{frame}`, `{camera}`), keep both or overwrite, one subfolder per roll, and Show in Finder (Explorer on Windows, the file manager on Linux). Every format gets the crop, rotation
and all adjustments, plus the roll's camera and film in its metadata.

| Format | Best for |
|---|---|
| JPEG | Sharing, anywhere |
| TIFF / PNG (16-bit) | Archiving, printing, editing without quality loss |
| HEIC (10-bit) | Small files with smooth gradients (Apple devices) |
| AVIF / WebP / JPEG XL | Very small files for the web |
| **DNG (compact, 14-bit, recommended)** | The best DNG for most uses. Editing further in any raw editor, at about the size of the original raw. The converted, cropped picture is stored like a camera raw: one 14-bit value per pixel in an RGGB pattern (linear, Adobe RGB primaries, lossless JPEG). The raw editor demosaics it again; fine colour edges soften slightly (avg error under 1 level out of 255). |
| **DNG (lossless, 16-bit, large files)** | The same picture with all three colours per pixel, exact 16-bit linear data. About 3× larger (≈85–105 MB for 24 MP), because film grain barely compresses. Saved as `*_lossless.dng`. |

JPEG XL has no Adobe RGB: with that colour space chosen, JPEG XL files are saved in Display P3.

## Contact sheet

Every roll can be printed on one sheet, like a darkroom contact print: open the roll's details in
the left column and press **Contact Sheet** (or right-click the roll, or press **P**). The frames lie
on the paper at their real size, with their adjustments, in the roll's order. Options: paper size,
film (35mm, half frame, 120 sizes, or detected from the frames), strips of film with perforations and
edge print or upright frames in rows, black or white around the film, 150 or 300 dpi, and a line with
the roll's name, film, camera and date. Saved as JPEG (sRGB) next to the roll's exports; a roll that
does not fit continues on a second sheet. The sheet is drawn by NegPy's contact-sheet code.

## The files

| File | What it does |
|---|---|
| `simpleneg/engine.py` | Maps DumbConvert's sliders onto NegPy's `WorkspaceConfig` and calls its renderer. **Start here.** |
| `simpleneg/app.py` | The window: film strip, preview, sliders, export. |
| `simpleneg/keys.py` | Key bindings: the list of actions, the defaults, the keyboard drawing, saving. |
| `vendor/negpy/` | NegPy's conversion engine (a trimmed copy, changed for this app). Its `docs/PIPELINE.md` explains the colour science. |

## How the controls map to NegPy

| DumbConvert | NegPy control |
|---|---|
| Contrast (1.00 = auto) | The contrast NegPy chose for the frame, scaled (+0.05 = about 3 % steeper). NegPy's own Grade cannot print softer than its automatic shadow floor, so DumbConvert reads the automatic result and prints it by hand with the new slope (`engine._with_contrast`) |
| Density (1.00 = auto) | Print Density (higher = darker) |
| Saturation | Chroma |
| C / M / Y | Cyan / Magenta / Yellow filtration, balanced so a step of each moves the picture equally (yellow ×1.40, magenta ×0.87). One step = 0.02; the fader covers the full range |
| L / Mirror / R | Rotation (quarter turns) / Flip horizontal |
| Crop: Edit / Auto | Manual crop rect / Auto Crop. Auto also levels a tilted scan (NegPy's frame-edge measure), unless you straightened it yourself |
| Auto WB | DumbConvert's own: makes the least colourful pixels neutral by setting C/M/Y |
| Refine ▸ Re-measure Inside Crop | Measures the film again inside the crop (`analysis_rect`), then Auto WB |
| Settings (gear tab) ▸ Flat field | Flat field gain map from a photo of the bare light panel (NegPy's `compute_gain`), all images: fixes uneven light and vignetting |
| Refine ▸ Light: Exposure, Highlights, Shadows, Whites, Blacks | Lightroom-like tone controls on the finished positive (`exposure_ev`, `tone_*`, Lab stage) |
| Refine ▸ Color: Temp / Tint | White balance on the positive, like a raw converter's (`wb_temp` / `wb_tint`); the dropper sets them from a spot that should be grey |
| Refine ▸ Color: Mixer | Hue, saturation and luminance per colour band, in Oklab (`hsl_*`); the dropper picks the band from the photo |
| Refine ▸ Finish: Print look | Lab print (the default, see below), None (a straight conversion) or one of NegPy's papers and minilab renderings (`paper_profile`) |
| (automatic) Dye separation | `dye_separation` 1.1 under white light, 1.0 under RGB light |
| Refine ▸ Finish: Glow | `glow_amount` (Lab stage): a soft glow around bright areas, like a minilab scan's soft focus |
| Refine ▸ Sharpen | `sharpen` (Lab stage), NegPy's default 0.25 |
| Refine ▸ Colour noise | `chroma_denoise` 0–3, default 1.0 (NegPy's 2.5 removes barely more speckle and costs some colour) |
| Heal Dust (H) | `manual_heal_strokes`: each click is a search area; NegPy repairs only what stands out from the film around it. Clicks are mapped to the raw scan with NegPy's UV grid, so crop and rotation can change later. Scroll for brush size; every spot is one undo step |
| Clean Area (A) | The shape you draw is filled with heal bands (`manual_heal_strokes`), so inside it NegPy repairs whatever stands out from the film around it and leaves clean film alone. Right-click a shape to remove it; each shape is one undo step. |
| Pick Grey (W) | DumbConvert's own: sets C/M/Y so the clicked spot comes out grey |
| (automatic) Film crosstalk | From the roll's film stock (`crosstalk_matrix`), the film's own dye matrix under either light |
| Settings ▸ Scanning light | White (a broadband panel: camera white balance in the decode, dye separation 1.1) or RGB (separate LEDs: a linear decode, no extra separation) |
| Settings ▸ Accent | The colour of what is chosen: purple, blue, gold or teal |

## Saving a session

There are no projects or sidecar files. The save icon in the tab bar (Cmd+S, Ctrl+S on Windows
and Linux) keeps the open rolls and their adjustments inside the app; a dot beside it shows unsaved
changes. Quitting with unsaved changes asks whether to save, and the next start offers the saved
session back. A crash loses nothing either: the work in progress is written to a small recovery
file every few seconds, in the app's data folder, and offered back at the next start.

The place search in the Exif tab sends the words you type to Photon (photon.komoot.io), a free
search over OpenStreetMap. No key or account is used, and nothing else leaves your computer.

## Why these defaults

DumbConvert starts from a more film-like setup than NegPy's neutral defaults:

| Setting | NegPy default | DumbConvert | Why |
|---|---|---|---|
| Look | Digital (a straight conversion) | Lab print | A straight scan conversion leaves out what printing on photo paper does to a negative's colour. Lab print adds it back, per film stock, from the makers' data sheets: a spectral simulation of each film exposed, scanned under a white LED by real camera sensors, and printed on the paper its maker pairs it with (`PRINT_CHARACTER` in `engine.py`, method in `tools/print_character/`). Kodak stocks turn yellows toward amber and blues toward teal with quieter greens; Fujifilm keeps lemon yellows and cooler greens. 70 % of the measured hue turns, with the overall colourfulness left as it is. |
| Crosstalk | off | the film's dye matrix (Generic C41 or the roll's stock), 1.0 | Under white light a camera mixes the dye layers a little; the film's own matrix unmixes the film. |
| Dye Separation | 1.0 | 1.1 (white light) | A small overall lift for what a broadband light and a Bayer sensor lose. |
| Cast Removal | 0.5 | 0.2 | Less neutralizing keeps the film's own colour character. |
| Skin Protection | 0.5 | 0.25 | Less damping of warm tones. |

To expose another NegPy control, find its field in `vendor/negpy/negpy/features/*/models.py`
(the comments there say what it does), add it to `Settings` in `engine.py`,
map it in `to_config()`, and add a `line(...)` row in `app.py`'s `_build_more` (the Refine section).

## Updating NegPy

```sh
cd vendor/negpy && git pull
cd ../.. && .venv/bin/pip install ./vendor/negpy/negpy_core
```

## License

DumbConvert includes NegPy, so the whole project is **GPLv3** (see `LICENSE`).
You can use and change it freely. If you share it, share the source under GPLv3 too.
