# Contributing to DumbConvert

Thanks for helping. DumbConvert is meant to stay simple: drop a roll, get good colour, make a few
tweaks, export. Fixes are very welcome; for a new feature, open an issue first so we can talk about
whether it fits (use the "Idea" form in Issues).

## Set up

Needs Python 3.13+ and Rust (`brew install rust` on a Mac, [rustup](https://rustup.rs) elsewhere).

```sh
git clone https://github.com/jiijii0/dumbconvert.git
cd dumbconvert
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pip install ./vendor/negpy/negpy_core
.venv/bin/pip install --no-deps -e ./vendor/negpy
./run.command
```

`run.command` starts the app on a Mac. On Linux, start it with `.venv/bin/python -m simpleneg`.
On Windows, use `.venv\Scripts\` wherever these steps say `.venv/bin/`, and start the app with
`.venv\Scripts\python -m simpleneg`.

## Before you open a pull request

1. **Run the tests** (they open the app without a window, about a minute):

   ```sh
   QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests
   ```

2. **Run the linter:** `.venv/bin/ruff check .`
3. **Add a test** for the bug you fixed, so it cannot come back. `tests/test_app.py` drives the
   window like a person would (keys, clicks, the crop tool); `tests/test_engine.py` checks the
   conversion itself. Test scans are made in code by `tests/support.py`.
4. **Try it in the real app** on your system, and say which one in the pull request.

The same tests run automatically on macOS, Windows and Linux for every pull request.

## Rules of the road

- **One fix per pull request.** Small pull requests get reviewed and merged quickly; a pull request
  that fixes five things waits until all five are right.
- **Colours.** `tests/test_colour.py` compares the colours of the test negatives with
  `tests/golden/colours.json`. If your change is *meant* to change colours, update the reference
  (`DUMBCONVERT_UPDATE_GOLDEN=1` in front of the test command) and explain why in the pull request,
  with a before / after of a real scan.
- **NegPy is part of this repository.** `vendor/negpy` is a copy of [NegPy](https://github.com/marcinz606/NegPy)
  with a few changes of DumbConvert's own. Fix a conversion bug where it is, NegPy included, and keep
  the change as small as the fix.
- **No scans or sample packs in the repository**, not even small ones. Use `tests/support.py` to make
  what a test needs.
- **AI help is fine**, as long as you have read the change, tested it, and can explain it.
- **Keep it simple.** Every option added is one more thing for everyone to understand. The defaults
  and the reasons behind them are in [docs/details.md](docs/details.md).
- **Settings are not touched by tests.** If a test needs `QSettings` or the recovery file, go through
  `tests/support.py`, which points them at a temporary folder.

## Reporting a bug

Use the bug report form in Issues. Say which system you use, what you did and what happened. If the
app has a log file (Windows: `%APPDATA%\DumbConvert\dumbconvert.log`), attach it.

By contributing you agree that your contribution is licensed under the GPLv3, like the rest of the
project.
