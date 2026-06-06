# Fonts (for synthetic data generation)

The synthetic generator pastes vocabulary text in varied typefaces. Font files
live here but are **gitignored** (`data/external/` is not committed) to avoid
redistributing fonts whose licenses don't permit it.

- **Safe to use/redistribute:** SIL Open Font License fonts — e.g.
  [Roboto](https://fonts.google.com/specimen/Roboto) and
  [Open Sans](https://fonts.google.com/specimen/Open+Sans). Download these into
  this folder to reproduce synthetic data.
- **Do NOT commit** proprietary system fonts (Arial, Calibri, Times New Roman,
  Helvetica, …). They are not redistributable.

Point the generator at this directory:

```bash
flashmask synth --backgrounds <bg_dir> --fonts data/external/fonts
```
