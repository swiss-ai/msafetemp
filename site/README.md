# MSafeTemp project page

Source of https://swiss-ai.github.io/msafetemp/ (this folder only; the rest of the repo is the code).

Project page for **How I Aligned Your Model: Scaling Up Multilingual Safety Training with Templates**.

- Dataset: https://huggingface.co/datasets/swiss-ai/MSafeTemp

## Build

```bash
cd site && python build.py   # writes site/dist/index.html (Python 3.8+, standard library only)
```

Open `dist/index.html` in a browser; it is a single self-contained file.

## Layout

| Path | Contents |
| --- | --- |
| `src/page.html` | Page markup with placeholders for CSS, JS and data |
| `src/style.css` | Styles, light and dark themes |
| `src/app.js` | Interactive figures and the example explorer |
| `data/site-data.json.gz` | Every number and example shown on the page (gzipped; contains harmful example prompts) |
| `build.py` | Inlines the above into `dist/index.html` |

Pushing changes under `site/` to `main` rebuilds and deploys the page with GitHub Actions (`.github/workflows/pages.yml` at the repo root).

## Content note

The page shows examples of harmful prompts from the dataset, including slurs and offensive language
that appear in the paper. Slurs and politician names outside the paper's printed examples are masked.
