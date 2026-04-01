# Hiding Cell Code in Marimo

There are two ways to hide the code for specific cells in marimo notebooks:

## 1. Programmatically via `hide_code=True`

Add the `hide_code=True` parameter to the `@app.cell` decorator:

```python
@app.cell(hide_code=True)
def _():
    # cell code here
    ...
```

Example from the marimo documentation: [Details example](https://docs.marimo.io/examples/markdown/details/).

## 2. Via the UI (Editor)

Click the three dots (⋮) at the top‑right of a cell and select **"Hide code"** from the context menu.

Reference: [Editor overview – Cell actions](https://docs.marimo.io/guides/editor_features/overview/#cell-actions).

## Notes

- `hide_code` is currently respected in **edit mode** (the code is collapsed/hidden).
- There is an open issue to extend `hide_code` to **run mode** and exported views: [Respect hide_code in run mode and exported views #5244](https://github.com/marimo-team/marimo/issues/5244).
- To explicitly **show** code in the output area, use `mo.show_code()` ([API: mo.show_code](https://docs.marimo.io/api/outputs/#marimo.show_code)).