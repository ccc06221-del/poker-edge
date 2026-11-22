# Poker percentile trainer

A browser exercise for learning starting-hand percentiles. Open `index.html` in a browser. Everything runs locally with the starting-hand grid embedded in the page.

The trainer is a standalone part of the Poker project. It can be used without the purchased hand histories, player-level data or offline results of the research. Percentiles describe the trainer’s own scoring assumptions.

## Flop research code

The `pokeredge` package includes the original card encoding, 98-class flop taxonomy and six hand-strength buckets from the offline research. These modules and their tests use generated card combinations. Install NumPy and pytest, then run `python -m pytest tests` from the repository root. The standalone trainer can be used without Python.
