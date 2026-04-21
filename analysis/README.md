# analysis/

Optional analysis scripts that operate on artifacts produced by the main
pipeline. Nothing in here is required to run the regressions or GARCH stage —
these scripts only consume already-produced outputs.

## Contents

```
src/
└── descriptive_stats.py   CLI: panel-level summary statistics + correlations
```

## Prerequisites

`descriptive_stats.py` reads the merged stock–day panel created by the
regression module:

```
regression/out/panel/panel_data.parquet
```

so `regression/src/build_panel.py` must have run first.

## Usage

```bash
cd analysis/src
python descriptive_stats.py
```

## Outputs

```
out/
└── descriptive_stats/
    ├── descriptive_stats.tex
    └── correlation_matrix.tex
```
