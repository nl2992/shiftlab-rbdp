# Risk-Budget Dynamic Programming for Causal-Null-Safe Model Promotion

When should a deployed model be replaced by a challenger using a small budget of labels bought where the two models disagree? The promotion gate must keep the probability of promoting a no-better challenger below alpha under an adaptive null, where the challenger's conditional win probability may change with the history but never exceeds one half. Policies calibrated to the i.i.d. null violate this. For one-round label delay the adaptive-null problem is an exact dynamic programme on time, belief, remaining labels and a risk budget (RB-DP), differing from the i.i.d. programme by one constraint. The repository holds the solver, the premium theorems and proofs, the synthetic validation, the real-stream studies, and the TMLR manuscript.

**Status.** Manuscript draft for TMLR.

## Layout

- `paper/`: manuscript sources
- `docs/`: protocols, results and notes
- `src/shiftlab/`: implementation
- `scripts/`: experiment, analysis and audit scripts
- `tests/`: unit and regression tests
- `prompts/`: review prompts

The manuscript sources are in `paper/p4_tmlr`. The main implementation is `src/shiftlab/paper4`; scripts are under `scripts/paper4_continuation/rbdp`. Solver plan, proofs and results are in `docs/paper4_continuation/rbdp/`; earlier premium theory is in `docs/paper4_continuation/theory_v2/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Large raw data and run outputs are not tracked. Scripts regenerate them, and each protocol or results note under `docs/` states the seeds and commands it used.

## Related repositories

The Python package `shiftlab` started as one shared codebase and is now split by paper. Each repository carries the part of it that the paper needs.

## License

MIT. See `LICENSE`.
