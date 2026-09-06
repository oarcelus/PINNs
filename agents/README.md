# PINNs agent workflow

This directory contains four reusable role prompts for work in this repository. They are deliberately checked-in documents, not a repository-wide `AGENTS.md`: provide the appropriate file to an agent when assigning its role. This keeps an experimental coding task from silently inheriting a role intended for a different task.

## Project context every role must preserve

This is a script-oriented Python/PyTorch project for battery state-of-health (SoH) prediction with a physics-informed neural network (PINN). It has no package metadata, dependency manifest, test suite, or existing agent instructions.

```text
raw battery records
  -> data/40V and data/41V collection, cleaning, interpolation, feature scripts
  -> per-condition normalized feature CSV
  -> model/dataloader.py adjacent-cycle tensors
  -> model/model.py PINN and residual
  -> model/train*.py / train_*_holdout.py checkpoints, histories, logs
  -> model/plot_*.py analysis plots
```

Important verified contracts:

- A feature CSV currently has `Cycle Number` in column 0, `SoH` in column 1, then 16 descriptors. `FeatureDataset` selects column 1 as `y` and column 0 plus columns 2 onward as `x`; it therefore produces 17 inputs and one target.
- `PINN.forward` calculates `u(x)` and differentiates it with respect to every input. It defines `ut = du_dxt[:, 0:1]` for input 0 (`Cycle Number`) and `ux = du_dxt[:, 1:]` for the 16 descriptor derivatives, then feeds `[x, u, ux, ut]` to the dynamics MLP. With 17 inputs, the expected shapes are `x=[B,17]`, `u=[B,1]`, `ux=[B,16]`, `ut=[B,1]`, and dynamics input `[B,35]`; `cfg/pinn_config.json` currently matches that contract.
- `Cycle Number` at input index 0 is the current temporal-coordinate contract. No role may silently change coordinate selection, input order, or residual semantics. A change requires an explicit Architect decision, a schema/checkpoint compatibility plan, and targeted tests. Checkpoints trained under a prior coordinate convention are not scientifically equivalent merely because their tensor shapes match.
- Training uses data loss, a monotonicity term, and a PDE-residual term. PINN evaluation still needs autograd because the residual is computed in `forward`; `torch.no_grad()` and `torch.inference_mode()` are not valid around a residual evaluation.
- Data, CSVs, plots, checkpoints, histories, and logs are ignored. Do not add generated artifacts or private source data as test fixtures. Scripts use direct imports and cwd-relative paths, and normally run CPU-only (with a Slurm launcher present).

## Normal handoff loop

```text
User request
  -> Architect: versioned, implementation-ready specification
  -> Implementer: focused code change and implementation report
  -> Code Reviewer: independently evidenced findings
  -> Tester: tests and a coverage matrix based on the spec and review
  -> Reviewer checks test adequacy; Implementer resolves findings
  -> repeat as needed; Architect revises the spec if intent changed
```

The Tester and Code Reviewer are a two-way loop: the reviewer identifies high-risk behavior and test gaps; the tester maps each finding to a test, an investigation, or a documented reason it cannot be automated; the reviewer then evaluates whether that response actually protects the risk.

For a nontrivial work item, use durable artifacts so handoffs do not depend on chat history:

```text
docs/work-items/<work-id>/spec.md
docs/work-items/<work-id>/review.md
docs/work-items/<work-id>/test-report.md
```

Create those artifacts only when working the item; `agents/` is the durable role library. Use stable IDs: requirements `R-<id>`, acceptance criteria `AC-<id>`, review findings `REV-<id>`, and tests `T-<id>`.

## Shared completion gate

Work is ready to hand back only when the approved acceptance criteria have evidence; tests appropriate to the change pass; every blocker or high-severity review finding is fixed or explicitly accepted by the Architect/user; and the final report distinguishes code-level verification from claims that require a full experiment or private data.
