# AGENTS.md

## Project context

This repository is `LTSM-AD`.

The directory opened by Codex is an SSHFS-mounted directory:

- Local mount path: `/home/wrj/LTSM-AD`
- Remote server path: `/workspace/code/LTSM-AD`
- Remote server host: `h100`
- Remote user: `root`

Files edited locally under `/home/wrj/LTSM-AD` are actually files on the remote server.

## Important workflow

Codex runs on the local laptop, which has network access.

The remote server has GPU resources but should be treated as having no external network access.

Therefore:

- It is OK to edit code directly in the local mounted directory.
- It is OK to run local Git commands such as `git status`, `git add`, `git commit`, and `git push` from the mounted directory.
- Do not run training or model experiments directly with local `python` from `/home/wrj/LTSM-AD`.
- Training, evaluation, and GPU experiments must be executed on the remote server through SSH.

## Remote execution

When running experiments, use commands like:

```bash
ssh root@h100 "cd /workspace/code/LTSM-AD && conda run -n ltsm python <script>.py"


Research code policy

This repository is a research prototype, not a production software library.

When modifying code, follow these principles:

Prefer explicit assumptions over hidden compatibility logic.
Prefer raising clear errors over silently guessing defaults.
Do not add compatibility branches unless they are explicitly needed by the current experiment.
Do not preserve old behavior just for backward compatibility unless asked.
Do not write broad try/except blocks unless the exception is expected and scientifically meaningful.
Do not auto-create missing experimental configuration values unless explicitly requested.
Do not infer important model, data, or training parameters from unrelated fields.