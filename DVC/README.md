# ml-dvc-titanic — Ready-to-run DVC project

This folder is a **fully working** copy of the project, already tested end-to-end
(clone → dvc pull → dvc repro). Everything DVC needs — the dataset, the trained
model, the local remote cache (dvc_storage/), and the full Git commit history —
is already inside this folder.

## Option A — Just run it as-is (fastest)

```
cd ml-dvc-titanic
pip install -r requirements.txt
dvc status        # should say "Data and pipelines are up to date."
dvc repro         # re-runs training only if dataset.csv or train.py changed
```

## Option B — Push it to your own GitHub repo (recommended for submission)

1. Create an empty repository on GitHub (do NOT initialize it with a README).
2. From inside this folder:

```
cd ml-dvc-titanic
git remote add origin https://github.com/reetish25/<your-repo-name>.git
git branch -M main
git push -u origin main
```

3. `dvc_storage/` is your **local** DVC remote — it works on this machine but
   won't travel with `git push` (Git only stores code + the small `.dvc`
   pointer files, by design). For the lab's "GitHub repository link" deliverable,
   the pushed repo is enough — a grader can `git clone`, then run `dvc repro`
   directly (no `dvc pull` needed) because `train.py` + `dataset.csv` are both
   in Git-trackable form and the pipeline will just retrain locally.

   If you want `dvc pull` to also work for someone else, you'd need a shared
   remote (e.g. Google Drive, S3) instead of a local folder — this wasn't
   required by the lab sheet but I can set one up if you want.

## Verifying it yourself

```
git clone ml-dvc-titanic test_clone
cd test_clone
pip install -r requirements.txt
dvc repro   # will retrain from scratch since dvc_storage isn't cloned by git — this is expected and fine
```

This confirms the pipeline is reproducible from code + data alone, which is
the actual point of the lab.
