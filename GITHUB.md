# Pushing this model to GitHub

This folder is a self-contained repository. Below are two ways to publish it to
your GitHub account. Your credentials stay on your machine.

## Option A — GitHub CLI (fastest)

Install the GitHub CLI from https://cli.github.com if you do not have it, then:

```bash
cd "deepfake-detector-model"

git init
git add .
git commit -m "Initial commit: hybrid multi-modal deepfake detector (SAFF + late-fusion)"

# Authenticate once (opens a browser):
gh auth login

# Create the GitHub repo and push in one step.
# Replace the name if you want something different; --public or --private as you prefer.
gh repo create hybrid-deepfake-detector --public --source=. --remote=origin --push
```

That single `gh repo create ... --push` command creates the remote repository
and pushes your first commit.

## Option B — Plain git (create the repo on github.com first)

1. Go to https://github.com/new and create an empty repository named
   `hybrid-deepfake-detector` (do not add a README, license, or .gitignore;
   this folder already has them).
2. Then run, replacing `YOUR_USERNAME`:

```bash
cd "deepfake-detector-model"

git init
git add .
git commit -m "Initial commit: hybrid multi-modal deepfake detector (SAFF + late-fusion)"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hybrid-deepfake-detector.git
git push -u origin main
```

If prompted for a password during `git push`, use a **personal access token**
(GitHub no longer accepts account passwords). Create one at
https://github.com/settings/tokens with the `repo` scope.

## Recommended order

1. Push to GitHub first (this guide) so you have version history.
2. Then publish to Hugging Face for model hosting and the rendered model card
   (see `PUBLISH.md`).

The two are complementary: GitHub holds the source and history; Hugging Face
hosts the loadable model with `trust_remote_code=True`.

## What gets pushed

All source files in this folder: the model code, config, demo and training
scripts, README, LICENSE, and requirements. The `.gitignore` excludes virtual
environments, Python caches, and trained weight files. Publish trained weights
through Hugging Face or Git LFS rather than committing them directly.
