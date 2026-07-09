# From Waveforms to Parameters — A Gravitational-Wave Tutorial

A hands-on Jupyter notebook tutorial for Bachelors-level Physics students, covering waveform modeling, likelihoods, the Fisher matrix, and Bayesian (MCMC) parameter estimation for gravitational waves — built up from a simple toy model to the same logical structure used by real LIGO-Virgo-KAGRA analysis pipelines.

There are two ways to run this notebook — pick whichever is easier for you. **No prior Python setup is needed for Option 1.**

---

## Option 1: Google Colab (recommended — nothing to install)

Click the badge below to open the notebook directly in your browser:

[Open In Colab, ](https://colab.research.google.com/drive/1y70ilcognmMVsm8sm5GN3ZsR7rEq-S6e?usp=sharing)


### First-time setup in Colab

1. Click the badge/link above. Colab will open the notebook in your browser — no installation required.
2. Run the very first cell (labeled **Setup**). It automatically installs the two packages Colab doesn't include by default (`emcee` and `corner`). This takes about 10–15 seconds and only needs to run once per session.
3. Run the remaining cells in order, from top to bottom (**Runtime → Run all**, or step through with Shift+Enter).

### A few Colab tips

- **If you edit an earlier cell** (e.g. changing a parameter), **re-run every cell below it** — Colab does not do this automatically, and stale variables are a common source of confusing bugs (we hit this ourselves while building the notebook — see Part X's discussion of the `log_posterior(theta_true)` sanity check).
- **If something looks broken after edits**, the safest fix is `Runtime → Restart session and run all` — slower, but guarantees every variable reflects your current code.
- Your changes in Colab are **not saved back to GitHub automatically**. To keep your edits, use `File → Save a copy in Drive`, or `File → Save a copy in GitHub` if you want to push changes back to your fork.
- No Google account is required to *view and run* a Colab notebook opened from a public GitHub link, but you do need to be signed in if you want to save your changes.

---

## Option 2: Run Locally with Jupyter (via conda)

Use this option if you prefer working offline, want faster iteration without Colab's occasional slow startup, or want to keep permanent local copies of your results.

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) installed on your computer. If you don't have either yet, Miniconda is the smaller, faster install and is all you need here.
- `git` installed (to clone the repository). Alternatively, you can just download the repo as a ZIP from GitHub and skip the `git clone` step below.

### Step-by-step setup

**1. Clone this repository** (or download it as a ZIP and extract it):

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. Create the conda environment** from the provided `environment.yml` file. This installs Python and every library the notebook needs — `numpy`, `scipy`, `matplotlib`, `jupyter`, `emcee`, and `corner` — all in one isolated environment that won't interfere with anything else on your computer:

```bash
conda env create -f environment.yml
```

This may take a few minutes the first time. You'll see conda downloading and installing packages.

**3. Activate the environment:**

```bash
conda activate gw-tutorial
```

*(You'll need to run this `conda activate gw-tutorial` command every time you open a new terminal to work on the notebook — it's how conda knows which set of installed packages to use.)*

**4. Launch Jupyter:**

```bash
jupyter notebook
```

or, if you prefer the newer JupyterLab interface:

```bash
jupyter lab
```

This opens a browser tab showing the files in this repository.

**5. Open `Tutorial_GW.ipynb`** from the file list, and run the cells from top to bottom (**Cell → Run All**, or step through with Shift+Enter).

### Updating the environment later

If the `environment.yml` file changes (e.g. a new dependency gets added later), update your existing environment with:

```bash
conda env update -f environment.yml --prune
```

### Removing the environment

If you want to clean up later:

```bash
conda deactivate
conda env remove -n gw-tutorial
```

---

## Which option should I use?

| | Google Colab | Local Jupyter (conda) |
|---|---|---|
| Setup required | None — just a browser | Install Miniconda once, then a few terminal commands |
| Speed | Free tier can be slower to start; fine once running | Fast, especially after the first setup |
| Works offline | No | Yes |
| Saving your work | Needs an extra step (Save a copy) | Saved locally automatically |
| Best for | Trying it out quickly, using a shared/lab computer | Regular use on your own laptop |

If you're not sure, **start with Colab** — it's the fastest way to get going, and you can always switch to a local setup later.

---

## Troubleshooting

- **"ModuleNotFoundError: No module named 'emcee'" (or 'corner')** — In Colab, make sure you ran the **Setup** cell at the top first. Locally, make sure you activated the conda environment (`conda activate gw-tutorial`) *before* launching Jupyter.
- **Symbols like `\mathcal{M}` show up as a blank box instead of rendering** — this is a known Colab font-loading quirk; a hard refresh (Ctrl+Shift+R) usually fixes it. The notebook itself uses plain `$M$` rather than `\mathcal{M}` specifically to avoid this.
- **Numbers don't match what you expect after editing a cell** — see the "re-run every downstream cell" tip above; this is the most common source of confusing results while working through the notebook.
