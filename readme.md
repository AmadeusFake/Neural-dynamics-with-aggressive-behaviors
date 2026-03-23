# Neural Population Analysis — Aggression Observation Paradigm

Analysis pipeline for Neuropixels population recordings during a head-fixed aggression observation paradigm. Four notebooks cover the full analysis from raw data loading to unsupervised latent state discovery.

---

## Dataset

Three `.mat` files are required:

| File | Contents |
|------|----------|
| `neuralData.mat` | Firing rate matrix (neurons × time bins, 20 ms bins, 50 Hz) |
| `peripheralData.mat` | Binary arrays: `gateOpen`, `whiteLight`, `ctBudSniff`, `lickMilk`, `lickEmpty` |
| `behaviorData.mat` | Manually annotated behaviours: `attackOfIntLeft/Right`, `chaseOfIntLeft/Right`, `submissionSignFromIntLeft/Right`, `attackDark` |

Place all three files in the same directory and set `DATA_DIR` in each notebook accordingly.

---

## Notebooks

### 1. `Analysis.ipynb`
**Data loading, preprocessing, and session structure**

- Loads all three `.mat` files
- Constructs phase masks: `Pre`, `WhiteAgg`, `DarkAgg`, `Post`
- Builds behaviour label vectors from annotated onsets
- Generates session timeline (context / behaviour / observer stimuli / phase channels)
- Entry point for all downstream notebooks — run this first

---

### 2. `population_state_space.ipynb`
**Population geometry, decoding, and context encoding**

Sections:
1. PCA dimensionality reduction — phase-labelled scatter and correlation matrix
2. Behaviour-conditioned population trajectories (attack onset PETH in PC space)
3. Centroid geometry — Euclidean distances between WA / WN / DA / DN conditions
4. Binary linear decoding — binwise and grouped cross-validation
   - White attack vs non-attack
   - Dark attack vs non-attack
   - White attack vs Dark attack (matched-behaviour, different context)
   - White non-attack vs Dark non-attack
5. Pre → Post remapping — sniff cross-phase analysis and population drift

Key result: cross-context distance is ~2.5× within-context distance; WA vs DA decodes at 100%.

---

### 3. `single_neuron_screening.ipynb`
**Single-neuron modulation and context sensitivity**

Sections:
1. Attack-aligned PSTH — population mean ± SEM (White vs Dark)
2. Per-neuron permutation test for attack modulation (block circular-shift, FDR corrected)
3. White vs Dark single-neuron contrast — label-shuffle permutation, volcano plot
4. Neuron gallery — individual PSTHs for significantly modulated cells

Key result: 9 neurons significantly modulated by attack; 0 neurons survive FDR correction for the White vs Dark contrast (n = 218 neurons).

---

### 4. `latent_state_hmm_analysis_v2.ipynb`
**Unsupervised latent state discovery via Gaussian HMM**

Sections:
1. Data loading and phase mask construction
2. Helper functions — geometry, HMM utilities, enrichment metrics
3. PCA for HMM input (N_PCS = 20) + N_PCS sensitivity sweep
4. Model selection — temporal cross-validation and ARI stability
   - BIC is invalid here: raw PC1 autocorrelation τ ≈ 65 s, n_eff ≈ 18
   - CV elbow selects K = 12; ARI > 0.6 confirms reproducibility
5. Viterbi decoding — latent state sequence
6. Session state-sequence overview
7. State × behaviour alignment — enrichment heatmaps
8. State occupancy by phase — Pre → Post bootstrap test (10/12 states significant)
9. Transition structure and dwell times
10. HMM emission means — neural fingerprint of each state
11. HMM states projected into supervised summary space
    - Nearest supervised condition: WA → state 0, DA → state 7
12. Attack-aligned latent state occupancy (PETH)
13. Compact results summary
14. Discussion

Key result: unsupervised HMM, with no behavioural labels, recovers the same context-dependent structure identified by supervised analysis.

---

## Dependencies

```
python >= 3.9
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn
hmmlearn
```

Install:
```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn hmmlearn
```

---

## Reproducibility

- HMM fitting uses 5 random seeds per K; best log-likelihood is selected
- All random seeds are fixed where applicable
- Bootstrap tests use n = 1000 iterations, seed = 0
- Analysis order: `Analysis.ipynb` → `single_neuron_screening.ipynb` → `population_state_space.ipynb` → `latent_state_hmm_analysis_v2.ipynb`