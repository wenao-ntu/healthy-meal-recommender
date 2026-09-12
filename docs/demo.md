# Streamlit demonstration

## Purpose

The application connects preprocessing, personalized preference scores, health-aware reranking, and explanation in one working interface. It does not train models during interaction and does not use a hand-authored recommendation list. Selecting a user or changing the target calls the same `gap_rank_order` function used by the offline experiment on that user's real NeuMF Top-100 candidates.

## Data flow

```text
Anonymous user and wellness target
                |
                v
Train-only user baseline and NeuMF Standard Top-100
                |
                v
healthy_meal_recommender.reranking.gap_rank_order
                |
                v
Live Top-10, before/after metrics, meal details, and evidence
```

The application reads:

- `meal_catalog.csv`: meal names, FSA/WHO scores, health utility, and compact nutrition fields.
- `user_health_train.csv`: anonymous user baselines derived only from training interactions.
- `neumf_standard_test_top100.csv`: candidates with training and validation history excluded and no test relevance labels.
- `results/reranking/gap_rank.json`: validation-selected parameters and frozen offline results.

The first three artifacts are rebuilt locally and are not committed. Test candidates provide a stable anonymous demonstration population, but test labels are never read during interaction; changing the UI target cannot affect offline experiment selection or results.

## Interaction design

1. Select one of 1,114 anonymous evaluation users. The selector also shows the user's train-only health baseline.
2. Adjust the wellness target from 0.20 to 0.55. The default is the experiment target of approximately 0.360.
3. Inspect the adaptive quota, mean Top-10 health utility, goal-meeting rate, and overlap with the original preference list.
4. Review three-course meal names, FSA/WHO values, NeuMF ranking scores, and available aggregate nutrition.
5. Compare health utility by rank and item replacements in the `Before & after` tab.
6. Inspect the three-step method summary and the four-method frozen test table in `Method & evidence`.

When the selected user's baseline already reaches the target, the quota becomes zero and the list remains unchanged. If the candidate set contains too few target-meeting meals, the application reports the feasible behavior rather than inventing candidates.

## Visual and accessibility choices

- The user and target controls appear on the first screen.
- Deep navy provides hierarchy, teal denotes wellness signals, and orange is reserved for important boundaries.
- Six metrics are arranged in two rows to remain readable in a narrow browser pane.
- Tables support search, full-screen inspection, and download; controls have visible labels and help text.
- The project-owned banner contains no text, brand, watermark, person, or medical symbol and has no effect on recommendation output.

## Launch

After completing the data, NeuMF, candidate-export, and reranking stages:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app\streamlit_app.py
```

Open `http://localhost:8501`. If a required artifact is missing, the application stops and displays the reconstruction commands instead of substituting synthetic recommendations.

## Automated verification

```powershell
.\.venv\Scripts\python.exe .\scripts\check_demo.py
```

The smoke test verifies that:

- 1,114 users, 100 candidates per user, and 3,817 meal display rows are loaded;
- the default user receives 10 unique meals drawn from the actual candidate set;
- repeated computation with identical inputs is deterministic;
- a target below the user baseline produces quota 0 and 100% Top-10 overlap;
- default and no-intervention Streamlit states render without exceptions;
- the page contains the expected title, selectors, slider, tabs, metrics, and tables.

Machine-readable evidence is stored in `results/demo/smoke_test.json`. Manual browser inspection also covered default behavior, the no-intervention case, the evidence tab, narrow-window layout, and browser console errors.

## Safety and limitations

- User IDs are dataset indices rather than names, accounts, or patient identifiers; the interface collects no personal data.
- Health utility is a project ranking proxy based on FSA and WHO values, not diagnosis, prescription, or a guarantee of health outcomes.
- Missing aggregate nutrition is shown as `Unavailable` rather than interpreted as zero.
- The demonstration uses locally precomputed preference candidates for speed and stability; it does not implement online training or user registration.
