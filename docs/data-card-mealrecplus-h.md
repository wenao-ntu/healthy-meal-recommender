# MealRec+H dataset card

## Suitability

MealRec+H supports the project's combination of personalized preference learning and health-aware reranking. It provides user-course interactions, user-meal interactions, meal-course affiliations, and FSA/WHO health scores.

The dataset is used under the following constraints: pin the source revision, cite the accompanying paper, do not redistribute raw files, process only necessary fields, and state both the simulated-data component and the absence of a declared license.

## Source and version

| Field | Value |
|---|---|
| Official repository | https://github.com/WUT-IDEA/MealRecPlus |
| Pinned commit | `8bd7d7d7effb505be4a7d2074b28abad802fe0f7` |
| Paper | MealRec+: A Meal Recommendation Dataset with Meal-Course Affiliation for Personalization and Healthiness |
| Venue | SIGIR 2024 |
| DOI | https://doi.org/10.1145/3626772.3657857 |
| Variant | MealRec+H |
| Retrieval date | 2026-09-11 |

The repository and paper link to each other. Original user-course interactions come from Allrecipes. Meal construction and user-meal interactions were produced through the dining-session simulation and collaborative interaction-mining procedures described by the authors. The data therefore represent neither clinical patient records nor complete observed meal diaries.

## License and citation status

- The repository is publicly accessible, publishes H and L variants, and requests citation of the SIGIR 2024 paper.
- No LICENSE or COPYING file and no SPDX license declaration were found at the pinned revision.
- This project cites the source but does not describe it as MIT, Creative Commons, or another explicit license.
- Raw data are not committed. The repository contains only the source URL, pinned commit, download procedure, and integrity evidence.
- Explicit redistribution rights should be confirmed with the authors before any use beyond the course context.

## Integrity checks

| File | Bytes | SHA-256 |
|---|---:|---|
| `course.zip` | 195,052,658 | `00f10920166e21bbf2a581c835027cc413c7bc96c79053e4a8cfe0774dbff7ce` |
| `user_course.zip` | 1,880,922 | `661892a6c73ad03dec66b74d8d92ddfb87cef3a6150ab0bfd178ae178c8e62e7` |

Hashes for all core text files are stored in `results/data_audit/mealrecplus_h_audit.json`.

## Audited scale

| Entity or relation | Count |
|---|---:|
| Users | 1,575 |
| Course catalog | 7,280 |
| Meals | 3,817 |
| User-course interactions | 151,148 |
| Meal-course relations | 11,451 |
| User-meal training interactions | 37,413 |
| User-meal validation interactions | 4,677 |
| User-meal test interactions | 4,677 |
| Total user-meal interactions | 46,767 |

The official split is approximately 8:1:1, with no duplicate interactions across splits.

## Meal structure

- Every meal contains exactly three courses.
- Every meal covers categories 0, 1, and 2: appetizer, main dish, and dessert.
- The 3,817 meals use 942 distinct courses. Other catalog courses appear only in the user-course interaction layer.
- `course_category.txt` has 11,451 rows because course-category relations repeat with meal occurrences. Deduplication produces 942 course-category pairs, with one category per used course.

The 11,451 relation rows must not be misinterpreted as 11,451 unique courses.

## Health attributes

Lower FSA scores and higher WHO scores represent healthier meals under the source definitions.

| Vector | Count | Minimum | Maximum | Mean |
|---|---:|---:|---:|---:|
| Course FSA | 7,280 | 4.0000 | 12.0000 | 9.1195 |
| Course WHO | 7,280 | 0.0000 | 7.0000 | 2.2120 |
| Meal FSA | 3,817 | 5.0000 | 11.6667 | 9.2758 |
| Meal WHO | 3,817 | 0.3333 | 5.3333 | 2.0602 |
| User FSA | 1,575 | 7.0000 | 10.6667 | 9.3623 |
| User WHO | 1,575 | 0.9167 | 3.6667 | 2.0554 |

All values are finite and vector lengths match the corresponding entity indices.

## Train-only user baselines

The audit recomputed mean FSA and WHO values for every user from training interactions and from all three splits:

- Official `user_fsa.txt` equals the train-only reconstruction for all 1,575 users, with MAE = 0.
- Official `user_who.txt` equals the train-only reconstruction for all 1,575 users, with MAE = 0.
- Reconstruction from train, validation, and test produces FSA MAE = 0.079926 and WHO MAE = 0.057947 relative to the official vectors.

The official user health vectors were therefore derived from training interactions rather than validation or test labels. The project still recomputes these baselines from training data and uses the official files only as consistency assertions.

## Metadata handling

The extracted `course.csv` is approximately 706 MB and contains 7,280 rows. Core fields `course_name`, `category`, `ingredients`, and `nutritions` are complete.

Nutrition values are Python dictionary literals rather than JSON. The pipeline parses them with `ast.literal_eval`; it never uses `eval`. All 7,280 rows parse successfully. The used courses provide numeric amounts for protein, carbohydrates, sugars, sodium, fat, saturated fat, and fibre. Five of the 942 used courses encode all eight nutrition amounts as `0` with missing units. These records are treated as source missing-value placeholders rather than genuine zero nutrition.

The `reviews` field is very large; the first inspected row exceeds 2.6 MB. Reviews, complete cooking directions, image binaries, and source usernames are unnecessary for the recommendation task and are not retained in processed data.

## Split and warm-start rules

Meal interactions have no timestamps. The project uses the official 8:1:1 assignment and applies the following reproducible warm-start rule:

1. Training contains all 1,575 users, 3,810 meals, and 37,413 interactions.
2. Validation contains nine interactions with six meals absent from training; removal leaves 4,668 interactions for 1,130 users.
3. Test contains four interactions with three meals absent from training; removal leaves 4,673 interactions for 1,114 users.
4. Main results cover warm-start meals. All 13 exclusions and their reasons are recorded separately.
5. Recommendation candidates exclude meals already present in the user's available history.

## Privacy and safety

- Models use contiguous numeric identifiers and do not require source usernames.
- `user2index.txt` maps source-site identifiers but is excluded from processed data and the demonstration.
- No patient identities, diagnoses, medical record numbers, contact details, or individual clinical measurements were observed.
- FSA and WHO values are general food-wellness indicators; they do not support diagnosis or personalized treatment.

## Limitations

- Meal composition and user-meal interactions include simulation and collaborative-mining components.
- The absence of timestamps prevents temporal extrapolation evaluation.
- FSA and WHO do not represent disease-specific, allergy-specific, or physiologically personalized suitability.
- Allrecipes may introduce regional, language, cuisine, and activity biases.
- The absence of an explicit license prevents this project from redistributing raw data.

## Reproduction

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_mealrecplus.ps1
.\.venv\Scripts\python.exe .\scripts\audit_mealrecplus.py
```
