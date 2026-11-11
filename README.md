# AuthenticityCheck

AuthenticityCheck studies supervised image classification using the GoldNet dataset. A model receives a photograph and predicts one of two dataset labels: **REAL** or **FAKE**.

**A photograph cannot prove that an item is genuine gold. The model learns image patterns associated with the reference labels and does not measure gold content or certify material authenticity.**

## Dataset source and labels

[GoldNet](https://github.com/zobeirraisi/GoldNet) accompanies Zobeir Raisi's GoldFormer paper. The README describes smartphone photographs from primarily Iran and the Persian Gulf region, with varied lighting, angles, distances, and backgrounds. GoldNet covers gold items broadly. A jewelry-only image count is not specified.

According to [Section 3.1 of the paper](https://www.mdpi.com/1999-4893/19/7/530), labels come from acquisition provenance:

- **REAL:** licensed jewelers/dealers supplied hallmarked items with receipts or assay documentation.
- **FAKE:** law-enforcement-confirmed counterfeit stock and imitations disclosed by cooperating pawnbrokers/dealers.

The author reports establishing labels independently of photographs. The separate visual expert benchmark did not create the reference labels. These are author-reported provenance claims. We have not independently verified the underlying receipts, assay records, or seizure documentation. A uniform laboratory assay for every item is not established by the collection description.

The audit below found repeated photographs, so the source's claim of 2,127 distinct physical items should not be treated as independently verified. No conflicting labels were found among byte-identical images. This consistency check cannot establish whether an individual label is chemically correct.

## Dataset characteristics and split status

| Characteristic | Number | Status |
| --- | ---: | --- |
| Total source images | 2,127 | Independently counted |
| REAL source images | 1,044 | Independently counted |
| FAKE source images | 1,083 | Independently counted |
| Requested training set | 1,488 | Original plan before duplicate removal |
| Requested validation set | 319 | Original plan before duplicate removal |
| Requested test set | 320 | Original plan before duplicate removal |
| Unique files after SHA-256 deduplication | 2,011 | Audited, including 1,007 REAL and 1,004 FAKE |
| Calculated training size after SHA-256 deduplication | 1,407 | Implied by current notebook splitting rule |
| Calculated validation size after SHA-256 deduplication | 302 | Implied by current notebook splitting rule |
| Calculated test size after SHA-256 deduplication | 302 | Implied by current notebook splitting rule |

The requested sizes sum to 2,127 and approximate a 70/15/15 split. The current [notebook](AuthenticityCheck.ipynb) first removes byte-identical images, then uses stratified `train_test_split` calls with `test_size=0.30`, followed by `test_size=0.50`, both with `random_state=42`. With 2,011 unique inputs, these calls imply **1,407 / 302 / 302**, not 1,488 / 319 / 320. These are calculated sizes, not saved outputs from a historical training run. Further grouping of near copies may change the final counts again.

The project split is separate from GoldNet's published five-fold evaluation protocol. The inspected notebook contains no saved execution outputs or persistent split manifest. Its filesystem traversal is not explicitly sorted, so the seed alone does not fully specify which filenames entered each historical split.

## Duplicate and leakage audit

Audit date: **11 November 2026**. GoldNet revision: [`f206c941a39d3126b6ce867e01abfad87e10f525`](https://github.com/zobeirraisi/GoldNet/tree/f206c941a39d3126b6ce867e01abfad87e10f525). Project notebook inspected at [`43862f71a1302a980d7b52a768ce188698a1fbbb`](https://github.com/zevqwlin-tech/AuthenticityCheck/tree/43862f71a1302a980d7b52a768ce188698a1fbbb).

The audit included only `gold/real` and `gold/fake`. The separate `gold/pairs` figure examples were excluded. All 2,127 audited files decoded successfully. Their SHA-256 values also matched independent downloads from the pinned GoldNet revision.

| Check | Finding |
| --- | --- |
| SHA-256 of original bytes | 115 duplicate groups, containing 116 redundant copies |
| EXIF-oriented RGB pixel hash, including dimensions | 115 duplicate groups |
| Opposite labels within a byte-identical group | 0 |
| Perceptual similarity screening | 26 candidate pairs beyond exact pixel matches |
| Visual review of all 26 candidates | 6 strong near-copy pairs, 11 uncertain similar-scene/item pairs, 9 screening false positives |
| Actual train/validation/test overlap | Not verified: the actual split manifest is absent |

Examples of byte-identical files in the pinned source:

- `gold/real/r_img_079.jpg` and `gold/real/r_img_092.jpg`
- `gold/real/r_img_137.jpg`, `gold/real/r_img_150.jpg`, and `gold/real/r_img_982.jpg`
- `gold/fake/f_img_017.jpg` and `gold/fake/f_img_018.jpg`

Strong visual near-copy examples with different original bytes and decoded pixels:

| First image | Second image | pHash distance | dHash distance |
| --- | --- | ---: | ---: |
| `real/r_img_956.jpg` | `real/r_img_988.jpg` | 0 | 0 |
| `real/r_img_004.jpg` | `real/r_img_005.jpg` | 0 | 0 |
| `fake/f_img_992.jpg` | `fake/f_img_997.jpg` | 0 | 0 |
| `real/r_img_953.jpg` | `real/r_img_990.jpg` | 0 | 2 |
| `real/r_img_950.jpg` | `real/r_img_986.jpg` | 2 | 1 |
| `real/r_img_959.jpg` | `real/r_img_984.jpg` | 2 | 1 |

Paths in this table are relative to GoldNet's `gold/` directory. The six pairs visually appear to reproduce the same photographs after image processing. This review does not identify physical items or validate gold authenticity.

Screening used 64-bit pHash, computed with an orthonormal DCT-II on a 32x32 grayscale thumbnail, the top-left 8x8 coefficients including DC, and a median threshold. It also used horizontal 64-bit dHash from a 9x8 grayscale thumbnail. Both used EXIF-oriented images and Lanczos resizing. The candidate rule was **pHash Hamming distance <= 6 OR dHash Hamming distance <= 4**. These are heuristic thresholds. Crops, rotations, alternate views, and other copies may remain undetected. Candidate pairs are not necessarily independent groups.

The notebook's SHA-256 deduplication and disjoint-hash assertions can prevent exact-byte leakage when executed successfully, but they do not prevent leakage from the six near-copy pairs. **We cannot claim that those pairs actually cross the project's splits without the saved filenames.** A high model score would not resolve this uncertainty.

Before final evaluation:

1. Review uncertain matches and place confirmed duplicate or related-item groups into a single split, or remove redundant copies before splitting.
2. Save a stable manifest containing `path`, `label`, `split`, `sha256`, and `group_id`. Sort inputs and record the source revision and seed.
3. Recheck both exact and perceptual matches across train, validation, and test. Keep augmentation in training only.
4. Freeze the test set. Use validation for model selection and assess possible shortcuts from backgrounds or capture sources.

The audit CSVs, JSON summary, and reusable audit script are supplied with the accompanying research deliverables. They are not yet tracked in this repository.

## Why this is binary Computer Vision classification

The input is an image, and the model learns visual patterns from pixels, making this a Computer Vision task. Each training image has one reference label, and the target has exactly two classes, making it supervised binary classification. The output is a label for the whole image rather than a location or segmentation mask.

In this project's notebook, `REAL = 0` and `FAKE = 1`. Thus binary precision, recall, and F1 computed with the default positive class refer to **FAKE**. This mapping must be kept consistent when interpreting metrics.

## Presentation materials

### Slide 1: Problem Statement

- REAL and FAKE images can look very similar.
- AuthenticityCheck aims to predict the dataset label of an unseen photograph.
- This is supervised Computer Vision with two output classes.
- A photograph cannot prove gold content or material authenticity.

### Slide 2: Dataset Overview

- GoldNet contains 2,127 source images: 1,044 REAL and 1,083 FAKE.
- Labels rely on documented acquisition provenance, as reported by the author.
- The initial split plan is 1,488 training / 319 validation / 320 test.
- The audit found 116 redundant byte copies. After SHA-256 cleaning, the current split rule implies 1,407 / 302 / 302.
- Six strong near-copy pairs remain a leakage risk. Actual split overlap requires the saved manifest.

### Slide 3: Research Hypothesis

- **Proposed hypothesis:** a classifier trained on GoldNet will achieve balanced accuracy above 50% on a held-out test set that separates duplicate and related-image groups.
- Use validation for model selection. Compare with a majority-class baseline, whose balanced accuracy is 50%.
- Report per-class recall and a confusion matrix to expose both error types.
- Check whether performance depends on background or capture-source shortcuts.
- This is a hypothesis, not a reported training result. Participant 1 supplies the verified experimental results.

## Training results

No training was performed for this documentation and audit task. Participant 1 owns experimental results and their verification. Existing files under `results/` are not reinterpreted here, and the original paper's scores are not presented as AuthenticityCheck results. Final metrics should identify the exact dataset revision, cleaning decisions, and split manifest used.

## References and license

1. [GoldNet repository and README](https://github.com/zobeirraisi/GoldNet).
2. Raisi, Zobeir. (2026). *GoldFormer: A Texture-Aware Vision Transformer-Based Algorithm for Detecting Near-Identical Images*. **Algorithms, 19**(7), Article 530. MDPI. Published 1 July 2026. [Paper](https://www.mdpi.com/1999-4893/19/7/530), [DOI: 10.3390/a19070530](https://doi.org/10.3390/a19070530). Zobeir Raisi is the sole listed author.
3. [GoldNet dataset license: CC BY 4.0](https://github.com/zobeirraisi/GoldNet/blob/main/LICENSE-DATA). Cite the dataset and paper when reusing images. The source repository's code has a separate [MIT license](https://github.com/zobeirraisi/GoldNet/blob/main/LICENSE).
