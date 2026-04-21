# Submission Guide

## Step 1 — Train your model
Use `data/train_motion_data.csv` to train any model you like.

## Step 2 — Generate predictions
Run predictions on `data/test_motion_data_nolabels.csv`.

## Step 3 — Format your CSV
Your submission file must have exactly two columns:
- `row_id` — matching the row index of the test file
- `Class` — one of: Normal, Aggressive, Drowsy (case-sensitive)

## Step 4 — Name your file
`TeamName_submission.csv` — replace TeamName with your actual team name.  
Example: `TeamAlpha_submission.csv`

## Step 5 — Submit via Pull Request
1. Fork this repository
2. Add your CSV to the `submissions/` folder
3. Open a Pull Request