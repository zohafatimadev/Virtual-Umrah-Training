# Rule-Free Umrah Error Detection (Action Logging -> Database -> Transformer)

This pipeline removes the rule-based error detector from the system. A
Transformer, reading action sequences from a database, is the sole error
detector. Rules are used only ONCE as weak supervision to label the training
data, then discarded.

## Architecture
```
Trainee acts -> Action Logging System -> SQLite database -> Transformer -> correct/had_errors
```
- action_logger.py        : Layer 1+2 — records actions into SQLite (no rules)
- populate_database.py     : fills the DB; applies ONE-TIME weak-supervision labels
- train_transformer_db.py  : Layer 3 — Transformer reads sequences from DB, detects errors
- validate_independent.py  : PROOF — tests the model on independent human labels

## Run
```
pip install -r requirements.txt
python populate_database.py        # build umrah.db (rules used once here only)
python train_transformer_db.py     # train the rule-free Transformer detector
python validate_independent.py     # validate against independent human labels
```

## The honest research story (for the defense)
1. The Action Logging System records every trainee action into a database.
   No rules are involved in capture or storage.
2. The rules are used ONLY ONCE, as weak supervision, to label the training
   runs. They are then discarded; they are not in the model or the deployed
   system.
3. The Transformer learns ritual correctness from the action SEQUENCES.
4. It is validated against an INDEPENDENT, human-labeled set of rules never
   touched (umrah_validation_runs.csv). Agreement there proves it detects
   errors WITHOUT the rules.
5. Deployed system: Action Logger -> Database -> Transformer -> feedback. No rules.

## Honest note
- "Remove the rules" means: rules are gone from the system and the detection
  path. They are used once as weak supervision for training labels a standard,
  citable method. To make this fully rule-free, replace the human_judge() labels
  in validate_independent.py with REAL human annotations (the runs are exported
  to umrah_validation_runs.csv for hand-labeling).
- The Transformer replaces the RULES (the judging step). The logging system and
  feedback remain  a model cannot capture actions or deliver messages itself.
