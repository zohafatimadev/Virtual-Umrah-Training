"""
populate_database.py
====================
Fills the SQLite database with Umrah runs (as action sequences) and applies
ONE-TIME weak-supervision labels using the rules.

IMPORTANT: this is the ONLY place rules are ever used. After this step the
rules are discarded — they are not part of the system, the model, or the
deployed pipeline. This is standard weak supervision / bootstrapping.

Run:  python populate_database.py
"""
import os
import random
from action_logger import ActionLogger, DB

random.seed(21)

ACTIONS_CORRECT_TAWAF = lambda: ['StartTawaf'] + ['Istilam', 'Walk'] * 7 + ['EndTawaf', 'PrayNawafil']
ACTIONS_CORRECT_SAI = lambda: ['StartSai'] + ['WalkSai', 'Dua'] * 7 + ['EndSai', 'DrinkZamzam', 'CutHair']


def correct_run():
    return ACTIONS_CORRECT_TAWAF() + ACTIONS_CORRECT_SAI()


def erroneous_run():
    seq = correct_run()
    k = random.random()
    if k < 0.25:      # drop some Istilam
        seq = [a for a in seq if not (a == 'Istilam' and random.random() < 0.4)]
    elif k < 0.5:     # remove walk pairs (fewer rounds)
        idx = [i for i, a in enumerate(seq) if a == 'Walk']
        for i in sorted(random.sample(idx, min(2, len(idx))), reverse=True):
            seq.pop(i)
    elif k < 0.7:     # skip hair
        seq = ['SkipHair' if a == 'CutHair' else a for a in seq]
    elif k < 0.85:    # forget nawafil
        seq = [a for a in seq if a != 'PrayNawafil']
    else:             # miss Dua
        seq = [a for a in seq if not (a == 'Dua' and random.random() < 0.4)]
    return seq


# ---- ONE-TIME rule labeller (weak supervision). Discarded after this. -------
def bootstrap_label(seq):
    istilam = seq.count('Istilam'); walk = seq.count('Walk')
    dua = seq.count('Dua')
    if istilam < 7 or walk < 7 or dua < 7: return 'had_errors'
    if 'PrayNawafil' not in seq: return 'had_errors'
    if 'SkipHair' in seq or 'CutHair' not in seq: return 'had_errors'
    return 'correct'


def main(n=4000):
    if os.path.exists(DB):
        os.remove(DB)
    lg = ActionLogger()
    n_correct = 0
    for r in range(n):
        gender = random.choice(['male', 'female'])
        seq = correct_run() if random.random() < 0.5 else erroneous_run()
        rid = f'run_{r:05d}'
        label = bootstrap_label(seq)          # rules used ONCE here
        lg.log_run_batch(rid, gender, seq, label=label, source='rule_bootstrap')
        if label == 'correct':
            n_correct += 1
    print(f'Populated {DB}: {n} runs ({n_correct} correct, {n - n_correct} had_errors)')
    print('Labels set by ONE-TIME weak supervision (rules now discarded).')


if __name__ == '__main__':
    main()
