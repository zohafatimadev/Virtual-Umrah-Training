"""
action_logger.py
================
THE ACTION LOGGING SYSTEM (Layer 1 + Layer 2 of the architecture).

Its only job is to RECORD what the trainee does and STORE it in a database.
It does NOT judge correctness — no rules live here. It is a flight recorder.

Architecture:
    Trainee acts  ->  ActionLogger.log()  ->  SQLite database (umrah.db)

Two tables:
    runs    : one row per Umrah run     (run_id, gender, label, started_at)
    actions : one row per action taken  (run_id, step, action, state...)

The Transformer later reads ORDERED SEQUENCES of actions from this database.
No rules are involved in capture or storage.
"""
import sqlite3
import time
import os

DB = 'umrah.db'


class ActionLogger:
    def __init__(self, db_path=DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id      TEXT PRIMARY KEY,
                gender      TEXT,
                label       TEXT,          -- filled later (bootstrap or human)
                label_source TEXT,         -- 'rule_bootstrap' or 'human'
                started_at  REAL
            )''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT,
                step       INTEGER,        -- order of the action in the run
                action     TEXT,           -- e.g. Istilam, Walk, Dua
                phase      TEXT,           -- tawaf / sai / ...
                round      INTEGER,
                lap        INTEGER,
                ts         REAL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )''')
        con.commit()
        con.close()

    def start_run(self, run_id, gender):
        con = sqlite3.connect(self.db_path)
        con.execute('INSERT OR REPLACE INTO runs (run_id, gender, started_at) VALUES (?,?,?)',
                    (run_id, gender, time.time()))
        con.commit(); con.close()

    def log(self, run_id, step, action, phase='', round_=0, lap=0):
        """Record ONE action. No judgement — just store it."""
        con = sqlite3.connect(self.db_path)
        con.execute('''INSERT INTO actions (run_id, step, action, phase, round, lap, ts)
                       VALUES (?,?,?,?,?,?,?)''',
                    (run_id, step, action, phase, round_, lap, time.time()))
        con.commit(); con.close()

    def log_run_batch(self, run_id, gender, actions, label=None, source=None):
        """Fast path: insert a whole run (all its actions) in ONE transaction.
        actions = list of action names in order."""
        con = sqlite3.connect(self.db_path)
        con.execute('INSERT OR REPLACE INTO runs (run_id, gender, label, label_source, started_at) VALUES (?,?,?,?,?)',
                    (run_id, gender, label, source, time.time()))
        rows = [(run_id, i, a, 'tawaf' if i < len(actions) // 2 else 'sai', 0, 0, time.time())
                for i, a in enumerate(actions)]
        con.executemany('''INSERT INTO actions (run_id, step, action, phase, round, lap, ts)
                           VALUES (?,?,?,?,?,?,?)''', rows)
        con.commit(); con.close()

    def set_label(self, run_id, label, source):
        con = sqlite3.connect(self.db_path)
        con.execute('UPDATE runs SET label=?, label_source=? WHERE run_id=?',
                    (label, source, run_id))
        con.commit(); con.close()

    def get_sequence(self, run_id):
        con = sqlite3.connect(self.db_path)
        cur = con.execute('SELECT action FROM actions WHERE run_id=? ORDER BY step', (run_id,))
        seq = [r[0] for r in cur.fetchall()]
        con.close()
        return seq

    def all_runs(self):
        con = sqlite3.connect(self.db_path)
        cur = con.execute('SELECT run_id, gender, label, label_source FROM runs')
        rows = cur.fetchall(); con.close()
        return rows


if __name__ == '__main__':
    # tiny demo
    if os.path.exists(DB):
        os.remove(DB)
    lg = ActionLogger()
    lg.start_run('demo_1', 'male')
    for i, a in enumerate(['StartTawaf', 'Istilam', 'Walk', 'EndTawaf']):
        lg.log('demo_1', i, a, 'tawaf')
    print('Logged sequence:', lg.get_sequence('demo_1'))
    print('Action logging system works.')
