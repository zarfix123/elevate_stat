import re
import unicodedata
import pandas as pd

_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def parse_sub(description):
    """'SUB: Niang FOR Allen' -> 'Niang' (the incoming player's name)."""
    m = _SUB_RE.search(str(description))
    return m.group(1).strip() if m else None


def _norm(name):
    """Lowercased, accent-stripped. Sub descriptions are ASCII ('Saric') but the
    playerName column is accented Unicode ('Šarić') — normalize both to match."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def _split_name(name):
    """Normalized name -> (first_tokens, last_name, suffix). Suffix (jr/iii/...) is
    kept separately because it disambiguates same-last-name teammates.
    'Jal. Williams' -> (['jal'], 'williams', ''); 'Jackson Jr.' -> ([], 'jackson', 'jr')."""
    toks = [t.strip(".") for t in _norm(name).split() if t.strip(".")]
    suffix = ""
    if toks and toks[-1] in _SUFFIXES:
        suffix, toks = toks[-1], toks[:-1]
    if not toks:
        return [], "", suffix
    return toks[:-1], toks[-1], suffix


class _Resolver:
    """Resolves a substitution's incoming player name to a personId, using full
    names (from id_fullname) so same-last-name teammates can be disambiguated by
    first-initial. Falls back to the last-name-only playerName when no full name."""

    def __init__(self, df, id_fullname=None):
        valid = df[df["personId"].notna() & (df["personId"] != 0) & df["teamId"].notna()]
        self.by_team_last = {}   # teamId -> {last_name: [pids]}
        self.first = {}          # pid -> first token (e.g. 'jalen')
        self.suffix = {}         # pid -> suffix ('jr' / '')
        seen = set()
        for tid, pid, pname in zip(valid["teamId"], valid["personId"], valid["playerName"]):
            if (tid, pid) in seen:
                continue
            seen.add((tid, pid))
            src = id_fullname.get(pid) if (id_fullname and pid in id_fullname) else pname
            firsts, last, suffix = _split_name(src)
            self.by_team_last.setdefault(tid, {}).setdefault(last, []).append(pid)
            self.first[pid] = firsts[0] if firsts else ""
            self.suffix[pid] = suffix

    def resolve(self, tid, description):
        firsts, last, suf = _split_name(parse_sub(description))
        if not last:
            return None
        cands = self.by_team_last.get(tid, {}).get(last, [])
        if len(cands) == 1:
            return cands[0]
        if not cands:
            return None
        # Score candidates by first-initial and suffix agreement; take a unique winner.
        best, best_score, tie = None, -1, False
        for p in cands:
            score = (2 if firsts and self.first.get(p, "").startswith(firsts[0]) else 0) \
                + (1 if self.suffix.get(p, "") == suf else 0)
            if score > best_score:
                best, best_score, tie = p, score, False
            elif score == best_score:
                tie = True
        return best if (best is not None and best_score > 0 and not tie) else None


def _period_starters(period_df, teams, resolver):
    """A player started the period if their first involvement isn't 'subbed in':
    they either act, or are subbed OUT, before being subbed in."""
    subbed_in = {t: set() for t in teams}
    starters = {t: set() for t in teams}
    for row in period_df.itertuples(index=False):
        tid, at, pid = row.teamId, row.actionType, row.personId
        if tid not in teams:
            continue
        if at == "Substitution":
            in_pid = resolver.resolve(tid, row.description)
            if pid not in subbed_in[tid] and pid not in starters[tid]:
                starters[tid].add(pid)
            if in_pid is not None:
                subbed_in[tid].add(in_pid)
        elif pd.notna(pid) and pid != 0 and pid not in subbed_in[tid]:
            starters[tid].add(pid)
    return starters


def reconstruct(game_pbp, id_fullname=None):
    """Return (df_with_lineups, resolution_ok). Adds frozenset columns `on_a`/`on_b`
    (team_a = min teamId, team_b = max) with the 5-man lineup active at each event.
    Pass id_fullname (personId -> full name) to disambiguate same-last-name players.
    resolution_ok is False if any period/state doesn't resolve to exactly 5 per team."""
    df = game_pbp.sort_values("actionNumber").reset_index(drop=True)
    teams = sorted(t for t in pd.unique(df["teamId"].dropna()) if t != 0)
    if len(teams) != 2:
        df["on_a"] = df["on_b"] = None
        return df, False
    team_a, team_b = teams
    resolver = _Resolver(df, id_fullname)

    on_a_col, on_b_col, valid_col = [], [], []
    for period in sorted(df["period"].dropna().unique()):
        pdf = df[df["period"] == period]
        starters = _period_starters(pdf, teams, resolver)
        starters_ok = all(len(starters[t]) == 5 for t in teams)
        oncourt = {t: set(starters[t]) for t in teams}
        for row in pdf.itertuples(index=False):
            tid, at, pid = row.teamId, row.actionType, row.personId
            if at == "Substitution" and tid in teams:
                in_pid = resolver.resolve(tid, row.description)
                oncourt[tid].discard(pid)
                if in_pid is not None:
                    oncourt[tid].add(in_pid)
            # An event is valid only if its period's starters resolved AND the
            # current lineup is exactly 5v5 (a bad sub invalidates a period's tail,
            # but the next period re-syncs from its own starters).
            valid = starters_ok and len(oncourt[team_a]) == 5 and len(oncourt[team_b]) == 5
            on_a_col.append(frozenset(oncourt[team_a]))
            on_b_col.append(frozenset(oncourt[team_b]))
            valid_col.append(valid)

    df["on_a"], df["on_b"], df["valid"] = on_a_col, on_b_col, valid_col
    ok = bool(df["valid"].all())
    df.attrs.update(team_a=team_a, team_b=team_b, resolution_ok=ok)
    return df, ok
