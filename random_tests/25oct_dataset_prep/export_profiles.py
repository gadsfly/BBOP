# ---- simple social alias CSV flow -------------------------------------------
# CSV schema we output & later reuse:
# token,animal_key,area,short_id
# - token: what we auto-detect from a session name (e.g., "20240809v1r1", "pmc13r1")
# - animal_key: grouping key; if you want two tokens to be the same animal, give them the same animal_key
# - area: MC or VC (we guess; you can fix)
# - short_id: MC1/VC3... (left blank first run; we fill next run — or when you call build)

import csv
# export_profiles.py
import os, re, csv, glob, fnmatch, shlex, subprocess, datetime as dt
import pyarrow as pa
import pyarrow.parquet as pq
import yaml



# --- stricter token extraction -----------------------------------------------
# uses helpers you already have:
#  - _session_dir_name, _split_path (paths), _session_split_pair (split by '_p')
#    (these are already defined above in your file)  # e.g. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
_BAD_CHUNKS = {"2social","social","mini","miniscope","single","test","only","aligned",
               "com","dannce","predict","vis","ao","bo","with","and"}

def _pick_animalish_chunk(seg: str) -> str:
    # pick first chunk with at least one letter and one digit, or starts with pmc/v1/le/re
    for raw in (seg or "").split("_"):
        c = raw.strip()
        if not c or c.lower() in _BAD_CHUNKS or c.isdigit():
            continue
        if re.search(r"[A-Za-z]", c) and re.search(r"\d", c):
            return c
        if re.match(r"(?i)^(pmc|v1|le|re)", c):
            return c
    # fallback: first non-empty chunk
    return (seg or "").split("_", 1)[0].strip()

def _detect_pair_tokens(row):
    session = _session_dir_name(_split_path(_to_str(row.get("rec_path"))))
    left, right = _session_split_pair(session)    # already in your file  # :contentReference[oaicite:2]{index=2}
    return _pick_animalish_chunk(left), _pick_animalish_chunk(right)



def _guess_area(tok, row):
    s = (tok or "").lower() + " " + _to_str(row.get("rec_path")).lower()
    if "pmc" in s: return "MC"
    if "v1"  in s: return "VC"
    return "MC"  # default; you can edit later

def write_social_animals_csv(filtered_table: pa.Table, out_csv: str, prev_csv: str=None):
    """First pass: auto spit CSV of tokens we see in SOCIAL rows only."""
    prev = {}
    if prev_csv and os.path.exists(prev_csv):
        with open(prev_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                prev[r["token"].lower()] = r

    rows_out, seen = [], set()
    for r in filtered_table.to_pylist():
        if not _truthy(r.get("social")):        # only social
            continue
        a, b = _detect_pair_tokens(r)
        for tok in (a, b):
            if not tok: continue
            key = tok.lower()
            if key in seen: continue
            seen.add(key)
            old = prev.get(key, {})
            rows_out.append({
                "token": tok,
                "animal_key": old.get("animal_key") or tok,          # default: itself
                "area": old.get("area") or _guess_area(tok, r),       # guessed
                "short_id": old.get("short_id") or "",                # assign later
            })

    rows_out.sort(key=lambda x: (x["area"], x["animal_key"], x["token"]))
    _ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["token","animal_key","area","short_id"])
        w.writeheader(); w.writerows(rows_out)
    print(f"Wrote proposal CSV with {len(rows_out)} rows → {out_csv}")

def _load_animals_csv(csv_path):
    m_by_token = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m_by_token[r["token"].lower()] = {
                "animal_key": r.get("animal_key") or r["token"],
                "area": (r.get("area") or "MC").upper(),
                "short_id": r.get("short_id") or "",
            }
    return m_by_token

def _assign_missing_short_ids(m_by_token, update_csv_path=None):
    """Fill empty short_id per animal_key & area with next MC#/VC#; optionally persist back."""
    # collect per (area, animal_key)
    by_key = {}
    for tok, rec in m_by_token.items():
        by_key.setdefault((rec["area"], rec["animal_key"]), []).append(tok)

    # find used numbers per area
    used = {"MC": set(), "VC": set()}
    for rec in m_by_token.values():
        sid = rec.get("short_id","")
        if sid.startswith("MC"): 
            n = sid[2:].isdigit() and int(sid[2:])
            if n: used["MC"].add(n)
        if sid.startswith("VC"):
            n = sid[2:].isdigit() and int(sid[2:])
            if n: used["VC"].add(n)

    def _next(area):
        n = 1
        while n in used[area]: n += 1
        used[area].add(n)
        return f"{area}{n}"

    # assign one short_id per (area, animal_key)
    key2sid = {}
    for (area, akey), toks in sorted(by_key.items()):
        # if any token under this key already had a short_id, keep it
        sid = None
        for t in toks:
            sid = sid or m_by_token[t]["short_id"]
        if not sid:
            sid = _next(area)
        key2sid[(area, akey)] = sid
        for t in toks:
            m_by_token[t]["short_id"] = sid

    # optionally persist
    if update_csv_path:
        # re-write rows ordered
        rows = []
        for tok, rec in m_by_token.items():
            rows.append({"token": tok, "animal_key": rec["animal_key"], "area": rec["area"], "short_id": rec["short_id"]})
        rows.sort(key=lambda x: (x["area"], x["animal_key"], x["token"]))
        with open(update_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["token","animal_key","area","short_id"])
            w.writeheader(); w.writerows(rows)
        print(f"Updated short_id back to CSV → {update_csv_path}")
    return m_by_token

def build_export_plan_social_v2(filtered_table: pa.Table, animals_csv: str, profile: dict=None, update_csv=True):
    """
    Output dest paths like: social/<MCx+VCy>_sN/
    No nested date/time; we keep originals only in the log metadata.
    """
    # load + ensure every animal_key has a short_id
    m = _load_animals_csv(animals_csv)
    m = _assign_missing_short_ids(m, update_csv_path=(animals_csv if update_csv else None))

    plan, pair_counts = [], {}
    rows = filtered_table.to_pylist()
    for r in rows:
        if not _truthy(r.get("social")):     # skip non-social for this v2
            continue

        ymd, hms = _extract_date_time(r)     # used only for stable ordering if needed
        a_raw, b_raw = _detect_pair_tokens(r)
        if not a_raw or not b_raw:
            continue
        a = m.get(a_raw.lower()); b = m.get(b_raw.lower())
        if not a or not b:                    # token not in CSV yet
            # you can re-run write_social_animals_csv to add the new tokens
            continue

        a_sid, b_sid = a["short_id"], b["short_id"]
        pair = "+".join(sorted([a_sid, b_sid]))

        # gather paths
        base_paths = _base_paths_for_row(r)
        final_globs, _ = _apply_profile_to_paths(r, base_paths, profile)

        # assign s-index per pair (order by date/time appearance)
        pair_counts.setdefault(pair, 0)
        pair_counts[pair] += 1
        s_idx = pair_counts[pair]

        dest_rel = _join("social", f"{pair}_s{s_idx}")
        plan.append({
            "source_rec_path": _to_str(r.get("rec_path")),
            "dest_rel_path": dest_rel,
            "path_globs": final_globs,
            "is_social": True,
            "id": pair,
            "ymd": ymd, "hms": hms,
            "session_index": s_idx,
            "orig_session_name": _session_dir_name(_split_path(_to_str(r.get("rec_path")))),
            "tokens": f"{a_raw}+{b_raw}",
        })
    return plan




# ---------------- basic utils ----------------

def _truthy(x):
    return str(x).strip().lower() in {"1","true","t","yes","y"}

def _to_str(x):
    return "" if x is None else str(x)

def _join(*a):
    return os.path.join(*[p for p in a if p not in (None, "", ".")])

def _ensure_dir(p):
    if p: os.makedirs(p, exist_ok=True)

def _run(cmd, dry):
    if dry:
        print("$", " ".join(shlex.quote(c) for c in cmd))
        return 0
    return subprocess.call(cmd)

# ---------------- aliases ----------------

def load_aliases(csv_path=None):
    """CSV with columns: seen_name, canonical_name"""
    m = {}
    if not csv_path or not os.path.exists(csv_path): return m
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seen = r.get("seen_name","").strip()
            canon = r.get("canonical_name","").strip()
            if seen:
                m[seen.lower()] = (canon or seen).strip()
    return m

def canonical(name, alias_map):
    if not name: return "UNK"
    k = name.strip().lower()
    return alias_map.get(k, name.strip())

# ---------------- date/time ----------------

def _split_path(path):
    return [p for p in path.replace("\\","/").split("/") if p]

def _extract_date_time(row):
    """
    Returns ('YYYY_MM_DD', 'HHMMSS').
    - date: prefer row['date_folder'] like '2025_05_16'; else parse from path; else UTC today
    - time: look for HH_MM_SS or HH_MM in row fields or session folder; else UTC now
    """
    rec_path = _to_str(row.get("rec_path"))
    parts = _split_path(rec_path)

    # date
    df = _to_str(row.get("date_folder"))
    m = re.fullmatch(r"(\d{4})_(\d{2})_(\d{2})", df) if df else None
    if m:
        ymd = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    else:
        seg = next((p for p in parts if re.fullmatch(r"\d{4}_\d{2}_\d{2}", p)), None)
        if seg: ymd = seg
        else:   ymd = dt.datetime.utcnow().strftime("%Y_%m_%d")

    # time (prefer HH_MM_SS then HH_MM -> HHMM00)
    candidates = [
        _to_str(row.get("time")),
        _to_str(row.get("time_folder")),
        _to_str(row.get("rec_file")),
        _session_dir_name(parts)  # often carries *_HH_MM
    ]
    hms = None
    for s in candidates:
        if not s: continue
        m3 = re.search(r"(\d{2})_(\d{2})_(\d{2})", s)
        if m3: hms = f"{m3.group(1)}{m3.group(2)}{m3.group(3)}"; break
        m2 = re.search(r"(\d{2})_(\d{2})(?!_)", s)
        if m2: hms = f"{m2.group(1)}{m2.group(2)}00"; break
    if not hms:
        hms = dt.datetime.utcnow().strftime("%H%M%S")

    return ymd, hms

def _session_dir_name(parts):
    """Return directory immediately after YYYY_MM_DD; else last component."""
    for i, p in enumerate(parts):
        if re.fullmatch(r"\d{4}_\d{2}_\d{2}", p):
            return parts[i+1] if i+1 < len(parts) else parts[-1]
    return parts[-1] if parts else ""

# ---------------- ID parsing ----------------
# Strategy:
#   - prefer explicit columns (animal / animal_a / animal_b / pair_id)
#   - else parse tokens from the session folder name (right after date)
#   - tokenize as letter+digit mixes; drop known noise; keep first 1 or 2 canonical tokens

_NOISE = {
    "mini","miniscope","social","single","test","videos",
    "predict","predict00","predict01","vis","only","aligned",
    "com","with","and","wnd","stp","max","diff","pnr","auto",
    "mir","aligned","h5","mat","dannce","com3d","trajectory","plot","speed","histogram"
}

def _candidate_tokens(name):
    # split to alnum chunks, keep those mixing letters+digits
    raw = re.findall(r"[A-Za-z]+[A-Za-z0-9]*\d+[A-Za-z0-9]*", name)
    toks = []
    for t in raw:
        tl = t.lower()
        if any(k in tl for k in _NOISE):    # drop obvious noise
            continue
        if len(t) < 3:                      # too short → ignore
            continue
        toks.append(t)
    # prefer tokens starting with V, PMC, LE, RE
    def score(t):
        up = t.upper()
        if up.startswith("V"): return (0, t)
        if up.startswith("PMC"): return (1, t)
        if up.startswith("LE") or up.startswith("RE"): return (2, t)
        return (3, t)
    toks = sorted(dict.fromkeys(toks), key=score)  # unique, stable order
    return toks
# --- helpers for pair parsing ---
def _strip_dates(s: str) -> str:
    # remove 8-digit dates and 'p' + date markers
    s = re.sub(r"(?i)\bp?\d{8}\b", "", s)
    return s

def _session_split_pair(session_name: str):
    """
    Split '<mini>_p<partner>' pattern if present.
    Returns (left, right) segments (raw, with everything else untouched).
    """
    m = re.search(r"(?i)^(.*?)[_\-]p(.*)$", session_name)
    if not m:
        return (session_name, "")
    return (m.group(1), m.group(2))

def _extract_best_token(seg: str, *, keep_leading_date: bool):
    """
    Extract an animal-like token from a segment.
    Rules:
      - Prefer pattern with optional leading date: ^(?:(\d{8}))?([A-Za-z][A-Za-z0-9]*\d[A-Za-z0-9]*)
      - Accept bare 'PMC...' tokens (even without digits)
      - If keep_leading_date=True and a date is present, prefix it to the token (e.g., '20240919v1l5r1')
      - Fallback: previous scored-token heuristic (tokens mixing letters+digits)
    """
    seg = seg.strip()
    if not seg:
        return ""

    # 1) strong pattern: optional 8-digit date + alnum code
    m = re.search(r"^(?:(\d{8}))?([A-Za-z][A-Za-z0-9]*\d[A-Za-z0-9]*)", seg)
    if m:
        date = m.group(1) or ""
        tok  = m.group(2)
        return (date + tok) if (date and keep_leading_date) else tok

    # 2) accept 'PMC...' (no digits) as a valid token
    m2 = re.search(r"^(?:(\d{8}))?(PMC[A-Za-z0-9]*)", seg, flags=re.IGNORECASE)
    if m2:
        date = m2.group(1) or ""
        tok  = m2.group(2)
        return (date + tok) if (date and keep_leading_date) else tok

    # 3) fallback: scan for mixed letter+digit tokens inside the segment
    raw = re.findall(r"[A-Za-z]+[A-Za-z0-9]*\d+[A-Za-z0-9]*", seg)
    if raw:
        return raw[0]
    return ""

def _pick_id(row, alias_map, is_social, idcfg):
    """Choose ID with mini-first / optional leading dates."""
    get = lambda k: row.get(k) if k in row else None

    # 0) explicit columns still win
    if is_social:
        a = _to_str(get("animal_a"))
        b = _to_str(get("animal_b"))
        pair = _to_str(get("pair_id"))
        vals = [v for v in (a, b) if v]
        if vals:
            canon = [canonical(v, alias_map) for v in vals]
            # mini-first: keep order; alpha: sort
            if (idcfg or {}).get("pair_sort", "mini_first") == "alpha":
                canon = sorted(dict.fromkeys(canon))
            else:
                canon = list(dict.fromkeys(canon))
            return "+".join(canon[:2]) if canon else "UNK"
        if pair:
            parts = [p for p in re.split(r"[+_, ]+", pair) if p]
            canon = [canonical(p, alias_map) for p in parts]
            if (idcfg or {}).get("pair_sort", "mini_first") == "alpha":
                canon = sorted(dict.fromkeys(canon))
            else:
                canon = list(dict.fromkeys(canon))
            return "+".join(canon[:2]) if canon else "UNK"
    else:
        a = _to_str(get("animal") or get("animal_id"))
        if a:
            return canonical(a, alias_map)

    # 1) infer from the session dir immediately after YYYY_MM_DD
    session = _session_dir_name(_split_path(_to_str(get("rec_path"))))
    left_seg, right_seg = _session_split_pair(session)

    keep_mini_date    = bool((idcfg or {}).get("keep_mini_date", True))
    keep_partner_date = bool((idcfg or {}).get("keep_partner_date", False))
    pair_sort         = (idcfg or {}).get("pair_sort", "mini_first")

    if is_social:
        mini_tok    = _extract_best_token(left_seg,  keep_leading_date=keep_mini_date)
        partner_tok = _extract_best_token(right_seg, keep_leading_date=keep_partner_date)

        A = canonical(mini_tok, alias_map) if mini_tok else ""
        B = canonical(partner_tok, alias_map) if partner_tok else ""
        ids = [x for x in (A, B) if x]

        if not ids:
            return "UNK"

        if pair_sort == "alpha":
            ids = sorted(dict.fromkeys(ids))
        else:
            # mini_first: keep order A, B, but de-dup
            ids = list(dict.fromkeys(ids))

        return "+".join(ids[:2])

    # single
    single_tok = _extract_best_token(left_seg or session, keep_leading_date=keep_mini_date)
    return canonical(single_tok, alias_map) if single_tok else "UNK"

# ---------------- tiers (base) ----------------
# Base tiers are code-defined, then profiles tweak them with flags + includes/excludes.

def _base_paths_for_row(row, include_videos=True, include_visuals=False):
    has_com   = _truthy(row.get("com"))
    has_dan   = _truthy(row.get("dannce"))
    has_align = _truthy(row.get("mini_rec_sync")) or _truthy(row.get("mini_rec_sync_com"))

    paths = []
    # T0 (raw/provenance)
    paths += ["metaData.json", "calib/"]
    if include_videos:
        paths.insert(1, "videos/")

    # T1 (core derived)
    if has_com:   paths += ["COM/predict*/com3d*.mat"]
    if has_dan:   paths += ["DANNCE/predict*/save_data_AVG.mat"]
    if has_align: paths += ["MIR_Aligned/only_com*.h5", "MIR_Aligned/aligned_predictions_with_ca*.h5"]

    # T2 visuals (optional via flag)
    if include_visuals:
        paths += [
            "COM/predict*/vis/*.jpg","COM/predict*/vis/*.png",
            "DANNCE/predict*/vis/*.jpg","DANNCE/predict*/vis/*.png",
        ]
    return paths

# ---------------- profiles (YAML) ----------------

def load_profiles(yaml_path):
    """
    YAML schema:

    profiles:
      paper-core:
        flags:
          include_videos: true
          include_visuals: false
        includes: []         # extra globs to add
        excludes: ["**/vis/**"]  # globs to remove
        conditions:          # optional row-aware tweaks
          - when: "social == 1"
            includes: ["MIR_Aligned/*with_ca*.h5"]
            excludes: []

      qc-review:
        flags: { include_videos: false, include_visuals: true }
        includes: ["COM/predict*/com3d*.mat"]
        excludes: []

      dannce-delta:
        flags: { include_videos: false, include_visuals: false }
        includes: ["DANNCE/predict01/save_data_AVG.mat"]
        excludes: ["DANNCE/predict00/**"]

      repro-full:
        flags: { include_videos: true, include_visuals: true }
        includes: []
        excludes: []

    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return (data.get("profiles") or {})

def _apply_profile_to_paths(row, base_paths, profile):
    """Return final list of path globs after flags/includes/excludes + simple conditions."""
    flags     = profile.get("flags", {}) if profile else {}
    includes  = list(profile.get("includes", []) if profile else [])
    excludes  = list(profile.get("excludes", []) if profile else [])
    conds     = profile.get("conditions", []) if profile else []

    # condition language: only expose 'social' and row fields as strings
    ctx = {k: row.get(k) for k in row.keys()}
    ctx["social"] = 1 if _truthy(row.get("social")) else 0

    # allow conditions to add includes/excludes
    for c in conds:
        expr = _to_str(c.get("when"))
        if not expr: continue
        try:
            ok = bool(eval(expr, {"__builtins__": {}}, ctx))
        except Exception:
            ok = False
        if ok:
            includes += c.get("includes", []) or []
            excludes += c.get("excludes", []) or []

    # flags that change base composition (rebuild base if needed)
    inc_vid = bool(flags.get("include_videos", True))
    inc_vis = bool(flags.get("include_visuals", False))
    # rebuild base to honor flags (so profile flags always win)
    base_paths = _base_paths_for_row(row, include_videos=inc_vid, include_visuals=inc_vis)

    # merge includes
    final = list(dict.fromkeys(base_paths + includes))

    # apply excludes as fnmatch over pattern strings (we'll also check on actual matches later)
    def keep(p):
        return not any(fnmatch.fnmatch(p, pat) for pat in excludes)
    final = [p for p in final if keep(p)]
    return final, flags

# ---------------- plan + export ----------------

def build_export_plan(filtered_table: pa.Table, alias_csv: str = None, profile: dict = None):
    """
    Returns list of dicts:
      - source_rec_path
      - dest_rel_path  = <single|social>/<YYYY_MM_DD>/<ID>_<HHMMSS>/
      - path_globs     = final list of relative src globs (already profile-adjusted)
      - meta: is_social, id, ymd, hms
    """
    alias_map = load_aliases(alias_csv)
    rows = filtered_table.to_pylist()
    plan = []

    for r in rows:
        src_root = _to_str(r.get("rec_path"))
        if not src_root: continue

        is_social = _truthy(r.get("social"))
        ymd, hms  = _extract_date_time(r)
        idcfg     = (profile or {}).get("id", {})  # pull per-profile id options
        sid       = _pick_id(r, alias_map, is_social, idcfg)
        side      = "social" if is_social else "single"
        dest_rel  = _join(side, ymd, f"{sid}_{hms}")

        base_paths = _base_paths_for_row(r)  # default: videos on, visuals off
        path_globs, _flags = _apply_profile_to_paths(r, base_paths, profile)

        plan.append({
            "source_rec_path": src_root,
            "dest_rel_path": dest_rel,
            "path_globs": path_globs,
            "is_social": is_social,
            "id": sid,
            "ymd": ymd,
            "hms": hms,
            # also carry a lean copy of row fields for debugging if needed
        })
    return plan

def export_with_rsync(
    plan,
    export_root,
    dry_run=True,
    mode="staged",
    dataset_version=None,
    profile_label="default",
    log_parquet_path=None,
    log_on_dry_run=False,
):
    """
    - dry_run=True: prints rsync commands only (no dirs created, no logs unless log_on_dry_run=True)
    - mode: 'staged' or 'snapshot' (snapshot writes under export_root/v<dataset_version>/)
    """
    if mode not in {"staged","snapshot"}:
        raise ValueError("mode must be 'staged' or 'snapshot'")

    root = _join(export_root, f"v{dataset_version}") if (mode=="snapshot" and dataset_version) else export_root

    log_rows = []
    for item in plan:
        src_root = item["source_rec_path"]
        dest     = _join(root, item["dest_rel_path"])
        if not dry_run:
            _ensure_dir(dest)

        # expand globs and dedupe files to avoid double rsync of the same path
        file_set = set()
        for rel in item["path_globs"]:
            for m in glob.glob(_join(src_root, rel)):
                file_set.add(os.path.abspath(m))

        for src in sorted(file_set):
            cmd = ["rsync", "-a"]
            if dry_run: cmd.append("-n")
            cmd += [src, dest + "/"]
            _run(cmd, dry_run)


        log_rows.append({
            "export_time_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "profile_label": profile_label,
            "export_mode": mode,
            "dataset_version": dataset_version or "",
            "source_rec_path": src_root,
            "dest_rel_path": item["dest_rel_path"],
            "id": item["id"],
            "is_social": int(item["is_social"]),
            "dry_run": bool(dry_run),
        })

    if log_parquet_path and (not dry_run or log_on_dry_run):
        log_path = _normalize_log_path(log_parquet_path)
        _ensure_dir(os.path.dirname(log_path) or ".")
        _append_parquet(log_path, pa.Table.from_pylist(log_rows))

# def _append_parquet(path, new_table: pa.Table):
#     if os.path.exists(path):
#         old = pq.read_table(path)
#         out = pa.concat_tables([old, new_table], promote=True)
#     else:
#         out = new_table
#     pq.write_table(out, path)

def _append_parquet(path, new_table: pa.Table):
    if os.path.exists(path):
        old = pq.read_table(path)
        out = pa.concat_tables([old, new_table], promote_options="default")
    else:
        out = new_table
    pq.write_table(out, path)


def _normalize_log_path(path):
    if not path: return None
    if os.path.isdir(path): return os.path.join(path, "export_log.parquet")
    return path
