# export_profiles.py
import os, re, csv, glob, fnmatch, shlex, subprocess, datetime as dt
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

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
