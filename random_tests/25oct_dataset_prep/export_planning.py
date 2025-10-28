from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import re
import json
import os

PAIR_COLS = [
    "source_rec_path", "animal1_key", "animal2_key",
    "area1", "area2", "short_id1", "short_id2", 
    "session_manual", "include", "notes"
]

# Animal key aliases - add your own here!
# Alias -> canonical animal key
ALIASES = {
    # Example: "mouse1": "pmc_L",
    # Example: "lefty": "v1_left",
}


def load_registry():
    """Load animal IDs from file."""
    if os.path.exists("animal_ids.json"):
        with open("animal_ids.json") as f:
            return json.load(f)
    return {}

def save_registry(animal_dict):
    """Save animal IDs to file."""
    with open("animal_ids.json", 'w') as f:
        json.dump(animal_dict, f, indent=2)

def seed_pairs_csv(filtered_table: pa.Table, out_csv: str, prev_csv: str = None):
    """Create/append pairs CSV with paths from filtered_table."""
    t = filtered_table
    if "is_social" in t.schema.names:
        t = t.filter(pc.field("is_social") == True)
    
    paths = pc.unique(t["rec_path"])
    blanks = {c: pa.array([""] * len(paths), pa.string()) for c in PAIR_COLS[1:]}
    new_rows = pa.table({"source_rec_path": paths, **blanks})
    
    if prev_csv and Path(prev_csv).exists():
        prev = pacsv.read_csv(prev_csv)
        is_dup = pc.is_in(new_rows["source_rec_path"], prev["source_rec_path"])
        out = pa.concat_tables([prev, new_rows.filter(pc.invert(is_dup))], promote=True)
    else:
        out = new_rows
    
    pacsv.write_csv(out, out_csv)
    return out_csv


def autofill_short_ids(pairs_csv: str):
    """Auto-assign short IDs based on animal_key. Area inferred from key name."""
    rows = pacsv.read_csv(pairs_csv).to_pylist()
    
    # Track: animal_key -> (area, num)
    animal_to_id = load_registry()
    used_nums = {"MC": set(), "VC": set(), "UNK": set()}
    

    # Load registry and convert format
    saved_registry = load_registry()
    for animal, value in saved_registry.items():
        if isinstance(value, str):
            # Format: "MC1", "VC1", "UNK1"
            m = re.match(r"^(MC|VC|UNK)(\d+)$", value)
            if m:
                area = m.group(1)
                num = int(m.group(2))
                animal_to_id[animal] = (area, num)
                used_nums[area].add(num)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            # Format: ["MC", 1] or ("MC", 1)
            area, num = value[0], value[1]
            animal_to_id[animal] = (area, num)
            used_nums[area].add(num)


    def resolve_alias(key: str) -> str:
        """Resolve alias to canonical animal key."""
        return ALIASES.get(key, key)
    
    def infer_area(key: str) -> str:
        """Guess area from animal key: v1/V1 -> VC, pmc -> MC, else UNK"""
        k = key.lower()
        if "v1" in k or k.startswith("v"):
            return "VC"
        if "pmc" in k or "mc" in k:
            return "MC"
        return "UNK"
    
    # First pass: collect existing IDs
    for r in rows:
        for animal_key, area_col, sid in [
            ("animal1_key", "area1", "short_id1"), 
            ("animal2_key", "area2", "short_id2")
        ]:
            key = resolve_alias(str(r.get(animal_key) or "").strip())
            if not key:
                continue
            
            # Parse existing short_id like "MC3", "VC12", or "UNK5"
            sid_val = str(r.get(sid) or "").strip().upper()
            m = re.match(r"^(MC|VC|UNK)(\d+)$", sid_val)
            if m:
                area, num = m.group(1), int(m.group(2))
                animal_to_id[key] = (area, num)
                used_nums[area].add(num)
    
    # Second pass: assign missing IDs
    for r in rows:
        for animal_key, area_col, sid in [
            ("animal1_key", "area1", "short_id1"), 
            ("animal2_key", "area2", "short_id2")
        ]:
            key = resolve_alias(str(r.get(animal_key) or "").strip())
            if not key:
                continue
            
            if key not in animal_to_id:
                area = infer_area(key)
                num = 1
                while num in used_nums[area]:
                    num += 1
                animal_to_id[key] = (area, num)
                used_nums[area].add(num)
            
            # Write back area and short_id
            area, num = animal_to_id[key]
            r[area_col] = area
            r[sid] = f"{area}{num}"
    
    # Save
    out_tab = pa.table({c: pa.array([r.get(c, "") for r in rows], pa.string()) for c in PAIR_COLS})
    pacsv.write_csv(out_tab, pairs_csv)
    
    registry_to_save = {animal: f"{area}{num}" for animal, (area, num) in animal_to_id.items()}
    save_registry(registry_to_save)
    
    return pairs_csv


def build_plan_from_pairs(pairs_csv: str, profile: dict) -> list[dict]:
    """Build export plan from pairs CSV."""
    rows = pacsv.read_csv(pairs_csv).to_pylist()
    
    # Filter: both animals filled, not excluded
    def _include(r):
        inc = str(r.get("include") or "").strip().lower()
        excluded = inc in {"0", "no", "n", "false", "skip", "exclude"}
        has_both = str(r.get("animal1_key") or "").strip() and str(r.get("animal2_key") or "").strip()
        return has_both and not excluded
    
    rows = [r for r in rows if _include(r)]
    if not rows:
        return []
    
    # Group by pair using short_id
    from collections import defaultdict
    groups = defaultdict(list)
    
    for r in rows:
        id1 = str(r.get("short_id1") or "").strip()
        id2 = str(r.get("short_id2") or "").strip()
        pair_key = f"{id1}+{id2}"  # KEEP ORDER - NO SORTING!
        groups[pair_key].append(r)
    
    # Build plan
    plan = []
    for pair_key, items in groups.items():
        items.sort(key=lambda r: r.get("source_rec_path", ""))
        for i, r in enumerate(items, 1):
            session = str(r.get("session_manual") or "").strip()
            sidx = int(session) if session and session.isdigit() else i
            
            plan.append({
                "dest_rel_path": f"social/{pair_key}_S{sidx}/",
                "source_rec_path": r.get("source_rec_path", ""),
                "id": pair_key,
                "session_index": sidx,
                "path_globs": profile.get("includes", []),
            })
    
    return plan