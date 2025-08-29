#!/usr/bin/env python3
import sys, pathlib, tomllib
from collections import defaultdict

def pick_deps(d):
    out = {}
    for sec in ("dependencies", "group", "dev-dependencies"):
        if sec == "group":
            dev = d.get("tool",{}).get("poetry",{}).get("group",{}).get("dev",{})
            if "dependencies" in dev: out.update(dev["dependencies"])
        else:
            out.update(d.get("tool",{}).get("poetry",{}).get(sec,{}))
    # drop python itself
    out.pop("python", None)
    # normalise table vs string forms
    norm = {}
    for k, v in out.items():
        if isinstance(v, str):
            norm[k] = v
        elif isinstance(v, dict):
            norm[k] = v.get("version", "*")
        else:
            norm[k] = "*"
    return norm

def load(path):
    with open(path, "rb") as f:
        return pick_deps(tomllib.load(f))

paths = [pathlib.Path(p) for p in sys.argv[1:]]
if not paths:
    print("Usage: check_deps.py <pyproject1> <pyproject2> <pyproject3>"); sys.exit(2)

depmap = {}
for p in paths:
    depmap[p.name] = load(p)

# build per-package versions
by_name = defaultdict(dict)
for proj, deps in depmap.items():
    for name, spec in deps.items():
        by_name[name][proj] = spec

mismatches = []
for name, specs in sorted(by_name.items()):
    if len(specs) < 2:  # only one project uses it
        continue
    if len(set(specs.values())) > 1:
        mismatches.append((name, specs))

if not mismatches:
    print("✅ Dependencies aligned across projects (where overlapping).")
    sys.exit(0)

print("❌ Version spec mismatches found:\n")
for name, specs in mismatches:
    left = "  - " + name
    print(left)
    for proj, spec in sorted(specs.items()):
        print(f"      {proj}: {spec}")
sys.exit(1)
