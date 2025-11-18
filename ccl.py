#!/usr/bin/env python3
"""
Categorical Coherence Linter (CCL) — entropy‑driven ghost detector.
"""
from __future__ import annotations
import argparse, ast, importlib, inspect, json, math, os, pkgutil, random, sys, types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

def H(values: List[Any]) -> float:
    """Compute Shannon entropy of values."""
    if not values: return 0.0
    counts: Dict[str, int] = {}
    for s in map(repr, values):
        counts[s] = counts.get(s, 0) + 1
    n = len(values)
    return -sum((c / n) * math.log(c / n, 2) for c in counts.values())

def isclose(a: Any, b: Any, rt: float = 1e-6, at: float = 1e-9) -> bool:
    """Safe floating-point comparison."""
    try:
        return math.isclose(float(a), float(b), rel_tol=rt, abs_tol=at)
    except Exception:
        return False

def deq(a: Any, b: Any) -> bool:
    """Deep equality with type safety."""
    if type(a) != type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return isclose(a, b)
        return False
    if isinstance(a, (int, float)):
        return isclose(a, b)
    if isinstance(a, (str, bytes, bool, type(None))):
        return a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(deq(a[k], b[k]) for k in a)
    return repr(a) == repr(b)

# Random value generators for probing
def _rand_int() -> int: return random.randint(-1000, 1000)
def _rand_float() -> float: return random.uniform(-1e3, 1e3)
def _rand_str() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(alphabet) for _ in range(random.randint(0, 12)))
def _rand_bool() -> bool: return random.choice([True, False])

def gen_value(depth: int = 0) -> Any:
    """Generate random test values for function probing."""
    if depth > 2:
        return random.choice([_rand_int(), _rand_float(), _rand_str(), _rand_bool()])
    r = random.random()
    if r < 0.25: return [_rand_int() for _ in range(random.randint(0, 5))]
    if r < 0.4: return (_rand_float(), _rand_float())
    if r < 0.6: return {str(i): _rand_str() for i in range(random.randint(0, 4))}
    return random.choice([_rand_int(), _rand_float(), _rand_str(), _rand_bool()])

# Side effect detection
SIDE_EFFECT_NAMES = {"print", "open", "write", "writelines", "flush", "remove", "unlink", "system", "popen", "exec", "eval", "seed"}
IMPURE_MODULES = {"random", "time", "os", "sys", "subprocess", "socket", "requests"}

def infer_impurity(fn: Callable) -> Dict[str, Any]:
    """Analyze function source for side effects."""
    try:
        src = inspect.getsource(fn)
    except OSError:
        return {"could_inspect": False, "impure": True, "reasons": ["no_source"]}
    try:
        tree = ast.parse(src)
    except Exception:
        return {"could_inspect": False, "impure": True, "reasons": ["parse_failed"]}

    imp = False
    reasons: List[str] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, n: ast.Call):
            nonlocal imp
            name: Optional[str] = None
            if isinstance(n.func, ast.Name):
                name = n.func.id
            elif isinstance(n.func, ast.Attribute):
                name = n.func.attr
                mod = n.func.value.id if isinstance(n.func.value, ast.Name) else None
                if mod in IMPURE_MODULES:
                    imp = True
                    reasons.append(f"module:{mod}")
            if name in SIDE_EFFECT_NAMES:
                imp = True
                reasons.append(f"call:{name}")
            self.generic_visit(n)

        def visit_Attribute(self, n: ast.Attribute):
            nonlocal imp
            if isinstance(n.value, ast.Name) and n.value.id in {"os", "sys"}:
                imp = True
                reasons.append(f"attr:{n.value.id}.{n.attr}")
            self.generic_visit(n)

        def visit_Global(self, node):
            nonlocal imp
            imp = True
            reasons.append("global")

        def visit_Nonlocal(self, node):
            nonlocal imp
            imp = True
            reasons.append("nonlocal")

    V().visit(tree)
    return {"could_inspect": True, "impure": imp, "reasons": reasons}

def safe(fn: Callable, *a: Any, **k: Any) -> Tuple[bool, Any]:
    """Safely execute function with error handling."""
    try:
        return True, fn(*a, **k)
    except Exception as e:
        return False, f"error:{type(e).__name__}:{e}"

def arity(fn: Callable) -> int:
    """Get function arity (number of required arguments)."""
    try:
        sig = inspect.signature(fn)
        c = 0
        for p in sig.parameters.values():
            if (p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is inspect._empty):
                c += 1
        return c
    except Exception:
        return 1

def probe(fn: Callable, samples: int, seed: int) -> Dict[str, Any]:
    """Probe function behavior with random inputs."""
    random.seed(seed)
    a = arity(fn)
    id_hits = id_total = 0
    comm_hits = comm_total = 0
    assoc_hits = assoc_total = 0
    outs: List[Any] = []
    sens: List[float] = []
    anomalies: List[str] = []

    for _ in range(samples):
        if a == 0:
            ok, y = safe(fn)
            ok2, y2 = safe(fn)
            if ok and ok2 and deq(y, y2):
                id_hits += 1
            id_total += 1
            if ok: outs.append(y)
        elif a == 1:
            x = gen_value()
            ok, y = safe(fn, x)
            if not ok:
                anomalies.append("raise:unary")
                continue
            ok2, yy = safe(fn, y)
            if ok2 and deq(y, yy):
                id_hits += 1
            id_total += 1
            outs.append(y)
            if isinstance(x, (int, float)):
                dx = (abs(float(x)) + 1.0) * 1e-6
                ok3, y2 = safe(fn, float(x) + dx)
                if ok3:
                    try:
                        diff = abs(float(y2) - float(y))
                        sens.append(diff / (abs(float(y)) + 1e-9))
                    except Exception:
                        pass
        else:
            x = gen_value()
            y = gen_value()
            ok, r1 = safe(fn, x, y)
            if not ok:
                anomalies.append("raise:binary")
                continue
            ok2, r2 = safe(fn, y, x)
            if ok2:
                comm_total += 1
                if deq(r1, r2):
                    comm_hits += 1
            z = gen_value()
            ok3, xy = safe(fn, x, y)
            ok4, yz = safe(fn, y, z)
            if ok3 and ok4:
                ok5, a1 = safe(fn, xy, z)
                ok6, a2 = safe(fn, x, yz)
                if ok5 and ok6:
                    assoc_total += 1
                    if deq(a1, a2):
                        assoc_hits += 1
            ok7, xx = safe(fn, x, x)
            ok8, xxx = safe(fn, xx, xx) if ok7 else (False, None)
            if ok7 and ok8 and deq(xx, xxx):
                id_hits += 1
                id_total += 1
                outs.append(r1)

    return {
        "idempotent_rate": (id_hits / id_total) if id_total else 0.0,
        "commutative_rate": (comm_hits / comm_total) if comm_total else None,
        "associative_rate": (assoc_hits / assoc_total) if assoc_total else None,
        "entropy_bits": H(outs),
        "sensitivity": (sum(sens) / len(sens)) if sens else 0.0,
        "sample_count": id_total,
        "anomalies": anomalies,
    }

def discover(mod: types.ModuleType) -> List[Tuple[str, Callable]]:
    """Discover functions in a module."""
    return [
        (f"{mod.__name__}.{n}", o)
        for n, o in vars(mod).items()
        if inspect.isfunction(o) and getattr(o, "__module__", None) == mod.__name__ and not n.startswith("_")
    ]

def load_modules(path: Path) -> List[types.ModuleType]:
    """Load Python modules from path."""
    mods: List[types.ModuleType] = []
    path = path.resolve()
    if path.is_file() and path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = m
            spec.loader.exec_module(m)
            mods.append(m)
        return mods

    if path.is_dir():
        sys.path.insert(0, str(path))
        for _, name, _ in pkgutil.iter_modules([str(path)]):
            try:
                mods.append(importlib.import_module(name))
            except Exception:
                pass
    return mods

def analyze(target: str | Path, samples: int, seed: int) -> Dict[str, Any]:
    """Analyze target for coherence metrics."""
    mods = load_modules(Path(target))
    funcs: List[Tuple[str, Callable]] = []
    for m in mods:
        funcs += discover(m)

    fn_reports: List[Dict[str, Any]] = []
    unaries: List[Tuple[str, Callable]] = []
    for qual, fn in funcs:
        p = probe(fn, samples, seed)
        pur = infer_impurity(fn)
        ar = max(0, arity(fn))
        if ar == 1:
            unaries.append((qual, fn))
        fn_reports.append({
            "qualname": qual,
            "arity": ar,
            "purity": pur,
            "probe": p,
        })

    # Pipeline analysis
    pipes: List[Dict[str, Any]] = []
    for i in range(len(unaries)):
        for j in range(i + 1, len(unaries)):
            (qf, f), (qg, g) = unaries[i], unaries[j]
            agree = tot = 0
            for _ in range(max(10, samples // 5)):
                x = gen_value()
                ok1, a = safe(lambda v: f(g(v)), x)
                ok2, b = safe(lambda v: g(f(v)), x)
                if ok1 and ok2:
                    tot += 1
                    agree += 1 if deq(a, b) else 0
            if tot:
                pipes.append({
                    "pair": [qf, qg],
                    "agreement_rate": agree / tot,
                    "samples": tot,
                })

    # Hotspot scoring
    hotspots: List[Tuple[float, Dict[str, Any]]] = []
    for fr in fn_reports:
        p = fr["probe"]
        score = (
            p["entropy_bits"] * 0.5
            + (1 - p["idempotent_rate"]) * 0.3
            + min(1.0, p["sensitivity"]) * 0.2
            + (0.1 if p["anomalies"] else 0.0)
            + (0.1 if fr["purity"].get("impure") else 0.0)
        )
        hotspots.append((score, fr))
    hotspots.sort(key=lambda t: t[0], reverse=True)

    hot = [
        {
            "function": h[1]["qualname"],
            "score": round(h[0], 4),
            "entropy_bits": round(h[1]["probe"]["entropy_bits"], 4),
            "idempotent_rate": round(h[1]["probe"]["idempotent_rate"], 4),
            "sensitivity": round(h[1]["probe"]["sensitivity"], 6),
            "anomalies": h[1]["probe"]["anomalies"],
            "impure": h[1]["purity"].get("impure", True),
            "reasons": h[1]["purity"].get("reasons", []),
        }
        for h in hotspots[:10]
    ]

    return {
        "summary": {
            "modules": [m.__name__ for m in mods],
            "functions_analyzed": len(fn_reports),
            "pipelines_checked": len(pipes),
            "top_hotspots": hot[:5],
        },
        "functions": fn_reports,
        "pipelines": pipes,
        "ghost_hotspots": hot,
    }

def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser(description="CCL — entropy-driven ghost detector")
    ap.add_argument("path")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--report", type=str, default="report.json")
    args = ap.parse_args()
    rep = analyze(args.path, args.samples, args.seed)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {args.report}")

if __name__ == "__main__":
    main()
