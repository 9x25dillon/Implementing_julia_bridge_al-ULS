"""
Minimal FastAPI mock for al-ULS /optimize so you can test the Julia bridge end-to-end.
Run: uvicorn mock_al_uls_server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np

app = FastAPI()

class OptimizeIn(BaseModel):
    adjacency: List[List[float]]
    labels: Optional[List[str]] = None
    mode: Optional[str] = "kfp"
    beta: Optional[float] = 0.8

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/optimize")
def optimize(inp: OptimizeIn):
    adj = inp.adjacency
    n = len(adj)
    out_sums = [(i, sum(adj[i])) for i in range(n)]
    routing = [i for i, _ in sorted(out_sums, key=lambda t: t[1], reverse=True)]
    return {
        "routing": routing,
        "objective": sum(sum(row) for row in adj),
        "notes": f"mock-al-ULS mode={inp.mode} beta={inp.beta}",
    }

# Generic function-dispatch endpoint used by TA ULS trainer's JuliaClient
class FunctionCall(BaseModel):
    function: str
    args: List[Any]

def _optimize_matrix(matrix: List[List[float]], method: str = "sparsity") -> Dict[str, Any]:
    A = np.array(matrix, dtype=float)
    if method == "sparsity":
        # Threshold small values to zero at the 60th percentile of absolute values
        thresh = np.percentile(np.abs(A), 60)
        B = A.copy()
        B[np.abs(B) < max(thresh, 1e-8)] = 0.0
    elif method == "svd_lowrank":
        # Keep top-k components (k=min( min(A.shape), 8 ))
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        k = int(max(1, min(min(A.shape), 8)))
        S = np.zeros_like(A)
        np.fill_diagonal(S, s)
        B = (U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :])
    else:
        B = A
    zeros = np.sum(B == 0.0)
    total = B.size
    return {
        "optimized_matrix": B.tolist(),
        "compression_ratio": float(zeros) / float(total) if total else 0.0,
        "method": method,
    }

def _create_polynomials(data: List[List[float]], variables: List[str]) -> Dict[str, Any]:
    X = np.array(data, dtype=float)  # shape: (n_samples, n_features)
    n, d = X.shape
    vars_ = variables if variables else [f"x{i}" for i in range(d)]
    # Build polynomial features up to degree 2: 1, xi, xi*xj
    feats = [np.ones((n, 1))]
    names = ["1"]
    # Linear terms
    for j in range(d):
        feats.append(X[:, [j]])
        names.append(vars_[j])
    # Quadratic/cross terms
    for i in range(d):
        for j in range(i, d):
            feats.append((X[:, [i]] * X[:, [j]]))
            names.append(f"{vars_[i]}*{vars_[j]}")
    Phi = np.concatenate(feats, axis=1)
    # Return the design matrix summary and feature names
    return {
        "feature_names": names,
        "num_features": int(Phi.shape[1]),
        "num_samples": int(Phi.shape[0]),
        "feature_stats": {
            "mean": Phi.mean(axis=0).tolist(),
            "std": (Phi.std(axis=0) + 1e-12).tolist(),
        },
    }

def _analyze_polynomials(polynomials: Dict[str, Any]) -> Dict[str, Any]:
    means = np.array(polynomials.get("feature_stats", {}).get("mean", []), dtype=float)
    stds = np.array(polynomials.get("feature_stats", {}).get("std", []), dtype=float)
    return {
        "num_features": polynomials.get("num_features", 0),
        "dominant_features": int(np.sum(means > (means.mean() + stds.mean()))),
        "mean_of_means": float(means.mean()) if means.size else 0.0,
        "mean_of_stds": float(stds.mean()) if stds.size else 0.0,
    }

@app.post("/")
def function_dispatch(call: FunctionCall):
    if call.function == "optimize_matrix":
        matrix = call.args[0] if len(call.args) > 0 else []
        method = call.args[1] if len(call.args) > 1 else "sparsity"
        return _optimize_matrix(matrix, method)
    if call.function == "create_polynomials":
        data = call.args[0] if len(call.args) > 0 else []
        variables = call.args[1] if len(call.args) > 1 else []
        return _create_polynomials(data, variables)
    if call.function == "analyze_polynomials":
        polys = call.args[0] if len(call.args) > 0 else {}
        return _analyze_polynomials(polys)
    return {"error": f"Unknown function: {call.function}"}
