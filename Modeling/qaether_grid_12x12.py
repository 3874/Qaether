"""qaether_grid_12x12.py — FCC-inspired 2D cellular complex

Mathematical framework
──────────────────────
K : 2-dimensional cellular complex, 1-skeleton ≅ FCC nearest-neighbour graph
K_2 = T ⊔ Q   (triangular + square 2-cells)
G  := R/2πZ    (coefficient group, additive notation)

A_raw(K)       = C^1(K; G)           edge phase configurations
A_adm(K,G)     ⊂ A_raw(K)            closure on bonded faces B(K,G)
A_0(K)         ⊂ A_adm(K,G)          globally flat (closure on all K_2)

B(K,G) = ⋃_{o∈Oct} F_Oct(o)  ∪  ⋃_{t∈Tet} F_Tet(t)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, Mapping, Optional, Sequence, Set, Tuple

TAU = 2.0 * math.pi
OrientedEdge = Tuple[Hashable, Hashable]


# ── helpers ─────────────────────────────────────────────────────

def mod_2pi(x: float) -> float:
    y = (x + math.pi) % TAU - math.pi
    if math.isclose(y, -math.pi, abs_tol=1e-12):
        return math.pi
    return y


def is_zero_mod_2pi(x: float, tol: float = 1e-9) -> bool:
    return abs(mod_2pi(x)) <= tol


# ── Face ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Face:
    """2-cell f ∈ K_2.  kind ∈ {"triangle","square"}."""
    face_id: str
    boundary: Tuple[OrientedEdge, ...]   # ∂^or f = (ė_1,…,ė_m)
    kind: str = "square"

    def __post_init__(self) -> None:
        if self.kind not in {"triangle", "square"}:
            raise ValueError(f"Unknown face kind: {self.kind}")
        expected = 3 if self.kind == "triangle" else 4
        if len(self.boundary) != expected:
            raise ValueError(
                f"Face {self.face_id!r}: kind={self.kind!r} but |∂f|={len(self.boundary)}"
            )
        for i, (_, v) in enumerate(self.boundary):
            nu, _ = self.boundary[(i + 1) % len(self.boundary)]
            if v != nu:
                raise ValueError(f"Boundary of {self.face_id!r} not cyclic at {i}.")
        if self.boundary[0][0] != self.boundary[-1][1]:
            raise ValueError(f"Boundary of {self.face_id!r} does not close.")


# ── CellularComplex2D ──────────────────────────────────────────

@dataclass
class CellularComplex2D:
    """2-dim cellular complex K with K_2 = T ⊔ Q."""
    vertices: Set[Hashable] = field(default_factory=set)
    reference_edges: Dict[frozenset, OrientedEdge] = field(default_factory=dict)
    faces: Dict[str, Face] = field(default_factory=dict)

    # K_2 = T ⊔ Q
    @property
    def T(self) -> Dict[str, Face]:
        return {fid: f for fid, f in self.faces.items() if f.kind == "triangle"}

    @property
    def Q(self) -> Dict[str, Face]:
        return {fid: f for fid, f in self.faces.items() if f.kind == "square"}

    def add_vertex(self, v: Hashable) -> None:
        self.vertices.add(v)

    def add_edge(self, u: Hashable, v: Hashable) -> OrientedEdge:
        if u == v:
            raise ValueError("Loop edges not allowed.")
        self.add_vertex(u); self.add_vertex(v)
        key = frozenset((u, v))
        if key not in self.reference_edges:
            ref = (u, v) if repr(u) <= repr(v) else (v, u)
            self.reference_edges[key] = ref
        return self.reference_edges[key]

    def canonicalize_edge(self, e: OrientedEdge) -> Tuple[OrientedEdge, int]:
        u, v = e
        ref = self.add_edge(u, v)
        if ref == (u, v):
            return ref, +1
        if ref == (v, u):
            return ref, -1
        raise RuntimeError("Edge orientation error.")

    def add_face(self, face_id: str, boundary: Sequence[OrientedEdge],
                 kind: str = "square") -> None:
        if face_id in self.faces:
            raise ValueError(f"Face {face_id!r} already exists.")
        face = Face(face_id=face_id, boundary=tuple(boundary), kind=kind)
        for e in face.boundary:
            self.add_edge(*e)
        self.faces[face_id] = face

    def raw_space_dimension(self) -> int:
        """dim A_raw(K) = |E^or(K)|."""
        return len(self.reference_edges)


# ── PhaseConfiguration  A_raw(K) ───────────────────────────────

@dataclass
class PhaseConfiguration:
    """φ ∈ A_raw(K) with φ(ē) = -φ(ė),  values in G = R/2πZ."""
    complex_: CellularComplex2D
    phases: Dict[OrientedEdge, float] = field(default_factory=dict)

    @classmethod
    def zero(cls, cmplx: CellularComplex2D) -> "PhaseConfiguration":
        return cls(complex_=cmplx)

    def set_phase(self, u: Hashable, v: Hashable, value: float) -> None:
        ref, sign = self.complex_.canonicalize_edge((u, v))
        self.phases[ref] = mod_2pi(sign * value)

    def get_phase(self, u: Hashable, v: Hashable) -> float:
        ref, sign = self.complex_.canonicalize_edge((u, v))
        return mod_2pi(sign * self.phases.get(ref, 0.0))

    def curvature(self, face_id: str) -> float:
        """(dφ)(f) = Σ φ(ė_j) ∈ G."""
        return mod_2pi(sum(self.get_phase(u, v)
                          for u, v in self.complex_.faces[face_id].boundary))

    def holonomy(self, face_id: str) -> complex:
        """Hol_φ(f) = exp(i·(dφ)(f)) ∈ U(1)."""
        c = self.curvature(face_id)
        return complex(math.cos(c), math.sin(c))

    def all_curvatures(self) -> Dict[str, float]:
        return {fid: self.curvature(fid) for fid in self.complex_.faces}


# ── Bonding Structures ─────────────────────────────────────────

@dataclass
class OctahedralBond:
    """o ∈ Oct(K,G).  F_Oct(o) ⊂ K_2."""
    bond_id: str
    faces: Set[str]


@dataclass
class TetrahedralBond:
    """t ∈ Tet(K,G).  F_Tet(t) ⊂ T."""
    bond_id: str
    faces: Set[str]


# ── FCCSpace (K, G) ────────────────────────────────────────────

@dataclass
class FCCSpace:
    """
    (K, G):  cellular complex + geometric data.
    B(K,G) = ⋃ F_Oct(o) ∪ ⋃ F_Tet(t)
    A_0(K) ⊂ A_adm(K,G) ⊂ A_raw(K)
    """
    complex_: CellularComplex2D
    oct_bonds: Dict[str, OctahedralBond] = field(default_factory=dict)
    tet_bonds: Dict[str, TetrahedralBond] = field(default_factory=dict)
    tol: float = 1e-9

    @property
    def bonded_faces(self) -> Set[str]:
        """B(K,G)."""
        b: Set[str] = set()
        for o in self.oct_bonds.values():
            b |= o.faces
        for t in self.tet_bonds.values():
            b |= t.faces
        return b

    def is_admissible(self, phi: PhaseConfiguration) -> bool:
        """φ ∈ A_adm(K,G) ⟺ (dφ)(f)=0 ∀f∈B(K,G)."""
        return all(is_zero_mod_2pi(phi.curvature(f), self.tol)
                   for f in self.bonded_faces)

    def is_globally_flat(self, phi: PhaseConfiguration) -> bool:
        """φ ∈ A_0(K) ⟺ (dφ)(f)=0 ∀f∈K_2."""
        return all(is_zero_mod_2pi(phi.curvature(f), self.tol)
                   for f in self.complex_.faces)

    def local_topological_defects(
        self, phi: PhaseConfiguration,
        bonded_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        if bonded_only is None:
            ids = set(self.complex_.faces)
        elif bonded_only:
            ids = self.bonded_faces
        else:
            ids = set(self.complex_.faces) - self.bonded_faces
        return {fid: phi.curvature(fid) for fid in sorted(ids)
                if not is_zero_mod_2pi(phi.curvature(fid), self.tol)}


# ── Build FCC-inspired 2D grid ─────────────────────────────────

def build_fcc_2d_grid(width: int = 12, height: int = 12) -> FCCSpace:
    """
    FCC-inspired 2D cellular complex (width × height cells).

    Structure
    ---------
    Vertices : (i,j)  0≤i≤width, 0≤j≤height
    Edges    : horizontal, vertical, + diagonal in even-parity cells
    Faces    :
      (i+j) even → diagonal (i,j)–(i+1,j+1) → 2 triangular faces  (∈ T)
      (i+j) odd  → square face                                      (∈ Q)

    Bonding
    -------
    Oct : at each interior vertex → 4 triangles + 2 squares = 6 faces
    Tet : pair of triangles in each triangulated cell
    """
    K = CellularComplex2D()

    for i in range(width + 1):
        for j in range(height + 1):
            K.add_vertex((i, j))

    for i in range(width):
        for j in range(height):
            v00, v10 = (i, j), (i + 1, j)
            v11, v01 = (i + 1, j + 1), (i, j + 1)

            if (i + j) % 2 == 0:
                # triangulated cell → 2 triangles sharing diagonal v00–v11
                K.add_face(f"t_{i}_{j}_l",
                           [(v00, v10), (v10, v11), (v11, v00)],
                           kind="triangle")
                K.add_face(f"t_{i}_{j}_u",
                           [(v00, v11), (v11, v01), (v01, v00)],
                           kind="triangle")
            else:
                # square cell
                K.add_face(f"q_{i}_{j}",
                           [(v00, v10), (v10, v11), (v11, v01), (v01, v00)],
                           kind="square")

    # ── Oct bonds: at each interior vertex ──
    oct_bonds: Dict[str, OctahedralBond] = {}
    for vi in range(1, width):
        for vj in range(1, height):
            surrounding: Set[str] = set()
            for di, dj in [(-1, -1), (0, -1), (-1, 0), (0, 0)]:
                ci, cj = vi + di, vj + dj
                if (ci + cj) % 2 == 0:
                    surrounding.add(f"t_{ci}_{cj}_l")
                    surrounding.add(f"t_{ci}_{cj}_u")
                else:
                    surrounding.add(f"q_{ci}_{cj}")
            bid = f"oct_{vi}_{vj}"
            oct_bonds[bid] = OctahedralBond(bond_id=bid, faces=surrounding)

    # ── Tet bonds: triangle pairs in each triangulated cell ──
    tet_bonds: Dict[str, TetrahedralBond] = {}
    for i in range(width):
        for j in range(height):
            if (i + j) % 2 == 0:
                bid = f"tet_{i}_{j}"
                tet_bonds[bid] = TetrahedralBond(
                    bond_id=bid,
                    faces={f"t_{i}_{j}_l", f"t_{i}_{j}_u"},
                )

    return FCCSpace(complex_=K, oct_bonds=oct_bonds, tet_bonds=tet_bonds)


# ── Phase utilities ─────────────────────────────────────────────

def flat_from_vertex_potential(
    space: FCCSpace,
    potential: Optional[Mapping[Tuple[int, int], float]] = None,
) -> PhaseConfiguration:
    """Globally flat φ via φ(u,v) = θ(v)−θ(u)."""
    K = space.complex_
    phi = PhaseConfiguration.zero(K)
    if potential is None:
        potential = {
            v: mod_2pi(0.17 * v[0] - 0.11 * v[1])
            for v in K.vertices
            if isinstance(v, tuple) and len(v) == 2
        }
    for ref in K.reference_edges.values():
        u, v = ref
        phi.set_phase(u, v, potential[v] - potential[u])
    return phi


def add_local_defect(phi: PhaseConfiguration,
                     i: int, j: int, flux: float) -> None:
    """Inject flux on bottom edge of cell (i,j), breaking local flatness."""
    phi.set_phase((i, j), (i + 1, j),
                  phi.get_phase((i, j), (i + 1, j)) + flux)


# ── Summary ─────────────────────────────────────────────────────

def grid_summary(space: FCCSpace,
                 phi: Optional[PhaseConfiguration] = None) -> Dict[str, object]:
    K = space.complex_
    out: Dict[str, object] = {
        "n_vertices": len(K.vertices),
        "n_edges": len(K.reference_edges),
        "n_faces": len(K.faces),
        "|T|": len(K.T),
        "|Q|": len(K.Q),
        "A_raw_dim": K.raw_space_dimension(),
        "|Oct|": len(space.oct_bonds),
        "|Tet|": len(space.tet_bonds),
        "|B(K,G)|": len(space.bonded_faces),
    }
    if phi is not None:
        out["is_admissible"] = space.is_admissible(phi)
        out["is_globally_flat"] = space.is_globally_flat(phi)
        out["n_defects"] = len(space.local_topological_defects(phi))
    return out


# ── Visualization ──────────────────────────────────────────────

def visualize_grid(
    space: FCCSpace,
    phi: Optional[PhaseConfiguration] = None,
    savepath: str = "grid_12x12.png",
    annotate_defects: bool = True,
) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon as MplPolygon

    K = space.complex_
    width = max(v[0] for v in K.vertices if isinstance(v, tuple))
    height = max(v[1] for v in K.vertices if isinstance(v, tuple))

    fig, ax = plt.subplots(figsize=(10, 10))

    # ── light face fills ──
    bonded = space.bonded_faces
    for fid, face in K.faces.items():
        verts = [face.boundary[0][0]]
        for _, dst in face.boundary:
            verts.append(dst)
        verts = verts[:-1]  # remove duplicate closing vertex
        if face.kind == "triangle":
            color = "#d0e8ff" if fid in bonded else "#eaf4ff"
        else:
            color = "#fff5d0" if fid in bonded else "#fffbe8"
        ax.add_patch(MplPolygon(verts, closed=True,
                                facecolor=color, edgecolor="none", zorder=0))

    # ── edges ──
    segments = []
    for u, v in K.reference_edges.values():
        if isinstance(u, tuple) and isinstance(v, tuple):
            segments.append([u, v])
    ax.add_collection(LineCollection(segments, linewidths=1.0,
                                    colors="#555555", zorder=1))

    # ── vertices ──
    vx = [v[0] for v in K.vertices if isinstance(v, tuple)]
    vy = [v[1] for v in K.vertices if isinstance(v, tuple)]
    ax.scatter(vx, vy, s=20, color="black", zorder=2)

    # ── defect annotation ──
    if annotate_defects and phi is not None:
        curvs = phi.all_curvatures()
        for fid, face in K.faces.items():
            if not is_zero_mod_2pi(curvs[fid], space.tol):
                verts = [face.boundary[0][0]]
                for _, dst in face.boundary:
                    verts.append(dst)
                cx = sum(v[0] for v in verts[:-1]) / len(verts[:-1])
                cy = sum(v[1] for v in verts[:-1]) / len(verts[:-1])
                ax.text(cx, cy, f"{curvs[fid]:.2f}",
                        ha="center", va="center",
                        fontsize=8, color="red", fontweight="bold", zorder=3)

    ax.set_xlim(-0.3, width + 0.3)
    ax.set_ylim(-0.3, height + 0.3)
    ax.set_aspect("equal")
    ax.set_title(f"FCC 2D cellular complex  ({width}×{height})"
                 f"   |T|={len(K.T)}  |Q|={len(K.Q)}", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return savepath


# ── main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    space = build_fcc_2d_grid(12, 12)
    phi = flat_from_vertex_potential(space)

    print("=== Flat 12×12 FCC grid ===")
    for k, v in grid_summary(space, phi).items():
        print(f"  {k}: {v}")

    phi_def = flat_from_vertex_potential(space)
    add_local_defect(phi_def, i=5, j=6, flux=0.7)

    print("\n=== After inserting one local defect ===")
    for k, v in grid_summary(space, phi_def).items():
        print(f"  {k}: {v}")

    defects = space.local_topological_defects(phi_def)
    print(f"  defect faces: {dict(list(defects.items())[:10])}")

    try:
        path = visualize_grid(space, phi_def, savepath="grid_12x12_defect.png")
        print(f"\nsaved: {path}")
    except Exception as e:
        print(f"visualization skipped: {e}")
