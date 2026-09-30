from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


class Hyperplane:
    """An N-dimensional hyperplane, defined by a unit normal and a signed offset.

    The hyperplane is the set of points x satisfying dot(normal, x) - offset = 0.
    Points whose signed distance is negative are "behind" the plane, positive are
    "in front". These terms are arbitrary but fixed so callers can reason about
    which subtree a primitive lands in.

    Normal is normalised on construction. We avoid dividing by a zero-length normal
    by raising immediately; callers building planes from two coincident points need
    to handle that upstream.
    """

    def __init__(self, normal: Sequence[float], offset: float, eps: float = 1e-9):
        if len(normal) == 0:
            raise ValueError("normal must be non-empty")
        m = 0.0
        for c in normal:
            m += float(c) * float(c)
        if m <= eps * eps:
            raise ValueError("normal is (near) zero; cannot define a plane")
        length = m ** 0.5
        self.normal = tuple(float(c) / length for c in normal)
        self.offset = float(offset) / length
        self.eps = float(eps)

    def signed_distance(self, point: Sequence[float]) -> float:
        if len(point) != len(self.normal):
            raise ValueError(
                f"point has {len(point)} dims; plane is {len(self.normal)}-d"
            )
        d = 0.0
        for c, n in zip(point, self.normal):
            d += float(c) * n
        return d - self.offset

    def classify(self, point: Sequence[float]) -> str:
        d = self.signed_distance(point)
        if d < -self.eps:
            return "behind"
        if d > self.eps:
            return "in_front"
        return "on"

    def __repr__(self) -> str:
        return f"Hyperplane(normal={self.normal!r}, offset={self.offset!r})"


@dataclass
class Leaf:
    """A bucket of primitive ids that share a region of space.

    ids is a list (not a set) so callers who pass duplicates see them preserved.
    Ordering is stable across builds given a stable input order.
    """
    ids: List[int] = field(default_factory=list)


class BSP:
    """A binary space partition tree over integer primitive ids.

    The tree stores only primitive ids (ints). The caller owns the actual
    primitive geometry. This keeps the library dependency-free and avoids any
    particular assumption about what a primitive looks like — points, boxes,
    triangles, whatever; the caller provides a centroid-like sample point per id.

    Interpretation chosen (documented in README): a primitive is assigned to
    exactly one leaf by the point its centroid sample maps to. We do not attempt
    to split primitives that straddle a plane. This is the simplest defensible
    reading of "organise primitives by splitting space", and it keeps the tree
    balanced by point count rather than by a more fragile geometric-split cost.
    """

    def __init__(self, plane: Optional[Hyperplane] = None, eps: float = 1e-9):
        # plane is None for leaves; for internal nodes both children are non-None.
        self.plane = plane
        self.eps = float(eps)
        self._behind: Optional[BSP] = None
        self._in_front: Optional[BSP] = None
        self._leaf: Optional[Leaf] = None
        self._dim: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        return self._leaf is not None and self.plane is None

    @property
    def leaf(self) -> Leaf:
        if self._leaf is None:
            raise ValueError("node is not a leaf")
        return self._leaf

    @property
    def behind(self) -> BSP:
        if self._behind is None:
            raise ValueError("node has no behind child")
        return self._behind

    @property
    def in_front(self) -> BSP:
        if self._in_front is None:
            raise ValueError("node has no in_front child")
        return self._in_front

    def build(self, ids: Sequence[int], points: Sequence[Sequence[float]]) -> BSP:
        """Construct the subtree from a flat list of ids and their sample points.

        points[i] is the sample for ids[i]. Duplicates in ids are kept and
        routed together. An empty input produces an empty leaf.
        """
        if len(ids) != len(points):
            raise ValueError("ids and points must have equal length")
        if len(ids) <= 1:
            self._leaf = Leaf(ids=list(ids))
            if len(points) > 0:
                self._dim = len(points[0])
            return self

        normal, offset = self._choose_split(points)
        try:
            plane = Hyperplane(normal, offset, self.eps)
        except ValueError:
            # All sample points are coincident; no plane can separate them.
            self._leaf = Leaf(ids=list(ids))
            self._dim = len(points[0])
            return self

        behind_ids: List[int] = []
        front_ids: List[int] = []
        behind_pts: List[Sequence[float]] = []
        front_pts: List[Sequence[float]] = []
        for i, pid in enumerate(ids):
            side = plane.classify(points[i])
            if side == "behind":
                behind_ids.append(pid)
                behind_pts.append(points[i])
            else:
                # "on" and "in_front" both go to the front. Putting coplanar
                # points consistently on one side avoids infinite recursion
                # when many points share the splitting plane.
                front_ids.append(pid)
                front_pts.append(points[i])

        if not behind_ids or not front_ids:
            # The split didn't actually partition the set (e.g. everything lay
            # on the plane and got routed to front). Bail out to a leaf rather
            # than recurse infinitely.
            self._leaf = Leaf(ids=list(ids))
            self._dim = len(points[0])
            return self

        self.plane = plane
        self._dim = len(points[0])
        self._behind = BSP(eps=self.eps).build(behind_ids, behind_pts)
        self._in_front = BSP(eps=self.eps).build(front_ids, front_pts)
        return self

    def _choose_split(self, points: Sequence[Sequence[float]]) -> tuple:
        """Pick a splitting plane from the sample points.

        Strategy: take the axis of greatest spread as the normal, and the mean
        coordinate along that axis as the offset. This is a cheap, deterministic
        heuristic (no floating-point ties on the choice of axis because we pick
        the first axis achieving the max, in index order).
        """
        dim = len(points[0])
        mins = [float("inf")] * dim
        maxs = [float("-inf")] * dim
        for p in points:
            if len(p) != dim:
                raise ValueError("all points must share the same dimension")
            for i in range(dim):
                v = float(p[i])
                if v < mins[i]:
                    mins[i] = v
                if v > maxs[i]:
                    maxs[i] = v
        spreads = [maxs[i] - mins[i] for i in range(dim)]
        best_axis = 0
        best_spread = spreads[0]
        for i in range(1, dim):
            if spreads[i] > best_spread:
                best_spread = spreads[i]
                best_axis = i
        normal = [0.0] * dim
        normal[best_axis] = 1.0
        mean = 0.0
        for p in points:
            mean += float(p[best_axis])
        mean /= len(points)
        return normal, mean

    def query(self, point: Sequence[float]) -> Leaf:
        """Return the leaf the given point falls into."""
        if self._dim is not None and len(point) != self._dim:
            raise ValueError(
                f"point has {len(point)} dims; tree is {self._dim}-d"
            )
        if self.is_leaf:
            return self._leaf  # type: ignore[return-value]
        if self.plane is None:
            raise RuntimeError("inconsistent node: not a leaf but has no plane")
        side = self.plane.classify(point)
        if side == "behind":
            return self.behind.query(point)
        return self.in_front.query(point)

    def walk(self):
        """Yield every node, leaves and internals, depth-first left (behind) first."""
        yield self
        if self._behind is not None:
            yield from self._behind.walk()
        if self._in_front is not None:
            yield from self._in_front.walk()

    def leaves(self) -> List[Leaf]:
        return [n.leaf for n in self.walk() if n.is_leaf]

    def depth(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + max(self.behind.depth(), self.in_front.depth())
