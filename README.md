# Binary Space Partition

A small, dependency-free Python library that organises integer primitive ids by
recursively splitting space with hyperplanes. Each primitive is represented by a
single sample point (its centroid, say); the tree routes that point to a leaf.

```python
from binary_space_partition import BSP, Hyperplane, Leaf

tree = BSP()
tree.build(ids=[1, 2, 3, 4], points=[[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]])

leaf = tree.query([3.0, 3.0])
print(leaf.ids)  # the primitives that share the region containing the query point

for node in tree.walk():
    if node.is_leaf:
        print(node.leaf.ids)
```

## Why this exists

The problem: you have a bag of primitives identified by integer ids and you want
to answer "which primitives live near this point?" repeatedly, without scanning
the whole set each time. A BSP gives you O(depth) point-location queries at the
cost of a one-time build.

The trade-off made here: primitives are **not** split when they straddle a
splitting plane. A primitive is assigned to exactly one leaf by the location of
its sample point. This keeps the implementation tiny and dependency-free, and it
keeps the tree balanced by point count rather than by a fragile geometric split
cost. If your primitives are large and span many leaves, you will get false
negatives from `query`; use a structure that clips primitives against planes if
that matters to you.

## The awkward edge you will hit

If every sample point is identical, no plane can separate them. The build detects
this case and falls back to a single leaf holding every id rather than recursing
forever. Likewise, if a chosen split leaves one side empty (which happens when
all points happen to lie on the splitting plane), the same fallback fires. The
result is a tree that always terminates, but which may be shallower than you
expect for degenerate inputs.

`Hyperplane` normalises its normal on construction and scales the offset
accordingly; passing a non-unit normal is fine and yields the same plane. A
near-zero normal raises `ValueError`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

