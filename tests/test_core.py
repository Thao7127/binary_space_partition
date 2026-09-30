import math
import unittest

from binary_space_partition import BSP, Hyperplane, Leaf


class TestHyperplane(unittest.TestCase):
    def test_normalises_to_unit(self):
        hp = Hyperplane([3.0, 4.0], 10.0)
        mag = math.sqrt(sum(c * c for c in hp.normal))
        self.assertAlmostEqual(mag, 1.0, places=12)
        self.assertEqual(len(hp.normal), 2)

    def test_offset_scales_with_normalisation(self):
        # normal [3,4] -> [0.6, 0.8]; offset 10 / 5 = 2
        hp = Hyperplane([3.0, 4.0], 10.0)
        self.assertAlmostEqual(hp.offset, 2.0, places=12)

    def test_signed_distance_on_plane_is_zero(self):
        hp = Hyperplane([1.0, 0.0], 5.0)
        self.assertAlmostEqual(hp.signed_distance([5.0, 999.0]), 0.0, places=12)

    def test_classify_behind_in_front_on(self):
        hp = Hyperplane([1.0, 0.0], 5.0)
        self.assertEqual(hp.classify([0.0, 0.0]), "behind")
        self.assertEqual(hp.classify([10.0, 0.0]), "in_front")
        self.assertEqual(hp.classify([5.0, 0.0]), "on")

    def test_rejects_zero_normal(self):
        with self.assertRaises(ValueError):
            Hyperplane([0.0, 0.0], 1.0)

    def test_rejects_empty_normal(self):
        with self.assertRaises(ValueError):
            Hyperplane([], 1.0)

    def test_point_dimension_mismatch(self):
        hp = Hyperplane([1.0, 0.0], 0.0)
        with self.assertRaises(ValueError):
            hp.signed_distance([1.0, 2.0, 3.0])


class TestBSPBuild(unittest.TestCase):
    def test_empty_input_yields_empty_leaf(self):
        tree = BSP().build([], [])
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.leaf.ids, [])

    def test_single_primitive_is_leaf(self):
        tree = BSP().build([7], [[1.0, 2.0]])
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.leaf.ids, [7])

    def test_two_primitives_split(self):
        # along x, ids 1 (x=0) behind, 2 (x=10) in front; mean = 5
        tree = BSP().build([1, 2], [[0.0, 0.0], [10.0, 0.0]])
        self.assertFalse(tree.is_leaf)
        self.assertEqual(tree.plane.normal, (1.0, 0.0))
        self.assertTrue(tree.behind.is_leaf)
        self.assertTrue(tree.in_front.is_leaf)
        self.assertEqual(tree.behind.leaf.ids, [1])
        self.assertEqual(tree.in_front.leaf.ids, [2])

    def test_duplicate_ids_are_preserved_and_routed_together(self):
        tree = BSP().build([5, 5, 5], [[0.0], [0.0], [0.0]])
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.leaf.ids, [5, 5, 5])

    def test_coincident_points_yield_leaf(self):
        # All points identical -> cannot split -> leaf with all ids.
        tree = BSP().build([1, 2, 3], [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.leaf.ids, [1, 2, 3])

    def test_coplanar_points_routed_to_front(self):
        # Points share the chosen split plane (mean along the spread axis).
        # behind_ids is empty, so we fall back to a single leaf rather than
        # recurse with an empty behind child.
        tree = BSP().build([1, 2, 3], [[1.0], [1.0], [1.0]])
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.leaf.ids, [1, 2, 3])

    def test_dimension_mismatch_in_points_raises(self):
        with self.assertRaises(ValueError):
            BSP().build([1, 2], [[0.0, 0.0], [1.0]])

    def test_ids_points_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            BSP().build([1, 2], [[0.0]])

    def test_chooses_axis_of_greatest_spread(self):
        # x spread 1, y spread 100 -> split along y.
        tree = BSP().build([1, 2], [[0.0, 0.0], [1.0, 100.0]])
        self.assertEqual(tree.plane.normal, (0.0, 1.0))

    def test_balanced_four_leaf_partition(self):
        pts = [[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]]
        ids = [1, 2, 3, 4]
        tree = BSP().build(ids, pts)
        leaves = tree.leaves()
        self.assertEqual(len(leaves), 4)
        all_ids = [i for lf in leaves for i in lf.ids]
        self.assertEqual(sorted(all_ids), [1, 2, 3, 4])
        self.assertEqual(tree.depth(), 3)

    def test_query_routes_to_expected_leaf(self):
        pts = [[0.0], [10.0]]
        tree = BSP().build([1, 2], pts)
        self.assertEqual(tree.query([-1.0]).ids, [1])
        self.assertEqual(tree.query([11.0]).ids, [2])
        # point exactly on the plane goes to front by design
        self.assertEqual(tree.query([5.0]).ids, [2])

    def test_query_dimension_mismatch_raises(self):
        tree = BSP().build([1], [[0.0, 0.0]])
        with self.assertRaises(ValueError):
            tree.query([1.0])

    def test_walk_yields_all_nodes(self):
        tree = BSP().build([1, 2], [[0.0, 0.0], [10.0, 0.0]])
        nodes = list(tree.walk())
        # root + two leaves
        self.assertEqual(len(nodes), 3)
        self.assertTrue(nodes[0] is tree)

    def test_query_on_empty_tree_returns_empty_leaf(self):
        tree = BSP().build([], [])
        result = tree.query([0.0, 0.0])
        self.assertEqual(result.ids, [])

    def test_build_is_stable_for_repeated_calls(self):
        pts = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
        ids = [1, 2, 3, 4]
        t1 = BSP().build(ids, pts)
        t2 = BSP().build(ids, pts)
        self.assertEqual(t1.leaves(), t2.leaves())


class TestBSPProperties(unittest.TestCase):
    def test_internal_node_children_raise_when_leaf(self):
        tree = BSP().build([1], [[0.0, 0.0]])
        with self.assertRaises(ValueError):
            _ = tree.behind
        with self.assertRaises(ValueError):
            _ = tree.in_front

    def test_leaf_access_on_internal_raises(self):
        tree = BSP().build([1, 2], [[0.0], [10.0]])
        with self.assertRaises(ValueError):
            _ = tree.leaf


if __name__ == "__main__":
    unittest.main()
