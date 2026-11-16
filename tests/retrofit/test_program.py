from __future__ import annotations

import unittest

from recurrent_retrofit.program import (Cell, GatedLoop, Program, Recirculation, Update, block_loop, from_spec,
    layer_loop, parent, route, rys)


class ProgramTests(unittest.TestCase):
    def test_parent_is_the_plain_layer_order(self):
        program = parent(6)
        self.assertEqual(program.steps, (0, 1, 2, 3, 4, 5))
        self.assertTrue(program.is_parent)
        self.assertEqual(program.spec, "parent")

    def test_one_pass_of_any_loop_is_the_parent(self):
        for build in (block_loop, layer_loop):
            program = build(6, 2, 4, 1)
            self.assertTrue(program.is_parent)
            self.assertEqual(program.steps, parent(6).steps)

    def test_block_and_layer_loops_differ_in_order_not_in_count(self):
        block = block_loop(6, 2, 4, 2)
        layer = layer_loop(6, 2, 4, 2)
        self.assertEqual(block.steps, (0, 1, 2, 3, 2, 3, 4, 5))
        self.assertEqual(layer.steps, (0, 1, 2, 2, 3, 3, 4, 5))
        self.assertEqual(block.calls_per_layer(), layer.calls_per_layer())
        self.assertEqual(block.calls_per_layer(), [1, 1, 2, 2, 1, 1])

    def test_layer_calls_follow_the_counting_convention(self):
        for layers, start, end, passes in [(28, 8, 20, 1), (28, 8, 20, 3), (36, 0, 36, 2), (5, 4, 5, 7)]:
            for build in (block_loop, layer_loop):
                program = build(layers, start, end, passes)
                self.assertEqual(program.layer_calls, layers + (passes - 1) * (end - start))
                self.assertFalse(program.is_parent and passes > 1)

    def test_rys_recipe_is_a_two_pass_block(self):
        # RYS (i, j): layers 0..j-1, then i..N-1.
        program = rys(8, 3, 6)
        self.assertEqual(program.steps, (*range(6), *range(3, 8)))
        self.assertEqual(program, block_loop(8, 3, 6, 2))

    def test_routes_may_skip_and_repeat(self):
        program = route(5, [0, 2, 2, 4])
        self.assertEqual(program.calls_per_layer(), [1, 0, 2, 0, 1])
        self.assertEqual(program.spec, "route[0,2,2,4]")
        self.assertFalse(program.is_parent)

    def test_spec_round_trip(self):
        for program in [parent(6), block_loop(6, 1, 5, 3), layer_loop(6, 0, 2, 4), route(6, [5, 0, 0])]:
            self.assertEqual(from_spec(program.spec, 6, program.passes), program)
        self.assertEqual(block_loop(28, 8, 20, 3).spec, "block[8:20]")
        self.assertEqual(block_loop(28, 8, 20, 3).describe()["layer_calls"], 52)

    def test_invalid_designs_are_rejected(self):
        for build in (block_loop, layer_loop):
            for start, end in [(3, 3), (4, 2), (-1, 2), (0, 7)]:
                with self.assertRaises(ValueError):
                    build(6, start, end, 2)
            with self.assertRaises(ValueError):
                build(6, 1, 3, 0)
        with self.assertRaises(ValueError):
            route(4, [])
        with self.assertRaises(ValueError):
            route(4, [0, 4])
        with self.assertRaises(ValueError):
            Program(4, (0, 1, 2, 3), "block", None, 2)
        for spec, passes in [("block[2:1]", 2), ("block[1:3", 2), ("route[0,1]", 2), ("parent", 2), ("loop[1:2]", 1)]:
            with self.assertRaises(ValueError):
                from_spec(spec, 4, passes)


class UpdateTests(unittest.TestCase):
    def test_specs_name_the_rule_and_round_trip(self):
        designs = {
            "block[15:19]/euler": block_loop(36, 15, 19, 3, Update("euler")),
            "block[15:19]/euler,anchor=0.5": block_loop(36, 15, 19, 3, Update("euler", anchor=0.5)),
            "layer[15:19]/euler": layer_loop(36, 15, 19, 2, Update("euler")),
            "block[10:14]/uniform": block_loop(36, 10, 14, 3, Update("uniform")),
            "block[10:14]/uniform,per=layer": block_loop(36, 10, 14, 3, Update("uniform", per="layer")),
            "block[10:14]/ema,eta=0.5": block_loop(36, 10, 14, 3, Update("ema", eta=0.5)),
            "layer[10:14]/ema,eta=0.25": layer_loop(36, 10, 14, 3, Update("ema", eta=0.25)),
            "block[10:14]": block_loop(36, 10, 14, 3),
        }
        for spec, program in designs.items():
            self.assertEqual(program.spec, spec)
            self.assertEqual(from_spec(spec, 36, program.passes), program)
        self.assertEqual(len(set(designs)), len({p.spec for p in designs.values()}))

    def test_an_update_changes_no_layer_calls(self):
        plain = block_loop(28, 9, 18, 3)
        damped = block_loop(28, 9, 18, 3, Update("euler", anchor=0.5))
        self.assertEqual(plain.steps, damped.steps)
        self.assertNotEqual(plain, damped)

    def test_cells_mark_where_the_rule_applies(self):
        # Positions: 0 1 | 2 3 | 2 3 | 2 3 | 4 5 for block[2:4] x3.
        block = block_loop(6, 2, 4, 3, Update("uniform"))
        self.assertEqual(block.cells(), [Cell(0, 0, 2, 3), Cell(0, 1, 4, 5), Cell(0, 2, 6, 7)])
        each = block_loop(6, 2, 4, 2, Update("uniform", per="layer"))
        self.assertEqual(each.cells(), [Cell(0, 0, 2, 2), Cell(1, 0, 3, 3), Cell(0, 1, 4, 4), Cell(1, 1, 5, 5)])
        # Positions: 0 1 | 2 2 2 | 3 3 3 | 4 5 for layer[2:4] x3.
        layer = layer_loop(6, 2, 4, 3, Update("euler"))
        self.assertEqual(layer.cells(), [Cell(0, 0, 2, 2), Cell(0, 1, 3, 3), Cell(0, 2, 4, 4),
            Cell(1, 0, 5, 5), Cell(1, 1, 6, 6), Cell(1, 2, 7, 7)])
        for program in (block, each, layer):
            for cell in program.cells():
                covered = program.steps[cell.first:cell.last + 1]
                self.assertTrue(all(2 <= step < 4 for step in covered))

    def test_one_pass_and_plain_loops_have_no_cells(self):
        self.assertEqual(block_loop(6, 2, 4, 1, Update("euler", anchor=0.5)).cells(), [])
        self.assertTrue(block_loop(6, 2, 4, 1, Update("uniform")).is_parent)
        self.assertEqual(block_loop(6, 2, 4, 3).cells(), [])
        self.assertEqual(parent(6).cells(), [])

    def test_invalid_rules_are_rejected(self):
        for arguments in [("damped",), ("euler", None, 0.5), ("ema",), ("uniform", 0.5), ("euler", 1.5),
                ("euler", 1), ("plain", None, None, "layer"), ("uniform", None, None, "block")]:
            with self.assertRaises(ValueError):
                Update(*arguments)
        with self.assertRaises(ValueError):
            layer_loop(6, 2, 4, 2, Update("uniform", per="layer"))
        with self.assertRaises(ValueError):
            block_loop(6, 2, 4, 2, Update("euler", per="layer"))
        with self.assertRaises(ValueError):
            Program(6, (0, 1, 2), "route", None, 1, Update("uniform"))
        for spec in ["block[2:4]/plain", "block[2:4]/euler,anchor=0.50", "block[2:4]/euler,beta=0.5",
                "block[2:4]/ema", "block[2:4]/uniform,per=unit", "block[02:4]", "block[2:4]/euler,anchor=0.5,anchor=0.5",
                "layer[2:4]/uniform,per=layer", "block[2:4]/", "route[0,1]/uniform"]:
            with self.assertRaises(ValueError, msg=spec):
                from_spec(spec, 6, 2)


class RecirculationTests(unittest.TestCase):
    def test_specs_round_trip(self):
        designs = {
            "recirculate[18to9]/convex,alpha=0.07": Recirculation(36, 18, 9, 0.07),
            "recirculate[35to16]/additive,alpha=0.1": Recirculation(36, 35, 16, 0.1, "additive"),
            "recirculate[1to0]/convex,alpha=0": Recirculation(36, 1, 0, 0.0),
        }
        for spec, design in designs.items():
            self.assertEqual(design.spec, spec)
            self.assertEqual(from_spec(spec, 36, 2), design)
            self.assertEqual(from_spec(spec, 36, 1), Recirculation(36, design.source, design.destination,
                design.alpha, design.mix, passes=1))

    def test_calls_count_the_second_pass_above_the_destination(self):
        design = Recirculation(6, 4, 1, 0.1)
        self.assertEqual(design.calls_per_layer(), [1, 1, 2, 2, 2, 2])
        self.assertEqual(design.layer_calls, 10)
        self.assertEqual(Recirculation(36, 18, 9, 0.07).layer_calls, 36 + 26)
        once = Recirculation(6, 4, 1, 0.1, passes=1)
        self.assertEqual(once.calls_per_layer(), [1] * 6)
        self.assertEqual(once.spec, design.spec)
        self.assertNotEqual(once, design)
        self.assertEqual(design.describe()["layer_calls_are_per"], "token")
        self.assertFalse(design.is_parent)
        self.assertEqual(design.cells(), [])

    def test_invalid_designs_are_rejected(self):
        for arguments in [(6, 1, 4, 0.1), (6, 3, 3, 0.1), (6, 6, 1, 0.1), (6, 4, -1, 0.1), (6, 4, 1, 1.5),
                (6, 4, 1, 1), (6, 4, 1, 0.1, "mean"), (6, 4, 1, 0.1, "convex", 3), (6, 4, 1, 0.1, "convex", 0),
                (1, 0, 0, 0.1)]:
            with self.assertRaises(ValueError, msg=arguments):
                Recirculation(*arguments)
        for spec in ["recirculate[4to1]", "recirculate[4to1]/convex", "recirculate[4to1]/convex,alpha=0.10",
                "recirculate[1to4]/convex,alpha=0.1", "recirculate[4to1]/mean,alpha=0.1", "recirculate[04to1]/convex,alpha=0.1",
                "recirculate[4to1]/convex,alpha=0.1,ramp=10", "recirculate[4to1]/convex,alpha=1.2.3",
                "recirculate[4to9]/convex,alpha=0.1"]:
            with self.assertRaises(ValueError, msg=spec):
                from_spec(spec, 6, 2)
        with self.assertRaises(ValueError):
            from_spec("recirculate[4to1]/convex,alpha=0.1", 6, 3)


class GatedLoopTests(unittest.TestCase):
    def test_specs_round_trip(self):
        designs = {
            "gated[2:27]": GatedLoop(28, 2, 27, 8),
            "gated[2:27]/halt,threshold=0.6": GatedLoop(28, 2, 27, 8, 0.6),
            "gated[0:28]/halt,threshold=1": GatedLoop(28, 0, 28, 8, 1.0),
            "gated[5:6]/halt,threshold=0": GatedLoop(28, 5, 6, 8, 0.0),
        }
        for spec, design in designs.items():
            self.assertEqual(design.spec, spec)
            self.assertEqual(from_spec(spec, 28, 8), design)
            self.assertNotEqual(from_spec(spec, 28, 1), design)

    def test_the_steps_are_a_block_loops_and_the_gate_follows_every_pass(self):
        for layers, start, end, passes in [(28, 2, 27, 1), (28, 2, 27, 8), (6, 0, 6, 3), (6, 5, 6, 2)]:
            design = GatedLoop(layers, start, end, passes)
            block = block_loop(layers, start, end, passes)
            self.assertEqual(design.steps, block.steps)
            self.assertEqual(design.calls_per_layer(), block.calls_per_layer())
            self.assertEqual(design.layer_calls, layers + (passes - 1) * (end - start))
            cells = design.cells()
            # Unlike an update rule, the gate applies on a single pass too.
            self.assertEqual(len(cells), passes)
            size = end - start
            self.assertEqual([(cell.index, cell.first, cell.last) for cell in cells],
                [(index, start + index * size, start + (index + 1) * size - 1) for index in range(passes)])
            self.assertFalse(design.is_parent)
        self.assertEqual(GatedLoop(28, 2, 27, 8).layer_calls, 203)
        self.assertEqual(GatedLoop(28, 2, 27, 8).describe()["layer_calls_are"], "exact")
        self.assertEqual(GatedLoop(28, 2, 27, 8, 0.6).describe()["layer_calls_are"], "the most a forward pass can make")

    def test_invalid_designs_are_rejected(self):
        for arguments in [(6, 4, 4, 1), (6, 4, 2, 1), (6, -1, 2, 1), (6, 2, 7, 1), (6, 2, 4, 0), (6, 2, 4, 1.0),
                (6, 2, 4, 2, 1.5), (6, 2, 4, 2, -0.1), (6, 2, 4, 2, 1), (0, 0, 0, 1)]:
            with self.assertRaises(ValueError, msg=arguments):
                GatedLoop(*arguments)
        for spec in ["gated[2:4]/halt", "gated[2:4]/halt,threshold=0.60", "gated[02:4]", "gated[2:4]/stop,threshold=0.6",
                "gated[2:4]/halt,threshold=0.6,every=2", "gated[4:2]", "gated[2:9]", "gated[2:4]/halt,threshold=1.2.3",
                "gated[2:4]/euler", "gated[2to4]"]:
            with self.assertRaises(ValueError, msg=spec):
                from_spec(spec, 6, 2)


if __name__ == "__main__":
    unittest.main()
