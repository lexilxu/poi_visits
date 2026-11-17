"""Integration checks for moved modules, frozen receipts and experiment routing."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from exp.common.paths import REPO_ROOT
from exp.common.provenance import source_file, source_hash
from exp.exp1.queue import specifications as core_specs
from exp.exp1.queue_controls import command, plan
from exp.exp2.train import specifications as density_specs


class LayoutTests(unittest.TestCase):
    def test_catalog_resolves_to_real_modules(self):
        catalog = json.loads((REPO_ROOT/'exp/catalog.json').read_text())
        for group in catalog.values():
            for module, _ in group['actions'].values():
                with self.subTest(module=module):
                    self.assertTrue(REPO_ROOT.joinpath(*module.split('.')).with_suffix('.py').is_file())

    def test_old_and_new_receipts_keep_the_recorded_hash(self):
        old = 'experiments/evolvegcn_correction_20261115/model.py'
        new = 'exp/common/evolvegcn.py'
        for name in [old, old.replace('/', '\\'), new]:
            self.assertEqual(source_hash({'source_sha256': {name: 'frozen-hash'}}, new), 'frozen-hash')
        with self.assertRaises(KeyError):
            source_hash({'source_sha256': {}}, new)
        self.assertEqual(source_file('checkpoint_io.py'), REPO_ROOT/'exp/common/evolvegcn_io.py')

    def test_core_and_density_preserve_final_followup_recipes(self):
        core, density = core_specs(), density_specs()
        self.assertEqual(sum(s['protocol'] == 'original_masked' for s in core), 125)
        self.assertEqual(sum(s['protocol'] == 'complete_h7' for s in core), 35)
        self.assertEqual(len(density), 60)
        self.assertEqual({s['seed'] for s in density}, set(range(42, 47)))
        self.assertEqual(len({s['density'] for s in density}), 12)
        self.assertTrue(all((s['T'], s['H'], s['batch_size'], s['precision']) == (28, 14, 6, 'legacy_amp') for s in density))

    def test_control_subprocess_uses_package_module(self):
        a = SimpleNamespace(bundle=Path('panel.zip'), epochs=100, batch_size=4,
                            device='cpu', graph_dir=None, extra_input_dir=None, city_panels=None)
        recipe = plan('core')['recipes'][0]
        args = command(a, recipe, 42, .001, Path('unused-output'))
        self.assertEqual(args[1:3], ['-m', 'exp.common.controlled_training'])


if __name__ == '__main__':
    unittest.main()
