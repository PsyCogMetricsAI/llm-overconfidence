"""Regression checks for portable input resolution and protocol tamper rejection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from adapt_public_seal_verifier import adapt_seal_verifier, load_binding, expected_source_sha, sha256_bytes, CHECK_BLOCK
spec = importlib.util.spec_from_file_location('public_driver', ROOT/'core_reproduction/run_core.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

class PublicInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        driver.HERE = self.base/'core'
        driver.WORK = self.base/'run'
        driver.RAW = self.base/'raw'
        driver.CORRECTED = False
        for p in [driver.HERE/'original_inputs', driver.WORK, driver.RAW]:p.mkdir(parents=True)
    def tearDown(self):self.tmp.cleanup()
    def test_relative_input_and_mutation(self):
        p=driver.HERE/'original_inputs/contract.json';p.write_text('{}')
        rec={'path':'contract.json','sha256':driver.sha(p)}
        self.assertEqual(driver.mapped_record(rec)['sha256'], rec['sha256'])
        p.write_text('{"changed":true}')
        with self.assertRaises(AssertionError):driver.mapped_record(rec)
    def test_path_escape_rejected(self):
        for path in ['../outside.json', str(self.base/'outside.json')]:
            with self.assertRaises(AssertionError):driver.mapped_record({'path':path,'sha256':'0'*64})
    def test_missing_cold_input_never_falls_back(self):
        rel='confidence_comparison/mainline_v2/features/outer_0/train.csv'
        p=driver.HERE/'original_inputs'/rel;p.parent.mkdir(parents=True);p.write_text('model_id,A\nm,1\n')
        with self.assertRaises(AssertionError):driver.mapped_record({'path':rel,'sha256':driver.sha(p)},require_cold=True)
    def test_raw_member_hash(self):
        p=driver.RAW/'arc/example.npz';p.parent.mkdir();p.write_bytes(b'raw fixture')
        rec={'path':'data_peritem_v1_20260913/arc/example.npz','sha256':driver.sha(p)}
        self.assertEqual(driver.mapped_record(rec)['sha256'],rec['sha256'])
        p.write_bytes(b'changed')
        with self.assertRaises(AssertionError):driver.mapped_record(rec)
    def test_protocol_binding_is_relocatable_and_tamper_evident(self):
        protocol=driver.HERE/'PROTOCOL.md';protocol.write_text('fixed formulas')
        (driver.HERE/'PUBLIC_INTEGRITY.json').write_text(json.dumps({'public_files':{'PROTOCOL.md':driver.sha(protocol)}}))
        self.assertEqual(load_binding(driver.HERE)[1],driver.sha(protocol))
        protocol.write_text('changed formulas')
        with self.assertRaises(AssertionError):load_binding(driver.HERE)
    def test_joint_public_contract_rejects_wrong_matrix_or_epochs(self):
        spec=importlib.util.spec_from_file_location('joint_runner_public',ROOT/'joint_reproduction/original_code/joint_runner.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        contract=ROOT/'joint_reproduction/input_contract/CONTRACT.json'
        expected=json.loads(contract.read_text())['benches']['arc']['matrix_archived_sha256']
        runner.check_input_contract(str(contract),bench='arc',matrix_sha256=expected,epochs=2500)
        with self.assertRaises(runner.RunnerError):runner.check_input_contract(str(contract),bench='arc',matrix_sha256='0'*64,epochs=2500)
        with self.assertRaises(runner.RunnerError):runner.check_input_contract(str(contract),bench='arc',matrix_sha256=expected,epochs=5)
    def test_joint_cell_converter_hash_is_checked(self):
        sys.path.insert(0,str(ROOT/'joint_reproduction'))
        import cold
        cold.contract()
        builder=cold.builder()
        builder.load_cell_recover(expected_sha256=builder.CELL_RECOVER_SHA256_PIN)
        with self.assertRaises(builder.BuilderError):builder.load_cell_recover(expected_sha256='0'*64)
    def test_seal_source_and_protocol_checks_survive_adaptation(self):
        core=ROOT/'core_reproduction'
        source=(core/'original_inputs/confidence_comparison/risk_metrics_v1/reviews/verify_seal_independent.py').read_text()
        protocol=self.base/'protocol.md';protocol.write_text('fixed settings')
        want=driver.sha(protocol)
        kwargs=dict(source_sha256=sha256_bytes(source.encode()),expected_source_sha256=expected_source_sha(core),public_protocol_sha256=want,public_protocol_relpath='PROTOCOL.md',plan_binding_sha256='0'*64)
        adapted,_=adapt_seal_verifier(source,**kwargs)
        self.assertIn(CHECK_BLOCK,adapted)
        kwargs['source_sha256']='0'*64
        with self.assertRaises(AssertionError):adapt_seal_verifier(source,**kwargs)
        root=self.base/'seal';root.mkdir();(root/'protocol.json').write_text('{}')
        def fail(*args):raise AssertionError(args)
        scope={'ROOT':root,'Path':Path,'pr':{'plan_path':str(protocol)},'PLAN_SHA':want,'digest':driver.sha,'fail':fail}
        exec(textwrap.dedent(CHECK_BLOCK),scope)
        protocol.write_text('modified settings')
        with self.assertRaises(AssertionError):exec(textwrap.dedent(CHECK_BLOCK),scope)

if __name__=='__main__':unittest.main()
