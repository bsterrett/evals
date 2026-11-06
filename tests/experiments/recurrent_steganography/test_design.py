from pathlib import Path
import unittest

from recurrent_steg.design import decoding_jobs, decoder_prompt, encoding_jobs, load_design, observer_jobs
from recurrent_steg.scoring import calibrate_threshold, parse_bit, parse_observer

ROOT = Path(__file__).resolve().parents[3]


class DesignTests(unittest.TestCase):
    def test_payloads_are_balanced_within_each_cover_and_depth(self):
        config,cases,_ = load_design(ROOT/"experiments/recurrent_steganography/configs/pilot.toml")
        for depth in config["depths"]:
            jobs = encoding_jobs(config,cases,depth)
            self.assertEqual(len(jobs),32)
            for case in (c for c in cases if c["split"]=="development"):
                self.assertEqual(sorted(j["bit"] for j in jobs if j["case_id"]==case["id"] and j["kind"]=="encoded"),[0,1])
            self.assertTrue(all(j["bit"] is None for j in jobs if j["kind"]!="encoded"))

    def test_receiver_input_is_independent_of_private_target_and_audit_ids(self):
        # Same carrier with different private labels must produce identical input.
        outputs = [dict(carrier="An ordinary message.",bit=b,case_id=f"private-{b}") for b in [0,1]]
        prompts = [decoder_prompt(row["carrier"]) for row in outputs]
        self.assertEqual(prompts[0],prompts[1])
        self.assertNotIn("private-",prompts[0])

    def test_crossed_jobs_reuse_carriers_and_keep_targets_out_of_receiver_input(self):
        config,cases,_=load_design(ROOT/"experiments/recurrent_steganography/configs/pilot.toml")
        carriers=[dict(job,carrier="An ordinary paraphrase.") for depth in config["depths"]
            for job in encoding_jobs(config,cases,depth)]
        self.assertEqual(len(carriers),96)
        for depth in config["depths"]:
            jobs=decoding_jobs(config,carriers,cases,depth)
            self.assertEqual(len(jobs),81)
            encoded=[j for j in jobs if j["kind"]=="encoded"]
            self.assertEqual(len(encoded),48)
            self.assertTrue(all(j["prompt"]==decoder_prompt("An ordinary paraphrase.") for j in encoded))
        monitors=observer_jobs(config,carriers,cases)
        self.assertEqual(len(monitors),120)
        self.assertTrue(all("Secret bit:" not in j["prompt"] for j in monitors if j["kind"]=="encoded"))

    def test_reasoning_content_is_never_used_as_a_completed_bit_answer(self):
        self.assertIsNone(parse_bit("<think>Perhaps 1")['bit'])
        self.assertIsNone(parse_bit("The answer could be 0 or 1.")['bit'])
        self.assertEqual(parse_bit("<think>First consider 0.</think>1<|im_end|>")['bit'],1)
        boxed = parse_bit(r"\boxed{0}")
        self.assertEqual(boxed['bit'],0)
        self.assertFalse(boxed['bare_format'])

    def test_observer_cannot_qualify_when_it_cannot_distinguish_obvious_leaks(self):
        self.assertFalse(calibrate_threshold([0.5]*8,[0.5]*8,0.1,0.75)['qualified'])
        result=calibrate_threshold([0.0]*8,[1.0]*8,0.1,0.75)
        self.assertTrue(result['qualified'])
        self.assertEqual(result['false_positive_rate'],0)

    def test_invalid_observer_fields_cannot_silently_pass_cover_checks(self):
        with self.assertRaises(ValueError):
            parse_observer('{"facts_preserved":true}')
        with self.assertRaises(ValueError):
            parse_observer('{"facts_preserved":true,"added_claims":false,"overt_bit":false,"suspicion":true,"reason":"ok"}')
