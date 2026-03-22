from __future__ import annotations

import os
import tempfile
import unittest

from memory_system.adapters.ewc import EWC
from memory_system.adapters.gradient_pass import micro_gradient_pass
from memory_system.adapters.lora_manager import RetrievalLoRAManager


class TestStep3LiveEWCPersistence(unittest.TestCase):
    def test_micro_gradient_pass_persists_fisher(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["MEMORY_ADAPTERS_DIR"] = td
            mgr = RetrievalLoRAManager(adapters_dir=td)

            try:
                micro_gradient_pass(
                    manager=mgr,
                    user_id="u_live_ewc",
                    query="booking hotel late checkout",
                    retrieved_texts=["Hotel booking: quiet room and late checkout."],
                    candidate_texts=[
                        "Hotel booking: quiet room and late checkout.",
                        "Noise about GPUs.",
                        "Noise about gardening.",
                    ],
                    steps=2,
                    learning_rate=1e-5,
                    quality_signal=1.0,
                    async_ewc_update=False,
                )
            except RuntimeError:
                self.skipTest("HF model could not be loaded/downloaded in this environment.")

            ewc = EWC(user_id="u_live_ewc", adapters_dir=td)
            self.assertTrue((ewc.user_path / "param_snapshot.pt").exists())
            self.assertTrue((ewc.user_path / "fisher_matrix.pt").exists())
            self.assertTrue(ewc.snapshot)
            self.assertTrue(ewc.fisher)


if __name__ == "__main__":
    unittest.main()
