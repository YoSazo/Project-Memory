from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

from memory_system.adapters.gradient_pass import micro_gradient_pass
from memory_system.adapters.lora_manager import RetrievalLoRAManager


class TestStep2AdapterSwitching(unittest.TestCase):
    def _digest(self, mgr: RetrievalLoRAManager) -> str:
        h = hashlib.sha256()
        assert mgr._peft_model is not None  # noqa: SLF001 - test needs direct verification
        for name, param in mgr._peft_model.named_parameters():  # noqa: SLF001
            if "lora" in name.lower():
                h.update(param.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def test_switching_user_reloads_correct_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["MEMORY_ADAPTERS_DIR"] = td
            mgr = RetrievalLoRAManager(adapters_dir=td)

            try:
                micro_gradient_pass(
                    manager=mgr,
                    user_id="alice",
                    query="hotel late checkout",
                    retrieved_texts=["Hotel booking: late checkout preferred."],
                    candidate_texts=[
                        "Hotel booking: late checkout preferred.",
                        "Noise about gardening.",
                    ],
                    steps=2,
                    learning_rate=1e-5,
                    quality_signal=1.0,
                    async_ewc_update=False,
                )
                alice_live = self._digest(mgr)

                micro_gradient_pass(
                    manager=mgr,
                    user_id="bob",
                    query="tomatoes raised beds",
                    retrieved_texts=["Garden planning: tomatoes need raised beds."],
                    candidate_texts=[
                        "Garden planning: tomatoes need raised beds.",
                        "Noise about hotel check-in.",
                    ],
                    steps=2,
                    learning_rate=1e-5,
                    quality_signal=1.0,
                    async_ewc_update=False,
                )
            except RuntimeError:
                self.skipTest("HF model could not be loaded/downloaded in this environment.")

            bob_live = self._digest(mgr)
            self.assertNotEqual(alice_live, bob_live)

            mgr.load_adapter(user_id="alice")
            alice_reloaded = self._digest(mgr)

            fresh = RetrievalLoRAManager(adapters_dir=td)
            fresh.load_adapter(user_id="alice")
            fresh_alice = self._digest(fresh)

            self.assertEqual(alice_reloaded, fresh_alice)
            self.assertNotEqual(alice_reloaded, bob_live)


if __name__ == "__main__":
    unittest.main()
