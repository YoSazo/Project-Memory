from __future__ import annotations

import os
import tempfile
import unittest

from memory_system.memory.chunk_manager import ChunkManager
from memory_system.memory.episode_log import EpisodeLog
from memory_system.middleware.ttt_layer import TTTLayer


class TestStep1AssistantChunkIngestion(unittest.TestCase):
    def test_assistant_chunks_can_be_persisted_for_benchmark_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "memory.sqlite")
            log = EpisodeLog(db_path)
            cm = ChunkManager(log)
            ttt = TTTLayer(
                episode_log=log,
                chunk_manager=cm,
                async_training=False,
                extract_assistant_chunks=True,
            )

            ttt.on_assistant_message(
                session_id="sess_bench",
                user_id="u_bench",
                assistant_text="The hotel is near Alexanderplatz in Berlin and checkout is at 11am.",
            )

            chunks = log.fetch_recent_chunks(user_id="u_bench", limit=20)
            texts = " ".join(c.text for c in chunks)
            log.close()

            self.assertIn("[assistant]", texts)
            self.assertIn("Alexanderplatz", texts)


if __name__ == "__main__":
    unittest.main()
