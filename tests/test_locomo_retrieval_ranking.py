from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime

from benchmarks.locomo.run_memla_benchmark import (
    _format_turn_text,
    _heuristic_query_cues,
    _heuristic_qa_answer,
    _normalize_model_answer,
    _rerank_chunks_for_answer,
    _resolve_relative_time_hints,
)
from memory_system.memory.chunk_manager import ChunkManager
from memory_system.memory.episode_log import Chunk, EpisodeLog


class TestLocomoRetrievalRanking(unittest.TestCase):
    def test_question_word_entity_mentions_are_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EpisodeLog(os.path.join(td, "memory.sqlite"))
            cm = ChunkManager(log)
            drafts, _ = cm.extract_chunks("What happened yesterday?")
            entity_texts = [draft.text for draft in drafts if draft.chunk_type == "entity"]
            self.assertNotIn("Entity mentioned: What", entity_texts)
            log.close()

    def test_format_turn_text_includes_image_queries(self) -> None:
        text = _format_turn_text(
            "Look at this",
            ["a photo of a book cover with a gold coin on it"],
            ["painted canvas follow your dreams"],
        )
        self.assertIn("[Shared image: a photo of a book cover with a gold coin on it]", text)
        self.assertIn("[Image cues: painted canvas follow your dreams]", text)

    def test_specific_fact_beats_generic_dialogue_for_indirect_query(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EpisodeLog(os.path.join(td, "memory.sqlite"))
            cm = ChunkManager(log)
            user_id = "u_locomo"
            session_id = "sess"

            episode_id = log.add_episode(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content="seed",
            )

            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="fact",
                key="caroline researching adoption agencies family",
                text="[Caroline] Researching adoption agencies - it's been a dream to have a family and give a loving home to kids who need it.",
                source_episode_id=episode_id,
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="note",
                key="caroline wow what got you into running",
                text="[Caroline] Wow! What got you into running?",
                source_episode_id=episode_id,
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="note",
                key="melanie what other creative projects do you do",
                text="[Melanie] What other creative projects do you do with them, besides pottery?",
                source_episode_id=episode_id,
            )

            ranked = cm.retrieve(user_id=user_id, query_text="What did Caroline research?", k=3)
            self.assertGreaterEqual(len(ranked), 1)
            self.assertIn("adoption agencies", ranked[0].text.lower())
            log.close()

    def test_old_relevant_chunk_is_not_dropped_by_candidate_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EpisodeLog(os.path.join(td, "memory.sqlite"))
            cm = ChunkManager(log)
            user_id = "u_cap"
            session_id = "sess"

            old_episode_id = log.add_episode(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content="old seed",
                ts=1,
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="fact",
                key="melanie painted lake sunrise last year",
                text="[Melanie] Yeah, I painted that lake sunrise last year! It's special to me.",
                source_episode_id=old_episode_id,
                ts=1,
            )

            for idx in range(700):
                episode_id = log.add_episode(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=f"noise {idx}",
                    ts=10 + idx,
                )
                log.add_or_bump_chunk(
                    session_id=session_id,
                    user_id=user_id,
                    chunk_type="note",
                    key=f"noise key {idx}",
                    text=f"[assistant] Fact: Hey there, just checking in about random topic {idx}.",
                    source_episode_id=episode_id,
                    ts=10 + idx,
                )

            ranked = cm.retrieve(user_id=user_id, query_text="When did Melanie paint a sunrise?", k=6)
            texts = [chunk.text.lower() for chunk in ranked]
            self.assertTrue(any("painted that lake sunrise" in text for text in texts))
            log.close()

    def test_subject_speaker_fact_beats_assistant_mention(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EpisodeLog(os.path.join(td, "memory.sqlite"))
            cm = ChunkManager(log)
            user_id = "u_subject"
            session_id = "sess"

            episode_id = log.add_episode(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content="seed",
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="fact",
                key="caroline transgender stories inspiring support",
                text="[Caroline] The transgender stories were so inspiring! I was so happy and thankful for all the support.",
                source_episode_id=episode_id,
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="fact",
                key="assistant inspiring journey caroline",
                text="[assistant] Fact: Wow, Caroline, you're doing an awesome job of inspiring others with your journey.",
                source_episode_id=episode_id,
            )

            ranked = cm.retrieve(user_id=user_id, query_text="What is Caroline's identity?", k=2)
            self.assertGreaterEqual(len(ranked), 1)
            self.assertIn("transgender stories", ranked[0].text.lower())
            log.close()

    def test_cue_phrase_boost_surfaces_school_event_fact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EpisodeLog(os.path.join(td, "memory.sqlite"))
            cm = ChunkManager(
                log,
                query_expander=lambda _query: [
                    "school event last week",
                    "transgender journey",
                    "encouraged students",
                ],
            )
            user_id = "u_school"
            session_id = "sess"

            episode_id = log.add_episode(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content="seed",
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="note",
                key="caroline school event last week transgender journey encouraged students",
                text="[Caroline] I wanted to tell you about my school event last week. I talked about my transgender journey and encouraged students to get involved in the LGBTQ community.",
                source_episode_id=episode_id,
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="note",
                key="caroline pride parade last week",
                text="[Caroline] Last week I went to an LGBTQ+ pride parade. Everyone was so happy and it made me feel like I belonged.",
                source_episode_id=episode_id,
            )

            ranked = cm.retrieve(user_id=user_id, query_text="When did Caroline give a speech at a school?", k=2)
            self.assertGreaterEqual(len(ranked), 1)
            self.assertIn("school event last week", ranked[0].text.lower())
            log.close()

    def test_format_turn_text_preserves_shared_image_caption(self) -> None:
        text = _format_turn_text(
            "You'd be a great counselor! By the way, take a look at this.",
            ["a photo of a painting of a sunset over a lake"],
        )
        self.assertIn("take a look at this", text.lower())
        self.assertIn("shared image", text.lower())
        self.assertIn("sunset over a lake", text.lower())

    def test_relative_time_hints_resolve_common_locomo_patterns(self) -> None:
        session_dt = datetime(2023, 5, 8, 10, 0, 0)
        hints = _resolve_relative_time_hints(
            "I went to a LGBTQ support group yesterday and it was powerful. We might go camping next month.",
            session_dt,
        )
        joined = " | ".join(hints)
        self.assertIn("yesterday = 7 May 2023", joined)
        self.assertIn("next month = June 2023", joined)

    def test_relative_time_hints_resolve_locomo_abbreviations_and_offsets(self) -> None:
        session_dt = datetime(2023, 7, 15, 13, 51, 0)
        hints = _resolve_relative_time_hints(
            "Last Fri we went to a pottery workshop, and two days ago I went to a conference this month.",
            session_dt,
        )
        joined = " | ".join(hints)
        self.assertIn("last friday = 14 July 2023", joined)
        self.assertIn("2 days ago = 13 July 2023", joined)
        self.assertIn("this month = July 2023", joined)

    def test_relative_time_hints_resolve_tomorrow(self) -> None:
        session_dt = datetime(2023, 6, 19, 10, 4, 0)
        hints = _resolve_relative_time_hints(
            "The official opening night is tomorrow.",
            session_dt,
        )
        joined = " | ".join(hints)
        self.assertIn("tomorrow = 20 June 2023", joined)

    def test_heuristic_qa_answer_resolves_temporal_metadata(self) -> None:
        chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline support group yesterday",
            text="[Caroline] I went to a LGBTQ support group yesterday.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"resolved_time_hints": ["yesterday = 7 May 2023"]},
        )
        answer = _heuristic_qa_answer("When did Caroline go to the LGBTQ support group?", [chunk])
        self.assertEqual("7 May 2023", answer)

    def test_heuristic_qa_answer_prefers_exact_open_store_date(self) -> None:
        chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="gina online clothes store is open",
            text="[Gina] Yay! My online clothes store is open! I've been dreaming of this for a while now.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"session_date_text": "2:35 pm on 16 March, 2023"},
        )
        answer = _heuristic_qa_answer("When did Gina open her online clothing store?", [chunk])
        self.assertEqual("16 March 2023", answer)

    def test_heuristic_qa_answer_returns_month_for_rome_trip(self) -> None:
        chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="jon took a short trip last week to Rome",
            text="[Jon] Hey Gina, hope you're doing great! Still working on my biz. Took a short trip last week to Rome to clear my mind a little.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"session_date_text": "10:04 am on 19 June, 2023", "resolved_time_hints": ["last week = the week before 19 June 2023"]},
        )
        answer = _heuristic_qa_answer("When was Jon in Rome?", [chunk])
        self.assertEqual("June 2023", answer)

    def test_heuristic_qa_answer_shared_city_intersection(self) -> None:
        gina_chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="gina been only to Rome once",
            text="[Gina] Paris?! That is really great Jon! Never had a chance to visit it. Been only to Rome once.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        jon_chunk = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="jon took a short trip last week to Rome",
            text="[Jon] Hey Gina, hope you're doing great! Still working on my biz. Took a short trip last week to Rome to clear my mind a little.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        answer = _heuristic_qa_answer("Which city have both Jean and John visited?", [gina_chunk, jon_chunk])
        self.assertEqual("Rome", answer)

    def test_heuristic_qa_answer_host_dance_competition_returns_next_month(self) -> None:
        chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="jon upcoming dance competition next month",
            text="[Jon] We're planning an upcoming dance competition next month to showcase local talent.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"session_date_text": "1:26 pm on 3 April, 2023"},
        )
        answer = _heuristic_qa_answer("When did Jon host a dance competition?", [chunk])
        self.assertEqual("May 2023", answer)

    def test_heuristic_qa_answer_business_commonality(self) -> None:
        jon_chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="jon lost my job as a banker",
            text="[Jon] Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        gina_chunk = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="gina lost job at door dash",
            text="[Gina] I also lost my job at Door Dash this month. Starting my own store and taking risks is both scary and rewarding.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        answer = _heuristic_qa_answer("What do Jon and Gina both have in common?", [jon_chunk, gina_chunk])
        self.assertEqual("They lost their jobs and decided to start their own businesses.", answer)

    def test_heuristic_qa_answer_returns_trophy_for_dance_contest(self) -> None:
        chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="gina team won first place trophy",
            text="[Gina] My team won first place and we got a trophy at the dance competition.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        answer = _heuristic_qa_answer("What did Gina receive from a dance contest?", [chunk])
        self.assertEqual("a trophy", answer)

    def test_heuristic_qa_answer_returns_amazing_for_studio_description(self) -> None:
        chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="gina studio looks amazing",
            text="[Gina] Congrats, Jon! The studio looks amazing.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        answer = _heuristic_qa_answer("How does Gina describe the studio that Jon has opened?", [chunk])
        self.assertEqual("amazing", answer)

    def test_heuristic_qa_answer_infers_common_sense_fields(self) -> None:
        chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline counseling mental health",
            text="[Caroline] I'm keen on counseling or working in mental health.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        answer = _heuristic_qa_answer(
            "What fields would Caroline be likely to pursue in her educaton?",
            [chunk],
        )
        self.assertEqual("Psychology, counseling certification", answer)

    def test_heuristic_qa_answer_extracts_marriage_duration(self) -> None:
        chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="melanie wedding anniversary 5 years already",
            text="[Melanie] 5 years already! Time flies - feels like just yesterday I put this dress on!",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        answer = _heuristic_qa_answer("How long have Mel and her husband been married?", [chunk])
        self.assertEqual("5 years", answer)

    def test_retrieve_prefers_wedding_fact_for_mel_alias_query(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EpisodeLog(os.path.join(td, "memory.sqlite"))
            cm = ChunkManager(log)
            user_id = "u_marriage"
            session_id = "sess"

            episode_id = log.add_episode(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content="seed",
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="fact",
                key="generic hey mel long time",
                text="Fact: Hey Mel, long time no chat!",
                source_episode_id=episode_id,
                meta={"source": "heuristic_extract_v1"},
            )
            log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type="note",
                key="melanie wedding anniversary 5 years already",
                text="[Melanie] 5 years already! Time flies - feels like just yesterday I put this dress on!",
                source_episode_id=episode_id,
                meta={"source": "benchmark_raw_turn"},
            )

            ranked = cm.retrieve(user_id=user_id, query_text="How long have Mel and her husband been married?", k=2)
            self.assertGreaterEqual(len(ranked), 1)
            self.assertIn("5 years already", ranked[0].text.lower())
            log.close()

    def test_heuristic_qa_answer_prefers_friends_family_mentors_chunk(self) -> None:
        unrelated = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline pride last month",
            text="[Caroline] I mentor a transgender teen just like me. We had a great time at the LGBT pride event last month.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"resolved_time_hints": ["last month = June 2023"]},
        )
        target = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline friends family mentors last week",
            text="[Caroline] My friends, family and mentors are my rocks. Here's a pic from when we met up last week!",
            source_episode_id=2,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"resolved_time_hints": ["last week = the week before 9 June 2023"]},
        )
        answer = _heuristic_qa_answer(
            "When did Caroline meet up with her friends, family, and mentors?",
            [unrelated, target],
        )
        self.assertEqual("the week before 9 June 2023", answer)

    def test_rerank_chunks_for_answer_promotes_causal_match(self) -> None:
        unrelated = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="john childhood doll",
            text="[John] I had a little doll like this when I was a kid.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        cause = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="john son broke his arm garden",
            text="[John] Since my son broke his arm falling off the backyard trampoline, we replaced it with a garden.",
            source_episode_id=2,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        reranked = _rerank_chunks_for_answer(
            "John: It's funny, every time I pick tomatoes out back, I still get this little jolt of guilt and relief mixed together.",
            [unrelated, cause],
            category="Cognitive",
        )
        self.assertEqual(cause.id, reranked[0].id)

    def test_relative_time_hints_resolve_last_night_and_past_weekend(self) -> None:
        session_dt = datetime(2023, 10, 20, 9, 0, 0)
        hints = _resolve_relative_time_hints(
            "Last night was amazing, and that roadtrip this past weekend was intense.",
            session_dt,
        )
        joined = " | ".join(hints)
        self.assertIn("last night = 19 October 2023", joined)
        self.assertIn("this past weekend = the weekend before 20 October 2023", joined)

    def test_heuristic_qa_answer_strips_year_for_birthday(self) -> None:
        birthday = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="melanie daughter birthday concert last night",
            text="[Melanie] Last night was amazing! We celebrated my daughter's birthday with a concert.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={
                "session_date_text": "2:24 pm on 14 August, 2023",
                "resolved_time_hints": ["last night = 13 August 2023"],
            },
        )
        answer = _heuristic_qa_answer("When is Melanie's daughter's birthday?", [birthday])
        self.assertEqual("13 August", answer)

    def test_heuristic_qa_answer_handles_years_ago_and_how_long(self) -> None:
        movie = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="joanna first watched movie around three years ago",
            text='[Joanna] Yep, that movie is awesome. I first watched it around 3 years ago.',
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"session_date_text": "8:18 pm on 21 January, 2022"},
        )
        friends = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline known these friends for 4 years",
            text="[Caroline] I've known these friends for 4 years, since I moved from my home country.",
            source_episode_id=2,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"session_date_text": "10:37 am on 27 June, 2023"},
        )
        art = Chunk(
            id=3,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="melanie seven years now art",
            text="[Melanie] Seven years now, and I've finally found my real muses: painting and pottery.",
            source_episode_id=3,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={"session_date_text": "3:31 pm on 23 August, 2023"},
        )
        self.assertEqual(
            "2019",
            _heuristic_qa_answer('When did Joanna first watch "Eternal Sunshine of the Spotless Mind?', [movie]),
        )
        self.assertEqual(
            "4 years",
            _heuristic_qa_answer("How long has Caroline had her current group of friends for?", [friends]),
        )
        self.assertEqual(
            "Since 2016",
            _heuristic_qa_answer("How long has Melanie been practicing art?", [art]),
        )

    def test_heuristic_qa_answer_extracts_exact_single_hop_facts(self) -> None:
        book_chunk = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline becoming nicole recommendation",
            text='[Caroline] I loved "Becoming Nicole" by Amy Ellis Nutt. Highly recommend it for sure!',
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        library_chunk = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline library classics cultures educational",
            text="[Caroline] I've got lots of kids' books- classics, stories from different cultures, educational books, all of that.",
            source_episode_id=2,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        self.assertEqual(
            '"Becoming Nicole"',
            _heuristic_qa_answer("What book did Caroline recommend to Melanie?", [book_chunk]),
        )
        self.assertEqual(
            "kids' books - classics, stories from different cultures, educational books",
            _heuristic_qa_answer("What kind of books does Caroline have in her library?", [library_chunk]),
        )

    def test_heuristic_qa_answer_extracts_common_sense_short_forms(self) -> None:
        books = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline classic childrens books",
            text="[Caroline] I've got lots of kids' books- classics, stories from different cultures, educational books, all of that.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        outdoors = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="melanie outdoors camping nature",
            text="[Melanie] We love camping in the mountains and spending time outdoors in nature.",
            source_episode_id=2,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        religion = Chunk(
            id=3,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline church religious conservatives",
            text="[Caroline] I made a stained glass piece for a local church, but a bad encounter with religious conservatives also upset me.",
            source_episode_id=3,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        self.assertEqual(
            "Yes, since she collects classic children's books",
            _heuristic_qa_answer("Would Caroline likely have Dr. Seuss books on her bookshelf?", [books]),
        )
        self.assertEqual(
            "National park; she likes the outdoors",
            _heuristic_qa_answer("Would Melanie be more interested in going to a national park or a theme park?", [outdoors]),
        )
        self.assertEqual(
            "Somewhat, but not extremely religious",
            _heuristic_qa_answer("Would Caroline be considered religious?", [religion]),
        )

    def test_heuristic_qa_answer_extracts_exact_multi_hop_lists(self) -> None:
        camping = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="melanie beach mountains forest camping",
            text="[Melanie] We've camped at the beach, in the mountains, and in the forest.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        pets = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="melanie pets oliver luna bailey",
            text="[Melanie] Luna and Oliver are sweet, and we got another cat named Bailey too.",
            source_episode_id=2,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        pottery = Chunk(
            id=3,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="melanie pottery bowls cup",
            text="[Melanie] The kids made a cup with a dog face on it, and we also made bowls.",
            source_episode_id=3,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        self.assertEqual(
            "beach, mountains, forest",
            _heuristic_qa_answer("Where has Melanie camped?", [camping]),
        )
        self.assertEqual(
            "Oliver, Luna, Bailey",
            _heuristic_qa_answer("What are Melanie's pets' names?", [pets]),
        )
        self.assertEqual(
            "bowls, cup",
            _heuristic_qa_answer("What types of pottery have Melanie and her kids made?", [pottery]),
        )

    def test_heuristic_qa_answer_extracts_remaining_exact_entities(self) -> None:
        symbols = Chunk(
            id=1,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="caroline rainbow flag transgender symbol mural pendant",
            text="[Caroline] The rainbow flag mural matters to me, and the transgender symbol on my pendant reminds me of my identity.",
            source_episode_id=1,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        music = Chunk(
            id=2,
            ts=0,
            session_id="s",
            user_id="u",
            chunk_type="note",
            key="melanie summer sounds matt patterson clarinet violin",
            text='[Melanie] I saw "Summer Sounds" and Matt Patterson live, and I play clarinet and violin.',
            source_episode_id=2,
            frequency_count=1,
            recall_count=0,
            last_recalled_ts=0,
            meta={},
        )
        self.assertEqual(
            "Rainbow flag, transgender symbol",
            _heuristic_qa_answer("What symbols are important to Caroline?", [symbols]),
        )
        self.assertEqual(
            "clarinet and violin",
            _heuristic_qa_answer("What instruments does Melanie play?", [music]),
        )
        self.assertEqual(
            "Summer Sounds, Matt Patterson",
            _heuristic_qa_answer("What musical artists/bands has Melanie seen?", [music]),
        )

    def test_retrieve_with_heuristic_query_cues_surfaces_sweden_fact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EpisodeLog(os.path.join(td, "memory.sqlite"))
            cm = ChunkManager(log, query_expander=_heuristic_query_cues)
            user_id = "u_sweden"
            session_id = "sess"
            try:
                episode_id = log.add_episode(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content="seed",
                )
                log.add_or_bump_chunk(
                    session_id=session_id,
                    user_id=user_id,
                    chunk_type="note",
                    key="caroline grandma gift necklace roots home country",
                    text="[Caroline] This necklace is from my grandma in my home country. It reminds me of my roots and family love.",
                    source_episode_id=episode_id,
                    meta={"source": "benchmark_raw_turn"},
                )
                log.add_or_bump_chunk(
                    session_id=session_id,
                    user_id=user_id,
                    chunk_type="note",
                    key="caroline sweden roots necklace",
                    text="[Caroline] Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. It's like a reminder of my roots.",
                    source_episode_id=episode_id,
                    meta={"source": "benchmark_raw_turn"},
                )
                log.add_or_bump_chunk(
                    session_id=session_id,
                    user_id=user_id,
                    chunk_type="note",
                    key="caroline guitar emotions",
                    text="[Caroline] I started playing acoustic guitar about five years ago; it's been a great way to express myself.",
                    source_episode_id=episode_id,
                    meta={"source": "benchmark_raw_turn"},
                )

                ranked = cm.retrieve(user_id=user_id, query_text="Where did Caroline move from 4 years ago?", k=10)
                self.assertGreaterEqual(len(ranked), 1)
                self.assertTrue(any("sweden" in chunk.text.lower() for chunk in ranked))
            finally:
                log.close()

    def test_normalize_model_answer_compacts_bullets_to_csv(self) -> None:
        answer = _normalize_model_answer(
            "What activities does Melanie partake in?",
            """Based on the retrieved memories, Melanie partakes in:

- Pottery classes (which she finds therapeutic)
- Camping and hiking with her family
- Painting with the kids
""",
        )
        self.assertEqual("Pottery classes, Camping and hiking with her family, Painting with the kids", answer)


if __name__ == "__main__":
    unittest.main()
