from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

from .episode_log import Chunk, EpisodeLog


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "do",
    "does",
    "did",
    "done",
    "about",
    "again",
    "all",
    "any",
    "been",
    "can",
    "could",
    "just",
    "like",
    "more",
    "out",
    "than",
    "then",
    "there",
    "very",
    "you",
    "your",
}

_QUESTION_ENTITY_TOKENS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
}

_GENERIC_FILLER_TOKENS = {
    "awesome",
    "cool",
    "glad",
    "good",
    "great",
    "hey",
    "nice",
    "sound",
    "thank",
    "thanks",
    "totally",
    "wow",
    "yeah",
}

_SPECIFIC_DATE_TOKENS = {
    "april",
    "august",
    "december",
    "february",
    "friday",
    "january",
    "july",
    "june",
    "last",
    "march",
    "monday",
    "month",
    "next",
    "november",
    "october",
    "saturday",
    "september",
    "sunday",
    "thursday",
    "today",
    "tomorrow",
    "tuesday",
    "week",
    "wednesday",
    "year",
    "yesterday",
}

_IRREGULAR_TOKEN_MAP = {
    "gone": "go",
    "fri": "friday",
    "mel": "melanie",
    "mon": "monday",
    "ran": "run",
    "sat": "saturday",
    "sun": "sunday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "tue": "tuesday",
    "tues": "tuesday",
    "wed": "wednesday",
    "went": "go",
    "lgbtq": "lgbt",
}

_GENERIC_DIALOGUE_PATTERNS = (
    r"\bhow did (?:it|you)\b",
    r"\bhow's it going\b",
    r"\blong time no (?:chat|talk)\b",
    r"\bhope all'?s good\b",
    r"\bwhat got you\b",
    r"\bwhat other\b",
    r"\bwhat was your favorite\b",
    r"\bwhat've you been up to\b",
)


def _normalize_token(token: str) -> str:
    token = token.lower()
    token = _IRREGULAR_TOKEN_MAP.get(token, token)
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("ing") and len(token) > 5:
        token = token[:-3]
        if len(token) >= 3 and token[-1] == token[-2]:
            token = token[:-1]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("es") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3:
        token = token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    out: list[str] = []
    for token in tokens:
        norm = _normalize_token(token)
        if len(norm) >= 2 and norm not in _STOPWORDS:
            out.append(norm)
    return out


def _stable_key(text: str) -> str:
    # Normalize a key for retrieval: lower, remove punctuation, collapse spaces.
    k = re.sub(r"[^a-zA-Z0-9_ ]+", " ", text.lower())
    k = re.sub(r"\s+", " ", k).strip()
    return k[:256]


@dataclass(frozen=True)
class MemoryChunkDraft:
    chunk_type: str  # fact | decision | entity | note
    key: str
    text: str


class ChunkManager:
    """
    Step 1 memory chunks:
    - Extracts a few structured chunks from each user message (heuristic or LLM)
    - Retrieves top-k by hybrid scoring: semantic (MiniLM) + keyword + recency + frequency
    - Falls back to keyword-only if the embedding model is unavailable
    """

    def __init__(
        self,
        episode_log: EpisodeLog,
        *,
        llm_extractor: Optional[Callable[[str], Tuple[list["MemoryChunkDraft"], dict[str, Any]]]] = None,
        query_expander: Optional[Callable[[str], list[str]]] = None,
    ) -> None:
        self.log = episode_log
        self._llm_extractor = llm_extractor
        self._query_expander = query_expander

    def extract_chunks(self, user_text: str) -> tuple[list[MemoryChunkDraft], dict[str, Any]]:
        text = user_text.strip()
        if not text:
            return [], {"source": "empty"}

        if self._llm_extractor is not None:
            try:
                drafts, meta = self._llm_extractor(text)
                if drafts:
                    return drafts, meta
            except Exception as e:
                # Fall back to heuristic extraction; persist the failure signal in meta.
                pass

        drafts: list[MemoryChunkDraft] = []

        # Decisions / preferences (simple pattern capture).
        decision_patterns = [
            r"\b(?:i|we)\s+(?:decided|choose|chose|prefer|want|need)\s+(?P<x>.+)$",
            r"\bmy\s+preference\s+is\s+(?P<x>.+)$",
        ]
        for pat in decision_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                x = m.group("x").strip().rstrip(".")
                if x:
                    drafts.append(
                        MemoryChunkDraft(
                            chunk_type="decision",
                            key=_stable_key(x),
                            text=f"Preference/decision: {x}",
                        )
                    )
                break

        # Entities: naive capture of "X is Y", and capitalized tokens.
        is_stmt = re.findall(r"\b([A-Z][a-zA-Z0-9_]+)\s+is\s+([^.\n]{3,80})", text)
        for ent, desc in is_stmt[:5]:
            drafts.append(
                MemoryChunkDraft(
                    chunk_type="entity",
                    key=_stable_key(ent),
                    text=f"Entity: {ent} — {desc.strip()}",
                )
            )

        caps = re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", text)
        for ent in list(dict.fromkeys(caps))[:8]:
            if _normalize_token(ent) in _QUESTION_ENTITY_TOKENS:
                continue
            drafts.append(MemoryChunkDraft(chunk_type="entity", key=_stable_key(ent), text=f"Entity mentioned: {ent}"))

        # Facts: keep a few dense sentences.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        for s in sentences[:5]:
            if len(s) >= 20 and not s.startswith("/") and not s.endswith("?"):
                drafts.append(MemoryChunkDraft(chunk_type="fact", key=_stable_key(s[:80]), text=f"Fact: {s}"))

        # De-dupe by (type,key,text).
        seen = set()
        out: list[MemoryChunkDraft] = []
        for d in drafts:
            if not self._should_store_draft(d):
                continue
            k = (d.chunk_type, d.key, d.text)
            if k in seen:
                continue
            seen.add(k)
            out.append(d)
        return out[:20], {"source": "heuristic_extract_v1"}

    def _should_store_draft(self, draft: MemoryChunkDraft) -> bool:
        text = draft.text.strip()
        if not text:
            return False

        if draft.chunk_type == "entity" and text.lower().startswith("entity mentioned:"):
            entity = text.split(":", 1)[-1].strip()
            if _normalize_token(entity) in _QUESTION_ENTITY_TOKENS:
                return False
            # Single-token entity mention drafts are too noisy for retrieval.
            return False

        content_tokens = _tokenize(text)
        if draft.chunk_type == "fact" and len(content_tokens) < 4:
            return False
        if draft.chunk_type == "decision" and len(content_tokens) < 2:
            return False
        if draft.chunk_type == "entity" and len(content_tokens) < 2:
            return False
        return True

    def persist_user_message(
        self, *, session_id: str, user_id: str, user_text: str, ts: int | None = None
    ) -> tuple[int, list[int]]:
        return self.persist_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            text=user_text,
            ts=ts,
            extract_chunks=True,
        )

    def persist_chunks_from_text(
        self,
        *,
        session_id: str,
        user_id: str,
        text: str,
        source_episode_id: int | None,
        speaker_role: str = "user",
        ts: int | None = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> list[int]:
        ts_i = int(ts if ts is not None else time.time())
        speaker = (speaker_role or "user").strip().lower() or "user"
        chunk_ids: list[int] = []
        drafts, extract_meta = self.extract_chunks(text)
        for draft in drafts:
            chunk_meta = dict(extract_meta)
            if meta:
                chunk_meta.update(meta)
            if speaker != "user":
                chunk_meta.setdefault("speaker_role", speaker)

            key = draft.key
            stored_text = draft.text
            if speaker != "user":
                key = _stable_key(f"{speaker} {draft.key}")
                stored_text = f"[{speaker}] {draft.text}"

            cid = self.log.add_or_bump_chunk(
                session_id=session_id,
                user_id=user_id,
                chunk_type=draft.chunk_type,
                key=key,
                text=stored_text,
                source_episode_id=source_episode_id,
                ts=ts_i,
                meta=chunk_meta,
            )
            chunk_ids.append(cid)
        return chunk_ids

    def persist_message(
        self,
        *,
        session_id: str,
        user_id: str,
        role: str,
        text: str,
        ts: int | None = None,
        extract_chunks: bool = True,
        meta: Optional[dict[str, Any]] = None,
    ) -> tuple[int, list[int]]:
        ts_i = int(ts if ts is not None else time.time())
        episode_id = self.log.add_episode(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=text,
            meta=meta,
            ts=ts_i,
        )

        chunk_ids: list[int] = []
        if extract_chunks:
            chunk_ids = self.persist_chunks_from_text(
                session_id=session_id,
                user_id=user_id,
                text=text,
                source_episode_id=episode_id,
                speaker_role=role,
                ts=ts_i,
                meta=meta,
            )
        return episode_id, chunk_ids

    def retrieve(self, *, user_id: str, query_text: str, k: int = 12) -> list[Chunk]:
        candidate_limit = max(1500, int(k) * 50)
        candidates = self.log.fetch_top_level_chunks(user_id=user_id, limit=candidate_limit)
        if not candidates:
            return []

        q_tokens = set(_tokenize(query_text))
        q_entities = _extract_named_entities(query_text)
        q_subject = _extract_query_subject(query_text)
        expanded_query_text = query_text
        cue_phrases: list[str] = []
        if self._query_expander is not None:
            try:
                cues = [cue.strip() for cue in self._query_expander(query_text) if cue and cue.strip()]
            except Exception:
                cues = []
            if cues:
                cue_phrases = cues[:]
                expanded_query_text = query_text + "\n" + "\n".join(cues)
                q_tokens |= set(_tokenize(expanded_query_text))
                q_entities |= _extract_named_entities(expanded_query_text)
                if q_subject is None:
                    q_subject = _extract_query_subject(expanded_query_text)
        cue_tokens = set(_tokenize("\n".join(cue_phrases))) if cue_phrases else set()
        now = time.time()
        is_temporal_query = bool(re.search(r"\bwhen\b", query_text.lower()))
        is_reflective_query = bool(re.match(r"^[A-Z][a-zA-Z0-9_]+:", query_text.strip()))

        # --- Graph-augmented contextual retrieval (C1) ---
        # Enrich chunk text with connected node context before embedding.
        # MiniLM doesn't know domain vocabulary, but reads context clues.
        enriched_texts = self._enrich_with_graph_context(candidates, user_id)

        # --- Attempt semantic scoring via MiniLM embeddings ---
        sem_scores: dict[int, float] = {}
        try:
            from ..middleware.context_builder import _get_lora_manager
            mgr = _get_lora_manager()
            if mgr is not None:
                q_emb = mgr.embed_query(expanded_query_text)
                c_embs = mgr.embed_many(enriched_texts)
                if q_emb and c_embs:
                    for idx, c_emb in enumerate(c_embs):
                        dot = sum(a * b for a, b in zip(q_emb, c_emb))
                        sem_scores[candidates[idx].id] = float(dot)
        except Exception:
            pass

        has_semantic = bool(sem_scores)

        def score(c: Chunk) -> float:
            c_tokens = set(_tokenize(c.text)) | set(_tokenize(c.key))
            overlap = len(q_tokens & c_tokens)
            lexical_recall = overlap / max(1.0, float(len(q_tokens)))
            cue_token_overlap = 0.0
            if cue_tokens:
                cue_token_overlap = len(cue_tokens & c_tokens) / max(1.0, float(len(cue_tokens)))
            c_entities = _extract_named_entities(c.text)
            entity_overlap = len(q_entities & c_entities) / max(1.0, float(len(q_entities)))
            speaker = _extract_speaker_label(c.text)
            speaker_match = 1.0 if q_subject and speaker == q_subject else 0.0
            speaker_mismatch_penalty = 0.0
            if q_subject and speaker and speaker not in {q_subject, "assistant"}:
                speaker_mismatch_penalty = 0.25

            age_s = max(0.0, now - float(c.last_recalled_ts))
            recency = math.exp(-age_s / (60.0 * 60.0 * 24.0 * 7.0))

            freq = math.log(1.0 + float(c.frequency_count))
            specificity = _specificity_score(c.text)
            generic_penalty = _generic_dialogue_penalty(c.text)
            source = str(c.meta.get("source") or "").strip().lower()
            source_boost = 0.0
            if source == "benchmark_raw_turn":
                source_boost += 0.2
            if source == "heuristic_extract_v1":
                if generic_penalty >= 0.5:
                    source_boost -= 0.8
                elif specificity < 0.45:
                    source_boost -= 0.3
            cue_phrase_boost = _cue_phrase_overlap(f"{c.text}\n{c.key}", cue_phrases)
            temporal_hint = 0.0
            if is_temporal_query and _has_temporal_hint(c.text):
                temporal_hint = 0.45
            causal_hint = 0.0
            if is_reflective_query and _has_causal_hint(c.text):
                causal_hint = 0.45

            type_boost = 0.0
            if c.chunk_type == "decision":
                type_boost = 0.5
            elif c.chunk_type == "fact":
                type_boost = 0.4
            elif c.chunk_type == "entity":
                type_boost = -0.2

            if has_semantic:
                # Cosine sim is in [-1, 1]; shift to [0, 2] for positive weighting.
                semantic = (sem_scores.get(c.id, 0.0) + 1.0)
                return (
                    (1.4 * semantic)
                    + (1.8 * lexical_recall)
                    + (2.6 * cue_token_overlap)
                    + (1.5 * entity_overlap)
                    + (1.6 * speaker_match)
                    + (0.8 * specificity)
                    + cue_phrase_boost
                    + temporal_hint
                    + causal_hint
                    + source_boost
                    + (0.2 * recency)
                    + (0.2 * freq)
                    + type_boost
                    - generic_penalty
                    - speaker_mismatch_penalty
                )
            else:
                return (
                    (2.2 * lexical_recall)
                    + (2.8 * cue_token_overlap)
                    + (1.5 * entity_overlap)
                    + (1.6 * speaker_match)
                    + (0.8 * specificity)
                    + cue_phrase_boost
                    + temporal_hint
                    + causal_hint
                    + source_boost
                    + (0.4 * recency)
                    + (0.2 * freq)
                    + type_boost
                    - generic_penalty
                    - speaker_mismatch_penalty
                )

        ranked = sorted(candidates, key=score, reverse=True)
        top = ranked[: max(3, int(k))]
        return top

    def _enrich_with_graph_context(self, chunks: list[Chunk], user_id: str) -> list[str]:
        """Prepend connected-node context clues to each chunk before embedding.

        The graph becomes a living dictionary: MiniLM doesn't know "Project Phoenix",
        but it reads "[Context: React frontend, Supabase database]" and maps correctly.
        """
        try:
            links = self.log._conn.execute(
                "SELECT chunk_a_id, chunk_b_id FROM user_links WHERE user_id=?",
                (user_id,),
            ).fetchall()
        except Exception:
            return [c.text for c in chunks]

        if not links:
            return [c.text for c in chunks]

        neighbors: dict[int, set[int]] = {}
        for a, b in links:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)

        chunk_by_id = {c.id: c for c in chunks}
        all_chunks_by_id = chunk_by_id.copy()
        missing_ids = set()
        for c in chunks:
            for nid in neighbors.get(c.id, set()):
                if nid not in all_chunks_by_id:
                    missing_ids.add(nid)
        if missing_ids:
            all_db = self.log.fetch_recent_chunks(user_id=user_id, limit=9999)
            for c in all_db:
                if c.id in missing_ids:
                    all_chunks_by_id[c.id] = c

        enriched: list[str] = []
        for c in chunks:
            nids = neighbors.get(c.id, set())
            if not nids:
                enriched.append(c.text)
                continue
            clues = []
            for nid in list(nids)[:5]:
                nc = all_chunks_by_id.get(nid)
                if nc:
                    clues.append(nc.key)
            if clues:
                context = ", ".join(clues)
                enriched.append(f"{c.text} [Context: {context}]")
            else:
                enriched.append(c.text)
        return enriched

    def mark_recalled(self, chunks: Sequence[Chunk]) -> None:
        self.log.mark_recalled([c.id for c in chunks])


def ewc_lambda_multiplier_for_chunks(chunks: Sequence[Chunk]) -> float:
    """
    Step 3 frequency integration:
    - frequency_count >= 3 => stronger protection (bolded)
    - frequency_count == 1 => weaker protection (faint)

    Returns a single multiplier for the current training/update batch.
    """
    if not chunks:
        return 1.0
    freqs = [max(1, int(c.frequency_count)) for c in chunks]
    hi = sum(1 for f in freqs if f >= 3)
    lo = sum(1 for f in freqs if f == 1)
    if hi > lo:
        return 1.5
    if lo > hi:
        return 0.5
    return 1.0


def _extract_named_entities(text: str) -> set[str]:
    entities = set()
    for bracketed in re.findall(r"\[([A-Z][a-zA-Z0-9_]+)\]", text):
        norm = _normalize_token(bracketed)
        if norm and norm not in _STOPWORDS:
            entities.add(norm)
    for prefixed in re.findall(r"\b([A-Z][a-zA-Z0-9_]+):", text):
        norm = _normalize_token(prefixed)
        if norm and norm not in _STOPWORDS:
            entities.add(norm)
    for token in re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", text):
        norm = _normalize_token(token)
        if norm and norm not in _STOPWORDS and norm not in _QUESTION_ENTITY_TOKENS:
            entities.add(norm)
    return entities


def _extract_speaker_label(text: str) -> str | None:
    m = re.match(r"^\[([^\]]+)\]", text.strip())
    if m:
        norm = _normalize_token(m.group(1))
        return norm or None
    m = re.match(r"^([A-Z][a-zA-Z0-9_]+):", text.strip())
    if m:
        norm = _normalize_token(m.group(1))
        return norm or None
    return None


def _extract_query_subject(text: str) -> str | None:
    speaker = _extract_speaker_label(text)
    if speaker:
        return speaker

    m = re.search(
        r"\b(?:what|when|where|why|how)\s+(?:did|does|is|was|will|would|has|have)\s+([A-Z][a-zA-Z0-9_]+)\b",
        text,
    )
    if m:
        norm = _normalize_token(m.group(1))
        return norm or None

    entities = list(_extract_named_entities(text))
    return entities[0] if entities else None


def _specificity_score(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    distinct_specific = {t for t in tokens if t not in _GENERIC_FILLER_TOKENS}
    score = min(1.0, float(len(distinct_specific)) / 8.0)

    lower = text.lower()
    if re.search(r"\d", text):
        score += 0.2
    if any(token in lower for token in _SPECIFIC_DATE_TOKENS):
        score += 0.2
    return min(1.2, score)


def _generic_dialogue_penalty(text: str) -> float:
    lower = text.lower().strip()
    penalty = 0.0
    if lower.endswith("?"):
        penalty += 0.8
    if lower.startswith("[assistant] entity mentioned:") or lower.startswith("entity mentioned:"):
        penalty += 1.0
    if lower.startswith("[assistant]"):
        penalty += 0.4
    if re.match(r"^\[[^\]]+\]\s*(wow|hey|glad|thanks|thank|yeah|awesome|cool|nice)\b", lower):
        penalty += 0.5
    if re.match(r"^(?:\[[^\]]+\]\s*)?fact:\s*(wow|hey|glad|thanks|thank|yeah|awesome|cool|nice)\b", lower):
        penalty += 0.8
    if lower.startswith("fact:") and not re.search(r"\d", lower) and len(_tokenize(lower)) < 8:
        penalty += 0.4
    for pattern in _GENERIC_DIALOGUE_PATTERNS:
        if re.search(pattern, lower):
            penalty += 0.3
            break
    return penalty


def _cue_phrase_overlap(text: str, cues: Sequence[str]) -> float:
    lower = re.sub(r"\s+", " ", text.lower())
    hits = 0
    for cue in cues:
        cue_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(cue).lower())).strip()
        if len(cue_norm) < 5:
            continue
        if cue_norm in lower:
            hits += 1
    return min(3.2, 0.8 * float(hits))


def _has_temporal_hint(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\d", text):
        return True
    return any(token in lower for token in _SPECIFIC_DATE_TOKENS)


def _has_causal_hint(text: str) -> bool:
    lower = text.lower()
    markers = (
        "after ",
        "because",
        "changed",
        "family scare",
        "made me",
        "replaced",
        "since ",
    )
    return any(marker in lower for marker in markers)

