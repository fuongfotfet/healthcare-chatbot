import json
import re


class CitationStreamTransformer:
    """Incrementally transforms <source> tags into inline JSON blocks for streaming UI."""

    _SOURCE_OPEN = "<source"
    _SOURCE_PATTERN = re.compile(r'<source\s+id="([^"]+)">(.*?)</source>', re.DOTALL)

    def __init__(self):
        self.buffer = ""
        self._last_non_space_emitted = ""
        self._recent_output_tail = ""

    @staticmethod
    def _last_non_space_char(text: str) -> str:
        for ch in reversed(text or ""):
            if not ch.isspace():
                return ch
        return ""

    @staticmethod
    def _first_non_space_char(text: str) -> str:
        for ch in text or "":
            if not ch.isspace():
                return ch
        return ""

    @staticmethod
    def _needs_word_separator(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left.isspace() or right.isspace():
            return False
        return left.isalnum() and right.isalnum()

    @staticmethod
    def _needs_sentence_separator(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left.isspace():
            return False
        if left in ".!?;:…":
            return False
        return left.isalnum() and right.isalpha() and right.isupper()

    @staticmethod
    def _normalize_used_text_for_json(text: str) -> str:
        # Preserve line-break intent as literal "\\n" so FE can parse JSON
        # without markdown being broken by real newlines in streamed text.
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\t", " ")
        normalized = re.sub(r"[ ]{2,}", " ", normalized)
        normalized = normalized.replace("\n", "\\n")
        return normalized.strip()

    @staticmethod
    def _single_line_json(data: dict) -> str:
        # Serialize compact JSON then harden against any accidental real line breaks.
        inline = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        inline = inline.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")
        inline = inline.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        return inline

    @staticmethod
    def _normalize_for_compare(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip().lower()
        return re.sub(r"[^\w\s%≥≤₁₂₃₄₅₆₇₈₉₀]", "", text)

    @staticmethod
    def _short_display_text(text: str, max_chars: int = 180) -> str:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if not normalized:
            return ""
        # Prefer first sentence-like segment for display to avoid repeating long lists.
        parts = re.split(r"(?<=[\.!?;:])\s+", normalized)
        first = parts[0].strip() if parts else normalized
        if len(first) > max_chars:
            return first[:max_chars].rstrip() + "..."
        return first

    @staticmethod
    def _is_list_like_text(text: str) -> bool:
        raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if "\n" not in raw:
            return False
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        if len(lines) < 2:
            return False
        bullet_like = 0
        for ln in lines:
            if re.match(r"^[-*•]\s+", ln):
                bullet_like += 1
            elif re.match(r"^\d+[\.)]\s+", ln):
                bullet_like += 1
        return bullet_like >= 1

    @staticmethod
    def _is_duplicate_suffix(prev_text: str, used_text: str) -> bool:
        prev_norm = re.sub(r"\s+", " ", (prev_text or "")).strip()
        used_norm = re.sub(r"\s+", " ", (used_text or "")).strip()
        if not prev_norm or not used_norm:
            return False
        # If the sentence just emitted already ends with used_text, don't emit used_text again.
        return prev_norm.endswith(used_norm)

    def _is_recent_duplicate(self, display_text: str) -> bool:
        if not display_text:
            return True
        tail_norm = self._normalize_for_compare(self._recent_output_tail)
        disp_norm = self._normalize_for_compare(display_text)
        if not tail_norm or not disp_norm:
            return False
        # Dedup when the display snippet is already present in recently emitted output.
        if disp_norm in tail_norm:
            return True
        # Also dedup by prefix containment for near-duplicate phrasing.
        probe = disp_norm[: max(30, len(disp_norm) // 2)]
        return probe in tail_norm

    @staticmethod
    def _pending_open_tag_tail_len(text):
        """Return minimal suffix length to keep if it may be start of '<source'."""
        max_len = min(len(text), len(CitationStreamTransformer._SOURCE_OPEN) - 1)
        for size in range(max_len, 0, -1):
            if text.endswith(CitationStreamTransformer._SOURCE_OPEN[:size]):
                return size
        return 0

    def _replace_match(self, match):
        raw_id = match.group(1).strip()
        used_text = match.group(2)

        chunk_id = raw_id[1:-1].strip() if raw_id.startswith("[") and raw_id.endswith("]") else raw_id
        if not chunk_id:
            return used_text

        normalized_used_text = self._normalize_used_text_for_json(used_text)
        single_source_data = {"chunk_id": chunk_id, "used_text": normalized_used_text}
        # Emit citation JSON inline WITHOUT markdown code fences so streaming
        # consumers can safely parse outer SSE JSON. The inner JSON will be
        # escaped when the server wraps the chunk into an outer JSON payload.
        inline_json = self._single_line_json(single_source_data)

        prev_chunk = match.string[:match.start()]
        prev_char = self._last_non_space_char(prev_chunk)
        first_char = self._first_non_space_char(used_text)
        prefix = ""
        if self._needs_sentence_separator(prev_char, first_char):
            prefix = ". "
        elif self._needs_word_separator(prev_char, first_char):
            prefix = " "

        is_list_like = self._is_list_like_text(used_text)
        display_text = "" if is_list_like else self._short_display_text(used_text)
        if self._is_duplicate_suffix(prev_chunk, used_text) or self._is_recent_duplicate(display_text):
            return f" {inline_json}"
        return f"{prefix}{display_text} {inline_json}"

    def feed(self, chunk_text):
        self.buffer += chunk_text
        output_parts = []

        while True:
            start_idx = self.buffer.find("<source")
            if start_idx == -1:
                # Keep only the suffix that can still become a '<source' prefix.
                keep_tail = self._pending_open_tag_tail_len(self.buffer)
                flush_upto = len(self.buffer) - keep_tail
                if flush_upto <= 0:
                    break
                output_parts.append(self.buffer[:flush_upto])
                self.buffer = self.buffer[flush_upto:]
                break

            if start_idx > 0:
                output_parts.append(self.buffer[:start_idx])
                self.buffer = self.buffer[start_idx:]

            end_idx = self.buffer.find("</source>")
            if end_idx == -1:
                break

            end_idx += len("</source>")
            block = self.buffer[:end_idx]
            self.buffer = self.buffer[end_idx:]
            replaced = self._SOURCE_PATTERN.sub(self._replace_match, block)
            if replaced:
                prev_char = self._last_non_space_char(output_parts[-1]) if output_parts else self._last_non_space_emitted
                first_char = self._first_non_space_char(replaced)
                if self._needs_sentence_separator(prev_char, first_char):
                    if output_parts:
                        output_parts[-1] = output_parts[-1].rstrip()
                    output_parts.append(". ")
                elif self._needs_word_separator(prev_char, first_char):
                    if output_parts:
                        output_parts[-1] = output_parts[-1].rstrip()
                    output_parts.append(" ")
            output_parts.append(replaced)

        output = "".join(output_parts)
        emitted_last = self._last_non_space_char(output)
        if emitted_last:
            self._last_non_space_emitted = emitted_last
        if output:
            self._recent_output_tail = (self._recent_output_tail + output)[-2000:]
        return output

    def flush(self):
        if not self.buffer:
            return ""
        output = self._SOURCE_PATTERN.sub(self._replace_match, self.buffer)
        first_char = self._first_non_space_char(output)
        if self._needs_sentence_separator(self._last_non_space_emitted, first_char):
            output = ". " + output.lstrip()
        elif self._needs_word_separator(self._last_non_space_emitted, first_char):
            output = " " + output.lstrip()

        emitted_last = self._last_non_space_char(output)
        if emitted_last:
            self._last_non_space_emitted = emitted_last
        if output:
            self._recent_output_tail = (self._recent_output_tail + output)[-2000:]
        self.buffer = ""
        return output
