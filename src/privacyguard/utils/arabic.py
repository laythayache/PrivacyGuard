"""Arabic language support for regional anonymization.

Handles:
- Script detection (Arabic vs Latin)
- Right-to-left (RTL) text handling
- Arabic plate format recognition
- Bilingual document processing
"""

from __future__ import annotations

import re
from enum import Enum


class ScriptType(Enum):
    """Detected text script type."""

    ARABIC = "arabic"
    LATIN = "latin"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ArabicProcessor:
    """Utilities for Arabic text processing and detection."""

    # Arabic Unicode ranges
    ARABIC_RANGES = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]

    # Lebanese license plate patterns
    LEBANESE_PLATE_PATTERNS = [
        r"[ش-ي]{1,2}\s?\d{1,6}",  # Arabic letters with numerals
        r"[A-Z]{1,3}\s?\d{1,6}",  # Latin letters (French plates)
    ]

    # Arabic characters that commonly appear in documents
    ARABIC_DOCUMENT_MARKERS = {
        "هوية": "id_card",  # ID card
        "جواز": "passport",  # Passport
        "رخصة": "license",  # License
        "وثيقة": "document",  # Document
    }

    @staticmethod
    def is_arabic_char(char: str) -> bool:
        """Check if character is Arabic."""
        code_point = ord(char)
        return any(start <= code_point <= end for start, end in ArabicProcessor.ARABIC_RANGES)

    @staticmethod
    def detect_script(text: str) -> ScriptType:
        """Detect script type in text (Arabic, Latin, or Mixed)."""
        if not text:
            return ScriptType.UNKNOWN

        arabic_count = sum(1 for c in text if ArabicProcessor.is_arabic_char(c))
        latin_count = sum(1 for c in text if c.isalpha() and not ArabicProcessor.is_arabic_char(c))

        total = arabic_count + latin_count
        if total == 0:
            return ScriptType.UNKNOWN

        arabic_ratio = arabic_count / total
        latin_ratio = latin_count / total

        # Threshold: > 70% is pure script
        if arabic_ratio > 0.7:
            return ScriptType.ARABIC
        if latin_ratio > 0.7:
            return ScriptType.LATIN
        return ScriptType.MIXED

    @staticmethod
    def is_likely_lebanese_plate(text: str) -> bool:
        """Check if text matches Lebanese license plate format."""
        return any(re.match(pattern, text) for pattern in ArabicProcessor.LEBANESE_PLATE_PATTERNS)

    @staticmethod
    def extract_arabic_text_regions(text: str) -> list[tuple[int, int, str]]:
        """Extract regions containing Arabic text with positions.

        Returns:
            List of (start_pos, end_pos, text) tuples for Arabic regions
        """
        regions = []
        in_arabic = False
        start = 0

        for i, char in enumerate(text):
            is_arabic = ArabicProcessor.is_arabic_char(char)

            if is_arabic and not in_arabic:
                start = i
                in_arabic = True
            elif not is_arabic and in_arabic:
                regions.append((start, i, text[start:i]))
                in_arabic = False

        if in_arabic:
            regions.append((start, len(text), text[start:]))

        return regions

    @staticmethod
    def is_likely_document(filename: str | None, text_hints: list[str] | None = None) -> bool:
        """Check if content is likely an identity document.

        Args:
            filename: Original filename
            text_hints: List of text snippets found in image

        Returns:
            True if likely an identity document
        """
        if filename:
            doc_keywords = ["id", "passport", "license", "وثيقة", "هوية", "جواز"]
            filename_lower = filename.lower()
            if any(kw in filename_lower for kw in doc_keywords):
                return True

        if text_hints:
            for hint in text_hints:
                if any(marker in hint for marker in ArabicProcessor.ARABIC_DOCUMENT_MARKERS):
                    return True

        return False


def detect_script(text: str) -> ScriptType:
    """Convenience function to detect script type."""
    return ArabicProcessor.detect_script(text)
