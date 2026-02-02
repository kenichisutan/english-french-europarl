"""
Data exploration for English–French parallel corpus (TMX format).
Loads a subset of en-fr.tmx and runs basic stats and trend analyses.

Usage:
  python data-exploration.py
  # Or in code:
  from data_exploration import load_tmx_subset_robust, run_exploration, basic_stats
  pairs = load_tmx_subset_robust(TMX_PATH, max_tu=10_000)
  basic_stats(pairs)
  run_exploration(subset_size=50_000)
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import re

# Default paths
DATA_DIR = Path(__file__).resolve().parent / "data"
TMX_PATH = DATA_DIR / "en-fr.tmx"

# Subset size for exploration (avoid loading 8M lines)
DEFAULT_SUBSET_SIZE = 50_000


def _get_seg_text(elem):
    """Extract segment text from a tuv element."""
    seg = elem.find(".//{*}seg") or elem.find("seg")
    if seg is not None:
        return ((seg.text or "") + "".join((e.tail or "") for e in seg)).strip()
    return ""


def load_tmx_subset_robust(path: Path = TMX_PATH, max_tu: int = DEFAULT_SUBSET_SIZE):
    """
    Stream-parse TMX; robust to namespaces. Yields (en_text, fr_text) for up to max_tu.
    Only clears each <tu> after reading it so segment text is still available.
    """
    pairs = []
    in_tu = False
    en_text = None
    fr_text = None

    for event, elem in ET.iterparse(path, events=("start", "end")):
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag == "tu":
            if event == "start":
                in_tu = True
                en_text = None
                fr_text = None
            else:
                if en_text is not None and fr_text is not None:
                    pairs.append((en_text, fr_text))
                    if len(pairs) >= max_tu:
                        break
                in_tu = False
                elem.clear()

        elif in_tu and tag == "tuv":
            if event == "end":
                lang = elem.get(
                    "{http://www.w3.org/XML/1998/namespace}lang",
                    elem.get("lang", ""),
                )
                text = _get_seg_text(elem)
                if lang == "en":
                    en_text = text
                elif lang == "fr":
                    fr_text = text

    return pairs


def basic_stats(pairs):
    """Print and return basic corpus statistics."""
    if not pairs:
        print("No pairs loaded.")
        return {}

    n = len(pairs)
    en_lens = [len(en) for en, _ in pairs]
    fr_lens = [len(fr) for _, fr in pairs]
    en_words = [len(en.split()) for en, _ in pairs]
    fr_words = [len(fr.split()) for _, fr in pairs]

    stats = {
        "n_pairs": n,
        "en_char_mean": sum(en_lens) / n,
        "fr_char_mean": sum(fr_lens) / n,
        "en_word_mean": sum(en_words) / n,
        "fr_word_mean": sum(fr_words) / n,
        "en_char_min": min(en_lens),
        "en_char_max": max(en_lens),
        "fr_char_min": min(fr_lens),
        "fr_char_max": max(fr_lens),
    }

    print("=== Basic statistics ===")
    print(f"Number of sentence pairs: {n:,}")
    print(f"English — chars: mean={stats['en_char_mean']:.1f}, min={stats['en_char_min']}, max={stats['en_char_max']}")
    print(f"French  — chars: mean={stats['fr_char_mean']:.1f}, min={stats['fr_char_min']}, max={stats['fr_char_max']}")
    print(f"English — words/sent: mean={stats['en_word_mean']:.1f}")
    print(f"French  — words/sent: mean={stats['fr_word_mean']:.1f}")
    return stats


def length_ratios(pairs, top_k=10):
    """Report length ratios (FR/EN) by character and by word; show extremes."""
    if not pairs:
        return
    ratios_char = []
    ratios_word = []
    for en, fr in pairs:
        l_en = len(en)
        l_fr = len(fr)
        w_en = len(en.split())
        w_fr = len(fr.split())
        if l_en > 0:
            ratios_char.append(l_fr / l_en)
        if w_en > 0:
            ratios_word.append(w_fr / w_en)

    n = len(ratios_char)
    mean_rc = sum(ratios_char) / n
    mean_rw = sum(ratios_word) / n

    print("\n=== Length ratios (FR/EN) ===")
    print(f"By character: mean={mean_rc:.3f}")
    print(f"By word:      mean={mean_rw:.3f}")

    # Pairs with highest/lowest character ratio (skip very short EN to avoid noise)
    min_en_len = 20
    with_ratio = [((en, fr), len(fr) / len(en)) for (en, fr) in pairs if len(en) >= min_en_len]
    with_ratio.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_k} pairs with highest FR/EN char ratio (EN length >= {min_en_len}):")
    for (en, fr), r in with_ratio[:top_k]:
        print(f"  {r:.2f}  en[{len(en)}] fr[{len(fr)}]  en: {en[:60]}...")
    return {"mean_ratio_char": mean_rc, "mean_ratio_word": mean_rw}


def sentence_length_distribution(pairs, num_buckets=10):
    """Bucket sentence lengths (by word count) and show distribution."""
    if not pairs:
        return
    en_words = [len(en.split()) for en, _ in pairs]
    fr_words = [len(fr.split()) for _, fr in pairs]
    max_en = max(en_words)
    max_fr = max(fr_words)
    bucket_size_en = max(1, (max_en + num_buckets - 1) // num_buckets)
    bucket_size_fr = max(1, (max_fr + num_buckets - 1) // num_buckets)

    buckets_en = defaultdict(int)
    buckets_fr = defaultdict(int)
    for w in en_words:
        b = min(w // bucket_size_en, num_buckets - 1)
        buckets_en[b] += 1
    for w in fr_words:
        b = min(w // bucket_size_fr, num_buckets - 1)
        buckets_fr[b] += 1

    print("\n=== Sentence length (words) distribution ===")
    print("English (bucket = word count range):")
    for b in sorted(buckets_en.keys()):
        lo = b * bucket_size_en
        hi = (b + 1) * bucket_size_en - 1 if b < num_buckets - 1 else max_en
        print(f"  [{lo:3d}-{hi:3d}]: {buckets_en[b]:,}")
    print("French:")
    for b in sorted(buckets_fr.keys()):
        lo = b * bucket_size_fr
        hi = (b + 1) * bucket_size_fr - 1 if b < num_buckets - 1 else max_fr
        print(f"  [{lo:3d}-{hi:3d}]: {buckets_fr[b]:,}")


def sample_pairs(pairs, k=5, indices=None):
    """Print k sample pairs (or at given indices)."""
    if not pairs:
        return
    n = len(pairs)
    if indices is None:
        step = max(1, n // (k + 1))
        indices = [i * step for i in range(k)]
    print("\n=== Sample pairs ===")
    for i, idx in enumerate(indices):
        if idx >= n:
            break
        en, fr = pairs[idx]
        print(f"--- Pair {i+1} (index {idx}) ---")
        print(f"EN: {en[:200]}{'...' if len(en) > 200 else ''}")
        print(f"FR: {fr[:200]}{'...' if len(fr) > 200 else ''}")
        print()


def vocabulary_trends(pairs, top_n=20):
    """Simple word frequency (EN and FR) on the subset."""
    if not pairs:
        return
    en_freq = defaultdict(int)
    fr_freq = defaultdict(int)
    word_re = re.compile(r"\b\w+\b", re.UNICODE)

    for en, fr in pairs:
        for m in word_re.finditer(en.lower()):
            en_freq[m.group()] += 1
        for m in word_re.finditer(fr.lower()):
            fr_freq[m.group()] += 1

    en_top = sorted(en_freq.items(), key=lambda x: -x[1])[:top_n]
    fr_top = sorted(fr_freq.items(), key=lambda x: -x[1])[:top_n]

    print("\n=== Top words (subset) ===")
    print("English:", [w for w, _ in en_top])
    print("French: ", [w for w, _ in fr_top])
    return {"en_top": en_top, "fr_top": fr_top}


def run_exploration(
    tmx_path: Path = TMX_PATH,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    num_samples: int = 5,
    num_buckets: int = 10,
    top_words: int = 20,
):
    """Load subset and run all exploration steps."""
    print(f"Loading up to {subset_size:,} translation units from {tmx_path}...")
    pairs = load_tmx_subset_robust(tmx_path, max_tu=subset_size)
    print(f"Loaded {len(pairs):,} pairs.\n")

    basic_stats(pairs)
    length_ratios(pairs, top_k=5)
    sentence_length_distribution(pairs, num_buckets=num_buckets)
    vocabulary_trends(pairs, top_n=top_words)
    sample_pairs(pairs, k=num_samples)

    return pairs


if __name__ == "__main__":
    run_exploration(
        subset_size=20_000,
        num_samples=5,
        num_buckets=10,
        top_words=25,
    )
