"""
response_cleaner.py — Post-processing pipeline for high quality medical responses
Place this file at: utils/response_cleaner.py
"""

import re


# ---------------------------------------------------------------------------
# Word-split artifact repair
# ---------------------------------------------------------------------------

# Comprehensive medical vocabulary — words commonly split by small models
MEDICAL_WORD_FIXES = {
    # Symptoms
    "head ache": "headache", "head aches": "headaches",
    "back ache": "backache", "tooth ache": "toothache",
    "sto mach": "stomach", "ab domen": "abdomen", "ab dominal": "abdominal",
    "n ause a": "nausea", "nause a": "nausea",
    "fat igue": "fatigue", "fati gue": "fatigue",
    "di arrhe a": "diarrhea", "diarr hea": "diarrhea",
    "vom iting": "vomiting",
    "swe ating": "sweating", "swe ats": "sweats",
    "thro bb ing": "throbbing",
    "short ness": "shortness",
    "weak ness": "weakness",
    "diz ziness": "dizziness", "diz zy": "dizzy",
    "conf usion": "confusion",
    "sei zures": "seizures", "sei zure": "seizure",
    "tre mors": "tremors", "tre mor": "tremor",
    "par alysis": "paralysis",

    # Body parts
    "mus cle": "muscle", "mus cles": "muscles",
    "kid ney": "kidney", "kid neys": "kidneys",
    "live r": "liver",
    "sple en": "spleen",
    "pan creas": "pancreas",
    "thy roid": "thyroid",
    "bron chus": "bronchus", "bron chi": "bronchi",
    "al veoli": "alveoli",
    "ner vous": "nervous",
    "vas cular": "vascular",

    # Conditions
    "hy pertension": "hypertension",
    "dia betes": "diabetes", "dia betic": "diabetic",
    "mal aria": "malaria",
    "pneu monia": "pneumonia",
    "tuber culosis": "tuberculosis",
    "an emia": "anemia", "an aemia": "anaemia",
    "j au ndice": "jaundice", "jaun dice": "jaundice",
    "hem ol ysis": "hemolysis",
    "c ere bral": "cerebral",
    "pul monary": "pulmonary",
    "card iac": "cardiac",
    "hep atitis": "hepatitis",
    "arth ritis": "arthritis",
    "ost eoporosis": "osteoporosis",
    "al zheimer": "alzheimer",
    "park inson": "parkinson",
    "epi lepsy": "epilepsy",
    "sch izophrenia": "schizophrenia",
    "dep ression": "depression",
    "anx iety": "anxiety",

    # Treatments / drugs
    "anti biotic": "antibiotic", "anti biotics": "antibiotics",
    "anti viral": "antiviral",
    "anti fungal": "antifungal",
    "para cetamol": "paracetamol",
    "ibu profen": "ibuprofen",
    "am oxicillin": "amoxicillin",
    "met formin": "metformin",
    "in sulin": "insulin",
    "vac cine": "vaccine", "vac cination": "vaccination",
    "transf usion": "transfusion",
    "chem otherapy": "chemotherapy",
    "radio therapy": "radiotherapy",
    "surg ery": "surgery", "surg ical": "surgical",

    # Medical terms
    "diag nosis": "diagnosis", "diag nostic": "diagnostic",
    "prog nosis": "prognosis",
    "symp toms": "symptoms", "symp tom": "symptom",
    "treat ment": "treatment",
    "pre vention": "prevention",
    "comp lications": "complications", "comp lication": "complication",
    "en larg ed": "enlarged",
    "in flammation": "inflammation", "in flammatory": "inflammatory",
    "in fection": "infection", "in fectious": "infectious",
    "im mune": "immune", "im munity": "immunity",
    "p ale": "pale", "p allor": "pallor",
    "yellow ing": "yellowing",
    "swell ing": "swelling",
    "re duced": "reduced",
    "th reat ening": "threatening",
    "hem orrhage": "hemorrhage",
    "sep sis": "sepsis",
    "hyp oxia": "hypoxia",
    "ed ema": "edema", "oed ema": "oedema",
    "ne crosis": "necrosis",
    "path ogen": "pathogen", "path ogenic": "pathogenic",
}


def fix_split_words(text: str) -> str:
    """Fix tokenization artifacts where words are split with spaces."""
    # Apply known medical word fixes (case-insensitive)
    for wrong, right in MEDICAL_WORD_FIXES.items():
        # Match with any capitalization
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(
            lambda m: right.capitalize() if m.group(0)[0].isupper() else right,
            text
        )

    # Fix pattern: Single capital letter + space + lowercase letters
    # e.g., "F ever" -> "Fever", "Ch ills" -> "Chills"
    text = re.sub(r'\b([A-Z])\s+([a-z]{2,})\b', r'\1\2', text)

    # Fix pattern: lowercase letters + space + 1-3 lowercase letters (mid-word splits)
    # e.g., "swe ats" -> "sweats" but NOT "in the" -> "inthe"
    # Only fix if the second part looks like a suffix, not a word
    common_suffixes = {'ing', 'ed', 'er', 'es', 'ly', 'al', 'ic', 'tion', 'ness', 'ment', 'ive', 'ous', 'ful', 'sis', 'itis'}
    def fix_suffix_split(m):
        word1, word2 = m.group(1), m.group(2)
        if word2.lower() in common_suffixes:
            return word1 + word2
        return m.group(0)
    text = re.sub(r'([a-z]{3,})\s+([a-z]{2,4})\b', fix_suffix_split, text)

    # Fix asterisk-wrapped bold with spaces: "** F ever **" -> "**Fever**"
    text = re.sub(r'\*\*\s*(.+?)\s*\*\*', lambda m: f'**{fix_split_words_simple(m.group(1))}**', text)

    return text


def fix_split_words_simple(text: str) -> str:
    """Simpler version for use inside regex replacements."""
    text = re.sub(r'\b([A-Z])\s+([a-z]{2,})\b', r'\1\2', text)
    return text


def clean_formatting(text: str) -> str:
    """Clean up markdown formatting issues."""
    # Remove excessive blank lines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix "‑" (non-breaking hyphen) to regular hyphen
    text = text.replace('‑', '-')

    # Fix weird quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    # Remove trailing spaces on lines
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    # Ensure disclaimer is properly formatted
    text = re.sub(
        r'\*Always consult.*?\*',
        '\n\n---\n> ⚕️ **Medical Disclaimer:** This information is for educational purposes only. Always consult a qualified healthcare provider for diagnosis, treatment, and personalized medical advice.',
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def add_structure(text: str, intent: str = "general") -> str:
    """Add proper structure if response lacks it."""
    # If response has no headers and is long, it's probably unstructured
    has_headers = '##' in text or '**' in text
    is_long = len(text) > 500

    if not has_headers and is_long:
        # Add a simple structure
        lines = text.split('\n')
        structured = []
        for line in lines:
            line = line.strip()
            if not line:
                structured.append('')
            elif line.endswith(':'):
                structured.append(f'\n**{line}**')
            else:
                structured.append(line)
        text = '\n'.join(structured)

    return text


def ensure_disclaimer(text: str) -> str:
    """Always ensure medical disclaimer is present."""
    disclaimer = (
        "\n\n---\n"
        "> ⚕️ **Medical Disclaimer:** This information is for educational purposes only. "
        "Always consult a qualified healthcare provider for diagnosis, treatment, and personalized medical advice. "
        "In case of emergency, call emergency services immediately."
    )

    disclaimer_keywords = ["consult", "disclaimer", "healthcare provider", "medical advice"]
    has_disclaimer = any(kw.lower() in text.lower() for kw in disclaimer_keywords)

    if not has_disclaimer:
        text = text + disclaimer

    return text


def clean_response(text: str, intent: str = "general") -> str:
    """
    Master cleaning pipeline — run all fixes in order.
    Call this on every response before sending to frontend.
    """
    if not text or not text.strip():
        return "I was unable to generate a response. Please try again."

    # Step 1: Fix split words
    text = fix_split_words(text)

    # Step 2: Clean formatting
    text = clean_formatting(text)

    # Step 3: Add structure if needed
    text = add_structure(text, intent)

    # Step 4: Ensure disclaimer
    text = ensure_disclaimer(text)

    return text.strip()
