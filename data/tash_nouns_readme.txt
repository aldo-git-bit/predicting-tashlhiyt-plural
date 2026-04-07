================================================================================
TASHLHIYT BERBER NOUN PLURAL DATASET - README
================================================================================

Dataset File: tash_nouns.csv
Last Updated: December 31, 2024
Total Records: 1,914 nouns
Total Columns: 53

This dataset contains Tashlhiyt Berber nouns with complete morphological,
semantic, and phonological annotations. It was compiled for computational
analysis of plural formation patterns in Tashlhiyt Berber.

================================================================================
COLUMN DESCRIPTIONS
================================================================================

--------------------------------------------------------------------------------
ANALYSIS FIELDS (Morphological and Phonological Analysis)
--------------------------------------------------------------------------------

analysisGender2Category
  Two-way gender classification: Masculine or Feminine
  ML Feature Name: m_gender
  Values: "Masculine", "Feminine"

analysisGender3Category
  Three-way gender classification (includes Neutral)
  Values: "Masculine", "Feminine", "Neutral"

analysisHasSecondaryMorphYN
  Indicates presence of secondary morphological forms
  Values: Y/N or blank

analysisInternalChanges
  Types of stem-internal changes in plural formation
  ML Feature Name: y_micro_6mutations, y_micro_8mutations
  Values: Ablaut, Templatic, Medial A, Final A, Final V/W, Insert C,
          Suppletion, V Deletion, Cʷ Labialization, Degemination, Vowel Copy
  Note: Multiple values separated by newline character
  Examples: "Ablaut", "Final A\nAblaut", "Templatic"

analysisMutability
  Gender mutability classification
  ML Feature Name: m_mutability
  Values: "Fixed-Masc", "Fixed-Fem", "Variable-Masc", "Variable-Fem", "Neutral"
  Note: Correlated with gender but includes morphological distinctions

analysisNotes
  Free-text annotations about exceptional or notable cases

analysisPluralPattern
  Primary plural formation strategy
  ML Feature Name: y_macro_3types, y_macro_suffix, y_macro_mutated
  Values: "External" (suffix only), "Internal" (stem change only),
          "Mixed" (both), "No Plural", "Only Plural", "id Plural"

analysisPluralPatternSecondary
  Secondary or alternative plural formation pattern
  Values: Same as analysisPluralPattern or blank

analysisPluralSplitYN
  Indicates split plural patterns
  Values: Y/N or blank

analysisPluralTheme
  Plural theme (stem form) in phonological transcription
  Example: "tiγmratn" (plural of tiγmrt "shoulder")

analysisPluralThemeSecondary
  Secondary or alternative plural theme
  Values: Phonological string or blank

analysisRAugSplitYN
  Indicates split patterns in R-augment vowel
  Values: Y/N or blank

analysisRAugVowel
  R-augment vowel classification
  ML Feature Name: m_r_aug
  Values: "Zero", "I", "A", "U/A"
  Note: Used for morphological feature engineering

analysisSemanticCoreYN
  Indicates whether noun is in semantic core vocabulary
  Values: Y/N

analysisSingularTheme
  Singular theme (stem form) in phonological transcription
  ML Feature Name: r_stem (basis for syllabification and n-grams)
  Example: "tiγmrt" (singular form)
  Note: This is the phonological form used for all feature extraction

--------------------------------------------------------------------------------
LEXICON FIELDS (Semantic and Etymological Information)
--------------------------------------------------------------------------------

lexiconAnimateYN
  Animacy classification
  ML Feature Name: s_animacy
  Values: "Y" (animate), "N" (inanimate)

lexiconGender
  Lexical gender (may differ from morphological gender in some cases)
  Values: "M" (masculine), "F" (feminine), "N" (neuter/neutral)

lexiconGlossEnglish
  English translation/gloss
  ML Feature Name: r_glossEnglish
  Example: "shoulder"

lexiconGlossFrench
  French translation/gloss (primary source for glosses)
  ML Feature Name: r_glossFrench
  Example: "épaule"

lexiconHumanYN
  Human classification (subset of animate)
  ML Feature Name: s_humanness
  Values: "Y" (human), "N" (non-human)

lexiconjGender
  Alternative gender field (rarely used)
  Values: "M", "F", "N"

lexiconLoanwordNotes
  Free-text notes about loanword etymology and adaptations
  Example: "last Arabic vowel 'a' replaced by 't'"

lexiconLoanwordSource
  Etymological source of loanword
  ML Feature Name: m_loanTypes
  Values: "Arabic-Assimilated", "Arabic-Unassimilated", "French",
          "Spanish", "Unknown", blank (native Berber)

lexiconPOS
  Part of speech (all entries are nouns)
  Values: "N"

lexiconSemanticField
  Semantic category classification
  ML Feature Name: s_semantic_field
  Values: 22 categories including "Body Parts & Functions",
          "Emotion", "Physical Acts & Materials", "Motion & Transportation",
          "Animals", "Food & Drink", etc.

lexiconSexGender
  Biological sex (for animate nouns)
  Values: "M" (male), "F" (female), blank (unspecified/inanimate)
  Note: 92.46% unspecified; excluded from ML feature set

--------------------------------------------------------------------------------
RECORD FIELDS (Data Source and Identification)
--------------------------------------------------------------------------------

recordDataSource
  Bibliographic source for the record
  Values: "Jebbour 1996", "Ouakrim 1995", etc.

recordID
  Unique identifier for each noun
  ML Feature Name: r_id
  Values: Integer (1-1914)

--------------------------------------------------------------------------------
WORD PARADIGM FIELDS (Full Inflectional Forms)
--------------------------------------------------------------------------------

word1MSF
  Masculine Singular Free State (citation form)
  Example: "!lkdran" (tar)

word2MSB
  Masculine Singular Bound State

word3MPF
  Masculine Plural Free State

word4MPB
  Masculine Plural Bound State

word5FSF
  Feminine Singular Free State

word6FSB
  Feminine Singular Bound State

word7FPF
  Feminine Plural Free State

word8FPB
  Feminine Plural Bound State

wordDerivedCategory
  Derivational morphology classification
  ML Feature Name: m_derivational_category
  Values: "Underived", "Action Noun", "Agentive", "Instrumental",
          "Occupational", "Tirrugza", "Diminutive", "Locative", etc.
  Note: 15 categories; grouped to 5 dimensions for ML

wordDerivedCategoryNotes
  Free-text notes about derivational processes

wordDerivedInput
  Base form from which derived noun was created
  Example: Base verb or noun before derivation

wordDerivedInputGloss
  English gloss of derivational input

wordFemBase
  Base form for feminine gender

wordFemSecondary
  Secondary feminine form (if applicable)

wordLexBitVector
  Binary feature vector for lexical properties
  Format: "0000-1100" (8-bit encoding of various features)

wordMascBase
  Base form for masculine gender

wordMascSecondary
  Secondary masculine form (if applicable)

wordNumSemanticCategory
  Numeric encoding of semantic category
  Values: 0.0-5.0 or blank

--------------------------------------------------------------------------------
ENGINEERED PHONOLOGICAL FEATURES
(Generated via rule-based syllabification and feature extraction)
--------------------------------------------------------------------------------

p_stem_sing_syllabified
  Syllabified singular stem using Tashlhiyt syllabification rules
  ML Feature Name: p_stem_syllables
  Example: "l.ʕs.sa" (syllables separated by periods)
  Coverage: 1,864/1,914 nouns (97.4%)
  Accuracy: 99.52% validated against gold standard

p_stem_sing_LH
  Light (L) and Heavy (H) syllable pattern
  ML Feature Name: p_LH
  Example: "LLL" (3 light syllables), "HH" (2 heavy syllables)
  Generation: Light = no coda, Heavy = has coda
  Coverage: 1,864/1,914 nouns

p_stem_sing_foot
  Foot structure parsed from LH pattern
  ML Feature Name: p_foot
  Example: "lF" (unparsed light + foot), "FF" (2 feet)
  Principle: Right-to-left moraic trochee
  Rules: H → F, LL → F, L → l (unparsed)
  Coverage: 1,864/1,914 nouns

p_LH_count_heavies
  Count of heavy syllables in stem
  Values: Integer (0-4)

p_LH_count_moras
  Total mora count (L=1 mora, H=2 moras)
  Values: Integer (1-8)

p_foot_count_feet
  Count of complete feet in prosodic structure
  Values: Integer (0-3)

p_foot_residue
  Presence of unparsed residue in foot structure
  Values: 0 (no residue) or 1 (has unparsed 'l')

Note: Phonological features were generated using rule-based algorithms
validated against 72 gold-standard annotations (97.22% accuracy).

================================================================================
ML FEATURE NAME MAPPINGS
================================================================================

For computational modeling, some column names were shortened or transformed.
Below is the complete mapping from database columns to ML feature names:

TARGET VARIABLES (Predicted Outcomes)
  analysisPluralPattern → y_macro_suffix (binary: has suffix?)
  analysisPluralPattern → y_macro_mutated (binary: has stem change?)
  analysisPluralPattern → y_macro_3types (3-way classification)
  analysisInternalChanges → y_micro_6mutations (6 mutation types)
  analysisInternalChanges → y_micro_8mutations (8-way classification)

MORPHOLOGICAL FEATURES
  analysisRAugVowel → m_r_aug
  analysisGender2Category → m_gender
  wordDerivedCategory → m_derivational_category
  analysisMutability → m_mutability
  lexiconLoanwordSource → m_loanTypes

SEMANTIC FEATURES
  lexiconAnimateYN → s_animacy
  lexiconHumanYN → s_humanness
  lexiconSemanticField → s_semantic_field

PHONOLOGICAL FEATURES
  [engineered] → p_ngrams (macro: 2,265 features)
  [engineered] → p_ngrams (micro: 1,356 features)
  [engineered] → p_ngrams_macro_master (2,019 LASSO-selected)
  [engineered] → p_ngrams_micro_master (1,149 LASSO-selected)
  p_stem_sing_LH → p_LH
  p_stem_sing_syllabified → p_stem_syllables
  p_stem_sing_foot → p_foot

RECORD IDENTIFIERS
  analysisSingularTheme → r_stem
  recordID → r_id
  lexiconGlossEnglish → r_glossEnglish
  lexiconGlossFrench → r_glossFrench

================================================================================
KNOWN ISSUES AND EXCEPTIONS
================================================================================

Syllabification Exceptions (14 forms, 0.75% of dataset):
  Forms with non-standard syllable structure requiring manual annotation.
  Examples: tuzzgt → tuzz.gt, ismmji → i.smm.ji, lmqqttʕ → l.mq.qt.tʕ

Missing Data:
  - 50 nouns lack analysisSingularTheme (cannot generate phonological features)
  - Some paradigm cells empty (word1MSF through word8FPB) for defective nouns
  - lexiconSexGender: 92.46% unspecified (excluded from ML models)

Data Imbalance (Binary Classification Tasks):
  - Final V/W: 19.8:1 ratio (27 minority samples)
  - Insert C: 13.7:1 ratio (38 minority samples)
  - Final A: 6.9:1 ratio (71 minority samples)

  Note: SMOTE (Synthetic Minority Over-sampling Technique) applied to these
  domains during ML experiments to improve cross-fold stability.

================================================================================
TECHNICAL NOTES
================================================================================

File Format: CSV (comma-separated values)
Encoding: UTF-8
Delimiter: Comma (,)
Header Row: Yes (row 1)
Missing Values: Blank cells (empty string)

Phonological Transcription:
  - IPA symbols used with some ASCII approximations
  - Geminate consonants: doubled letters (e.g., "tt", "mm")
  - Pharyngealized consonants: "ʕ", "ħ", etc.
  - Labialization: superscript "ʷ" (e.g., "kʷ", "gʷ")
  - Syllable boundaries: period "." (in p_stem_sing_syllabified only)

Multi-value Fields:
  - analysisInternalChanges: Multiple values separated by newline ("\n")
  - Maximum 2 values per noun (most common: mutation + Ablaut)

Engineered Features:
  - Generated via rule-based algorithms (not manual annotation)
  - Deterministic: same input → same output (100% reproducible)
  - Validated against gold standard: 97-99% accuracy

Class Balancing:
  - All ML models use class_weight='balanced' parameter
  - SMOTE applied to 4 domains with extreme imbalance (final_a, final_vw,
    insert_c, medial_a)
  - Random seed: 42 (for reproducible train/test splits)


================================================================================
END OF README
================================================================================
