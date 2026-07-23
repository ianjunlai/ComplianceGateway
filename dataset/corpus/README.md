# Legal corpus (thesis 4.2.1)

Place the raw English corpus here before running `ingestion.build_indexes`
and `generate_qa.py`. Target: ~100-150 chunks total.

## Expected layout

```
corpus/
├── gdpr/
│   ├── gdpr_articles.txt          # GDPR plain text, "Article N ..." headings
│   └── gdpr_recitals.txt          # optional: recitals as "(N) ..." paragraphs
└── universities/
    ├── uni_a/
    │   └── data_protection_policy.txt
    ├── uni_b/
    │   └── ...
    └── ...                         # 3-5 universities, English policies
```

## Sources

- GDPR English text: https://eur-lex.europa.eu/eli/reg/2016/679/oj
  (export as plain text; keep core Articles + selected Recitals, target 80-120 chunks)
- University data-protection policies: download from official university sites
  (choose English-language policies; record URL + retrieval date per file in
  `sources.md` for the thesis appendix), 10-20 chunks each.

## Notes

- Plain UTF-8 `.txt` only; chunking splits GDPR on `Article N` headings and
  policies on `1. ` style numbered sections (see `ingestion/chunking.py` —
  adjust the regex there if a policy uses different numbering).
- Create `sources.md` here listing every document's origin URL, retrieval
  date, and license/terms — cited in thesis 4.2.1.
