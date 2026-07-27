# Legal corpus

Place the raw English corpus here before running `ingestion.build_indexes`
and `generate_qa.py`. Target: ~100-150 chunks total.

## Expected layout

```
corpus/
├── gdpr/
│   ├── gdpr_articles.txt          # GDPR plain text, "Article N ..." headings
│   └── gdpr_recitals.txt          # optional: recitals as "(N) ..." paragraphs
└── universities/
    ├── data_protection_policy_Cambridge.txt
    ├── data_protection_policy_TCD.txt
    └── ...                         # 3-5 universities, English policies
```

## Sources

- GDPR English text: https://eur-lex.europa.eu/eli/reg/2016/679/oj
  (export as plain text; keep core Articles + selected Recitals, target 80-120 chunks)
- University data-protection policies: download from official university sites
  (choose English-language policies; record URL + retrieval date per file in
  `sources.md` for the thesis appendix), 10-20 chunks each.

## Preparing a university policy file

GDPR is parsed automatically from its Article/Recital numbering, but real
university policies vary too much in structure (some are flat, some nested,
some mix both in the same document) to parse safely — so chunk boundaries in
these files are marked by hand, once, while cleaning up the source text
(removing footnotes, page numbers, and other PDF-extraction noise).

1. Copy the policy's text into `data_protection_policy_<Name>.txt`
   (`<Name>` becomes the university's internal ID, e.g. `Cambridge` -> `cambridge`;
   accented names are transliterated automatically, e.g. `Göttingen` -> `goettingen`).
2. Before every passage that should become its own chunk, add a line:
   ```
   ### 3.3
   3.3 The independent University Data Protection Officer is responsible for:
   ...
   ```
   The id after `###` becomes the chunk's suffix (`cambridge-policy-3.3`) — use
   the clause's own number if there is one, or any short label otherwise.
3. Everything between one `### ` line and the next belongs to that chunk,
   verbatim. Don't mark sub-items that should stay inside their parent clause
   (e.g. Cambridge's `1.6.1`-`1.6.4` under `1.6` — just leave them unmarked).
4. If the document opens with unnumbered text worth keeping (a title, an
   introduction), mark it too: `### preamble` as the very first line.
5. Text before the first `### ` marker is discarded — make sure nothing
   important precedes it.

## Notes

- Plain UTF-8 `.txt` only.
- Create `sources.md` here listing every document's origin URL, retrieval
  date, and license/terms — cited in the thesis's corpus section.
