# Corpus sources

Provenance for every document in `dataset/corpus/`, for the thesis appendix.

The corpus is frozen before indexing and dataset generation: changing any source
text shifts chunk boundaries and invalidates the ground-truth chunk ids that the
evaluation sets are built from.

## Regional tier — EU

| | |
|---|---|
| Document | Regulation (EU) 2016/679 (GDPR) — Articles and selected Recitals |
| Source | https://eur-lex.europa.eu/eli/reg/2016/679/oj |
| Files | `gdpr/gdpr_articles.txt`, `gdpr/gdpr_recitals.txt` |
| Chunks | 288 |

## National tier

| | |
|---|---|
| Document | **Data Protection Act 2018** (UK), c. 12 |
| Point-in-time | 2026-01-01 snapshot; `dc:modified` 2026-07-09 |
| Source | http://www.legislation.gov.uk/ukpga/2018/12/2026-01-01 (`DocumentURI` recorded inside the file) |
| Identifier | http://www.legislation.gov.uk/id/ukpga/2018/12 |
| Extent | E+W+S+N.I. |
| File | `nations/UK_Data Protection Act 2018_2026.xml` — **not in git, see below** |
| MD5 | `ba7c50e3e67ef2873b3bb844de9052ba` (5.42 MB) |
| Chunks | 285 |

| | |
|---|---|
| Document | **Data Protection Act 2018** (Ireland), Number 7 of 2018 |
| Source | Irish Statute Book, `irishstatutebook.ie` — official English text |
| File | `nations/Ireland_DATA PROTECTION ACT 2018.txt` |
| Chunks | 243 |

| | |
|---|---|
| Document | **Federal Data Protection Act (BDSG)** (Germany) |
| Version | Act of 30 June 2017 (Federal Law Gazette I p. 2097), as last amended |
| Source | `gesetze-im-internet.de` — official English translation |
| File | `nations/Germany_Federal Data Protection Act.txt` |
| Note | Tables of contents were removed by hand before chunking |
| Chunks | 86 |

## Institutional tier

Four university data-protection policies, English-language, from each
institution's official site. Chunk boundaries are marked by hand with `###`
markers — see `README.md` for why.

| University | File | Chunks |
|---|---|---|
| Trinity College Dublin | `universities/data_protection_policy_TCD.txt` | 20 |
| University of Cambridge | `universities/data_protection_policy_Cambridge.txt` | 19 |
| Georg-August-Universität Göttingen | `universities/data_protection_policy_Goettingen.txt` | 14 |
| University of Limerick | `universities/data_protection_policy_UL.txt` | 4 |

> **TODO before submission** — record the exact retrieval URL and date for the
> Irish Act, the German Act and each university policy. The institution and
> publisher are certain; the specific page URLs were not captured at download
> time and should not be reconstructed from memory for an appendix.

## Why the UK XML is not in the repository

GitHub push protection rejects it. The file contains 8,112 occurrences of
`key-<32 hex>` inside URIs of the form
`http://www.legislation.gov.uk/id/effect/key-<md5>` — legislation.gov.uk's
public identifier scheme for amending effects. That pattern is
indistinguishable from a Mailgun API key, so the scanner reports each one, and
the push is blocked with hundreds of findings that cannot be dismissed
individually.

They are not secrets: all 8,112 are pure hexadecimal (Mailgun keys are mixed
alphanumeric), they appear inside public URIs, and the document is a public
Act of Parliament.

**Nothing in the pipeline reads the XML.** It is only the input to the one-off
chunking step:

```
UK_...xml  --chunk_nations.py-->  nations.chunks.json  --build_full_corpus.py-->  full_corpus.json
  (not in git)                      (in git, 0.96 MB)         (in git, 1.37 MB)
```

`run_full_experiment.sh` starts from `build_full_corpus.py`, which reads
`nations.chunks.json`. To re-derive from the original instead, download the XML
to `nations/`, check the MD5 above, and run:

```bash
python dataset/chunk_nations.py
```
