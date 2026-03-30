# Entity Cleanup Pass (after extraction completes)

## Problems Found
1. **280 `unknown` type** — many are people (Vyōmēśvara, Aparādityadēva), locations (Vēharali village), dates (Saka year)
2. **124 `concept` type** — includes people (Pōnnaladēvī), literary works (Harshacharita, Kādambarī), cyclic years (Kilaka, Sarvadhārin)
3. **173 entries typed as full pipe-separated enum** (`person|dynasty|temple|...`) — model echoed the type list instead of picking one
4. **67 more with partial pipe types** (`person|dynasty|event|...`)
5. Minor: custom types crept in — `king` (22), `prince` (11), `donor` (7), `village` (6), `organization` (10) — should map to standard types

## Fix Plan
- Write a reclassification script that:
  1. Merges `king`/`prince`/`donor` → `person`
  2. Merges `village` → `location`
  3. Merges `religious_building` → `temple`
  4. For `unknown`/`concept`/pipe-types: use a fast LLM call with just the name + description to pick the correct type
  5. Batch — group all ~650 mistyped entities, send in batches of 50 to reclassify
- Estimated: ~13 LLM calls, very fast

## Stats Snapshot (during extraction)
- 3,191 entity nodes total
- 869 location, 610 person, 285 date, 135 deity, 106 dynasty, 78 temple — these are good
