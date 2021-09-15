---
title: HP style models for modeling semantic innovation cascades in NLP venues.
---

Modeling pipeline:

- `make_venue_cascades.py`
  - `python make_venue_cascades.py --innovs-file data/innovs.pkl --cascades-file data/cascades.pkl --offset 0.2 --start-year 1989 --num-cascades 200 --num-events 100 --data-file data/hp-data-v1-001.pkl --summary-file data/hp-data-summary.json`
- `collapse_and_filter_venues.py` (optional: needed when venues need to be filtered or collapsed)
  - `python collapse_and_filter_venues.py --input-data-file data/hp-data-v1-001.pkl --venues-file data/venues.tsv --keep-venues-file data/keep_venues.txt --output-data-file data/hp-data-v2-001.pkl --log-file data/collapse_and_filter_venues.log`
- `sem_influence.py`
  - `python sem_influence.py --data-file data/hp-data-v2-001.pkl  --num-cd-iterations 50 --vanilla-hp-params-file data/v2_001_vanilla_hp_params.pkl --init-params-file data/v2_001_init_params.pkl --coarsened-hp-params-file data/v2_001_coarsened_hp_params.pkl --log-file data/sem_influence.log`
- `citation_analysis.py`
  - `python citation_analysis.py --data-file data/hp-data-v2-001.pkl --params-file data/v2_001_coarsened_hp_params.pkl --citations-file data/venue_citations.pkl --venues-file data/venues.tsv --keep-venues-file data/keep_venues.txt --output-file data/v2_001_influence_relations.csv --log-file data/citation_analysis.log`
