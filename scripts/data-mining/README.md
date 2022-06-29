Scripts in this directory process the raw data from the [ACL anthology](https://aclanthology.org/) bib file and the [s2orc dataset](https://allenai.org/data/s2orc) and produce a JSONL format that is easy to process for subsequent tasks.

Files
=====

* `acl_anthology_bib_to_json.py`: Converts bib file from ACL anthology to JSONL file
* `aggregate_s2orc_metadata.py`: Aggregates the metadata entries from s2orc and matches with bib entries in ACL anthology (by title and year)
* `aggregate_s2orc_fulltext.py`: Aggregates the full text entries from s2orc with the metadata entry
* `filter_lang_and_year.py`: Filter the papers based on year range and the language id extracted from the full text
* `acl_anthology_venue_scrape.py`: Get all the venues from the ACL Anthology website
  * This file is not used anymore. We ended up labeling the venues from the bib instances.
* `makeOneFile.py`: Scrub some text; create a single file that contains text and additional metadata
* `make_file_with_citations.py`: Create a file that contains the distribution of citations per year.
  * Most of this is taken care of by `makeOneFile.py` so we end up not using this script.
