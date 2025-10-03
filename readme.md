# DPA Enforcement Data Pipeline

This repository organizes raw and AI-annotated GDPR enforcement decisions, providing a reproducible pipeline for parsing, cleaning, exploratory diagnostics, and policy analysis. The goal of the project is to develop academic insights into decision-making patterns of highest academic quality.

Raw decision data is saved under /raw_data - which contains folders and files:

/raw_data/full-decisions/ - contains the raw decisions, in subfolders sorted by authority type, then country, each decision having a unique filename and saved as .md (for example: Norway_22.md)/
raw_data/AI_analysis/AI-prompt.md: This is the full text of the prompt sent to the AI. It contains the prompt with all the questions asked. When unclear on what the schema field represents, consult
raw_data/AI_analysis/AI_responses: This folder contains the responses from the AI. There are three formats: all-responses.txt, which contains all the responses from the AI in a single file, delimited with ----- RESPONSE DELIMITER ----- , id field, ----- RESPONSE DELIMITER -----, answers. JSON contains the full response from AI including misc metadata.
THE MAIN FILE FOR ANALYSIS IS /raw_data/main_dataset.csv - this is the file that contains the full dataset as csv. CONSULT THE SCHEMA CAREFULLY BEFORE IMPLEMENTING ANY FEATURES, AND DOUBLE-CHECK CONFORMANCE. Another file, /raw_data/data_with_errors.csv, contains additional rows with errors which will need troubleshooting down the line.

/schema/ - contains the CSV schema and annotated allowed values for each field. AI prompt is copied there for extra context.
/scripts/ - contains the scripts used to clean the data, analyze the data, and generate the reports, as well as other utility scripts.

It is vital to bear the following in mind:
1. When designing new scripts, always use the existing scripts as a reference, and make sure no duplication takes place. Full consistency is key.
2. Scripts must be able to operate in a logical order, and must be able to operate on the full dataset.
3. Outputs should be organized per phase, each phase having its own folder under /outputs. Each script file name must contain the phase reference (for example: 1_clean_data.py, 2_analyze_data.py, 3_generate_reports.py) for easy traceability. This document must be updated with a concise phase plan, as they develop, and concisely describe data processing taking place, referencing inputs, scripts and outputs in a concise way.