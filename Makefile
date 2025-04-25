process_archive_2_files:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_files.py 1 data/raw_data/archive-2 data/prepared_data bmp data/raw_data/archive-2/tensile-properties.csv $(target_column) data/prepared_data/answer.csv

process_images:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_images.py data/prepared_data/answer.csv 1

prepare_data: process_archive_2_files process_images
