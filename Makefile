process_files_stage1:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_files.py 1 data/raw_data/microstructure_dataset data/prepared_data png high data/prepared_data/answer.csv

process_files_stage2:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_files.py 0 data/raw_data/archive-2 data/prepared_data bmp low data/prepared_data/answer.csv      

process_images:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_images.py data/prepared_data/answer.csv

prepare_data: process_files_stage1 process_files_stage2 process_images