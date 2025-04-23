process_files_stage1:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_files.py 1 data/raw_data/microstructure_dataset data/prepared_data png 1 data/prepared_data/answer.csv

process_files_stage2:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_files.py 0 data/raw_data/archive-2 data/prepared_data bmp 0 data/prepared_data/answer.csv     

interpolate_uhcsdb: 
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/proccess_uhcsdb.py data/raw_data/UHCSDB data/raw_data/UHCSDB/metadata.csv png

 process_files_stage3:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_files.py 0 data/raw_data/UHCSDB data/prepared_data png 2 data/prepared_data/answer.csv     

process_images:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_images.py data/prepared_data/answer.csv 1

process_images_without_crop:
	python3 /Users/aleksandr/Kursovay_METALs/Metals/src/data/process_images.py data/prepared_data/answer.csv 0

prepare_data: process_files_stage1 process_files_stage2 process_images interpolate_uhcsdb process_files_stage3 process_images_without_crop
