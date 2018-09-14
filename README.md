# VideoQA
## Usage

### Preprocess
python preprocess_resnet.py

### Train
python run_gra_xxx.py --dataset msrvtt_qa --mode train --log log_dir_path

### Test
python run_gra_xxx.py --dataset msrvtt_qa --mode test --log log_dir_path

## Project file structure
* data
Saved preprocessed data & origin data.

* VideoQA
Project source code path.
    * run_*.py: Train/test python script files.
    * preprocess_*.py: Preprocess data python script files.
    * util/: Dir. about preprocess and dataloader.
    * model/: Models.

