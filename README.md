# Receipt-data-extraction :raising_hand:

## 0. Installation :smiley:

- `git clone https://github.com/manhph2211/Receipt-Data-Extraction.git`
- `cd Receipt-Data-Extraction`
- `pip3 install -r requirements.txt`

## 1. Dataset and Annotations 

- Data & annotations can be found at [this](https://drive.google.com/drive/folders/1fkJ_1M5C4Xr0ppbDaHSABkKvg8zOD2XA?usp=sharing)
- Download and put them into `./data`

## 2. Tasks :sleepy:

### 2.1 Scanned Receipt Text Localisation

- Following these steps to train:

```
cd task1
python3 utils.py
python3 train.py

```

- And test: `python3 test.py`

### 2.2 Scanned Receipt OCR

- In this task, I used CRNN+CTCLoss for getting text from image. Therefore, I had to cut the given image to images and each of them just contains a single line of text. Save data in folder `./data/For_task_2`, like [this](https://drive.google.com/drive/folders/1BIdbIMDfeL69QymbsPQsv-xsc-klKbSG?usp=sharing) .Anyway, all you need is just run `python3 utils.py` 

- For trainning, make sure you are in `./task2`, just try `python3 train.py`

### 2.3 Key Information Extraction from Scanned Receipts


