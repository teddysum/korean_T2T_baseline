# Table-to-Text Baseline
본 소스 코드는 '국립국어원 인공 지능 언어 능력 평가' 시범 운영 과제 중 '표 기반 문장 생성' 과제 베이스라인 모델 및 학습과 평가를 위한 코드입니다.

학습 및 추론, 평가는 아래의 실행 방법(How to Run)에서 확인하실 수 있습니다.  
<br>
본 베이스라인 코드를 이용해서 pretrained model을 학습한 결과입니다.
|Model|ROUGE-1|BLUE|
|:---:|---|---|
|TeddyBART|0.4446|0.4398|
|KoBART|0.4147|0.4246|

## 디렉토리 구조(Directory Structure)
```
# 학습에 필요한 리소스들이 들어있습니다.
resource
├── data
└── tokenizer

# 실행 가능한 python 스크립트가 들어있습니다.
run
├── infernece.py
└── train.py

# 학습에 사용될 커스텀 함수들이 구현되어 있습니다.
src
├── data.py     # torch dataloader
├── module.py   # pytorch-lightning module
└── utils.py
```

## 데이터(Data)
### 제공 데이터
```
{
    "id": "nikluge-2022-table-dev-000001",
    "input": {
        "metadata": {
            "title": "안전·표시기준 위반 148개 생활화학제품 제조금지 등 조치",
            "date": "2020-12-29",
            "publisher": "환경부",
            "url": "...",
            "table_title": "안전확인대상생활화학제품 지정현황",
            "highlighted_cells": [[1, 1], [5, 1], [5, 2]]
        },
        "table": [
            [
                {"value": "구 분", "is_header": True, "row_span": "2", "column_span": "1"},
                {"value": "협약 사업장(톤, %)", "is_header": True, "row_span": 1, "column_span": "3"},
                {"value": "비협약 사업장(톤, %)", "is_header": True, "row_span": 1, "column_span": "3"}
            ],
            [
                {"value": "‘19.12", "is_header": True, "row_span": 1, "column_span": 1},
                .
                .
                .
                {"value": "58", "is_header": False, "row_span": 1, "column_span": 1},
                {"value": "58", "is_header": False, "row_span": 1, "column_span": 1},
                {"value": "0", "is_header": False, "row_span": 1, "column_span": 1}
            ]
        ]
    }
    "output": [
        "협약 사업장의 감축량은 4,571톤, 비협약 사업장의 감축량은 539톤이다.",
        "협약 사업장의 감축량은 4,571톤인데 비해 비협약 사업장의 감축량은 539톤에 그쳤다.",
        "굴뚝원격감시체계 설치 사업장 중 협약 사업장의 감축량은 4,571톤, 비협약 사업장의 감축량은 539톤으로 나타났다.",
        "굴뚝원격감시체계 설치 사업장의 오염물질 감축량은 협약 사업장 4,571톤, 비협약 사업장 539톤으로 나타났다."
    ]
}
```
데이터의 경우 입력 `"table"`, 출력 `"output"`이 됩니다. Baseline 모델에서 사용한 데이터 전처리 과정을 통하면 아래와 같은 형태가 됩니다.

### 데이터 전처리
```
{
    "table": [
        "구 분[TAB]협약 사업장(톤, %)[TAB]비협약 사업장(톤, %)[NL]‘19.12[TAB]...58[TAB]58[TAB]0",
        "구 분[TAB]협약 사업장(톤, %)[TAB]비협약 사업장(톤, %)[NL]‘19.12[TAB]...58[TAB]58[TAB]0",
        "구 분[TAB]협약 사업장(톤, %)[TAB]비협약 사업장(톤, %)[NL]‘19.12[TAB]...58[TAB]58[TAB]0",
        "구 분[TAB]협약 사업장(톤, %)[TAB]비협약 사업장(톤, %)[NL]‘19.12[TAB]...58[TAB]58[TAB]0"
    ],
    "text": [
        "협약 사업장의 감축량은 4,571톤, 비협약 사업장의 감축량은 539톤이다.",
        "협약 사업장의 감축량은 4,571톤인데 비해 비협약 사업장의 감축량은 539톤에 그쳤다.",
        "굴뚝원격감시체계 설치 사업장 중 협약 사업장의 감축량은 4,571톤, 비협약 사업장의 감축량은 539톤으로 나타났다.",
        "굴뚝원격감시체계 설치 사업장의 오염물질 감축량은 협약 사업장 4,571톤, 비협약 사업장 539톤으로 나타났다."
    ]
}
```
학습 데이터는 대부분 복수 정답으로 구성되어 있기 때문에, baseline 모델의 경우 위와 같이 중복된 입력 데이터를 이용했습니다.

## 설치(Installation)
Execute it, if mecab is not installed
```
./install_mecab.sh
```

Install python dependency
```
pip install -r requirements.txt
```

## 실행 방법(How to Run)
### 학습(Train)
```
python -m run train \
    --output-dir outputs/ttt \
    --tokenizer "resource/tokenizer/kobart-base-v2(ttt)" \
    --seed 42 --epoch 10 --gpus 4 --warmup-rate 0.1 \
    --max-learning-rate 2e-4 --min-learning-rate 1e-5 \
    --batch-size=32 --valid-batch-size=64 \
    --logging-interval 100 --evaluate-interval 1 \
    --wandb-project <wandb-project-name>
```
- 기본 모델은 `SKT-KoBART`를 이용합니다.
- BART 외의 pretrained model을 이용해서 학습하고 싶은 경우, 코드를 수정할 필요가 있습니다.
- 학습 로그 및 모델은 지정한 `output-dir`에 저장됩니다.

### 추론(Inference)
```
python -m run inference \
    --model-ckpt-path outputs/ttt/<your-model-ckpt-path> \
    --tokenizer "resource/tokenizer/kobart-base-v2(ttt)" \
    --output-path test_output.jsonl \
    --batch-size=64 \
    --summary-max-seq-len 512 \
    --num-beams 5 \
    --device cuda:2
```
- `transformers` 모델을 불러와 inference를 진행합니다.
- Inference 시 출력 데이터는 jsonl format으로 저장되며, "output"의 경우 입력 데이터와 다르게 `list`가 아닌 `string`이 됩니다.

### 채점(scoring)
```
python -m run scoring \
    --candidate-path <your-candidate-file-path>
```
- Inference output을 이용해 채점을 진행합니다.
- 기본적으로 Rouge-1과 BLEU를 제공합니다.

### Reference

huggingface/transformers (https://github.com/huggingface/transformers)  
SKT-AI/KoBART (https://github.com/SKT-AI/KoBART)  
국립국어원 모두의말뭉치 (https://corpus.korean.go.kr/)  
