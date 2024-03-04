# Birdirectional-LSTM Based Stock Prediction

## Introduction
이 연구는 Birdirectional-LSTM 을 기반하여 주가 데이터 분석 및 예측을 목적으로 진행하였습니다.
연구를 위해 KOSPI 상위 10가지 종목들에 대해서
크롤링 기술을 사용하여 종목 정보와 기업 관련 뉴스 기사를 수집하였습니다.
또한 수정 종가, Rolling과 Lagging, 기술적 분석, 감성 분석, Stacked Variational AutoEncoder 등 패턴 캡쳐 기술을 구성하고 있습니다.

## Requirements
- numpy
- pandas
- chromedriver_autoinstaller
- selenium
- bs4
- urllib
- Python-Io
- datetime
- FinanceDataReader
- pykrx
- Ta-lib
- typing
- xgboost
- statsmodels
- torch

## License
This repository is licensed under [Apache 2.0](https://github.com/paulms77/BiLSTM-StockPrediction-Algorithm/blob/main/LICENSE).

## Structure
```bash
└─ stock_prediction
    └─ stock_prediction
        ├─ setup.py
        └─ src
            ├─ stock_prediction_program
                ├─ __init__.py
                ├─ program_func.py
                ├─ my_data
                │   └─ *.csv
                ├─ my_news_data
                │   └─ *.csv
                ├─ my_package
                │   ├─ metrics.py
                │   ├─ model.py
                │   ├─ preprocessing.py
                │   ├─ train.py
                │   └─ visualization.py
                └─ my_path
                    ├─ bilstm
                    │  └─ *.pt
                    ├─ vae
                    │  └─ *.pt
                    └─ xgb
                       └─ *.joblib   
```
## Usage

## Performance
<img width="612" alt="image" src="https://github.com/paulms77/BiLSTM-StockPrediction-Algorithm/assets/69188065/5d692899-d1b6-4b20-a80d-58f9b0b527e5">

## Additional Development Plans
- 분석 기법 다양화
- 분석 결과 출력 시스템 개발
- 최신 트렌드 모델 도입
- 트레이딩 봇 서비스 1.0 출시
