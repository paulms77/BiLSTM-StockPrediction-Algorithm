import pandas as pd
import numpy as np
from talib import abstract
import talib as ta
from typing import Tuple
import re

def get_technical_indicators(dfs:pd.DataFrame(), symbol:str, value:int)->pd.DataFrame():
    """
        기술분석 수행
        Args:
            dfs (pd.DataFrame): 기술분석을 수행할 데이터
            symbol (str): 종목코드
            value (int): 결측치 처리 기능 타입
        Returns:
            pd.DataFrame: 기술분석을 수행 후 결과 데이터
    """
    temp = dfs[dfs.SYMBOL == symbol]
    O = temp.OPEN
    L = temp.LOW
    H = temp.HIGH
    C = temp.CLOSE
    V = temp.VOLUME
    A = temp.AMOUNT
    Period = pd.to_datetime(temp.DATE)

    # Cycle Indicator Functions
    indicator_names = ta.get_function_groups()['Cycle Indicators']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name == 'HT_PHASOR':
            result_df['INPHASE'], result_df['QUADRATURE'] = indicator_func(C)
        elif indicator_name == 'HT_SINE':
            result_df['SINE'], result_df['LEADSINE'] = indicator_func(C)
        else:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)

    # Math Operator Functions
    indicator_names = ['ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'MULT', 'SUB', 'SUM']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['ADD', 'DIV', 'MULT', 'SUB']:
            result_df[indicator_name] = indicator_func(H, L)
        elif indicator_name in ['MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'SUM']:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)

    # Momentum Indicator Functions
    indicator_names = ta.get_function_groups()['Momentum Indicators']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['BOP']:
            result_df[indicator_name] = indicator_func(O, H, L, C)
        elif indicator_name in ['MFI']:
            result_df[indicator_name] = indicator_func(H, L, C, V)
        elif indicator_name in ['ADX', 'ADXR', 'CCI', 'DX', 'MINUS_DI', 'PLUS_DI', 'ULTOSC', 'WILLR']:
            result_df[indicator_name] = indicator_func(H, L, C)
        elif indicator_name == 'STOCH':
            result_df['SLOWK'], result_df['SLOWD'] = indicator_func(H, L, C)
        elif indicator_name == 'STOCHF':
            result_df['FASTK'], result_df['FASTD'] = indicator_func(H, L, C)
        elif indicator_name in ['AROONOSC', 'MINUS_DM', 'PLUS_DM']:
            result_df[indicator_name] = indicator_func(H, L)
        elif indicator_name == 'AROON':
            result_df['ARROONDOWN'], result_df['AROONUP'] = indicator_func(H, L)
        elif indicator_name in ['MACD', 'MACDEXT', 'MACDFIX']:
            result_df[indicator_name], result_df[indicator_name+'SIGNAL'], result_df[indicator_name+'HIST'] = indicator_func(C)
        elif indicator_name == 'STOCHRSI':
            result_df[indicator_name+'FASTK'], result_df[indicator_name+'FASTD'] = indicator_func(C)
        else:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)
    
    # Overlap Studies Functions
    indicator_names = ta.get_function_groups()['Overlap Studies']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['MIDPRICE', 'SAR', 'SAREXT']:
            result_df[indicator_name] = indicator_func(H, L)
        elif indicator_name == 'BBANDS':
            result_df['UPPERBAND'], result_df['MIDDLEBAND'], result_df['LOWERBAND'] = indicator_func(C)
        elif indicator_name == 'MAVP':
            result_df[indicator_name] = indicator_func(C, Period)
        elif indicator_name == 'MAMA':
            result_df[indicator_name], result_df['FAMA'] = indicator_func(C)
        else:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)
    
    # Pattern Recognition Functions
    indicator_names = ta.get_function_groups()['Pattern Recognition']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        result_df[indicator_name] = indicator_func(O, H, L, C)
    temp = pd.concat([temp, result_df], axis=1)

    # Price Transform Functions
    indicator_names = ta.get_function_groups()['Price Transform']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name == 'AVGPRICE':
            result_df[indicator_name] = indicator_func(O, H, L, C)
        elif indicator_name in ['TYPPRICE', 'WCLPRICE']:
            result_df[indicator_name] = indicator_func(H, L, C)
        else:
            result_df[indicator_name] = indicator_func(H, L)
    temp = pd.concat([temp, result_df], axis=1)

    # Statistic Functions
    indicator_names = ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['BETA', 'CORREL']:
            result_df[indicator_name] = indicator_func(H, L)
        else:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)

    # Volatility Indicators
    indicator_names = ta.get_function_groups()['Volatility Indicators']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        result_df[indicator_name] = indicator_func(H, L, C)
    temp = pd.concat([temp, result_df], axis=1)

    # Volume Indicators
    indicator_names = ta.get_function_groups()['Volume Indicators']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['AD', 'ADOSC']:
            result_df[indicator_name] = indicator_func(H, L, C, V)
        else:
            result_df[indicator_name] = indicator_func(C, V)
    temp = pd.concat([temp, result_df], axis=1)

    # Add Feature Engineering
    # 이동 평균 (Rolling)
    timeperiods = [5, 6, 7, 10, 20, 21, 50, 60, 100, 120]
    for timeperiod in timeperiods:
        temp[f'MA{timeperiod}'] = C.rolling(window = timeperiod).mean()
        
    # 과거 종가 (Lagging)
    temp[f'CLOSE_BEFORE1'] = C.shift(-1)
    temp[f'CLOSE_BEFORE3'] = C.shift(-3)
    temp[f'CLOSE_BEFORE5'] = C.shift(-5)
    temp[f'CLOSE_BEFORE7'] = C.shift(-7)
    
    # 누적 거래대금
    temp['CUMULATIVE_AMOUNT'] = A.cumsum()
    
    # 누적 거래량
    temp['CUMULATIVE_VOLUME'] = V.cumsum()
    
    # HIGH/OPEN
    temp['HIGH/OPEN'] = H / O
    
    # LOW/OPEN
    temp['LOW/OPEN'] = L / O
    
    # CLOSE/OPEN
    temp['CLOSE/OPEN'] = C / O
    
    # CHG_PCT
    temp['CHG_PCT'] = ((C - C.shift(-1)) / C.shift(-1)) * 100

    # 결측치 처리 기능
    temp = temp.replace([np.inf, -np.inf], np.nan)
    if isinstance(value, int) and value == -1:
        temp = temp.fillna(method="ffill").fillna(method='bfill')
    else:
        temp = temp.fillna(method="ffill").fillna(value)
    
    return temp

def tokenizer_filter(text:str)->list:
    """
        커스텀 토크나이져 (Custom Tokenizer)
        Args:
            text (str): 토크나이져를 수행할 텍스트
        Returns:
            list: 토크나이져를 수행 후 결과 리스트
    """
    words = re.findall(r'\b\w+\b', text)
    filtered_words = [word for word in words if len(word) > 1 and not word.isdigit()]
    return filtered_words

def get_freq_word_set(grouped_news_temp:pd.DataFrame())->set:
    """
        자주 등장하는 단어 셋 추출
        Args:
            grouped_news_temp (pd.DataFrame): 토크나이져를 수행한 데이터
        Returns:
            Set: 단어의 빈도 수 가 50 이상인 단어 모음(셋)
    """
    all_words = [word for words in grouped_news_temp['FILTERED_TITLE'] for word in words]
    
    word_freq = pd.Series(all_words).value_counts().reset_index()
    
    word_freq.columns = ['WORD', 'FREQ']
    word_freq['PERC'] = (word_freq['FREQ'].rank(ascending = True) * 100)
    
    min_word_freq = 50 #word_freq.FREQ.value_counts().mean()
    filtered_word_freq = word_freq[word_freq['FREQ'] >= 50]
    filtered_word_set = set(filtered_word_freq['WORD'])

    return filtered_word_set

def get_emotional_score(final_temp:pd.DataFrame(), filtered_word_set:set)->pd.DataFrame():
    """
        감성점수 구하기
        Args:
            final_temp (pd.DataFrame): 전체 데이터 (날짜, 종목코드, 토크나이져가 수행된 뉴스 데이터, 수익률 정보가 포함된)
            filtered_word_set (Set): 자주 등장하는 단어 셋
        Returns:
            Set: 단어의 빈도 수 가 50 이상인 단어 모음(셋)
    """
    scores_by_date_symbol_word = {}
    for index, row in final_temp.iterrows():
        date = row['DATE']
        symbol = row['SYMBOL']
        words = row['FILTERED_TITLE']
        pct = row['CHG_PCT']
    
        #next_day_row = final_temp.loc[(final_temp['DATE'] > date) & (final_temp['SYMBOL'] == symbol)].head(1)
        #if not next_day_row.empty:
        #    pct = next_day_row.iloc[0]['CHG_PCT']
        #else:
        #    pct = 0
        
        for word in words:
            if word in filtered_word_set:
                if word not in scores_by_date_symbol_word:
                    scores_by_date_symbol_word[word] = {'total_pct': 0, 'word_count': 0}
                    
                scores_by_date_symbol_word[word]['total_pct'] += pct
                scores_by_date_symbol_word[word]['word_count'] += 1
    
    emotional_scores = []
    for index, row in final_temp.iterrows():
        words = row['FILTERED_TITLE']
        emotional_scores_by_word = {}
    
        if len(words) > 0:
            n_count = 0
            for word in words:
                if word in scores_by_date_symbol_word:
                    n_count += 1
                    emotional_scores_by_word[word] = scores_by_date_symbol_word[word]['total_pct'] / scores_by_date_symbol_word[word]['word_count']
                else:
                    emotional_scores_by_word[word] = 0
    
            emotional_scores.append(np.sum(emotional_scores_by_word[word]) / n_count)
        else:
            emotional_scores.append(0)
            
        
    final_temp['EMOTIONAL_SCORES'] = emotional_scores
    
    # 자주 등장하는 단어가 없는 경우 = 감성점수 NaN (결측치 0으로 처리)
    final_temp['EMOTIONAL_SCORES'] = final_temp['EMOTIONAL_SCORES'].fillna(0)
    
    # 'TITLE', 'FILTERED_TITLE' 처리 과정에서 생긴 불필요한 열 제거
    final_temp.drop(['TITLE', 'FILTERED_TITLE'], axis = 1, inplace = True)

    return final_temp

def train_test_split(dataset:pd.DataFrame(), close_name:str)->tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """
        Custom train test split (학습 데이터, 테스트 데이터 분리 기능)
        Args:
            dataset (pd.DataFrame): train test split를 수행할 데이터프레임
            close_name (Set): target에 해당하는 열 이름
        Returns:
            tuple, tuple: 학습 데이터 모음, 테스트 데이터 모음
    """
    dataset = dataset.copy()
    y = dataset[close_name]
    X = dataset.drop([close_name], axis = 1)
    train_samples = int(np.ceil(0.8 * X.shape[0]))
    
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]
    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)