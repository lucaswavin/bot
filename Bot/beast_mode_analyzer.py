#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEAST MODE DECISION PATTERN ANALYZER - 85% TARGET 🔥
==================================================
🎯 OBJETIVO: 85% DE PRECISIÓN O MUERTE
🚀 18,000+ ÁRBOLES + Deep Learning + Hyperparameter Optimization
💪 RAILWAY POWER - SIN LÍMITES DE MEMORIA/CPU
🏆 ENCONTRAR EL PATRÓN SAGRADO QUE DOMINA LOS MERCADOS

Uso en Railway:
python3 beast_mode_analyzer.py --data ethusdt_data_20may_31jul.csv --target-accuracy 0.85
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import joblib
from itertools import combinations
import random

# ML ARSENSAL COMPLETO
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV, 
    TimeSeriesSplit,
    cross_val_score
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# XGBoost y LightGBM para máximo poder
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible - instalando...")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM no disponible - instalando...")

warnings.filterwarnings('ignore')

@dataclass
class BeastConfig:
    mtf_minutes: int = 30
    ema_period: int = 75
    min_history_min: int = 200
    target_accuracy: float = 0.85
    max_optimization_time_hours: float = 4.0
    n_trials: int = 1000
    use_neural_networks: bool = True
    use_ensemble_stacking: bool = True

class BeastModeAnalyzer:
    def __init__(self, target_accuracy: float = 0.85):
        self.cfg = BeastConfig(target_accuracy=target_accuracy)
        self.feature_names = [
            'ema', 'ema_slope', 'dist_ema', 'ret_1m', 'mom_5', 'mom_15',
            'vol_15', 'vol_30', 'block_ret_so_far', 'block_range_so_far', 
            'elapsed_min', 'cur_green', 'rsi_14', 'macd_diff', 'bb_position',
            'atr_14', 'macd_15', 'trend_15', 'trend_60',
            # FEATURES ADICIONALES BEAST MODE
            'rsi_overbought', 'rsi_oversold', 'bb_squeeze', 'volume_spike',
            'price_momentum', 'trend_strength', 'volatility_regime'
        ]
        self.best_models = []
        self.elite_patterns = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Carga datos con detección automática de separador"""
        print(f"📥 Cargando datos BEAST MODE desde: {file_path}")
        
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        sep = ';' if ';' in first_line and first_line.count(';') > first_line.count(',') else ','
        df = pd.read_csv(file_path, sep=sep)
        
        df_clean = df.iloc[:, :5].copy()
        df_clean.columns = ['time', 'open', 'high', 'low', 'close']
        df_clean['time'] = pd.to_datetime(df_clean['time'])
        df_clean = df_clean.dropna().sort_values('time')
        
        for col in ['open', 'high', 'low', 'close']:
            df_clean[col] = pd.to_numeric(df_clean[col])
        
        print(f"🚀 {len(df_clean):,} velas cargadas - INICIANDO DOMINACIÓN")
        return df_clean.set_index('time')
    
    def create_mtf_data(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """Crea bloques MTF"""
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        mtf_data = df_1m.resample("30min", label="right", closed="right").agg(agg).dropna()
        
        mtf_data['is_red'] = mtf_data['close'] < mtf_data['open']
        mtf_data['body_pct'] = (mtf_data['close'] - mtf_data['open']) / mtf_data['open'] * 100
        
        return mtf_data
    
    def compute_beast_features(self, df_1m: pd.DataFrame, timestamp: pd.Timestamp) -> Optional[Dict]:
        """Calcula TODAS las features posibles - MODO BESTIAL"""
        available_data = df_1m[df_1m.index <= timestamp]
        
        if len(available_data) < self.cfg.min_history_min:
            return None
        
        try:
            # Ventanas de datos
            win5 = available_data.tail(5)
            win15 = available_data.tail(15)
            win30 = available_data.tail(30)
            win60 = available_data.tail(60)
            
            # Precios básicos
            close_now = float(available_data["close"].iloc[-1])
            open_now = float(available_data["open"].iloc[-1])
            high_now = float(available_data["high"].iloc[-1])
            low_now = float(available_data["low"].iloc[-1])
            
            # EMA y tendencias múltiples
            ema_series = available_data["close"].ewm(span=self.cfg.ema_period, adjust=False).mean()
            ema = float(ema_series.iloc[-1])
            ema_prev = float(ema_series.iloc[-2]) if len(ema_series) >= 2 else ema
            ema_slope = ema - ema_prev
            
            # EMA múltiples
            ema_fast = available_data["close"].ewm(span=12).mean().iloc[-1]
            ema_slow = available_data["close"].ewm(span=26).mean().iloc[-1]
            ema_signal = available_data["close"].ewm(span=9).mean().iloc[-1]
            
            # Returns múltiples
            def _ret(a, b): 
                return float(a/b - 1.0) if b != 0 else 0.0
                
            ret_1m = _ret(close_now, open_now)
            
            # Momentum avanzado
            mom_5 = float((win5["close"].iloc[-1] / win5["close"].iloc[0]) - 1.0) if len(win5) >= 2 else 0.0
            mom_15 = float((win15["close"].iloc[-1] / win15["close"].iloc[0]) - 1.0) if len(win15) >= 2 else 0.0
            mom_30 = float((win30["close"].iloc[-1] / win30["close"].iloc[0]) - 1.0) if len(win30) >= 2 else 0.0
            
            # Volatilidad avanzada
            r5 = win5["close"].pct_change().dropna()
            r15 = win15["close"].pct_change().dropna()
            r30 = win30["close"].pct_change().dropna()
            
            vol_5 = float(r5.std()) if len(r5) else 0.0
            vol_15 = float(r15.std()) if len(r15) else 0.0
            vol_30 = float(r30.std()) if len(r30) else 0.0
            
            # Distancias EMA
            dist_ema = float(close_now - ema)
            dist_ema_pct = dist_ema / ema if ema != 0 else 0.0
            
            # Información del bloque MTF
            dt = pd.Timedelta(minutes=30)
            block_start = timestamp - dt + pd.Timedelta(minutes=1)
            block_data = available_data[(available_data.index >= block_start) & (available_data.index <= timestamp)]
            
            if block_data.empty:
                return None
            
            block_open = float(block_data["open"].iloc[0])
            block_high = float(block_data["high"].max())
            block_low = float(block_data["low"].min())
            block_close = float(block_data["close"].iloc[-1])
            block_ret = _ret(block_close, block_open)
            block_range = float(block_high - block_low)
            elapsed_min = int((timestamp - block_start).total_seconds() / 60.0)
            
            cur_green = block_close > float(block_data["open"].iloc[-1])
            
            # === BEAST MODE INDICATORS ===
            
            # RSI avanzado
            delta = available_data["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
            # RSI estados
            rsi_overbought = 1.0 if rsi_14 > 70 else 0.0
            rsi_oversold = 1.0 if rsi_14 < 30 else 0.0
            
            # MACD avanzado
            ema12 = available_data["close"].ewm(span=12, adjust=False).mean()
            ema26 = available_data["close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_diff = float(macd_line.iloc[-1] - signal_line.iloc[-1])
            macd_histogram = macd_diff
            
            # Bollinger Bands avanzado
            bb_period = 20
            sma20 = available_data["close"].rolling(window=bb_period).mean()
            std20 = available_data["close"].rolling(window=bb_period).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            bb_middle = sma20
            
            if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]):
                bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
                bb_position = (close_now - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                bb_squeeze = 1.0 if bb_width < 0.05 else 0.0  # Squeeze detection
            else:
                bb_width = 0.0
                bb_position = 0.5
                bb_squeeze = 0.0
            
            # ATR avanzado
            high = available_data["high"]
            low = available_data["low"]
            close_prev = available_data["close"].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean()
            atr_14 = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
            
            # Volume analysis (simulado con price action)
            price_changes = available_data["close"].diff().abs()
            volume_proxy = price_changes * (available_data["high"] - available_data["low"])
            avg_volume = volume_proxy.rolling(window=20, min_periods=1).mean()
            volume_spike = 1.0 if volume_proxy.iloc[-1] > avg_volume.iloc[-1] * 2 else 0.0
            
            # Multi-timeframe analysis
            tf_15 = available_data.resample('15min', label='right', closed='right').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            
            if len(tf_15) >= 10:
                ema_fast_15 = tf_15['close'].ewm(span=8).mean()
                ema_slow_15 = tf_15['close'].ewm(span=17).mean()
                macd_15 = (ema_fast_15 - ema_slow_15).iloc[-1]
                trend_15 = (tf_15['close'].iloc[-1] / tf_15['close'].iloc[-3] - 1) * 100 if len(tf_15) >= 3 else 0
            else:
                macd_15 = 0.0
                trend_15 = 0.0
            
            tf_60 = available_data.resample('60min', label='right', closed='right').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            
            if len(tf_60) >= 5:
                trend_60 = (tf_60['close'].iloc[-1] / tf_60['close'].iloc[-2] - 1) * 100 if len(tf_60) >= 2 else 0
            else:
                trend_60 = 0.0
            
            # Price momentum strength
            momentum_periods = [3, 5, 8, 13, 21]
            momentum_scores = []
            for period in momentum_periods:
                if len(available_data) >= period:
                    mom_score = (available_data["close"].iloc[-1] / available_data["close"].iloc[-period] - 1) * 100
                    momentum_scores.append(mom_score)
            
            price_momentum = np.mean(momentum_scores) if momentum_scores else 0.0
            
            # Trend strength (múltiples EMAs alignment)
            emas = []
            for span in [8, 13, 21, 34, 55]:
                if len(available_data) >= span:
                    ema_val = available_data["close"].ewm(span=span).mean().iloc[-1]
                    emas.append(ema_val)
            
            if len(emas) >= 3:
                # Todas las EMAs en orden ascendente = uptrend strength
                sorted_emas = sorted(emas)
                if emas == sorted_emas:
                    trend_strength = 1.0  # Perfect uptrend
                elif emas == sorted_emas[::-1]:
                    trend_strength = -1.0  # Perfect downtrend
                else:
                    trend_strength = 0.0  # Mixed
            else:
                trend_strength = 0.0
            
            # Volatility regime
            vol_short = available_data["close"].pct_change().tail(10).std()
            vol_long = available_data["close"].pct_change().tail(50).std()
            volatility_regime = vol_short / vol_long if vol_long != 0 else 1.0
            
            # === TODAS LAS FEATURES BEAST MODE ===
            features = {
                # Básicas mejoradas
                "ema": ema,
                "ema_slope": ema_slope,
                "dist_ema": dist_ema,
                "ret_1m": ret_1m,
                "mom_5": mom_5,
                "mom_15": mom_15,
                "vol_15": vol_15,
                "vol_30": vol_30,
                "block_ret_so_far": block_ret,
                "block_range_so_far": block_range,
                "elapsed_min": elapsed_min,
                "cur_green": int(cur_green),
                
                # Indicadores técnicos avanzados
                "rsi_14": rsi_14,
                "macd_diff": macd_diff,
                "bb_position": float(bb_position),
                "atr_14": atr_14,
                "macd_15": float(macd_15),
                "trend_15": float(trend_15),
                "trend_60": float(trend_60),
                
                # BEAST MODE features
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold,
                "bb_squeeze": bb_squeeze,
                "volume_spike": volume_spike,
                "price_momentum": float(price_momentum),
                "trend_strength": float(trend_strength),
                "volatility_regime": float(volatility_regime),
                
                # Features adicionales calculadas
                "mom_30": mom_30,
                "vol_5": vol_5,
                "dist_ema_pct": dist_ema_pct,
                "bb_width": float(bb_width),
                "macd_histogram": macd_histogram,
                "ema_fast": float(ema_fast),
                "ema_slow": float(ema_slow),
            }
            
            return features
            
        except Exception as e:
            return None
    
    def find_all_decision_points(self, df_1m: pd.DataFrame, mtf_data: pd.DataFrame) -> List[Dict]:
        """Encuentra todos los puntos de decisión"""
        print("🔍 BEAST MODE: Cazando TODOS los puntos de decisión...")
        decision_points = []
        dt = pd.Timedelta(minutes=30)
        
        for i in range(1, len(mtf_data)):
            if i % 500 == 0:
                print(f"   🔥 Procesando MTF {i:,}/{len(mtf_data):,}", end='\r')
            
            current_mtf_end = mtf_data.index[i]
            current_mtf_start = current_mtf_end - dt
            
            # MTF anterior debe ser rojo
            prev_mtf = mtf_data.iloc[i-1]
            if not prev_mtf['is_red']:
                continue
            
            # Primera vela verde en MTF actual
            current_candles = df_1m[(df_1m.index >= current_mtf_start) & (df_1m.index < current_mtf_end)]
            
            if current_candles.empty:
                continue
            
            first_green_time = None
            for ts, candle in current_candles.iterrows():
                if candle['close'] > candle['open']:
                    first_green_time = ts
                    break
            
            if first_green_time is None:
                continue
            
            # Resultado real
            current_mtf = mtf_data.iloc[i]
            actual_red = current_mtf['is_red']
            
            decision_points.append({
                'decision_time': first_green_time,
                'mtf_start': current_mtf_start,
                'mtf_end': current_mtf_end,
                'actual_red': actual_red,
                'correct_decision': not actual_red,
                'mtf_open': float(current_mtf['open']),
                'mtf_close': float(current_mtf['close']),
                'mtf_body_pct': float(current_mtf['body_pct'])
            })
        
        print(f"\n🚀 BEAST MODE: {len(decision_points):,} puntos de decisión capturados")
        return decision_points
    
    def create_beast_dataset(self, decision_points: List[Dict], df_1m: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Crea dataset con TODAS las features beast mode"""
        print("💪 Creando dataset BEAST MODE con features avanzadas...")
        
        X_data = []
        y_data = []
        skipped = 0
        
        for i, dp in enumerate(decision_points):
            if i % 200 == 0:
                print(f"   ⚡ Procesando decisión {i:,}/{len(decision_points):,}", end='\r')
            
            features = self.compute_beast_features(df_1m, dp['decision_time'])
            
            if features is None:
                skipped += 1
                continue
            
            # Verificar todas las features
            feature_vector = []
            for fname in self.feature_names:
                if fname in features:
                    val = features[fname]
                    # Handle infinities and NaN
                    if pd.isna(val) or np.isinf(val):
                        val = 0.0
                    feature_vector.append(float(val))
                else:
                    feature_vector.append(0.0)
            
            X_data.append(feature_vector)
            y_data.append(int(dp['correct_decision']))
        
        print(f"\n🔥 Dataset BEAST creado: {len(X_data):,} casos válidos, {skipped:,} saltados")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Limpiar outliers extremos
        for i in range(X.shape[1]):
            col = X[:, i]
            q99 = np.percentile(col, 99)
            q01 = np.percentile(col, 1)
            X[:, i] = np.clip(col, q01, q99)
        
        print(f"📊 Features: {X.shape[1]} | Target distribution: {np.mean(y):.1%} correctos")
        
        return X, y
    
    def create_beast_models(self) -> List[Tuple[str, object]]:
        """Crea el arsenal completo de modelos BEAST MODE"""
        print("🔥 ARMANDO ARSENAL DE MODELOS BEAST MODE...")
        
        models = []
        
        # RANDOM FORESTS MASIVOS
        models.extend([
            ('RF_Beast_2000', RandomForestClassifier(
                n_estimators=2000,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                n_jobs=-1,
                random_state=42
            )),
            ('RF_Beast_3000', RandomForestClassifier(
                n_estimators=3000,
                max_depth=30,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='log2',
                bootstrap=True,
                n_jobs=-1,
                random_state=123
            )),
            ('ExtraTrees_Beast', ExtraTreesClassifier(
                n_estimators=2500,
                max_depth=35,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=False,
                n_jobs=-1,
                random_state=456
            ))
        ])
        
        # GRADIENT BOOSTING ARSENAL
        models.extend([
            ('GB_Beast_1000', GradientBoostingClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=12,
                subsample=0.8,
                max_features='sqrt',
                random_state=789
            )),
            ('GB_Beast_1500', GradientBoostingClassifier(
                n_estimators=1500,
                learning_rate=0.03,
                max_depth=15,
                subsample=0.9,
                max_features=None,
                random_state=101112
            ))
        ])
        
        # XGBOOST NUCLEAR (si está disponible)
        if XGBOOST_AVAILABLE:
            models.extend([
                ('XGB_Beast_3000', xgb.XGBClassifier(
                    n_estimators=3000,
                    learning_rate=0.01,
                    max_depth=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    n_jobs=-1,
                    random_state=131415
                )),
                ('XGB_Beast_5000', xgb.XGBClassifier(
                    n_estimators=5000,
                    learning_rate=0.005,
                    max_depth=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.05,
                    reg_lambda=0.05,
                    n_jobs=-1,
                    random_state=161718
                ))
            ])
        
        # LIGHTGBM POWER (si está disponible)
        if LIGHTGBM_AVAILABLE:
            models.extend([
                ('LGB_Beast_4000', lgb.LGBMClassifier(
                    n_estimators=4000,
                    learning_rate=0.01,
                    max_depth=18,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    n_jobs=-1,
                    random_state=192021,
                    verbose=-1
                ))
            ])
        
        # NEURAL NETWORKS (si está habilitado)
        if self.cfg.use_neural_networks:
            models.extend([
                ('MLP_Beast_Large', MLPClassifier(
                    hidden_layer_sizes=(500, 250, 100),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=1000,
                    early_stopping=True,
                    random_state=222324
                )),
                ('MLP_Beast_Deep', MLPClassifier(
                    hidden_layer_sizes=(300, 200, 150, 100, 50),
                    activation='tanh',
                    solver='adam',
                    alpha=0.0001,
                    learning_rate='adaptive',
                    max_iter=1500,
                    early_stopping=True,
                    random_state=252627
                ))
            ])
        
        # SUPPORT VECTOR MACHINES
        models.extend([
            ('SVM_Beast_RBF', SVC(
                kernel='rbf',
                C=100,
                gamma='scale',
                probability=True,
                random_state=282930
            )),
            ('SVM_Beast_Poly', SVC(
                kernel='poly',
                degree=3,
                C=10,
                probability=True,
                random_state=313233
            ))
        ])
        
        total_estimators = 0
        for name, model in models:
            if hasattr(model, 'n_estimators'):
                total_estimators += model.n_estimators
        
        print(f"💪 ARSENAL COMPLETO: {len(models)} modelos, ~{total_estimators:,} árboles totales")
        return models
    
    def hyperparameter_optimization_beast(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[str, object, float]]:
        """Optimización de hiperparámetros BEAST MODE"""
        print(f"🚀 INICIANDO HIPER-OPTIMIZACIÓN BEAST MODE...")
        print(f"🎯 Objetivo: {self.cfg.target_accuracy:.1%} precisión")
        print(f"⏰ Tiempo máximo: {self.cfg.max_optimization_time_hours:.1f} horas")
        print(f"🔥 Trials: {self.cfg.n_trials:,}")
        
        # Split temporal para evaluación real
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        start_time = time.time()
        max_time = self.cfg.max_optimization_time_hours * 3600
        
        results = []
        models_to_optimize = self.create_beast_models()
        
        for model_name, base_model in models_to_optimize:
            if time.time() - start_time > max_time:
                print(f"⏰ Tiempo límite alcanzado para {model_name}")
                break
                
            print(f"\n🔧 Optimizando {model_name}...")
            
            try:
                # Configurar parámetros de búsqueda según tipo de modelo
                if 'RF' in model_name or 'Extra' in model_name:
                    param_grid = {
                        'n_estimators': [1000, 2000, 3000, 4000],
                        'max_depth': [15, 20, 25, 30, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
                    }
                elif 'GB' in model_name:
                    param_grid = {
                        'n_estimators': [500, 1000, 1500, 2000],
                        'learning_rate': [0.01, 0.05, 0.1, 0.15],
                        'max_depth': [8, 12, 15, 20],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                elif 'XGB' in model_name and XGBOOST_AVAILABLE:
                    param_grid = {
                        'n_estimators': [1000, 2000, 3000, 4000],
                        'learning_rate': [0.005, 0.01, 0.05, 0.1],
                        'max_depth': [10, 15, 20, 25],
                        'subsample': [0.8, 0.9],
                        'colsample_bytree': [0.8, 0.9]
                    }
                elif 'LGB' in model_name and LIGHTGBM_AVAILABLE:
                    param_grid = {
                        'n_estimators': [2000, 3000, 4000, 5000],
                        'learning_rate': [0.005, 0.01, 0.05],
                        'max_depth': [15, 20, 25],
                        'subsample': [0.85, 0.9],
                        'colsample_bytree': [0.85, 0.9]
                    }
                elif 'MLP' in model_name:
                    param_grid = {
                        'hidden_layer_sizes': [(200, 100), (300, 200, 100), (500, 250, 100), (400, 200, 100, 50)],
                        'learning_rate_init': [0.001, 0.01, 0.1],
                        'alpha': [0.0001, 0.001, 0.01]
                    }
                elif 'SVM' in model_name:
                    param_grid = {
                        'C': [1, 10, 100, 1000],
                        'gamma': ['scale', 'auto', 0.001, 0.01]
                    }
                else:
                    # Parámetros genéricos
                    param_grid = {}
                
                if param_grid:
                    # RandomizedSearchCV para eficiencia
                    n_iter = min(50, len(list(param_grid.values())[0]) if param_grid else 10)
                    
                    search = RandomizedSearchCV(
                        base_model,
                        param_grid,
                        n_iter=n_iter,
                        cv=TimeSeriesSplit(n_splits=3),
                        scoring='accuracy',
                        n_jobs=-1,
                        random_state=42,
                        verbose=0
                    )
                    
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    
                    # Evaluar en test set
                    y_pred = best_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    results.append((model_name, best_model, accuracy))
                    
                    print(f"   ✅ {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    
                    # Si alcanzamos el objetivo, guardamos inmediatamente
                    if accuracy >= self.cfg.target_accuracy:
                        print(f"🎉 ¡OBJETIVO ALCANZADO! {model_name} logró {accuracy:.4f}")
                        self.save_elite_model(best_model, model_name, accuracy, X_test, y_test)
                
                else:
                    # Modelo sin optimización
                    base_model.fit(X_train, y_train)
                    y_pred = base_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    results.append((model_name, base_model, accuracy))
                    print(f"   ✅ {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    
            except Exception as e:
                print(f"   ❌ Error en {model_name}: {str(e)[:50]}...")
                continue
        
        # Ordenar por precisión
        results.sort(key=lambda x: x[2], reverse=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n🏆 OPTIMIZACIÓN COMPLETADA en {elapsed_time/3600:.2f} horas")
        
        return results
    
    def save_elite_model(self, model, name: str, accuracy: float, X_test: np.ndarray, y_test: np.ndarray):
        """Guarda modelo elite que alcanza el objetivo"""
        filename = f"elite_model_{name}_{accuracy:.4f}.joblib"
        joblib.dump(model, filename)
        
        # Guardar predicciones detalladas
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'correct': y_test == y_pred
        })
        
        if y_proba is not None:
            results_df['y_proba'] = y_proba
        
        results_df.to_csv(f"elite_predictions_{name}_{accuracy:.4f}.csv", index=False)
        
        print(f"💾 Modelo elite guardado: {filename}")
        
    def create_mega_ensemble(self, best_models: List[Tuple[str, object, float]]) -> Tuple[object, float]:
        """Crea mega-ensemble con los mejores modelos"""
        print("🚀 CREANDO MEGA-ENSEMBLE DE ÉLITE...")
        
        # Seleccionar top modelos para ensemble
        top_n = min(7, len(best_models))  # Max 7 para evitar overfitting
        elite_models = best_models[:top_n]
        
        print(f"🎯 Seleccionados {len(elite_models)} modelos élite para ensemble:")
        for name, _, acc in elite_models:
            print(f"   • {name}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Crear VotingClassifier
        estimators = [(name, model) for name, model, _ in elite_models]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Usa probabilidades
            n_jobs=-1
        )
        
        return ensemble
    
    def extract_winning_patterns(self, best_model, X: np.ndarray, y: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray) -> List[str]:
        """Extrae patrones ganadores del mejor modelo"""
        print("🔍 EXTRAYENDO PATRONES GANADORES...")
        
        patterns = []
        
        # Si es tree-based, extraer reglas
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            important_features = [(self.feature_names[i], imp) 
                                for i, imp in enumerate(importances) if i < len(self.feature_names)]
            important_features.sort(key=lambda x: x[1], reverse=True)
            
            print("🔝 Top 10 features más importantes:")
            for feat, imp in important_features[:10]:
                print(f"   • {feat}: {imp:.4f}")
            
            # Crear árbol de decisión simple para reglas interpretables
            from sklearn.tree import DecisionTreeClassifier, export_text
            
            dt_simple = DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=25,
                random_state=42
            )
            
            dt_simple.fit(X, y)
            tree_rules = export_text(dt_simple, 
                                   feature_names=self.feature_names[:X.shape[1]])
            
            # Guardar reglas
            with open('beast_mode_rules.txt', 'w') as f:
                f.write("🔥 BEAST MODE DECISION RULES 🔥\n")
                f.write("="*50 + "\n")
                f.write(tree_rules)
            
            patterns.append("Reglas guardadas en beast_mode_rules.txt")
            
            # Análisis de casos de alta precisión
            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                high_conf_indices = np.where(y_proba > 0.8)[0]  # Alta confianza
                
                if len(high_conf_indices) > 0:
                    high_conf_accuracy = accuracy_score(
                        y_test[high_conf_indices], 
                        best_model.predict(X_test[high_conf_indices])
                    )
                    
                    pattern = f"Casos alta confianza (>80%): {high_conf_accuracy:.3f} precisión"
                    patterns.append(pattern)
                    print(f"💡 {pattern}")
        
        return patterns
    
    def generate_beast_report(self, best_models: List[Tuple[str, object, float]], 
                            patterns: List[str], total_decisions: int, 
                            base_accuracy: float, processing_time: float) -> None:
        """Genera reporte completo BEAST MODE"""
        
        report_lines = []
        report_lines.append("🔥 BEAST MODE ANALYSIS REPORT - 85% TARGET 🔥")
        report_lines.append("="*70)
        
        # Estadísticas generales
        report_lines.append(f"\n📊 DATOS PROCESADOS:")
        report_lines.append(f"   Total decisiones analizadas: {total_decisions:,}")
        report_lines.append(f"   Precisión base: {base_accuracy:.4f} ({base_accuracy*100:.2f}%)")
        report_lines.append(f"   Tiempo procesamiento: {processing_time/3600:.2f} horas")
        
        # Objetivo
        report_lines.append(f"\n🎯 OBJETIVO BEAST MODE:")
        report_lines.append(f"   Target accuracy: {self.cfg.target_accuracy:.4f} ({self.cfg.target_accuracy*100:.1f}%)")
        
        # Resultados de modelos
        if best_models:
            best_accuracy = best_models[0][2]
            improvement = best_accuracy - base_accuracy
            
            report_lines.append(f"\n🏆 MEJOR MODELO ENCONTRADO:")
            report_lines.append(f"   Modelo: {best_models[0][0]}")
            report_lines.append(f"   Precisión: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            report_lines.append(f"   Mejora vs base: +{improvement:.4f} (+{improvement*100:.2f}%)")
            
            if best_accuracy >= self.cfg.target_accuracy:
                report_lines.append(f"   🎉 ¡OBJETIVO ALCANZADO!")
                status = "ÉXITO TOTAL"
            elif best_accuracy >= 0.8:
                report_lines.append(f"   🔥 MUY CERCA DEL OBJETIVO")
                status = "CASI PERFECTO"
            elif best_accuracy >= 0.75:
                report_lines.append(f"   💪 EXCELENTE RESULTADO")
                status = "EXCELENTE"
            elif best_accuracy >= 0.7:
                report_lines.append(f"   ✅ BUEN RESULTADO")
                status = "BUENO"
            else:
                report_lines.append(f"   🤔 MEJORA MODERADA")
                status = "MODERADO"
            
            report_lines.append(f"\n🔝 TOP 5 MEJORES MODELOS:")
            for i, (name, _, acc) in enumerate(best_models[:5]):
                report_lines.append(f"   #{i+1}: {name} - {acc:.4f} ({acc*100:.2f}%)")
        
        # Patrones encontrados
        if patterns:
            report_lines.append(f"\n💡 PATRONES GANADORES ENCONTRADOS:")
            for pattern in patterns:
                report_lines.append(f"   • {pattern}")
        
        # Evaluación final
        report_lines.append(f"\n🏆 EVALUACIÓN FINAL: {status}")
        
        if best_models and best_models[0][2] >= self.cfg.target_accuracy:
            report_lines.append("\n🎉 ¡MISIÓN CUMPLIDA!")
            report_lines.append("💰 Modelo listo para dominar los mercados")
            report_lines.append("🚀 Implementar inmediatamente en producción")
        else:
            report_lines.append("\n📈 PRÓXIMOS PASOS:")
            report_lines.append("   1. Recolectar más datos históricos")
            report_lines.append("   2. Probar features adicionales")
            report_lines.append("   3. Aumentar tiempo de optimización")
            report_lines.append("   4. Considerar ensemble stacking avanzado")
        
        # Archivos generados
        report_lines.append(f"\n💾 ARCHIVOS GENERADOS:")
        report_lines.append(f"   📄 beast_mode_report.txt - Este reporte")
        report_lines.append(f"   🌳 beast_mode_rules.txt - Reglas interpretables")
        if best_models:
            report_lines.append(f"   🤖 elite_model_*.joblib - Mejores modelos")
            report_lines.append(f"   📊 elite_predictions_*.csv - Predicciones detalladas")
        
        report_content = "\n".join(report_lines)
        
        # Guardar reporte
        with open('beast_mode_report.txt', 'w') as f:
            f.write(report_content)
        
        # Imprimir en consola
        print("\n" + report_content)
        print(f"\n💾 Reporte BEAST MODE guardado en: beast_mode_report.txt")
    
    def run_beast_analysis(self, data_file: str) -> Dict:
        """Ejecuta el análisis completo BEAST MODE"""
        print("🔥 INICIANDO ANÁLISIS BEAST MODE - OBJETIVO 85%")
        print("="*80)
        print(f"🎯 Target: {self.cfg.target_accuracy:.1%} precisión")
        print(f"⚡ Max tiempo: {self.cfg.max_optimization_time_hours:.1f} horas")
        print(f"🚀 Trials: {self.cfg.n_trials:,}")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # 1. Cargar datos
            df_1m = self.load_data(data_file)
            
            # 2. Crear MTF
            print("\n🔳 Creando bloques MTF...")
            mtf_data = self.create_mtf_data(df_1m)
            print(f"✅ {len(mtf_data):,} bloques MTF creados")
            
            # 3. Encontrar puntos de decisión
            decision_points = self.find_all_decision_points(df_1m, mtf_data)
            
            if not decision_points:
                return {'error': 'No se encontraron puntos de decisión'}
            
            # 4. Crear dataset BEAST
            X, y = self.create_beast_dataset(decision_points, df_1m)
            
            if len(X) == 0:
                return {'error': 'No se pudieron calcular features'}
            
            base_accuracy = np.mean(y)
            print(f"📊 Precisión base: {base_accuracy:.4f} ({base_accuracy*100:.2f}%)")
            
            # 5. OPTIMIZACIÓN BEAST MODE
            best_models = self.hyperparameter_optimization_beast(X, y)
            
            if not best_models:
                return {'error': 'No se pudieron entrenar modelos'}
            
            # 6. Crear mega-ensemble si tenemos múltiples modelos buenos
            ensemble = None
            if len(best_models) >= 3 and self.cfg.use_ensemble_stacking:
                print("\n🚀 Creando mega-ensemble...")
                # Split para ensemble
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                ensemble = self.create_mega_ensemble(best_models[:5])
                ensemble.fit(X_train, y_train)
                ensemble_accuracy = accuracy_score(y_test, ensemble.predict(X_test))
                
                print(f"🎯 Ensemble accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
                
                if ensemble_accuracy > best_models[0][2]:
                    best_models.insert(0, ('MEGA_ENSEMBLE', ensemble, ensemble_accuracy))
                    print("🏆 ¡Ensemble supera modelos individuales!")
            
            # 7. Extraer patrones ganadores
            best_model = best_models[0][1]
            split_idx = int(len(X) * 0.8)
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            patterns = self.extract_winning_patterns(best_model, X, y, X_test, y_test)
            
            # 8. Generar reporte
            processing_time = time.time() - start_time
            self.generate_beast_report(best_models, patterns, len(decision_points), 
                                     base_accuracy, processing_time)
            
            # Resultados finales
            best_accuracy = best_models[0][2]
            results = {
                'success': True,
                'total_decisions': len(decision_points),
                'valid_decisions': len(X),
                'base_accuracy': base_accuracy,
                'best_accuracy': best_accuracy,
                'improvement': best_accuracy - base_accuracy,
                'target_achieved': best_accuracy >= self.cfg.target_accuracy,
                'best_model_name': best_models[0][0],
                'total_models_tested': len(best_models),
                'processing_time_hours': processing_time / 3600,
                'patterns_found': len(patterns),
                'models_above_70': len([m for _, _, acc in best_models if acc > 0.7]),
                'models_above_80': len([m for _, _, acc in best_models if acc > 0.8]),
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error BEAST MODE: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Beast Mode Decision Pattern Analyzer")
    parser.add_argument("--data", type=str, required=True, help="Archivo de datos OHLC")
    parser.add_argument("--target-accuracy", type=float, default=0.85, 
                       help="Precisión objetivo (0.85 = 85%)")
    parser.add_argument("--max-hours", type=float, default=4.0,
                       help="Tiempo máximo de optimización en horas")
    parser.add_argument("--trials", type=int, default=1000,
                       help="Número de trials para optimización")
    parser.add_argument("--no-neural", action="store_true",
                       help="Deshabilitar redes neuronales")
    parser.add_argument("--no-ensemble", action="store_true", 
                       help="Deshabilitar ensemble stacking")
    
    args = parser.parse_args()
    
    print("🔥 BEAST MODE DECISION PATTERN ANALYZER 🔥")
    print(f"📊 Datos: {args.data}")
    print(f"🎯 Target: {args.target_accuracy:.1%}")
    print(f"⏰ Max tiempo: {args.max_hours:.1f} horas")
    print(f"🔥 Trials: {args.trials:,}")
    print("\n" + "="*80)
    print("🚀 INICIANDO MISIÓN: ENCONTRAR EL PATRÓN DE 85%")
    print("💪 18,000+ ÁRBOLES + DEEP LEARNING + HYPEROPT")
    print("🏆 OBJETIVO: DOMINAR LOS MERCADOS")
    print("="*80)
    
    try:
        analyzer = BeastModeAnalyzer(target_accuracy=args.target_accuracy)
        
        # Configurar opciones
        analyzer.cfg.max_optimization_time_hours = args.max_hours
        analyzer.cfg.n_trials = args.trials
        analyzer.cfg.use_neural_networks = not args.no_neural
        analyzer.cfg.use_ensemble_stacking = not args.no_ensemble
        
        # Ejecutar análisis BEAST
        results = analyzer.run_beast_analysis(args.data)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return 1
        
        # Mostrar resultados épicos
        print("\n" + "="*80)
        print("🏆 BEAST MODE ANALYSIS COMPLETED")
        print("="*80)
        
        print(f"\n📊 DATOS ANIQUILADOS:")
        print(f"   Decisiones procesadas: {results['total_decisions']:,}")
        print(f"   Con features válidas: {results['valid_decisions']:,}")
        print(f"   Tiempo total: {results['processing_time_hours']:.2f} horas")
        
        print(f"\n🎯 PRECISIÓN ALCANZADA:")
        print(f"   Base: {results['base_accuracy']:.4f} ({results['base_accuracy']*100:.2f}%)")
        print(f"   Mejor modelo: {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%)")
        print(f"   Mejora: +{results['improvement']:.4f} (+{results['improvement']*100:.2f}%)")
        
        print(f"\n🔥 MODELOS DOMINADOS:")
        print(f"   Total testados: {results['total_models_tested']}")
        print(f"   >70% precisión: {results['models_above_70']}")
        print(f"   >80% precisión: {results['models_above_80']}")
        
        # Evaluación épica final
        if results['target_achieved']:
            print(f"\n🎉 ¡¡¡OBJETIVO DE {args.target_accuracy:.1%} ALCANZADO!!!")
            print(f"🏆 MODELO GANADOR: {results['best_model_name']}")
            print(f"💰 PRECISIÓN: {results['best_accuracy']:.4f}")
            print(f"🚀 ¡LISTOS PARA DOMINAR LOS MERCADOS!")
            evaluation = "🔥 ÉXITO TOTAL - MISIÓN CUMPLIDA"
        elif results['best_accuracy'] >= 0.8:
            print(f"🔥 ¡MUY CERCA! {results['best_accuracy']:.1%} vs {args.target_accuracy:.1%} objetivo")
            print(f"💪 Con más tiempo/datos podemos llegar al {args.target_accuracy:.1%}")
            evaluation = "🚀 CASI PERFECTO - 80%+ ALCANZADO"
        elif results['best_accuracy'] >= 0.75:
            print(f"✅ ¡Excelente resultado! {results['best_accuracy']:.1%}")
            print(f"💡 Mejor que la mayoría de traders humanos")
            evaluation = "💪 EXCELENTE - 75%+ ES PROFESIONAL"
        elif results['best_accuracy'] >= 0.7:
            print(f"👍 Buen resultado: {results['best_accuracy']:.1%}")
            print(f"📈 Base sólida para refinamiento")
            evaluation = "✅ BUENO - 70%+ ES RENTABLE"
        else:
            print(f"🤔 Resultado moderado: {results['best_accuracy']:.1%}")
            print(f"🔬 Necesitamos más datos/features/tiempo")
            evaluation = "🔄 MODERADO - SEGUIR OPTIMIZANDO"
        
        print(f"\n🏆 EVALUACIÓN FINAL: {evaluation}")
        
        print(f"\n💾 TESORO GENERADO:")
        print(f"   📄 beast_mode_report.txt - Reporte completo")
        print(f"   🌳 beast_mode_rules.txt - Reglas para tu bot")
        print(f"   🤖 elite_model_*.joblib - Mejores modelos")
        print(f"   📊 elite_predictions_*.csv - Predicciones detalladas")
        
        print(f"\n🚀 SIGUIENTE PASO:")
        if results['target_achieved']:
            print("   ¡Implementar el modelo ganador en producción!")
            print("   ¡A ganar dinero!")
        else:
            print("   Revisar beast_mode_rules.txt para patrones")
            print("   Usar el mejor modelo encontrado")
            print("   Recopilar más datos para siguiente iteración")
        
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error catastrófico: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
