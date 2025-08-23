#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Decision Pattern Analyzer - DESCUBRE LOS PATRONES GANADORES
-------------------------------------------------------------------
🎯 OBJETIVO: Analizar TODAS las decisiones mayo-julio y encontrar patrones que funcionan
📊 Genera reglas concretas: "Si RSI<30 Y mom_5>0.5 → 85% acierto"
🔥 Exporta CSV completo con todas las decisiones para análisis manual
💡 Identifica los TOP patrones más rentables

Uso:
python3 enhanced_decision_pattern_analyzer.py \
  --data ethusdt_data_20may_31jul.csv \
  --min-confidence 65 \
  --export-decisions
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings
import time
from collections import defaultdict
warnings.filterwarnings('ignore')

@dataclass
class Config:
    mtf_minutes: int = 30
    ema_period: int = 75
    min_history_min: int = 200

@dataclass
class PatternRule:
    """Una regla de patrón descubierta"""
    description: str
    condition: str
    accuracy: float
    total_cases: int
    correct_cases: int
    confidence_level: str
    feature_conditions: Dict[str, Tuple[str, float]]  # {'rsi': ('<=', 30.0)}

class EnhancedDecisionPatternAnalyzer:
    def __init__(self):
        self.cfg = Config()
        self.feature_names = [
            'ema', 'ema_slope', 'dist_ema', 'ret_1m', 'mom_5', 'mom_15',
            'vol_15', 'vol_30', 'block_ret_so_far', 'block_range_so_far', 
            'elapsed_min', 'cur_green', 'rsi_14', 'macd_diff', 'bb_position',
            'atr_14', 'macd_15', 'trend_15', 'trend_60'
        ]
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Carga datos OHLC con detección automática de separador"""
        print(f"📥 Cargando datos desde: {file_path}")
        
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
        
        print(f"✅ {len(df_clean)} velas cargadas desde {df_clean['time'].min()} hasta {df_clean['time'].max()}")
        return df_clean.set_index('time')
    
    def create_mtf_data(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """Crea bloques MTF 30min"""
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        mtf_data = df_1m.resample("30min", label="right", closed="right").agg(agg).dropna()
        
        mtf_data['is_red'] = mtf_data['close'] < mtf_data['open']
        mtf_data['body_pct'] = (mtf_data['close'] - mtf_data['open']) / mtf_data['open'] * 100
        
        return mtf_data
    
    def compute_advanced_features(self, df_1m: pd.DataFrame, timestamp: pd.Timestamp) -> Optional[Dict]:
        """Calcula las 19 features avanzadas"""
        available_data = df_1m[df_1m.index <= timestamp]
        
        if len(available_data) < self.cfg.min_history_min:
            return None
        
        try:
            # Ventanas de datos
            win5 = available_data.tail(5)
            win15 = available_data.tail(15)
            win30 = available_data.tail(30)
            
            # EMA y tendencia
            ema_series = available_data["close"].ewm(span=self.cfg.ema_period, adjust=False).mean()
            ema = float(ema_series.iloc[-1])
            ema_prev = float(ema_series.iloc[-2]) if len(ema_series) >= 2 else ema
            ema_slope = ema - ema_prev
            
            # Returns y volatilidad
            def _ret(a, b): 
                return float(a/b - 1.0) if b != 0 else 0.0
            
            close_now = float(available_data["close"].iloc[-1])
            open_now = float(available_data["open"].iloc[-1])
            ret_1m = _ret(close_now, open_now)
            
            # Momentum
            mom_5 = float((win5["close"].iloc[-1] / win5["close"].iloc[0]) - 1.0) if len(win5) >= 2 else 0.0
            mom_15 = float((win15["close"].iloc[-1] / win15["close"].iloc[0]) - 1.0) if len(win15) >= 2 else 0.0
            
            # Volatilidad
            r15 = win15["close"].pct_change().dropna()
            vol_15 = float(r15.std()) if len(r15) else 0.0
            
            r30 = win30["close"].pct_change().dropna()
            vol_30 = float(r30.std()) if len(r30) else 0.0
            
            # Distancia a EMA
            dist_ema = float(close_now - ema)
            
            # Información del bloque MTF actual
            dt = pd.Timedelta(minutes=30)
            block_start = timestamp - dt + pd.Timedelta(minutes=1)
            block_data = available_data[(available_data.index >= block_start) & (available_data.index <= timestamp)]
            
            if block_data.empty:
                return None
            
            block_open = float(block_data["open"].iloc[0])
            block_high_so_far = float(block_data["high"].max())
            block_low_so_far = float(block_data["low"].min())
            block_close_now = float(block_data["close"].iloc[-1])
            block_ret_so_far = _ret(block_close_now, block_open)
            block_range_so_far = float(block_high_so_far - block_low_so_far)
            elapsed_min = int((timestamp - block_start).total_seconds() / 60.0)
            
            # ¿Vela actual es verde?
            cur_green = block_close_now > float(block_data["open"].iloc[-1])
            
            # === FEATURES ADICIONALES AVANZADAS ===
            
            # RSI (14 períodos)
            delta = available_data["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
            # MACD
            ema12 = available_data["close"].ewm(span=12, adjust=False).mean()
            ema26 = available_data["close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_diff = float(macd_line.iloc[-1] - signal_line.iloc[-1])
            
            # Bollinger Bands
            bb_period = 20
            sma20 = available_data["close"].rolling(window=bb_period).mean()
            std20 = available_data["close"].rolling(window=bb_period).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            bb_position = (close_now - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) != 0 else 0.5
            
            # ATR (Average True Range)
            high = available_data["high"]
            low = available_data["low"]
            close_prev = available_data["close"].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean()
            atr_14 = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
            
            # Multi-timeframe
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
            
            features = {
                "ema": ema,
                "ema_slope": ema_slope,
                "dist_ema": dist_ema,
                "ret_1m": ret_1m,
                "mom_5": mom_5,
                "mom_15": mom_15,
                "vol_15": vol_15,
                "vol_30": vol_30,
                "block_ret_so_far": block_ret_so_far,
                "block_range_so_far": block_range_so_far,
                "elapsed_min": elapsed_min,
                "cur_green": int(cur_green),
                "rsi_14": rsi_14,
                "macd_diff": macd_diff,
                "bb_position": float(bb_position),
                "atr_14": atr_14,
                "macd_15": float(macd_15),
                "trend_15": float(trend_15),
                "trend_60": float(trend_60),
            }
            
            return features
            
        except Exception as e:
            return None
    
    def find_all_decision_points(self, df_1m: pd.DataFrame, mtf_data: pd.DataFrame) -> List[Dict]:
        """Encuentra TODOS los puntos de decisión (MTF rojo → primera vela verde)"""
        
        print("🔍 Buscando TODOS los puntos de decisión...")
        decision_points = []
        dt = pd.Timedelta(minutes=30)
        
        for i in range(1, len(mtf_data)):
            if i % 200 == 0:
                print(f"   Procesando MTF {i}/{len(mtf_data)}", end='\r')
            
            current_mtf_end = mtf_data.index[i]
            current_mtf_start = current_mtf_end - dt
            
            # MTF anterior debe ser rojo
            prev_mtf = mtf_data.iloc[i-1]
            if not prev_mtf['is_red']:
                continue
            
            # Buscar primera vela verde en MTF actual
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
            
            # Resultado real del MTF
            current_mtf = mtf_data.iloc[i]
            actual_red = current_mtf['is_red']
            
            decision_point = {
                'decision_time': first_green_time,
                'mtf_start': current_mtf_start,
                'mtf_end': current_mtf_end,
                'actual_red': actual_red,
                'correct_decision': not actual_red,  # Si predecimos GREEN y fue GREEN = correcto
                'mtf_open': float(current_mtf['open']),
                'mtf_close': float(current_mtf['close']),
                'mtf_body_pct': float(current_mtf['body_pct'])
            }
            
            decision_points.append(decision_point)
        
        print(f"\n✅ {len(decision_points)} puntos de decisión encontrados")
        return decision_points
    
    def analyze_winning_patterns(self, df_features: pd.DataFrame, min_confidence: float = 0.65) -> List[PatternRule]:
        """Encuentra patrones específicos que funcionan"""
        
        print(f"🔍 Analizando patrones con >={min_confidence*100}% precisión...")
        
        patterns = []
        
        # Crear árbol de decisión interpretable
        feature_cols = [col for col in self.feature_names if col in df_features.columns]
        X = df_features[feature_cols].values
        y = df_features['correct_decision'].values
        
        # Árbol simple para reglas interpretables
        dt = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        dt.fit(X, y)
        
        # Extraer reglas del árbol
        tree_rules = export_text(dt, feature_names=feature_cols)
        
        # Guardar reglas en archivo
        with open('decision_tree_rules.txt', 'w') as f:
            f.write("REGLAS DEL ÁRBOL DE DECISIÓN\n")
            f.write("="*50 + "\n")
            f.write(tree_rules)
        
        print("💾 Reglas guardadas en: decision_tree_rules.txt")
        
        # Análisis por cuantiles para cada feature importante
        if hasattr(dt, 'feature_importances_'):
            important_features = [(feature_cols[i], imp) for i, imp in enumerate(dt.feature_importances_)]
            important_features.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🔝 Top 5 features más importantes:")
            for feat, imp in important_features[:5]:
                print(f"   {feat}: {imp:.3f}")
            
            # Crear reglas por cuantiles para top features
            for feat_name, _ in important_features[:8]:  # Top 8 features
                if feat_name not in df_features.columns:
                    continue
                
                # Dividir en quintiles
                try:
                    df_features[f'{feat_name}_bin'] = pd.qcut(df_features[feat_name], q=5, duplicates='drop')
                    
                    # Analizar cada bin
                    bin_analysis = df_features.groupby(f'{feat_name}_bin')['correct_decision'].agg(['mean', 'count'])
                    
                    for bin_name, row in bin_analysis.iterrows():
                        if row['count'] >= 10 and row['mean'] >= min_confidence:
                            # Extraer thresholds del bin
                            bin_str = str(bin_name)
                            if ',' in bin_str:
                                # Formato: (0.123, 0.456]
                                try:
                                    threshold = float(bin_str.split(',')[1].strip(' ]'))
                                    
                                    pattern = PatternRule(
                                        description=f"Cuando {feat_name} <= {threshold:.3f}",
                                        condition=f"{feat_name} <= {threshold:.3f}",
                                        accuracy=row['mean'],
                                        total_cases=int(row['count']),
                                        correct_cases=int(row['mean'] * row['count']),
                                        confidence_level="ALTA" if row['mean'] >= 0.75 else "MEDIA",
                                        feature_conditions={feat_name: ('<=', threshold)}
                                    )
                                    patterns.append(pattern)
                                except:
                                    continue
                except:
                    continue
        
        # Combinaciones de 2 features (más complejo pero más preciso)
        print("🔍 Buscando patrones combinados (2 features)...")
        
        top_features = [feat for feat, _ in important_features[:6]]
        
        for i, feat1 in enumerate(top_features):
            for j, feat2 in enumerate(top_features[i+1:], i+1):
                if feat1 not in df_features.columns or feat2 not in df_features.columns:
                    continue
                
                # Dividir cada feature en 3 partes (low, med, high)
                try:
                    f1_low = df_features[feat1].quantile(0.33)
                    f1_high = df_features[feat1].quantile(0.67)
                    
                    f2_low = df_features[feat2].quantile(0.33)
                    f2_high = df_features[feat2].quantile(0.67)
                    
                    # Probar combinaciones
                    combinations = [
                        (f"({feat1} <= {f1_low:.3f}) & ({feat2} <= {f2_low:.3f})", 
                         (df_features[feat1] <= f1_low) & (df_features[feat2] <= f2_low)),
                        (f"({feat1} >= {f1_high:.3f}) & ({feat2} >= {f2_high:.3f})", 
                         (df_features[feat1] >= f1_high) & (df_features[feat2] >= f2_high)),
                        (f"({feat1} <= {f1_low:.3f}) & ({feat2} >= {f2_high:.3f})", 
                         (df_features[feat1] <= f1_low) & (df_features[feat2] >= f2_high)),
                    ]
                    
                    for condition_desc, condition_mask in combinations:
                        subset = df_features[condition_mask]
                        if len(subset) >= 15:  # Mínimo 15 casos
                            accuracy = subset['correct_decision'].mean()
                            if accuracy >= min_confidence:
                                pattern = PatternRule(
                                    description=f"Cuando {condition_desc}",
                                    condition=condition_desc,
                                    accuracy=accuracy,
                                    total_cases=len(subset),
                                    correct_cases=int(accuracy * len(subset)),
                                    confidence_level="ALTA" if accuracy >= 0.75 else "MEDIA",
                                    feature_conditions={}  # Más complejo para combinaciones
                                )
                                patterns.append(pattern)
                except:
                    continue
        
        # Ordenar por precisión
        patterns.sort(key=lambda x: x.accuracy, reverse=True)
        
        print(f"✅ {len(patterns)} patrones encontrados con >={min_confidence*100}% precisión")
        
        return patterns
    
    def export_all_decisions_csv(self, decision_points: List[Dict], df_1m: pd.DataFrame, 
                                export_file: str = "all_decisions_analysis.csv") -> str:
        """Exporta CSV completo con TODAS las decisiones + features + resultados"""
        
        print(f"📊 Creando dataset completo de decisiones...")
        
        rows = []
        skipped = 0
        
        for i, dp in enumerate(decision_points):
            if i % 100 == 0:
                print(f"   Procesando decisión {i}/{len(decision_points)}", end='\r')
            
            # Calcular features en el momento de la decisión
            features = self.compute_advanced_features(df_1m, dp['decision_time'])
            
            if features is None:
                skipped += 1
                continue
            
            # Combinar todo en una fila
            row = {
                'decision_time': dp['decision_time'],
                'mtf_start': dp['mtf_start'],
                'mtf_end': dp['mtf_end'],
                'actual_red': dp['actual_red'],
                'correct_decision': dp['correct_decision'],
                'mtf_open': dp['mtf_open'],
                'mtf_close': dp['mtf_close'],
                'mtf_body_pct': dp['mtf_body_pct'],
            }
            
            # Añadir features
            row.update(features)
            
            rows.append(row)
        
        print(f"\n📊 {len(rows)} decisiones procesadas, {skipped} saltadas")
        
        # Crear DataFrame y exportar
        df_decisions = pd.DataFrame(rows)
        df_decisions.to_csv(export_file, index=False)
        
        print(f"💾 Dataset completo guardado en: {export_file}")
        
        return export_file
    
    def generate_summary_report(self, patterns: List[PatternRule], df_decisions: pd.DataFrame) -> None:
        """Genera reporte de resumen con los mejores patrones"""
        
        report_lines = []
        report_lines.append("🏆 REPORTE DE PATRONES DE DECISIÓN")
        report_lines.append("="*60)
        
        # Estadísticas generales
        total_decisions = len(df_decisions)
        correct_decisions = df_decisions['correct_decision'].sum()
        base_accuracy = correct_decisions / total_decisions
        
        report_lines.append(f"\n📊 ESTADÍSTICAS GENERALES:")
        report_lines.append(f"   Total decisiones analizadas: {total_decisions:,}")
        report_lines.append(f"   Decisiones correctas: {correct_decisions:,}")
        report_lines.append(f"   Precisión base: {base_accuracy:.3f} ({base_accuracy*100:.1f}%)")
        
        # Distribución por mes
        df_decisions['month'] = pd.to_datetime(df_decisions['decision_time']).dt.strftime('%Y-%m')
        monthly_stats = df_decisions.groupby('month')['correct_decision'].agg(['count', 'mean'])
        
        report_lines.append(f"\n📅 PRECISIÓN POR MES:")
        for month, stats in monthly_stats.iterrows():
            report_lines.append(f"   {month}: {stats['mean']:.3f} ({stats['mean']*100:.1f}%) - {stats['count']} decisiones")
        
        # Top patrones
        report_lines.append(f"\n🔝 TOP 10 MEJORES PATRONES:")
        for i, pattern in enumerate(patterns[:10]):
            improvement = pattern.accuracy - base_accuracy
            report_lines.append(f"\n   #{i+1}: {pattern.confidence_level}")
            report_lines.append(f"       Patrón: {pattern.description}")
            report_lines.append(f"       Precisión: {pattern.accuracy:.3f} ({pattern.accuracy*100:.1f}%)")
            report_lines.append(f"       Casos: {pattern.total_cases} (aciertos: {pattern.correct_cases})")
            report_lines.append(f"       Mejora: +{improvement:.3f} (+{improvement*100:.1f}%)")
        
        # Análisis de features importantes
        report_lines.append(f"\n💡 ANÁLISIS DE FEATURES:")
        numeric_features = df_decisions.select_dtypes(include=[np.number]).columns
        feature_correlations = []
        
        for feat in numeric_features:
            if feat in self.feature_names:
                corr = df_decisions[feat].corr(df_decisions['correct_decision'])
                if not pd.isna(corr):
                    feature_correlations.append((feat, abs(corr)))
        
        feature_correlations.sort(key=lambda x: x[1], reverse=True)
        
        report_lines.append(f"   Top 10 features más correlacionadas con acierto:")
        for feat, corr in feature_correlations[:10]:
            report_lines.append(f"   {feat}: {corr:.3f}")
        
        # Guardar reporte
        report_content = "\n".join(report_lines)
        
        with open('decision_patterns_report.txt', 'w') as f:
            f.write(report_content)
        
        # También imprimir en consola
        print("\n" + report_content)
        print(f"\n💾 Reporte completo guardado en: decision_patterns_report.txt")
    
    def run_complete_analysis(self, data_file: str, min_confidence: float = 0.65, 
                             export_decisions: bool = True) -> Dict:
        """Ejecuta el análisis completo de patrones de decisión"""
        
        print("🚀 INICIANDO ANÁLISIS COMPLETO DE PATRONES DE DECISIÓN")
        print("="*70)
        
        start_time = time.time()
        
        try:
            # 1. Cargar datos
            df_1m = self.load_data(data_file)
            
            # 2. Crear MTF
            print("\n🔳 Creando bloques MTF 30min...")
            mtf_data = self.create_mtf_data(df_1m)
            print(f"✅ {len(mtf_data)} bloques MTF creados")
            
            # 3. Encontrar todos los puntos de decisión
            decision_points = self.find_all_decision_points(df_1m, mtf_data)
            
            if not decision_points:
                return {'error': 'No se encontraron puntos de decisión'}
            
            # 4. Exportar CSV completo (opcional)
            csv_file = None
            if export_decisions:
                csv_file = self.export_all_decisions_csv(decision_points, df_1m)
                df_decisions = pd.read_csv(csv_file)
            else:
                # Crear DataFrame solo con las features necesarias
                rows = []
                for dp in decision_points:
                    features = self.compute_advanced_features(df_1m, dp['decision_time'])
                    if features:
                        row = dp.copy()
                        row.update(features)
                        rows.append(row)
                df_decisions = pd.DataFrame(rows)
            
            if df_decisions.empty:
                return {'error': 'No se pudieron calcular features para las decisiones'}
            
            print(f"✅ Dataset de decisiones creado: {len(df_decisions)} filas")
            
            # 5. Analizar patrones ganadores
            patterns = self.analyze_winning_patterns(df_decisions, min_confidence)
            
            # 6. Generar reporte
            self.generate_summary_report(patterns, df_decisions)
            
            elapsed_time = time.time() - start_time
            
            # Resultados finales
            results = {
                'total_decision_points': len(decision_points),
                'valid_decisions_with_features': len(df_decisions),
                'base_accuracy': df_decisions['correct_decision'].mean(),
                'patterns_found': len(patterns),
                'high_confidence_patterns': len([p for p in patterns if p.confidence_level == "ALTA"]),
                'best_pattern_accuracy': patterns[0].accuracy if patterns else 0,
                'csv_exported': csv_file,
                'processing_time_seconds': elapsed_time,
                'top_patterns': patterns[:5]  # Top 5 para mostrar
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error durante el análisis: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Enhanced Decision Pattern Analyzer")
    parser.add_argument("--data", type=str, required=True, help="Archivo de datos OHLC")
    parser.add_argument("--min-confidence", type=float, default=0.65, help="Mínima confianza para patrones (0.65 = 65%)")
    parser.add_argument("--export-decisions", action="store_true", help="Exportar CSV con todas las decisiones")
    
    args = parser.parse_args()
    
    print("🔥 Enhanced Decision Pattern Analyzer - MAYO A JULIO 2025")
    print(f"📊 Datos: {args.data}")
    print(f"🎯 Confianza mínima: {args.min_confidence*100:.0f}%")
    print(f"📋 Exportar CSV: {'SÍ' if args.export_decisions else 'NO'}")
    print("\n" + "="*80)
    print("🎯 OBJETIVO: Encontrar patrones que realmente funcionan")
    print("📊 Analizando TODAS las decisiones mayo-julio")
    print("🔥 Generando reglas concretas para mejorar precisión")
    print("="*80)
    
    try:
        analyzer = EnhancedDecisionPatternAnalyzer()
        
        # Ejecutar análisis completo
        results = analyzer.run_complete_analysis(
            data_file=args.data,
            min_confidence=args.min_confidence,
            export_decisions=args.export_decisions
        )
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return 1
        
        # Mostrar resultados finales
        print("\n" + "="*80)
        print("🏆 ANÁLISIS COMPLETADO - RESULTADOS FINALES")
        print("="*80)
        
        print(f"\n📊 DATOS PROCESADOS:")
        print(f"   Puntos de decisión encontrados: {results['total_decision_points']:,}")
        print(f"   Con features válidas: {results['valid_decisions_with_features']:,}")
        print(f"   Tiempo de procesamiento: {results['processing_time_seconds']:.1f} segundos")
        
        print(f"\n🎯 PRECISIÓN:")
        print(f"   Precisión base (sin patrones): {results['base_accuracy']:.3f} ({results['base_accuracy']*100:.1f}%)")
        
        if results['patterns_found'] > 0:
            print(f"   Mejor patrón encontrado: {results['best_pattern_accuracy']:.3f} ({results['best_pattern_accuracy']*100:.1f}%)")
            improvement = results['best_pattern_accuracy'] - results['base_accuracy']
            print(f"   Mejora potencial: +{improvement:.3f} (+{improvement*100:.1f}%)")
        
        print(f"\n🔍 PATRONES DESCUBIERTOS:")
        print(f"   Total patrones encontrados: {results['patterns_found']}")
        print(f"   Patrones alta confianza: {results['high_confidence_patterns']}")
        
        if results['top_patterns']:
            print(f"\n🔝 TOP 5 MEJORES PATRONES:")
            for i, pattern in enumerate(results['top_patterns']):
                print(f"   #{i+1}: {pattern.description}")
                print(f"        Precisión: {pattern.accuracy:.3f} ({pattern.accuracy*100:.1f}%)")
                print(f"        Casos: {pattern.total_cases} (aciertos: {pattern.correct_cases})")
                print()
        
        print(f"💾 ARCHIVOS GENERADOS:")
        print(f"   📋 all_decisions_analysis.csv - Dataset completo")
        print(f"   📄 decision_patterns_report.txt - Reporte detallado")
        print(f"   🌳 decision_tree_rules.txt - Reglas interpretables")
        
        # Evaluación final
        base_acc = results['base_accuracy']
        best_acc = results['best_pattern_accuracy']
        
        if best_acc >= 0.8:
            evaluation = "🔥 EXCEPCIONAL - Patrones muy fuertes encontrados"
        elif best_acc >= 0.75:
            evaluation = "🟢 EXCELENTE - Patrones sólidos para trading"
        elif best_acc >= 0.7:
            evaluation = "🟡 BUENO - Patrones útiles identificados"
        elif best_acc >= 0.65:
            evaluation = "🟠 REGULAR - Algunos patrones prometedores"
        else:
            evaluation = "🔴 LIMITADO - Patrones débiles"
        
        print(f"\n🏆 EVALUACIÓN FINAL: {evaluation}")
        
        if best_acc > base_acc + 0.1:  # Mejora significativa >10%
            print("🎉 ¡ÉXITO! Patrones con mejora significativa encontrados")
            print("💡 Implementa estas reglas en tu estrategia de trading")
        elif best_acc > base_acc + 0.05:  # Mejora moderada >5%
            print("✅ Buena mejora encontrada - Prueba estos patrones")
        else:
            print("🤔 Mejora moderada - Continúa refinando la estrategia")
        
        print("\n📈 PRÓXIMOS PASOS:")
        print("   1. Revisa decision_patterns_report.txt para patrones específicos")
        print("   2. Analiza all_decisions_analysis.csv para insights adicionales")
        print("   3. Implementa los mejores patrones en backtesting")
        print("   4. Valida con datos out-of-sample")
        
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())