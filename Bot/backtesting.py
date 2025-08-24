import pandas as pd
import numpy as np
import talib
from itertools import product
import warnings
import json
from datetime import datetime
import sys
import argparse

warnings.filterwarnings('ignore')

class MegaETHBacktester:
    def __init__(self, data):
        """
        Sistema MEGA backtesting 2000+ combinaciones - SOLO CCI COMO SALIDA
        """
        self.data = data.copy()
        self.prepare_indicators()
        
    def prepare_indicators(self):
        """Calcula todos los indicadores necesarios"""
        df = self.data
        print("🔄 Calculando indicadores técnicos avanzados...")
        
        # CCI - SOLO PARA SALIDA
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # EMAs expandidos
        for period in [5, 8, 9, 13, 17, 21, 26, 34, 50, 55, 89, 100, 144, 200]:
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # RSI con diferentes períodos
        for period in [9, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
        
        # Estocástico
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'],
                                                   fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k_fast'], df['stoch_d_fast'] = talib.STOCH(df['high'], df['low'], df['close'],
                                                            fastk_period=5, slowk_period=3, slowd_period=3)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['macd_fast'], df['macd_signal_fast'], df['macd_hist_fast'] = talib.MACD(
            df['close'], fastperiod=8, slowperiod=17, signalperiod=9)
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_10'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=10)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ADX
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volumen
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ma_50'] = df['volume'].rolling(window=50).mean()
        
        # Price action
        df['is_green'] = df['close'] > df['open']
        df['is_red'] = df['close'] < df['open']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        
        print("✅ Indicadores calculados correctamente")
        self.data = df
        
    def base_entry_filter(self, i, start_level=-100, mode='wick'):
        """
        Filtro de entrada base - SOLO PARA DETECTAR SEÑALES, CCI NO ES FILTRO DE ENTRADA
        """
        if i < 50:
            return False
            
        df = self.data
        
        # Aquí pones TU filtro de entrada base que quieras
        # Por ejemplo, momentum simple o precio vs EMA
        
        # Ejemplo: Precio rompe EMA 21 hacia arriba
        if pd.isna(df.iloc[i]['ema_21']):
            return False
            
        price_above_ema = df.iloc[i]['close'] > df.iloc[i]['ema_21']
        
        # Rompe máximo anterior
        if mode == 'wick':
            breaks_high = df.iloc[i]['high'] > df.iloc[i-1]['high']
        elif mode == 'close':
            breaks_high = df.iloc[i]['close'] > df.iloc[i-1]['high']
        else:  # body
            breaks_high = df.iloc[i]['close'] > df.iloc[i-1]['close']
        
        # Filtro de velas rojas
        last_two_red = (df.iloc[i]['is_red'] and df.iloc[i-1]['is_red'])
        color_ok = not last_two_red
        
        return price_above_ema and breaks_high and color_ok
    
    def advanced_filters(self, i, config):
        """Filtros avanzados configurables - SIN CCI COMO FILTRO"""
        df = self.data
        
        # EMA Filter
        if 'ema_period' in config:
            ema_col = f"ema_{config['ema_period']}"
            if pd.isna(df.iloc[i][ema_col]):
                return False
            
            if config.get('ema_direction') == 'above':
                if df.iloc[i]['close'] <= df.iloc[i][ema_col]:
                    return False
            elif config.get('ema_direction') == 'below':
                if df.iloc[i]['close'] >= df.iloc[i][ema_col]:
                    return False
            elif config.get('ema_direction') == 'cross_above':
                if not (df.iloc[i]['close'] > df.iloc[i][ema_col] and 
                       df.iloc[i-1]['close'] <= df.iloc[i-1][ema_col]):
                    return False
        
        # EMA Double Filter
        if 'ema_fast' in config and 'ema_slow' in config:
            ema_fast_col = f"ema_{config['ema_fast']}"
            ema_slow_col = f"ema_{config['ema_slow']}"
            if (pd.isna(df.iloc[i][ema_fast_col]) or pd.isna(df.iloc[i][ema_slow_col])):
                return False
            if config.get('emas_bullish'):
                if df.iloc[i][ema_fast_col] <= df.iloc[i][ema_slow_col]:
                    return False
        
        # RSI Filter
        rsi_period = config.get('rsi_period', 14)
        rsi_col = f"rsi_{rsi_period}"
        if 'rsi_min' in config and 'rsi_max' in config:
            if rsi_col not in df.columns:
                rsi_col = 'rsi_14'
            rsi = df.iloc[i][rsi_col]
            if pd.isna(rsi) or rsi < config['rsi_min'] or rsi > config['rsi_max']:
                return False
        
        # RSI Momentum
        if config.get('rsi_momentum'):
            rsi_col = f"rsi_{config.get('rsi_period', 14)}"
            if rsi_col not in df.columns:
                rsi_col = 'rsi_14'
            rsi = df.iloc[i][rsi_col]
            rsi_prev = df.iloc[i-1][rsi_col]
            if pd.isna(rsi) or pd.isna(rsi_prev) or rsi <= rsi_prev:
                return False
        
        # Volume Filter
        if config.get('volume_factor'):
            vol_period = config.get('volume_period', 20)
            vol_col = f"volume_ma_{vol_period}"
            if vol_col not in df.columns:
                vol_col = 'volume_ma_20'
            vol_ma = df.iloc[i][vol_col]
            if pd.isna(vol_ma) or df.iloc[i]['volume'] < vol_ma * config['volume_factor']:
                return False
        
        # Stochastic Filter
        stoch_col = 'stoch_k_fast' if config.get('stoch_fast') else 'stoch_k'
        if config.get('stoch_oversold'):
            stoch_k = df.iloc[i][stoch_col]
            if pd.isna(stoch_k) or stoch_k > config['stoch_oversold']:
                return False
        
        # MACD Filter
        macd_suffix = '_fast' if config.get('macd_fast') else ''
        if config.get('macd_bullish'):
            macd_col = f"macd{macd_suffix}"
            macd_signal_col = f"macd_signal{macd_suffix}"
            macd = df.iloc[i][macd_col]
            macd_signal = df.iloc[i][macd_signal_col]
            if pd.isna(macd) or pd.isna(macd_signal) or macd <= macd_signal:
                return False
        
        # Williams %R Filter
        if config.get('williams_oversold'):
            williams = df.iloc[i]['williams_r']
            if pd.isna(williams) or williams > config['williams_oversold']:
                return False
        
        # ADX Filter
        if config.get('adx_min'):
            adx = df.iloc[i]['adx']
            if pd.isna(adx) or adx < config['adx_min']:
                return False
        
        # Bollinger Bands Filter
        if config.get('bb_position'):
            bb_lower = df.iloc[i]['bb_lower']
            bb_middle = df.iloc[i]['bb_middle']
            bb_upper = df.iloc[i]['bb_upper']
            price = df.iloc[i]['close']
            
            if pd.isna(bb_lower) or pd.isna(bb_upper):
                return False
            
            if config['bb_position'] == 'lower_half' and price >= bb_middle:
                return False
            elif config['bb_position'] == 'lower_third':
                threshold = bb_lower + (bb_upper - bb_lower) * 0.33
                if price >= threshold:
                    return False
        
        return True
    
    def calculate_exit(self, entry_idx, entry_price, exit_config):
        """Sistema de salidas - CCI COMO SALIDA PRINCIPAL"""
        df = self.data
        max_bars = exit_config.get('max_bars', 200)
        
        # CCI EXIT - LA SALIDA PRINCIPAL
        if 'cci_exit_level' in exit_config:
            for i in range(entry_idx + 1, min(len(df), entry_idx + max_bars)):
                cci = df.iloc[i]['cci']
                if not pd.isna(cci) and cci > exit_config['cci_exit_level']:
                    return df.iloc[i]['close'], i, f"CCI_{exit_config['cci_exit_level']}"
        
        # ATR Trailing Stop
        elif 'atr_multiplier' in exit_config:
            atr_period = exit_config.get('atr_period', 14)
            atr_col = f"atr_{atr_period}" if f"atr_{atr_period}" in df.columns else 'atr'
            atr_value = df.iloc[entry_idx][atr_col]
            
            if pd.isna(atr_value):
                atr_value = entry_price * 0.02
            
            atr_stop = entry_price - (atr_value * exit_config['atr_multiplier'])
            highest_price = entry_price
            
            for i in range(entry_idx + 1, min(len(df), entry_idx + max_bars)):
                current_high = df.iloc[i]['high']
                
                if current_high > highest_price:
                    highest_price = current_high
                    current_atr = df.iloc[i][atr_col]
                    if pd.isna(current_atr):
                        current_atr = atr_value
                    atr_stop = highest_price - (current_atr * exit_config['atr_multiplier'])
                
                if df.iloc[i]['low'] <= atr_stop:
                    return atr_stop, i, f"ATR_{exit_config['atr_multiplier']}"
        
        # EMA Trailing
        elif 'ema_trailing_period' in exit_config:
            ema_col = f"ema_{exit_config['ema_trailing_period']}"
            for i in range(entry_idx + 1, min(len(df), entry_idx + max_bars)):
                if not pd.isna(df.iloc[i][ema_col]) and df.iloc[i]['close'] < df.iloc[i][ema_col]:
                    return df.iloc[i]['close'], i, f"EMA_{exit_config['ema_trailing_period']}"
        
        # RSI Exit
        elif 'rsi_exit_level' in exit_config:
            rsi_period = exit_config.get('rsi_period', 14)
            rsi_col = f"rsi_{rsi_period}"
            for i in range(entry_idx + 1, min(len(df), entry_idx + max_bars)):
                rsi = df.iloc[i][rsi_col]
                if not pd.isna(rsi) and rsi > exit_config['rsi_exit_level']:
                    return df.iloc[i]['close'], i, f"RSI_{exit_config['rsi_exit_level']}"
        
        # Fixed Profit %
        elif 'fixed_profit_pct' in exit_config:
            target_price = entry_price * (1 + exit_config['fixed_profit_pct'] / 100)
            for i in range(entry_idx + 1, min(len(df), entry_idx + max_bars)):
                if df.iloc[i]['high'] >= target_price:
                    return target_price, i, f"Fixed_{exit_config['fixed_profit_pct']}%"
        
        # Stop + Take Profit
        elif 'stop_loss_pct' in exit_config and 'take_profit_pct' in exit_config:
            stop_price = entry_price * (1 - exit_config['stop_loss_pct'] / 100)
            target_price = entry_price * (1 + exit_config['take_profit_pct'] / 100)
            
            for i in range(entry_idx + 1, min(len(df), entry_idx + max_bars)):
                if df.iloc[i]['low'] <= stop_price:
                    return stop_price, i, f"Stop_{exit_config['stop_loss_pct']}%"
                elif df.iloc[i]['high'] >= target_price:
                    return target_price, i, f"TP_{exit_config['take_profit_pct']}%"
        
        # Default exit
        stop_loss = entry_price * 0.97
        max_idx = min(entry_idx + 100, len(df) - 1)
        for i in range(entry_idx + 1, max_idx + 1):
            if df.iloc[i]['low'] <= stop_loss:
                return stop_loss, i, "Default_Stop"
        
        return df.iloc[max_idx]['close'], max_idx, "Time_Exit"
    
    def backtest_strategy(self, entry_config, exit_config):
        """Backtest sin logging individual"""
        df = self.data
        trades = []
        in_position = False
        
        for i in range(50, len(df) - 1):
            if not in_position:
                # Buscar entrada
                if self.base_entry_filter(i, entry_config.get('start_level', -100), entry_config.get('mode', 'wick')):
                    if self.advanced_filters(i, entry_config):
                        # Entrada confirmada
                        in_position = True
                        entry_idx = i
                        entry_price = df.iloc[i]['close']
                        
                        # Calcular salida
                        exit_price, exit_idx, exit_reason = self.calculate_exit(entry_idx, entry_price, exit_config)
                        
                        if exit_price:
                            profit_pct = (exit_price - entry_price) / entry_price * 100
                            
                            trades.append({
                                'entry_idx': entry_idx,
                                'entry_price': entry_price,
                                'exit_idx': exit_idx,
                                'exit_price': exit_price,
                                'profit_pct': profit_pct,
                                'exit_reason': exit_reason,
                                'bars_in_trade': exit_idx - entry_idx
                            })
                            
                            # Skip ahead
                            i = exit_idx
                            in_position = False
        
        return trades
    
    def run_backtest_flexible(self, max_combinations=2000):
        """BACKTEST flexible - puedes especificar cuántas combinaciones"""
        print(f"🚀 Iniciando BACKTEST {max_combinations} combinaciones...")
        
        # Configuraciones base
        base_ema_single = [None, 9, 13, 21, 34, 50, 89]
        base_ema_directions = [None, 'above', 'below']
        base_rsi_configs = [
            None,
            {'rsi_period': 14, 'rsi_min': 30, 'rsi_max': 70},
            {'rsi_period': 14, 'rsi_min': 20, 'rsi_max': 80},
        ]
        base_volume_configs = [None, 1.5, 2.0]
        base_stoch_configs = [None, 20]
        base_macd_configs = [None, True]
        base_modes = ['wick', 'close']
        
        # Exit strategies base
        base_exit_strategies = []
        
        # CCI exits principales
        for level in [100, 120, 140, 150, 160, 180]:
            base_exit_strategies.append({'cci_exit_level': level})
        
        # ATR básicos
        for mult in [1.5, 2.0, 2.5]:
            base_exit_strategies.append({'atr_multiplier': mult})
        
        # EMA trailing
        for period in [21, 50]:
            base_exit_strategies.append({'ema_trailing_period': period})
        
        # Calcular factor de reducción basado en max_combinations
        estimated_combinations = (
            len(base_modes) * len(base_ema_single) * len(base_ema_directions) * 
            len(base_rsi_configs) * len(base_volume_configs) * len(base_stoch_configs) * 
            len(base_macd_configs) * len(base_exit_strategies)
        )
        
        print(f"📊 Configuraciones base estimadas: {estimated_combinations}")
        
        # Ajustar configuraciones según max_combinations
        if max_combinations <= 100:
            # Ultra reducido
            ema_single = [None, 21]
            ema_directions = [None, 'above']
            rsi_configs = [None, {'rsi_period': 14, 'rsi_min': 30, 'rsi_max': 70}]
            volume_configs = [None, 1.5]
            stoch_configs = [None]
            macd_configs = [None, True]
            modes = ['wick']
            exit_strategies = [
                {'cci_exit_level': 150}, {'atr_multiplier': 2.0}, {'ema_trailing_period': 21}
            ] * 8  # Duplicar para llegar a ~100
            
        elif max_combinations <= 300:
            # Reducido
            ema_single = [None, 21, 50]
            ema_directions = [None, 'above']
            rsi_configs = base_rsi_configs
            volume_configs = [None, 1.5]
            stoch_configs = [None, 20]
            macd_configs = [None, True]
            modes = ['wick', 'close']
            exit_strategies = base_exit_strategies[:12]  # Solo primeros 12
            
        elif max_combinations <= 1000:
            # Medio
            ema_single = base_ema_single[:5]  # Solo primeros 5
            ema_directions = base_ema_directions
            rsi_configs = base_rsi_configs
            volume_configs = base_volume_configs
            stoch_configs = base_stoch_configs
            macd_configs = base_macd_configs
            modes = base_modes
            exit_strategies = base_exit_strategies
            
        else:
            # Completo 2000+
            ema_single = base_ema_single + [5, 8, 17, 26, 100]
            ema_directions = base_ema_directions + ['cross_above']
            rsi_configs = base_rsi_configs + [
                {'rsi_period': 9, 'rsi_min': 25, 'rsi_max': 75},
                {'rsi_period': 14, 'rsi_momentum': True}
            ]
            volume_configs = base_volume_configs + [1.2, 2.5]
            stoch_configs = base_stoch_configs + [15, 25, 30]
            macd_configs = base_macd_configs + ['fast']
            modes = base_modes + ['body']
            
            # Exit strategies expandidas
            exit_strategies = []
            for level in [70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200]:
                exit_strategies.append({'cci_exit_level': level})
            for mult in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
                exit_strategies.append({'atr_multiplier': mult})
            for period in [9, 13, 21, 34, 50]:
                exit_strategies.append({'ema_trailing_period': period})
            for pct in [2, 5, 10]:
                exit_strategies.append({'fixed_profit_pct': pct})
        
        total_combinations = 0
        all_strategies = []
        strategy_id = 0
        
        print("📊 Generando y ejecutando configuraciones...")
        
        # Generar combinaciones
        for mode in modes:
            for ema_period, ema_dir in product(ema_single, ema_directions):
                if ema_period is None and ema_dir is not None:
                    continue
                    
                for rsi_config in rsi_configs:
                    for vol_factor in volume_configs:
                        for stoch_level in stoch_configs:
                            for macd_config in macd_configs:
                                
                                # Build entry config
                                entry_config = {'mode': mode}
                                
                                if ema_period:
                                    entry_config['ema_period'] = ema_period
                                    entry_config['ema_direction'] = ema_dir
                                
                                if rsi_config:
                                    entry_config.update(rsi_config)
                                
                                if vol_factor:
                                    entry_config['volume_factor'] = vol_factor
                                
                                if stoch_level:
                                    entry_config['stoch_oversold'] = stoch_level
                                
                                if macd_config == 'fast':
                                    entry_config['macd_fast'] = True
                                elif macd_config:
                                    entry_config['macd_bullish'] = True
                                
                                # Test with exits
                                for exit_config in exit_strategies:
                                    strategy_id += 1
                                    total_combinations += 1
                                    
                                    # Stop si alcanzamos max_combinations
                                    if total_combinations > max_combinations:
                                        break
                                    
                                    try:
                                        trades = self.backtest_strategy(entry_config, exit_config)
                                        metrics = self.calculate_metrics(trades)
                                        
                                        if metrics and metrics['total_trades'] >= 3:
                                            strategy_result = {
                                                'strategy_id': strategy_id,
                                                'entry_config': entry_config.copy(),
                                                'exit_config': exit_config.copy(),
                                                'metrics': metrics,
                                                'score': self.calculate_score(metrics)
                                            }
                                            all_strategies.append(strategy_result)
                                    
                                    except Exception as e:
                                        continue
                                    
                                    # Progress
                                    if total_combinations % max(1, max_combinations // 20) == 0:
                                        progress_pct = (total_combinations / max_combinations) * 100
                                        print(f"✅ Procesando configuración {total_combinations}/{max_combinations} ({progress_pct:.1f}%)... Válidas: {len(all_strategies)}")
                                
                                if total_combinations > max_combinations:
                                    break
                            if total_combinations > max_combinations:
                                break
                        if total_combinations > max_combinations:
                            break
                    if total_combinations > max_combinations:
                        break
                if total_combinations > max_combinations:
                    break
            if total_combinations > max_combinations:
                break
        
        print(f"🎯 BACKTEST COMPLETADO! {total_combinations} combinaciones probadas")
        print(f"🏆 {len(all_strategies)} estrategias válidas encontradas")
        
        # Sort by score
        all_strategies.sort(key=lambda x: x['score'], reverse=True)
        return all_strategies[:min(100, len(all_strategies))]  # Top 100 max
        
        total_combinations = 0
        all_strategies = []
        strategy_id = 0
        
        print("📊 Generando y ejecutando configuraciones...")
        
        # Single EMA configs
        for mode in modes:
            for ema_period, ema_dir in product(ema_single, ema_directions):
                if ema_period is None and ema_dir is not None:
                    continue
                    
                for rsi_config in rsi_configs:
                    for vol_factor in volume_configs:
                        for stoch_level in stoch_configs:
                            for macd_config in macd_configs:
                                for williams_level in williams_configs:
                                    for adx_min in adx_configs:
                                        for bb_pos in bb_configs:
                                            
                                            # Build entry config
                                            entry_config = {'mode': mode}
                                            
                                            if ema_period:
                                                entry_config['ema_period'] = ema_period
                                                entry_config['ema_direction'] = ema_dir
                                            
                                            if rsi_config:
                                                entry_config.update(rsi_config)
                                            
                                            if vol_factor:
                                                entry_config['volume_factor'] = vol_factor
                                            
                                            if stoch_level:
                                                entry_config['stoch_oversold'] = stoch_level
                                            
                                            if macd_config == 'fast':
                                                entry_config['macd_fast'] = True
                                            elif macd_config:
                                                entry_config['macd_bullish'] = True
                                            
                                            if williams_level:
                                                entry_config['williams_oversold'] = williams_level
                                            
                                            if adx_min:
                                                entry_config['adx_min'] = adx_min
                                            
                                            if bb_pos:
                                                entry_config['bb_position'] = bb_pos
                                            
                                            # Test with all exits
                                            for exit_config in exit_strategies:
                                                strategy_id += 1
                                                total_combinations += 1
                                                
                                                try:
                                                    trades = self.backtest_strategy(entry_config, exit_config)
                                                    metrics = self.calculate_metrics(trades)
                                                    
                                                    if metrics and metrics['total_trades'] >= 3:
                                                        strategy_result = {
                                                            'strategy_id': strategy_id,
                                                            'entry_config': entry_config.copy(),
                                                            'exit_config': exit_config.copy(),
                                                            'metrics': metrics,
                                                            'score': self.calculate_score(metrics)
                                                        }
                                                        all_strategies.append(strategy_result)
                                                
                                                except Exception as e:
                                                    continue
                                                
                                                # Progress cada 100
                                                if total_combinations % 100 == 0:
                                                    print(f"✅ Procesando configuración {total_combinations}/2000+... Válidas: {len(all_strategies)}")
        
        # Double EMA configs
        for mode in modes:
            for ema_fast, ema_slow in ema_pairs:
                for rsi_config in rsi_configs[:3]:  # Solo primeros 3 para no explotar
                    for vol_factor in volume_configs[:3]:
                        for stoch_level in stoch_configs[:3]:
                            
                            entry_config = {
                                'mode': mode,
                                'ema_fast': ema_fast,
                                'ema_slow': ema_slow,
                                'emas_bullish': True
                            }
                            
                            if rsi_config:
                                entry_config.update(rsi_config)
                            if vol_factor:
                                entry_config['volume_factor'] = vol_factor
                            if stoch_level:
                                entry_config['stoch_oversold'] = stoch_level
                            
                            for exit_config in exit_strategies[:20]:  # Solo primeros 20 exits
                                strategy_id += 1
                                total_combinations += 1
                                
                                try:
                                    trades = self.backtest_strategy(entry_config, exit_config)
                                    metrics = self.calculate_metrics(trades)
                                    
                                    if metrics and metrics['total_trades'] >= 3:
                                        strategy_result = {
                                            'strategy_id': strategy_id,
                                            'entry_config': entry_config.copy(),
                                            'exit_config': exit_config.copy(),
                                            'metrics': metrics,
                                            'score': self.calculate_score(metrics)
                                        }
                                        all_strategies.append(strategy_result)
                                
                                except Exception as e:
                                    continue
                                
                                if total_combinations % 100 == 0:
                                    print(f"✅ Procesando configuración {total_combinations}/2000+... Válidas: {len(all_strategies)}")
        
        print(f"🎯 MEGA BACKTEST COMPLETADO! {total_combinations} combinaciones probadas")
        print(f"🏆 {len(all_strategies)} estrategias válidas encontradas")
        
        # Sort by score
        all_strategies.sort(key=lambda x: x['score'], reverse=True)
        return all_strategies[:100]  # Top 100
    
    def calculate_metrics(self, trades):
        """Calcula métricas de performance"""
        if not trades:
            return None
        
        profits = [t['profit_pct'] for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        
        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100
        total_return = sum(profits)
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.01
        profit_factor = gross_profit / gross_loss
        
        # Drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean(profits)
        
        sharpe_ratio = avg_trade / np.std(profits) if len(profits) > 1 and np.std(profits) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'sharpe_ratio': sharpe_ratio
        }
    
    def calculate_score(self, metrics):
        """Score balanceado para ranking"""
        score = (
            metrics['profit_factor'] * 0.25 +
            metrics['sharpe_ratio'] * 0.20 +
            (metrics['win_rate'] / 100) * 0.15 +
            max(0, metrics['total_return'] / 1000) * 0.15 +
            max(0, (100 - metrics['max_drawdown']) / 100) * 0.15 +
            (metrics['total_trades'] / 1000) * 0.10  # Reward frequency
        )
        return max(0, score)
    
    def print_results(self, strategies):
        """Imprime TOP 50 como el formato que quieres"""
        print("\n" + "="*100)
        print("🏆 TOP 50 MEJORES ESTRATEGIAS - MEGA BACKTEST 2000+")
        print("="*100)
        
        for i, strategy in enumerate(strategies[:50]):
            metrics = strategy['metrics']
            entry = strategy['entry_config']
            exit_cfg = strategy['exit_config']
            
            print(f"\n#{i+1} - SCORE: {strategy['score']:.3f}")
            print(f"📊 Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate']:.1f}%")
            print(f"💰 Return: {metrics['total_return']:.2f}% | PF: {metrics['profit_factor']:.2f}")
            print(f"📉 Max DD: {metrics['max_drawdown']:.2f}% | Sharpe: {metrics['sharpe_ratio']:.2f}")
            
            # Config string
            config_parts = [f"Mode: {entry['mode']}"]
            
            if 'ema_period' in entry:
                config_parts.append(f"EMA {entry['ema_period']} ({entry.get('ema_direction', 'N/A')})")
            elif 'ema_fast' in entry:
                config_parts.append(f"EMA {entry['ema_fast']}/{entry['ema_slow']}")
            
            filters = []
            if 'rsi_min' in entry:
                rsi_p = entry.get('rsi_period', 14)
                filters.append(f"RSI{rsi_p}({entry['rsi_min']}-{entry['rsi_max']})")
            if entry.get('rsi_momentum'):
                filters.append("RSI↗")
            if 'volume_factor' in entry:
                filters.append(f"Vol>{entry['volume_factor']}x")
            if 'stoch_oversold' in entry:
                filters.append(f"Stoch<{entry['stoch_oversold']}")
            if entry.get('macd_bullish') or entry.get('macd_fast'):
                macd_type = "Fast" if entry.get('macd_fast') else "Std"
                filters.append(f"MACD{macd_type}")
            if 'williams_oversold' in entry:
                filters.append(f"Williams<{entry['williams_oversold']}")
            if 'adx_min' in entry:
                filters.append(f"ADX>{entry['adx_min']}")
            if 'bb_position' in entry:
                bb_pos = entry['bb_position'].replace('_', ' ').title()
                filters.append(f"BB-{bb_pos}")
            
            if filters:
                config_parts.extend(filters)
            
            # Exit config
            exit_parts = []
            for key, value in exit_cfg.items():
                if key == 'cci_exit_level':
                    exit_parts.append(f"CCI Exit {value}")
                elif key == 'atr_multiplier':
                    atr_period = exit_cfg.get('atr_period', 14)
                    exit_parts.append(f"ATR {value}x (P{atr_period})")
                elif key == 'ema_trailing_period':
                    exit_parts.append(f"EMA Trail {value}")
                elif key == 'rsi_exit_level':
                    rsi_period = exit_cfg.get('rsi_period', 14)
                    exit_parts.append(f"RSI{rsi_period} Exit {value}")
                elif key == 'fixed_profit_pct':
                    exit_parts.append(f"Fixed TP {value}%")
                elif key == 'stop_loss_pct':
                    tp_pct = exit_cfg.get('take_profit_pct', 'N/A')
                    exit_parts.append(f"SL {value}% / TP {tp_pct}%")
            
            print(f"🔹 Config: {' | '.join(config_parts)} | Exit: {' | '.join(exit_parts)}")
            print("-" * 80)


def load_eth_data(filename):
    """Carga datos ETH"""
    try:
        df = pd.read_csv(filename)
        print(f"📊 Archivo: {filename}")
        print(f"📈 Columnas: {df.columns.tolist()}")
        
        # Column mapping
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'time': 'timestamp'
        }
        df = df.rename(columns=column_mapping)
        
        # Required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing columns: {required_cols}")
        
        # Smart volume
        df['volume'] = ((df['high'] - df['low']) / df['close'] * 1000000).astype(int)
        
        # Clean NaN
        df = df.dropna(subset=required_cols)
        
        print(f"✅ Data processed: {len(df)} candles")
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# MAIN EXECUTION CON ARGUMENTOS
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ETH Mega Backtesting System')
    parser.add_argument('--combinations', '-c', type=int, default=2000, 
                       help='Número de combinaciones a probar (default: 2000)')
    parser.add_argument('--top', '-t', type=int, default=50,
                       help='Número de top estrategias a mostrar (default: 50)')
    parser.add_argument('--data', '-d', type=str, default="data/ethusdt_data_20may_31jul.csv",
                       help='Ruta del archivo de datos (default: data/ethusdt_data_20may_31jul.csv)')
    
    args = parser.parse_args()
    
    print(f"🚀🚀🚀 MEGA BACKTEST RAILWAY - {args.combinations} COMBINATIONS 🚀🚀🚀")
    print("💡 CCI usado SOLO como estrategia de SALIDA")
    
    # Load data
    print("📈 Loading ETH data...")
    data = load_eth_data(args.data)
    
    if data is not None:
        print(f"✅ Data loaded: {len(data)} candles")
        
        # Create mega backtester
        mega_backtester = MegaETHBacktester(data)
        
        # Run flexible backtest
        best_strategies = mega_backtester.run_backtest_flexible(args.combinations)
        
        if best_strategies:
            # Print results (usando args.top)
            print("\n" + "="*100)
            print(f"🏆 TOP {min(args.top, len(best_strategies))} MEJORES ESTRATEGIAS - {args.combinations} COMBINACIONES")
            print("="*100)
            
            for i, strategy in enumerate(best_strategies[:args.top]):
                metrics = strategy['metrics']
                entry = strategy['entry_config']
                exit_cfg = strategy['exit_config']
                
                print(f"\n#{i+1} - SCORE: {strategy['score']:.3f}")
                print(f"📊 Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate']:.1f}%")
                print(f"💰 Return: {metrics['total_return']:.2f}% | PF: {metrics['profit_factor']:.2f}")
                print(f"📉 Max DD: {metrics['max_drawdown']:.2f}% | Sharpe: {metrics['sharpe_ratio']:.2f}")
                
                # Config string
                config_parts = [f"Mode: {entry['mode']}"]
                
                if 'ema_period' in entry:
                    config_parts.append(f"EMA {entry['ema_period']} ({entry.get('ema_direction', 'N/A')})")
                elif 'ema_fast' in entry:
                    config_parts.append(f"EMA {entry['ema_fast']}/{entry['ema_slow']}")
                
                filters = []
                if 'rsi_min' in entry:
                    rsi_p = entry.get('rsi_period', 14)
                    filters.append(f"RSI{rsi_p}({entry['rsi_min']}-{entry['rsi_max']})")
                if entry.get('rsi_momentum'):
                    filters.append("RSI↗")
                if 'volume_factor' in entry:
                    filters.append(f"Vol>{entry['volume_factor']}x")
                if 'stoch_oversold' in entry:
                    filters.append(f"Stoch<{entry['stoch_oversold']}")
                if entry.get('macd_bullish') or entry.get('macd_fast'):
                    macd_type = "Fast" if entry.get('macd_fast') else "Std"
                    filters.append(f"MACD{macd_type}")
                if 'williams_oversold' in entry:
                    filters.append(f"Williams<{entry['williams_oversold']}")
                if 'adx_min' in entry:
                    filters.append(f"ADX>{entry['adx_min']}")
                if 'bb_position' in entry:
                    bb_pos = entry['bb_position'].replace('_', ' ').title()
                    filters.append(f"BB-{bb_pos}")
                
                if filters:
                    config_parts.extend(filters)
                
                # Exit config
                exit_parts = []
                for key, value in exit_cfg.items():
                    if key == 'cci_exit_level':
                        exit_parts.append(f"CCI Exit {value}")
                    elif key == 'atr_multiplier':
                        atr_period = exit_cfg.get('atr_period', 14)
                        exit_parts.append(f"ATR {value}x (P{atr_period})")
                    elif key == 'ema_trailing_period':
                        exit_parts.append(f"EMA Trail {value}")
                    elif key == 'rsi_exit_level':
                        rsi_period = exit_cfg.get('rsi_period', 14)
                        exit_parts.append(f"RSI{rsi_period} Exit {value}")
                    elif key == 'fixed_profit_pct':
                        exit_parts.append(f"Fixed TP {value}%")
                    elif key == 'stop_loss_pct':
                        tp_pct = exit_cfg.get('take_profit_pct', 'N/A')
                        exit_parts.append(f"SL {value}% / TP {tp_pct}%")
                
                print(f"🔹 Config: {' | '.join(config_parts)} | Exit: {' | '.join(exit_parts)}")
                print("-" * 80)
            
            # Save results
            filename = f'backtest_results_{args.combinations}_combinations.json'
            with open(filename, 'w') as f:
                # Convert numpy types
                for strategy in best_strategies:
                    for key, value in strategy['metrics'].items():
                        if isinstance(value, (np.integer, np.floating)):
                            strategy['metrics'][key] = float(value)
                json.dump(best_strategies, f, indent=2)
            
            print(f"\n🎯 BACKTEST COMPLETED!")
            print(f"📊 Total valid strategies found: {len(best_strategies)}")
            if best_strategies:
                print(f"🏆 Best strategy return: {best_strategies[0]['metrics']['total_return']:.2f}%")
                print(f"🥇 Best profit factor: {max(s['metrics']['profit_factor'] for s in best_strategies):.2f}")
                print(f"🎯 Best win rate: {max(s['metrics']['win_rate'] for s in best_strategies):.1f}%")
            print(f"💾 Results saved to {filename}")
        else:
            print("❌ No valid strategies found")
    else:
        print("❌ Failed to load data")
