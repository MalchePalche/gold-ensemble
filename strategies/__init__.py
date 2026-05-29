from .base import BaseStrategy, StrategyResult
from .s1_ma_crossover import S1MACrossover
from .s2_donchian import S2Donchian
from .s4_52w_momentum import S452WeekMomentum
from .s5_macd import S5MACD

__all__ = [
    "BaseStrategy", "StrategyResult",
    "S1MACrossover", "S2Donchian", "S452WeekMomentum", "S5MACD",
]
