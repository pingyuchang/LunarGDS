# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:06:20 2025

@author: pingy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# 讀取 Apollo 15 數據
#kny = pd.read_csv("D:/Sat_MV/AP12_1969_1min.csv", sep=',').iloc[5520:6960]
kny = pd.read_csv("D:/Sat_MV/APollo15_0527.csv", sep=',').iloc[242875:286202]
#kny = pd.read_csv("D:/Sat_MV/APollo16_0527.csv",  sep=',')

# 重新命名可能衝突的欄位
kny.rename(columns={'min': 'minute', 'sec': 'second'}, inplace=True)

# 組合時間戳記
kny['timestamp'] = pd.to_datetime(kny[['year', 'month', 'day', 'hour', 'minute', 'second']])

# 磁場擾動參數
perturb_Mag = 0

# 使用中值濾波並加上偏移修正 X、Y、Z 分量
#knyX = signal.medfilt(kny['BX'] + 25.8 + perturb_Mag, 1)  
#knyY = signal.medfilt(kny['BY'] - 11.9 + perturb_Mag, 1)
#knyZ = signal.medfilt(kny['BZ'] + 25.8 + perturb_Mag, 1)

knyX = signal.medfilt(kny['BX'] + 0.2 + perturb_Mag, 1) 
knyY = signal.medfilt(kny['BY'] - 0.9 + perturb_Mag, 1)
knyZ = signal.medfilt(kny['BZ'] - 3.3 + perturb_Mag, 1)

#knyX = signal.medfilt(kny['BX'] + 186 + perturb_Mag, 1) 
#knyY = signal.medfilt(kny['BY'] + 48 + perturb_Mag, 1)
#knyZ = signal.medfilt(kny['BZ'] - 135 + perturb_Mag, 1)

# 繪製磁場變化圖
plt.figure(figsize=(15, 6))
plt.plot(kny['timestamp'], knyX, label='Magnetic Field X', linewidth=2, alpha=0.8)
plt.plot(kny['timestamp'], knyY, label='Magnetic Field Y', linewidth=2, alpha=0.8)
plt.plot(kny['timestamp'], knyZ, label='Magnetic Field Z', linewidth=2, alpha=0.8)

# 設定標題與標籤大小
plt.title("Magnetic Field Variation Over Time (Apollo 15)", fontsize=18)
plt.xlabel("Time (UTC)", fontsize=16, labelpad=10)
plt.ylabel("Magnetic Field (nT)", fontsize=16, labelpad=10)

# 動態設置 Y 軸範圍
plt.ylim(min(knyX.min(), knyY.min(), knyZ.min()) - 5, max(knyX.max(), knyY.max(), knyZ.max()) + 5)

# 增大圖例字體大小
plt.legend(fontsize=14)

# 調整座標軸數字（ticks）字體大小
plt.xticks(fontsize=14, rotation=45)  # 旋轉 X 軸時間標籤以避免重疊
plt.yticks(fontsize=14)

# 增加格線
plt.grid(True)

# 自動調整圖形布局
plt.tight_layout()
plt.show()

