
## Important Thing to notice
### 1) Do not use this strategy on live. It is still undergoing dry-run. 
Using it live at your own risk.
### 2) If you want to use it on other exchanges (not kucoin), 
You may need to re-hyperopt, the strategy is probably overfitted on kucoin. It will behave wildly different on other exchanges. (ex. Binance) <br>
The ```BB_RPB_TSL_BI``` version is for Binance. 
### 3) Don't use volume list or use volume list with very strict filters
This strategy is not optimized towards all trading pairs, and it cannot predict and prevent P&D (pump and dump).
### 4) If you want to show appreciation, 
Please notice a huge part of this strategy is a re-combination of NFI, I don't deserve the credit.
If you want to show appreciation, please buy a coffee for [@iterativ](https://github.com/iterativv/NostalgiaForInfinity).
### 5) About NFIX_BB_RPB
This version is just NFIX with BB_RPB conditions added. I will keep merge latest NFIX if I have time. Do notice NFIX is really heavy
, and it is recommended to get a high frequency cpu with at least 2G RAM. 
If you want to know more, there is [NFI wiki](https://github.com/iterativv/NostalgiaForInfinity/wiki) available.
### 6) How to use trailing buy
Run ```freqtrade trade -c <name of your config file>.json -s BB_RPB_TSL_Trailing```
