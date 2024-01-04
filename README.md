 # Robust Social Recommendation based on Contrastive Learning and Dual-Stage Graph Neural Network

This is the PyTorch implementation for **CLDS** proposed in the paper **Robust Social Recommendation based on Contrastive Learning and Dual-Stage Graph Neural Network**.

> Gang-Feng Ma, Xu-Hua Yang, Haixia Long, Yanbo Zhou, and Xin-Li Xu. 2024.

## Datasets

| Dataset  | # Users  | # Items   | # $E_I$ | Density of $E_I$ | $E_S$  | Density of $E_S$ |   
|----------|----------|-----------|---------|------------------|--------|------------------|
| LastFM   | 1,892    | 17,632    | 92,834  | 0.278%           | 25,434 | 0.711%           |
| Ciao     | 7,375    | 105,114   | 284,086 | 0.037%           | 57,544 | 0.106%           |
| Douban   | 2,848    | 39,586    | 894,887 | 0.793%           | 35,770 | 0.441%           |



# Example to run the codes
```
python main.py --dataset lastfm
python main.py --dataset ciao
python main.py --dataset douban
```