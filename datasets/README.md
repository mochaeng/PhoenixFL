# Datasets

## NF-ToN-IoT-v2

- Original dataset has 16_940_496 records. No data missing. No duplicate data. 
- [Article](https://ieeexplore.ieee.org/document/9625505) uses 213_460 records.
- PhoenixFL uses 254_107 records (1.5% from the original one).

By following the original distribution, the stratified sample should be close to:

| Attack     | Amount | Distribution |
|------------|--------|--------------|
|   Benign   | 91_492  | 36.01        |
|  scanning  | 56_721  | 22.32        |
|     xss    | 36_825  | 14.49        |
|    ddos    | 30_394  | 11.96        |
|  password  | 17_300  | 6.81         |
|     dos    | 10_689  | 4.21         |
|  injection | 10_267  | 4.04         |
| backdoor   | 252    | 0.10         |
| mitm       | 116    | 0.05         |
| ransomware | 51     | 0.02         |

The final result is as follows:

| Attack     | Amount | Distribution |
|------------|--------|--------------|
|   Benign   | 90_332  | 36.61        |
|  scanning  | 50_736  | 20.56        |
|     xss    | 36_824  | 14.92        |
|    ddos    | 30_300  | 12.28        |
|  password  | 17_245  | 6.99         |
|     dos    | 10_657  | 4.32         |
|  injection | 10_257  | 4.16         |
| backdoor   | 252    | 0.10         |
| mitm       | 116    | 0.05         |
| ransomware | 51     | 0.02         |

## NF-UNSW-NB15-v2

- Original dataset has 2_390_275 records. No data missing. No duplicate data. 
- [Article](https://ieeexplore.ieee.org/document/9625505) uses 467_539 records.
- PhoenixFL uses 358_541 (15% from the original one).

By following the original distribution, the stratified sample should be close to:

|   **Attack**   | **Amount** | **Distribution** |
|:--------------:|:----------:|:----------------:|
|     Benign     | 344283     |     0.960233     |
|    Exploits    | 4733       |     0.013200     |
|     Fuzzers    | 3346       |     0.009334     |
|     Generic    | 2484       |     0.006928     |
| Reconnaissance | 1917       |     0.005346     |
|       DoS      | 869        |     0.002424     |
| Backdoor       | 345        | 0.000962         |
| Analysis       | 325        | 0.000907         |
| Shellcode      | 214        | 0.000597         |
| Worms          | 25         | 0.000069         |

The final result is as follows:

|   **Attack**   | **Amount** | **Distribution** |
|:--------------:|:----------:|:----------------:|
|     Benign     | 330160     |     0.961786     |
|    Exploits    | 4611       |     0.013432     |
|     Fuzzers    | 3255       |     0.009482     |
|     Generic    | 1905       |     0.005549     |
| Reconnaissance | 1833       |     0.005340     |
|       DoS      | 765        |     0.002229     |
| Backdoor       | 256        | 0.000746         |
| Analysis       | 254        | 0.000740         |
| Shellcode      | 214        | 0.000623         |
| Worms          | 25         | 0.000073         |


## NF-BoT-IoT-v2

- Original dataset has 37_763_497 records. 
- [Article](https://ieeexplore.ieee.org/document/9625505) uses 278_780.
- PhoenixFL uses 339_871 records. 

By following the original distribution, the stratified sample should be close to: 

|   **Attack**   | **distribution**  | **Amount**  |
|:--------------:|:-----------------:|:----------:|
|      DDoS      |      0.485438     |   164_986  |
|       DoS      |      0.441516     |   150_058  |
| Reconnaissance |      0.069406     |   23_589   |
|     Benign     |      0.003576     |    1_215   |
|      Theft     |      0.000064     |     21     |

The final result is as follows:

|   **Attack**   | **Amount** |
|:--------------:|:----------:|
|      DDoS      |   164_479   |
|       DoS      |   149_633   |
| Reconnaissance |    23_561   |
|     Benign     |    1_216    |
|      Theft     |     24     | 

# MLP

### Complete dataset

- 927934 records. 
- **Attack**: 54.5% 
- **Benign**: 45.4%

### Train set

- 557_376 records (60%).
- **Attack**: 54.6% 
- **Benign**: 45.3%

### Evaluation set

- 185_792 records (20%)
- **Attack**: 54.6% 
- **Benign**: 45.3%

### Test set

- 185_793 records (20%)
- **Attack**: 54.6% 
- **Benign**: 45.3%
