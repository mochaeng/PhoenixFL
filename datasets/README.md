# Description

PhoenixFL currently uses four datasets for its experiments. Descriptions of each one are provided below.

A **stratified random sample (SRS)** was employed to reduce the size of the original datasets, which are too large. This process creates a smaller sample that reflects the same proportions of classes found in the original data.  A centralized dataset is also created by combining the four SRS datasets.

After the SRS, each dataset is divided into training and testing sets.

## Running

There are two Jupyter Notebook files: [`datasets-process.ipynb`](datasets-process.ipynb) and [`big-datasets-process.ipynb`](big-datasets-process.ipynb). These notebooks are used to create the stratified random samples (SRS) from each dataset. **Processing Requirements:** You will need at least 30GB of RAM to process the large datasets (you can use Kaggle, like I did LOL). There are already pre-processed SRS datasets by me located in the `pre-processed/datasets/` directory.

The script `creating-train-test-datasets.py` is used to split the pre-processed SRS datasets into training and testing sets. You can specify the desired size of the test set using the following command:

```sh
python creating-train-test-datasets.py --test-perc 0.1
```

The `--test-perc` argument must be a value between `0.1` and `0.5`, default to `0.2`. After running the script, the training and testing sets will be created in the `pre-processed/train-test/` directory.

### NF-ToN-IoT-v2

- Original dataset contains 16_940_496 records with no missing or duplicated data.
- [Article](https://ieeexplore.ieee.org/document/9625505) uses 213_460 records.
- PhoenixFL utilizes approximately 1.5% (254_107 records).

| Attack     | Amount  | Ratio (%) | Expected amount | Actual amount |
|------------|---------|-----------|-----------------|---------------|
|   Benign   | 6099469 | 36.01     | 91492           | 91492         |
|  scanning  | 3781419 | 22.32     | 56721           | 56721         |
|     xss    | 2455020 | 14.49     | 36825           | 36825         |
|    ddos    | 2026234 | 11.96     | 30394           | 30394         |
|  password  | 1153323 | 6.81      | 17300           | 17300         |
|     dos    | 712609  | 4.21      | 10689           | 10689         |
|  injection | 684465  | 4.04      | 10267           | 10267         |
| backdoor   | 16809   | 0.10      | 252             | 252           |
| mitm       | 7723    | 0.05      | 116             | 116           |
| ransomware | 3425    | 0.02      | 51              | 51            |

### NF-BoT-IoT-v2

- Original dataset contains 37_763_497 records with no missing or duplicated data.
- [Article](https://ieeexplore.ieee.org/document/9625505) uses 278_780 records.
- PhoenixFL utilizes approximately 0.6% (226_581 records).

|     Attack     |  Amount  | Ratio (%) | Expected amount | Actual amount |
|:--------------:|:--------:|:---------:|:---------------:|:-------------:|
|      Theft     |   2431   |  0.000064 |        15       |       15      |
|     Benign     |  135037  |  0.003576 |       810       |      810      |
| Reconnaissance |  2620999 |  0.069406 |      15726      |     15726     |
|       DoS      | 16673183 |  0.441516 |      100039     |     100039    |
|      DDoS      | 18331847 |  0.485438 |      109991     |     109991    |


### NF-UNSW-NB15-v2

- Original dataset contains 2_390_275 records with no missing or duplicated data.
- [Article](https://ieeexplore.ieee.org/document/9625505) uses 467_539 records.
- PhoenixFL utilizes approximately 15% (358_541 records).

| Attack         | Amount  | Ratio (%) | Expected amount | Actual amount |
|----------------|---------|-----------|-----------------|---------------|
|     Benign     | 2295222 | 96.02     | 344283          | 344283        |
|    Exploits    | 31551   | 1.32      | 4733            | 4733          |
|     Fuzzers    | 22310   | 0.93      | 3346            | 3346          |
|     Generic    | 16560   | 0.69      | 2484            | 2484          |
| Reconnaissance | 12779   | 0.53      | 1917            | 1917          |
|       DoS      | 5794    | 0.24      | 869             | 869           |
|    Analysis    | 2299    | 0.10      | 345             | 345           |
| Backdoor       | 2169    | 0.09      | 325             | 325           |
| Shellcode      | 1427    | 0.06      | 214             | 214           |
| Worms          | 164     | 0.01      | 25              | 25            |


### NF-CSE-CIC-IDS2018-v2

- Original dataset contains 18_893_708 records with no missing or duplicated data.
- [Article](https://ieeexplore.ieee.org/document/9625505) uses 177_138 records.
- PhoenixFL utilizes approximately 1% (188_936 records).

|          Attack          |  Amount  | Ratio (%) | Expected amount | Actual amount |
|:------------------------:|:--------:|:---------:|:---------------:|:-------------:|
|       SQL Injection      |    432   |  0.000023 |        4        |       4       |
|     Brute Force -XSS     |    927   |  0.000049 |        9        |       9       |
|   DDOS attack-LOIC-UDP   |   2112   |  0.000112 |        21       |       21      |
|     Brute Force -Web     |   2143   |  0.000113 |        21       |       21      |
|   DoS attacks-Slowloris  |   9512   |  0.000503 |        95       |       95      |
| DoS attacks-SlowHTTPTest |   14116  |  0.000747 |       141       |      141      |
|      FTP-BruteForce      |   25933  |  0.001373 |       259       |      259      |
|   DoS attacks-GoldenEye  |   27723  |  0.001467 |       277       |      277      |
|      SSH-Bruteforce      |   94979  |  0.005027 |       950       |      950      |
|       Infilteration      |  116361  |  0.006159 |       1164      |      1164     |
|            Bot           |  143097  |  0.007574 |       1431      |      1431     |
|  DDoS attacks-LOIC-HTTP  |  307300  |  0.016265 |       3073      |      3073     |
|     DoS attacks-Hulk     |  432648  |  0.022899 |       4326      |      4326     |
|     DDOS attack-HOIC     |  1080858 |  0.057207 |      10809      |     10809     |
|          Benign          | 16635567 |  0.880482 |      166356     |     166356    |


### Centralized dataset

- Centralized datasets contains 1_028_165 records. 

|          Attack          | Amount |
|:------------------------:|:------:|
|                   Benign | 602941 |
|                     DDoS | 109991 |
|                      DoS | 100908 |
|                 scanning |  56721 |
|                      xss |  36825 |
|                     ddos |  30394 |
|           Reconnaissance |  17643 |
|                 password |  17300 |
|         DDOS attack-HOIC |  10809 |
|                      dos |  10689 |
|                injection |  10267 |
|                 Exploits |   4733 |
|         DoS attacks-Hulk |   4326 |
|                  Fuzzers |   3346 |
|   DDoS attacks-LOIC-HTTP |   3073 |
|                  Generic |   2484 |
|                      Bot |   1431 |
|            Infilteration |   1164 |
|           SSH-Bruteforce |    950 |
|                 Analysis |    345 |
|                 Backdoor |    325 |
|    DoS attacks-GoldenEye |    277 |
|           FTP-BruteForce |    259 |
|                 backdoor |    252 |
|                Shellcode |    214 |
| DoS attacks-SlowHTTPTest |    141 |
|                     mitm |    116 |
|    DoS attacks-Slowloris |     95 |
|               ransomware |     51 |
|                    Worms |     25 |
|         Brute Force -Web |     21 |
|     DDOS attack-LOIC-UDP |     21 |
|                    Theft |     15 |
|         Brute Force -XSS |      9 |
|            SQL Injection |      4 |


Since PhoenixFL focuses on binary classification, the "Attack" column is removed. This removal results in 528 duplicate rows, which are then deleted.

- 1_027_637 records

| Label | Amount | Ratio (%) |
|:-----:|:------:|:---------:|
|     0 | 602941 |     58.67 |
|     1 | 425224 |     41.33 |
