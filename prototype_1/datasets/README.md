# Description

PhoenixFL currently uses four datasets for its experiments. Descriptions of each one are provided below.

A **stratified random sample (SRS)** was employed to reduce the size of the original datasets, which are too large. This process creates a smaller sample that reflects the same proportions of classes found in the original data. A centralized dataset is also created by combining the four SRS datasets.

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
- PhoenixFL utilizes approximately ?% (250_00 records) of the original.

<!-- - I removed the indexes `325792 (benign)`, `356361 (Benign)` from the column `DST_TO_SRC_SECOND_BYTES` because they were equals to: $2.3 \cdot 10^{139}$ and $2.6 \cdot 10^{53}$. Clearly outliers -->

| Attack     | Amount  | Original Ratio (%) | SRS   |
| ---------- | ------- | ------------------ | ----- |
| Benign     | 6099469 | 36.01              | 90013 |
| scanning   | 3781419 | 22.32              | 55804 |
| xss        | 2455020 | 14.49              | 36230 |
| ddos       | 2026234 | 11.96              | 29902 |
| password   | 1153323 | 6.81               | 17020 |
| dos        | 712609  | 4.21               | 10516 |
| injection  | 684465  | 4.04               | 10101 |
| backdoor   | 16809   | 0.10               | 248   |
| mitm       | 7723    | 0.05               | 114   |
| ransomware | 3425    | 0.02               | 51    |

### NF-BoT-IoT-v2

- Original dataset contains 37_763_497 records with no missing or duplicated data.
- PhoenixFL utilizes approximately ?% (250_000 records) of the original.

|     Attack     |  Amount  | Ratio (%) | Actual amount |
| :------------: | :------: | :-------: | :-----------: |
|     Theft      |   2431   | 0.000064  |      16       |
|     Benign     |  135037  | 0.003576  |      894      |
| Reconnaissance | 2620999  | 0.069406  |     17351     |
|      DoS       | 16673183 | 0.441516  |    110379     |
|      DDoS      | 18331847 | 0.485438  |    121360     |

### NF-UNSW-NB15-v2

- Original dataset contains 2_390_275 records with no missing or duplicated data.
- PhoenixFL utilizes approximately ?% (250_000 records).

| Attack         | Amount  | Ratio (%) | Actual amount |
| -------------- | ------- | --------- | ------------- |
| Benign         | 2295222 | 96.02     | 240058        |
| Exploits       | 31551   | 1.32      | 3300          |
| Fuzzers        | 22310   | 0.93      | 2333          |
| Generic        | 16560   | 0.69      | 1732          |
| Reconnaissance | 12779   | 0.53      | 1337          |
| DoS            | 5794    | 0.24      | 606           |
| Analysis       | 2299    | 0.10      | 240           |
| Backdoor       | 2169    | 0.09      | 227           |
| Shellcode      | 1427    | 0.06      | 149           |
| Worms          | 164     | 0.01      | 17            |

### NF-CSE-CIC-IDS2018-v2

- Original dataset contains 18_893_708 records with no missing or duplicated data.
- PhoenixFL utilizes approximately ?% (250_000 records).

- I removed the index `111639 (Benign)`, `98557 (Benign)`, `237729 (Benign)`, `166566 (Benign)` from the column `SRC_TO_DST_SECOND_BYTES` because the values were equal to $8.8 \cdot 10^{213}$, $8.8 \cdot 10^{201}$, $8.8 \cdot 10^{201}$, $8.8 \cdot 10^{199}$. Clearly outliers that were causing the std to be `inf`.

|          Attack          |  Amount  | Ratio (%) | Actual amount |
| :----------------------: | :------: | :-------: | :-----------: |
|      SQL Injection       |   432    | 0.000023  |       6       |
|     Brute Force -XSS     |   927    | 0.000049  |      12       |
|   DDOS attack-LOIC-UDP   |   2112   | 0.000112  |      28       |
|     Brute Force -Web     |   2143   | 0.000113  |      28       |
|  DoS attacks-Slowloris   |   9512   | 0.000503  |      126      |
| DoS attacks-SlowHTTPTest |  14116   | 0.000747  |      187      |
|      FTP-BruteForce      |  25933   | 0.001373  |      343      |
|  DoS attacks-GoldenEye   |  27723   | 0.001467  |      367      |
|      SSH-Bruteforce      |  94979   | 0.005027  |     1257      |
|      Infilteration       |  116361  | 0.006159  |     1540      |
|           Bot            |  143097  | 0.007574  |     1893      |
|  DDoS attacks-LOIC-HTTP  |  307300  | 0.016265  |     4066      |
|     DoS attacks-Hulk     |  432648  | 0.022899  |     5725      |
|     DDOS attack-HOIC     | 1080858  | 0.057207  |     14302     |
|          Benign          | 16635567 | 0.880482  |    220120     |

### Centralized dataset

- Centralized datasets contains ? records.

|          Attack          | Amount |
| :----------------------: | :----: |
|          Benign          |        |
|           DDoS           |        |
|           DoS            |        |
|         scanning         |        |
|           xss            |        |
|           ddos           |        |
|      Reconnaissance      |        |
|         password         |        |
|     DDOS attack-HOIC     |        |
|           dos            |        |
|        injection         |        |
|         Exploits         |        |
|     DoS attacks-Hulk     |        |
|         Fuzzers          |        |
|  DDoS attacks-LOIC-HTTP  |        |
|         Generic          |        |
|           Bot            |        |
|      Infilteration       |        |
|      SSH-Bruteforce      |        |
|         Analysis         |        |
|         Backdoor         |        |
|  DoS attacks-GoldenEye   |        |
|      FTP-BruteForce      |        |
|         backdoor         |        |
|        Shellcode         |        |
| DoS attacks-SlowHTTPTest |        |
|           mitm           |        |
|  DoS attacks-Slowloris   |        |
|        ransomware        |        |
|          Worms           |        |
|     Brute Force -Web     |        |
|   DDOS attack-LOIC-UDP   |        |
|          Theft           |        |
|     Brute Force -XSS     |        |
|      SQL Injection       |        |
