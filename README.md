# ED Claims: CDI and Claim Funding POCs

Two proofs of concept that share one hospital claims feed. Both use synthetic data only.

| Folder | What it is |
|---|---|
| [cdi-poc/](cdi-poc/) | Clinical documentation improvement: payer rules find documentation gaps before submission; a logistic-regression model scores denial risk |
| [funding-poc/](funding-poc/) | Pre-submission claim funding: LightGBM models predict approval, paid ratio and days to cash at the lock point, and a policy turns them into tiered funding offers with a backtest |

Each folder has its own README, `requirements.txt` and tests.
