# Sequential RL Robustness Sweep

Sequential card-session env · Dueling Double DQN · fraction=0.3 · 35 episodes/seed · 3/8 seeds done.

**Metric:** total cost £ (fraud lost + friction) on the full card set — lower is better. DQN also reports fraud recall by value.

| Seed | Threshold only | Block(1) | Block(2) | **Dueling DQN** | DQN recall | DQN vs threshold |
|-----:|---------------:|---------:|---------:|----------------:|-----------:|-----------------:|
| 1 | £20,083 | £23,977 | £29,465 | **£11,355** | 0.998 | +43% |
| 2 | £29,438 | £29,102 | £34,505 | **£10,634** | 0.996 | +64% |
| 3 | £23,049 | £28,164 | £30,212 | **£8,713** | 0.999 | +62% |
| **mean±std** | £24,190±4,781 | £27,081±2,728 | £31,394±2,720 | **£10,234±1,366** | 0.998 | **+57%±11** |

**Headline:** over 3 seeds the Dueling DQN cuts total cost by **57% ± 11%** vs the static threshold (£10,234±1,366 vs £24,190±4,781), at 99.8% fraud recall by value.
