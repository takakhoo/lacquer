SI-SDR dB (higher is better) / log-spectral distance dB (lower is better), against the clean mix

| scenario | n | input | full-mix model | stem pipeline | auto (two-level) |
|---|---:|---:|---:|---:|---:|
| room | 24 | 4.9 dB / 4.8 | 8.4 dB / 4.5 | 5.0 dB / 4.8 | 8.2 dB / 4.5 |
| wet_vocal | 24 | 6.4 dB / 3.7 | 6.6 dB / 3.7 | 7.2 dB / 3.6 | 7.4 dB / 3.6 |
| clean | 24 | 124.1 dB / 0.0 | 124.1 dB / 0.0 | 114.8 dB / 0.2 | 119.6 dB / 0.0 |

Auto decisions: room: room reduce 21/24, vocal reduce 1/24; wet_vocal: room reduce 3/24, vocal reduce 8/24; clean: room reduce 0/24, vocal reduce 1/24

Decisions (count of songs per action):
- room: vocals reduce 2, keep 22; drums reduce 10, keep 14; bass reduce 2, keep 21, skip 1; other reduce 10, keep 14
- wet_vocal: vocals reduce 9, keep 15; drums keep 23, skip 1; bass keep 23, skip 1; other keep 24
- clean: vocals reduce 2, keep 22; drums keep 23, skip 1; bass keep 23, skip 1; other keep 24
