SI-SDR dB (higher is better) / log-spectral distance dB (lower is better), against the clean mix

| scenario | n | input | full-mix model | stem pipeline | auto (two-level) |
|---|---:|---:|---:|---:|---:|
| room | 24 | 4.7 dB / 4.0 | 9.8 dB / 3.8 | 5.3 dB / 4.2 | 9.7 dB / 3.9 |
| wet_vocal | 24 | 6.7 dB / 2.9 | 7.3 dB / 2.9 | 7.7 dB / 3.0 | 7.8 dB / 2.9 |
| clean | 24 | 124.1 dB / 0.0 | 124.1 dB / 0.0 | 114.8 dB / 0.2 | 119.6 dB / 0.0 |

Auto decisions: room: room reduce 24/24, vocal reduce 0/24; wet_vocal: room reduce 7/24, vocal reduce 7/24; clean: room reduce 0/24, vocal reduce 1/24

Decisions (count of songs per action):
- room: vocals reduce 4, keep 20; drums reduce 14, keep 10; bass reduce 3, keep 20, skip 1; other reduce 16, keep 8
- wet_vocal: vocals reduce 9, keep 15; drums keep 23, skip 1; bass keep 23, skip 1; other keep 24
- clean: vocals reduce 2, keep 22; drums keep 23, skip 1; bass keep 23, skip 1; other keep 24
