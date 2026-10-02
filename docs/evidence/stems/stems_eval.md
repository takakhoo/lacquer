SI-SDR dB (higher is better) / log-spectral distance dB (lower is better), against the clean mix

| scenario | n | input | full-mix model | stem pipeline | auto (two-level) |
|---|---:|---:|---:|---:|---:|
| room | 24 | 4.7 dB / 4.0 | 9.1 dB / 3.9 | 5.3 dB / 4.3 | 9.0 dB / 4.0 |
| wet_vocal | 24 | 6.7 dB / 2.9 | 7.9 dB / 3.0 | 8.0 dB / 3.0 | 8.7 dB / 3.0 |
| clean | 24 | 124.1 dB / 0.0 | 119.5 dB / 0.0 | 111.7 dB / 0.2 | 114.9 dB / 0.1 |

Auto decisions: room: room reduce 24/24, vocal reduce 1/24; wet_vocal: room reduce 13/24, vocal reduce 9/24; clean: room reduce 1/24, vocal reduce 1/24

Decisions (count of songs per action):
- room: vocals reduce 9, keep 15; drums reduce 15, keep 9; bass reduce 3, keep 20, skip 1; other reduce 13, keep 11
- wet_vocal: vocals reduce 11, keep 13; drums reduce 1, keep 22, skip 1; bass keep 23, skip 1; other keep 24
- clean: vocals reduce 2, keep 22; drums reduce 1, keep 22, skip 1; bass keep 23, skip 1; other keep 24
