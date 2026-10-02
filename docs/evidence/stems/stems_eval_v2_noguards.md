SI-SDR dB (higher is better) / log-spectral distance dB (lower is better), against the clean mix

| scenario | n | input | full-mix model | stem pipeline | auto (two-level) |
|---|---:|---:|---:|---:|---:|
| room | 16 | 4.4 dB / 3.8 | 8.8 dB / 3.7 | 5.2 dB / 3.9 | 8.7 dB / 3.7 |
| wet_vocal | 16 | 6.2 dB / 3.0 | 7.0 dB / 3.0 | 8.3 dB / 3.1 | 8.8 dB / 3.1 |
| clean | 16 | 124.2 dB / 0.0 | 117.3 dB / 0.0 | 106.1 dB / 0.2 | 103.5 dB / 0.3 |

Auto decisions: room: room reduce 16/16, vocal reduce 2/16; wet_vocal: room reduce 7/16, vocal reduce 9/16; clean: room reduce 1/16, vocal reduce 2/16

Decisions (count of songs per action):
- room: vocals reduce 10, keep 6; drums reduce 10, keep 6; bass reduce 2, keep 14; other reduce 9, keep 7
- wet_vocal: vocals reduce 11, keep 5; drums reduce 1, keep 14, skip 1; bass keep 16; other keep 16
- clean: vocals reduce 2, keep 14; drums reduce 1, keep 14, skip 1; bass keep 16; other keep 16
