# q4_k RTF-080 queue receipt

This receipt records the already measured warm resident comparison on the AMD
Radeon 780M (`q4_k`, 213 frames, 17.04 s audio, six flushes per configuration).
The compact raw rows are in [rtf-080-queue.json](rtf-080-queue.json), copied
from `.beehive/agent/BREEZE-RTF-080-IMPL/queue-comparison.json`.

The benchmark CLI was commit `0ed7de69817e3b3380664c6d689606fb83541220`,
binary MD5 `a1bb5070e538f2bc45c117431207d023`. The same seeded request was used
for all twelve resident runs. Every run produced SHA-256
`67984c45dd90ffe4f6bba7c34b6c39401a793706f4dedbe9e0897705c1dddb3e`.

| configuration | runs | mean wall | mean wall RTF | first audio mean (range) | max playback deficit (range) |
| --- | ---: | ---: | ---: | ---: | ---: |
| control, queue variable absent | 6 | 18,263.2 ms | 1.07178599 | 3,026.3 ms (2,954–3,119) | 904.5 ms (761.4–1,042.0) |
| graphics queue enabled | 6 | 18,046.8 ms | 1.05908646 | 2,990.8 ms (2,955–3,083) | 717.1 ms (586.6–898.5) |

The graphics queue reduces mean resident wall time by 216.4 ms per 17.04-second
take, or `1.015962 ms` per audio frame. This is below the earlier 1.66 ms/frame
paired estimate. The wall RTF remains above 0.8, and the recorded playback
deficits do not support a no-underrun claim; the target is not achieved.

Retain the queue selection as an explicit Radeon 780M runtime recipe only:

```sh
# control: presence switch must be absent; assigning 0 still enables it
env -u GGML_VK_ALLOW_GRAPHICS_QUEUE ... breeze-cli ...

# measured candidate
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ... breeze-cli ...
```

No global engine or production default changed. Other implementation stages
remain in progress.
