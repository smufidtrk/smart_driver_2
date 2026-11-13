# sdcap: Silion (E710) capture on the Sapphire board

`sdcap` reads the chosen antennas of the Silion reader at a set power for N seconds. It writes one CSV row per read, in the same columns as the R700 captures plus the end phase:

```
timestamp_ms,epc,phase_deg,freq_mhz,rssi_dbm,antenna,end_phase_deg,phase_raw
```

- `phase_deg` is the start phase and `end_phase_deg` the end phase, both 0–360°. They're decoded the same way `trkappspsf` decodes them.
- `phase_raw` is the SDK's `TAGINFO.Phase`, which reads 0 on this module.

The firmware's `readTags` tool can't do a 4-antenna test because it only sets antenna 1's power. That's why this tool exists.

## On the board (root@172.16.14.3)

The tool is in `/home/root/sdcap/`, built there against the board's `libModuleAPIGen22.so`:

```
gcc -O2 -Wall -o sdcap sdcap.c -Iinc -lModuleAPIGen22 -lpthread    # inc/ = ModuleReader.h from Silion_SDK_GEN22/src
```

The main app holds the reader, so stop it first and always start it again afterwards:

```
systemctl stop trk-app-spsf
./sdcap.sh -a 1,2,3,4 -w 20 -t 60 -s 0 -g 0 -o /tmp/cap.csv
systemctl start trk-app-spsf
```

`sdcap.sh` powers reader 2 (`/dev/fudanR0` → `ttyUSB1`) through the same GPIO lines and in the same order as the firmware. It holds those lines until the capture ends.

## Options

| Option | Meaning | Default |
|---|---|---|
| `-a` | antennas | `1,2,3,4` |
| `-w` | read power, dBm | 20 |
| `-t` | seconds | 20 |
| `-m` | E710 mode: 51xx = autoset, 41xx = Impinj inventory mode (same IDs as the R700's) | 5123 |
| `-s` | Gen2 session | 1 |
| `-g` | target: 0 = A, 1 = B, 2 = A→B, 3 = B→A | 3 |
| `-q` | Q (−1 = dynamic) | −1 |
| `-r` | region (1 = North America) | 1 |
| `-d` | ms per antenna per cycle | 250 |

## First results (Nov 12, 20 dBm, 20 s, shelf as set up that day)

| Settings | Reads | A1 | A2 | A3 | A4 |
|---|---|---|---|---|---|
| S1, target B→A (the firmware's defaults) | 1,759 | 1,198 | 354 | 135 | 72 |
| **S0, target A** | **3,410** | 856 | 967 | 803 | 784 |

- **Use S0 to compare antennas.** In S1, a tag read on antenna 1 stays quiet for a while, so antennas 2–4 miss it.
- **All 4 antennas are detected and read.** The S0 capture saw 44 tags on all 50 channels.
- **Phase is clean.** Start and end phase differ by a median of 0.5°, and no reads flipped by 180°.
- **Use 60 s for phase-pattern work.** In 20 s each tag gets only about 20 channels per antenna, and the lookup wants 40 or more.

The captures are in `sessions/silion_test/`.
