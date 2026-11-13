/*
 * sdcap - Silion (E710, GEN22 SDK) capture for the Smart Driver shelf tests.
 * Reads chosen antennas at a set power for N seconds and writes one CSV row per read,
 * in the same columns as the R700 captures plus the end phase:
 *   timestamp_ms,epc,phase_deg,freq_mhz,rssi_dbm,antenna,end_phase_deg,phase_raw
 * phase_deg / end_phase_deg = start / end phase from the reader (12-bit, 0-360 deg,
 * decoded like trkappspsf: CRC bytes = start, Res bytes = end); phase_raw = TAGINFO.Phase.
 *
 * Build on the board:  gcc -O2 -o sdcap sdcap.c -I<sdk headers> -lModuleAPIGen22 -lpthread
 * Run via sdcap.sh (it powers the reader the way the firmware does; stop trk-app-spsf first).
 *
 *   sdcap -p /dev/fudanR0 -a 1,2,3,4 -w 20 -t 20 -o out.csv [-m 5123] [-s 1] [-g 3] [-q -1] [-r 1] [-d 250]
 */
#include "ModuleReader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <pthread.h>

static FILE *out;
static double t0;
static volatile long n_reads, n_ant[5];
static volatile int stop_now;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void on_tag(int h, TAGINFO *t, void *cookie)
{
    (void)h; (void)cookie;
    if (t->protocol == 100 || t->Epclen <= 0 || t->Epclen > MAXEPCBYTESCNT)
        return;                                         /* 100 = reader status packet */
    char epc[2 * MAXEPCBYTESCNT + 1];
    for (int i = 0; i < t->Epclen; i++)
        sprintf(epc + 2 * i, "%02X", t->EpcId[i]);
    double sp = (((t->CRC[0] << 8) | t->CRC[1]) & 0x0FFF) * 360.0 / 4096.0;
    double ep = (((t->Res[0] << 8) | t->Res[1]) & 0x0FFF) * 360.0 / 4096.0;
    pthread_mutex_lock(&mu);
    fprintf(out, "%.2f,%s,%.3f,%.3f,%d,%d,%.3f,%d\n", now_ms() - t0, epc, sp, t->Frequency / 1000.0,
            (int)(signed char)t->RSSI, (int)t->AntennaID, ep, t->Phase);
    n_reads++;
    if (t->AntennaID >= 1 && t->AntennaID <= 4)
        n_ant[t->AntennaID]++;
    pthread_mutex_unlock(&mu);
}

static void on_err(int h, READER_ERR e, void *cookie)
{
    (void)cookie;
    int code; char *desc = NULL;
    GetLastDetailError(h, &code, &desc);
    fprintf(stderr, "reader error %d (%d): %s\n", (int)e, code, desc ? desc : "");
}

static void on_sig(int s) { (void)s; stop_now = 1; }

static int set_mode(int h, int mode)
{
    /* E710 mode: 51xx = AUTOSET (cmd 0x11), 41xx = Impinj inventory mode (cmd 0x12), as in the firmware */
    MODUSPARMS mm; IMPINJ_FASTPARMS f;
    unsigned char buf[100] = "gen2op/cc_33_cmd";
    int len = 0;
    memset(&mm, 0, sizeof mm); memset(&f, 0, sizeof f);
    mm.CMD = 1; mm.isReTemp = 1;
    f.CMD = (mode >= 5000) ? 0x11 : 0x12;
    f.IMPINJ_MODE_ID = (unsigned short)mode;
    f.sub = &mm;
    if (SpecialStructDataConvert(TYPE_IMPINJFPARMS, &f, buf + 50, &len) != MT_OK_ERR) return -1;
    CustomParam_ST cp; memset(&cp, 0, sizeof cp);
    cp.pCusParam = buf; cp.CParamlen = len;
    return ParamSet(h, MTR_PARAM_CUSTOM, &cp) == MT_OK_ERR ? 0 : -1;
}

#define CHECK(call, what) do { READER_ERR _e = (call); if (_e != MT_OK_ERR) { \
    fprintf(stderr, "%s failed: %d\n", what, (int)_e); if (fatal) goto done; } else printf("  %s ok\n", what); } while (0)

int main(int argc, char **argv)
{
    const char *port = "/dev/fudanR0", *path = "sdcap.csv";
    char antstr[32] = "1,2,3,4", pwstr[32] = "20";
    int power = 20, secs = 20, mode = 5123, session = 1, target = 3, q = -1, region = 1, dur = 250;
    int c, fatal = 1, rc = 1;
    while ((c = getopt(argc, argv, "p:a:w:t:o:m:s:g:q:r:d:h")) != -1) {
        switch (c) {
        case 'p': port = optarg; break;
        case 'a': strncpy(antstr, optarg, sizeof antstr - 1); break;
        case 'w': strncpy(pwstr, optarg, sizeof pwstr - 1); power = atoi(optarg); break;  /* "20" or "12,12,14,14" (per -a antenna) */
        case 't': secs = atoi(optarg); break;
        case 'o': path = optarg; break;
        case 'm': mode = atoi(optarg); break;
        case 's': session = atoi(optarg); break;
        case 'g': target = atoi(optarg); break;
        case 'q': q = atoi(optarg); break;
        case 'r': region = atoi(optarg); break;
        case 'd': dur = atoi(optarg); break;
        default:
            printf("usage: sdcap -p port -a 1,2,3,4 -w dBm -t secs -o out.csv [-m mode] [-s session] "
                   "[-g target 0:A 1:B 2:A->B 3:B->A] [-q Q|-1] [-r region 1=NA] [-d ms per antenna]\n");
            return 1;
        }
    }
    if (power < 0 || power > 30 || secs < 1 || secs > 3600) { fprintf(stderr, "bad power or time\n"); return 1; }

    int ants[4], nant = 0;
    char antlist[32];
    strcpy(antlist, antstr);                            /* strtok below cuts antstr */
    for (char *s = strtok(antstr, ","); s && nant < 4; s = strtok(NULL, ",")) {
        int a = atoi(s);
        if (a >= 1 && a <= 4) ants[nant++] = a;
    }
    if (!nant) { fprintf(stderr, "no antennas\n"); return 1; }
    int pw[4], npw = 0;
    char pwlist[32];
    strcpy(pwlist, pwstr);
    for (char *s = strtok(pwstr, ","); s && npw < 4; s = strtok(NULL, ",")) pw[npw++] = atoi(s);
    if (npw != 1 && npw != nant) { fprintf(stderr, "-w needs 1 value or one per antenna\n"); return 1; }
    for (int i = 0; i < nant; i++) {
        if (npw == 1) pw[i] = pw[0];
        if (pw[i] < 0 || pw[i] > 30) { fprintf(stderr, "bad power %d\n", pw[i]); return 1; }
    }

    out = fopen(path, "w");
    if (!out) { perror(path); return 1; }
    fprintf(out, "timestamp_ms,epc,phase_deg,freq_mhz,rssi_dbm,antenna,end_phase_deg,phase_raw\n");

    int h = 0;
    printf("connect %s\n", port);
    CHECK(InitReader_Notype(&h, (char *)port, 4), "InitReader");

    ConnAnts_ST ca; memset(&ca, 0, sizeof ca);
    if (ParamGet(h, MTR_PARAM_READER_CONN_ANTS, &ca) == MT_OK_ERR) {
        printf("  antennas detected as connected:");
        for (int i = 0; i < ca.antcnt; i++) printf(" %d", ca.connectedants[i]);
        printf(ca.antcnt ? "\n" : " none reported\n");
    }

    Inv_Potls_ST ip; memset(&ip, 0, sizeof ip);
    ip.potlcnt = 1; ip.potls[0].potl = SL_TAG_PROTOCOL_GEN2; ip.potls[0].weight = 30;
    CHECK(ParamSet(h, MTR_PARAM_TAG_INVPOTL, &ip), "protocol Gen2");

    {   /* same as firmware setReaderCustomParam: general fast mode */
        unsigned char b[72] = "Reader/Ex10fastmode";
        b[50] = 0; b[51] = 20;
        CustomParam_ST cp; memset(&cp, 0, sizeof cp); cp.pCusParam = b; cp.CParamlen = 22;
        CHECK(ParamSet(h, MTR_PARAM_CUSTOM, &cp), "Ex10fastmode");
    }
    CHECK(ParamSet(h, MTR_PARAM_FREQUENCY_REGION, &region), "region");

    AntPowerConf apc; memset(&apc, 0, sizeof apc);
    for (int i = 0; i < nant; i++) {
        apc.Powers[i].antid = ants[i];
        apc.Powers[i].readPower = (unsigned short)(pw[i] * 100);
        apc.Powers[i].writePower = (unsigned short)(pw[i] * 100);
    }
    apc.antcnt = nant;
    CHECK(ParamSet(h, MTR_PARAM_RF_ANTPOWER, &apc), "antenna power");

    fatal = 0;                                          /* the rest: warn and carry on */
    if (set_mode(h, mode) != 0) fprintf(stderr, "mode %d failed (reader keeps its current mode)\n", mode);
    else printf("  mode %d ok\n", mode);
    CHECK(ParamSet(h, MTR_PARAM_POTL_GEN2_TARGET, &target), "target");
    CHECK(ParamSet(h, MTR_PARAM_POTL_GEN2_Q, &q), "Q");
    CHECK(ParamSet(h, MTR_PARAM_POTL_GEN2_SESSION, &session), "session");
    fatal = 1;

    OnReadingErrorBlock eb = { on_err, NULL };
    SetReadingErrorHandler(h, eb);
    OnTagReadBlock tb = { on_tag, NULL };
    SetTagReadHandler(h, tb);

    BackReadOption bro; memset(&bro, 0, sizeof bro);
    bro.IsFastRead = 1;
    bro.TMFlags.IsRSSI = 1; bro.TMFlags.IsAntennaID = 1; bro.TMFlags.IsFrequency = 1;
    bro.TMFlags.IsRFU = 1; bro.TMFlags.IsProtocol = 1; bro.TMFlags.IsEmdData = 0;
    bro.ReadDuration = (unsigned short)(dur * nant);
    bro.ReadInterval = 0;
    bro.FastReadDutyRation = 0;

    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    printf("reading antennas %s at %s dBm for %d s (mode %d, S%d, target %d, Q %d)\n",
           antlist, pwlist, secs, mode, session, target, q);
    t0 = now_ms();
    CHECK(StartReading(h, ants, nant, &bro), "StartReading");
    for (int s = 0; s < secs * 10 && !stop_now; s++)
        usleep(100000);
    StopReading(h);
    rc = 0;

done:
    if (h) CloseReader(h);
    fclose(out);
    printf("%ld reads  (A1 %ld, A2 %ld, A3 %ld, A4 %ld)  -> %s\n", n_reads, n_ant[1], n_ant[2], n_ant[3], n_ant[4], path);
    return rc;
}
