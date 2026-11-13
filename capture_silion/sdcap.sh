#!/bin/sh
# Run sdcap on the Sapphire board with reader 2 (/dev/fudanR0) powered the way trk-app-spsf does it.
# The main app must be stopped first (it holds the reader):  systemctl stop trk-app-spsf
#   ./sdcap.sh -a 1,2,3,4 -w 20 -t 20 -o /tmp/sdcap.csv
# GPIO lines (application/common/gpio/src/gpio_handler.cpp): RFID power gpiochip2:25,
# RFID2_ON gpiochip3:27, RFID2_EN rfid_gpiochip5:14, RFID2 reset rfid_gpiochip5:11 (all active high).
HERE=$(dirname "$0")
pids=""
cleanup() {
    for p in $pids; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# same order as RfidReader_powerOn(): supply on, reader 2 off, reader 2 on, reset pulse.
# Each line is held by its own gpioset (--mode=signal) until this script exits.
gpioset --mode=signal gpiochip2 25=1 & pids="$pids $!"   # RFID supply on
sleep 0.05
gpioset gpiochip3 27=0; gpioset rfid_gpiochip5 14=0      # reader 2 off
sleep 0.05
gpioset --mode=signal gpiochip3 27=1 & pids="$pids $!"   # reader 2 ON
gpioset --mode=signal rfid_gpiochip5 14=1 & pids="$pids $!"  # reader 2 EN
sleep 0.05
gpioset rfid_gpiochip5 11=0; sleep 0.05                  # reset low ...
gpioset --mode=signal rfid_gpiochip5 11=1 & pids="$pids $!"  # ... and high
sleep 1.5                                                # let the module boot

"$HERE/sdcap" -p /dev/fudanR0 "$@"
