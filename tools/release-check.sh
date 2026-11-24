#!/bin/bash
# The release check. tools/publish-app.sh refuses to publish a commit that has not passed it.
#   tools/release-check.sh [phone serial ...]      (no serials: every phone on the cable)
# 1. the quick checks (no phone): unit tests, including "every downloadable model is on the Create menu"
# 2. every model download link answers, with the right files and sizes (tools/check-downloads.py)
# 3. on each phone: installs this build and runs androidTest/ReleaseCheck, which taps through the app: every page,
#    every model in the menu, a picture from each downloaded model, picture edits and Wan, Anima's guide and 4x,
#    chat answering and drawing from its card, the gallery, and the theme. It removes everything it added.
# A pass is written to build/release-check/<commit>.pass, which names the phones it ran on.
set -e
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Commit your changes first: the check must cover exactly the code that gets published."; exit 1
fi
HEAD=$(git rev-parse HEAD)
OUT=build/release-check; mkdir -p $OUT
LOG=$OUT/$HEAD.log; : > $LOG

echo "== 1. quick checks and build"
./gradlew testDebugUnitTest assembleDebug assembleDebugAndroidTest -q 2>&1 | tee -a $LOG | grep -E "^e:|FAILED|tests completed" || true
[ "${PIPESTATUS[0]}" = 0 ] || { echo "FAILED: the quick checks or the build (see $LOG)"; exit 1; }

echo "== 2. download links"
python3 -I tools/check-downloads.py | tee -a $LOG | tail -3
[ "${PIPESTATUS[0]}" = 0 ] || { echo "FAILED: a model download is broken"; exit 1; }

echo "== 3. on the phone"
PHONES=("$@"); [ ${#PHONES[@]} = 0 ] && mapfile -t PHONES < <(adb devices | awk 'NR>1 && $2=="device" {print $1}')
[ ${#PHONES[@]} = 0 ] && { echo "FAILED: no phone on the cable. The on-phone check is part of every release."; exit 1; }
APP=app/build/outputs/apk/debug/app-debug.apk
TEST=app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
for P in "${PHONES[@]}"; do
  NAME=$(adb -s $P shell getprop ro.product.model | tr -d '\r')
  echo "-- $NAME ($P)"
  # never cut into a picture being drawn
  if adb -s $P shell dumpsys activity services com.anima.npu | grep -q QueueService; then
    echo "FAILED: Anima is drawing on $NAME; try again when its queue is empty"; exit 1
  fi
  adb -s $P install -r $APP > /dev/null      # -r keeps the downloaded models
  adb -s $P install -r $TEST > /dev/null
  adb -s $P logcat -c
  adb -s $P shell am instrument -w -e class com.anima.npu.ReleaseCheck \
    com.anima.npu.test/androidx.test.runner.AndroidJUnitRunner 2>&1 | tee $OUT/$HEAD.$P.txt | tail -15
  adb -s $P logcat -d -s ReleaseCheck:I | tee -a $LOG | grep -E "done in|no |nothing" || true
  if ! grep -q "^OK (" $OUT/$HEAD.$P.txt; then echo "FAILED on $NAME (see $OUT/$HEAD.$P.txt)"; exit 1; fi
  echo "$NAME $P $(date -Is)" >> $OUT/$HEAD.pass.tmp
done
mv $OUT/$HEAD.pass.tmp $OUT/$HEAD.pass
echo "PASSED: $(git log --oneline -1) on ${#PHONES[@]} phone(s)"
