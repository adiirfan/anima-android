#!/bin/bash
# Build the app stamped with the build time, and publish it as a new release of the GitHub repo
# adiirfan/anima-android (anima.apk + latest.json), which every installed copy checks: they compare this build
# number with their own and offer the update when it is newer. Needs the GitHub CLI, logged in (gh auth login).
#   tools/publish-app.sh "one line about what changed"
set -e
cd "$(dirname "$0")/.."
REPO=${REPO:-adiirfan/anima-android}
BUILD=$(date +%s)
./gradlew assembleDebug -PappBuild=$BUILD -q
OUT=build/release-$BUILD; mkdir -p $OUT
cp app/build/outputs/apk/debug/app-debug.apk $OUT/anima.apk
python3 - "$OUT" "$BUILD" "${1:-}" <<'PYEOF'
import hashlib, json, os, sys
out, build, notes = sys.argv[1], int(sys.argv[2]), sys.argv[3]
apk = os.path.join(out, "anima.apk")
sha = hashlib.sha256(open(apk, "rb").read()).hexdigest()
json.dump({"build": build, "size": os.path.getsize(apk), "sha256": sha, "notes": notes}, open(os.path.join(out, "latest.json"), "w"))
PYEOF
gh release create "b$BUILD" $OUT/anima.apk $OUT/latest.json --repo "$REPO" --title "Build $BUILD" --notes "${1:-}" --latest
echo "published build $BUILD ($OUT)"
