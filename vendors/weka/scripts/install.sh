#!/usr/bin/env bash

set -e

SRC_DIR="${1:-.}"

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: '$SRC_DIR' is not a directory"
    echo "Usage: $0 [source_directory]"
    exit 1
fi

SRC_DIR="$(cd "$SRC_DIR" && pwd)"

AGENT_INSTALL_TMP=$(mktemp -d)
AGENT_EXE_PATH=$AGENT_INSTALL_TMP/weka
CLI_VERSION="5.1.0.516"
SPEC_FILE="${CLI_VERSION}.spec"

check_sha256sum() {
    if [ -n "${WEKA_SKIP_INSTALL_VERIFICATION}" ]; then
        echo "Skipping SHA256 checksum verification."
        return
    fi

    if ! jq --version &>/dev/null; then
        echo "Warning: jq is not installed." \
             "Skipping checksum verification."
        return
    fi

    if ! sha256sum --version &>/dev/null; then
        echo "Warning: sha256sum is not installed." \
             "Skipping checksum verification."
        return
    fi

    if [ ! -f "$SRC_DIR/$SPEC_FILE" ]; then
        echo "Error: $SPEC_FILE not found in $SRC_DIR"
        exit 1
    fi

    set +e
    sums_and_files=$(jq -cr \
        '.containers
         | map(.images)[]
         | to_entries | .[]
         | (.value.sha256+" "+.key)' \
        "$SRC_DIR/$SPEC_FILE")
    jq_exit_code=$?
    set -e

    if [[ $jq_exit_code -ne 0 ]]; then
        echo "Error: jq failed with exit code" \
             "$jq_exit_code."
        exit $jq_exit_code
    fi

    set +e
    sha256_output="$(
        cd "$SRC_DIR" && \
        echo "$sums_and_files" | \
        sha256sum --quiet -c --status - 2>&1
    )"
    sha256sum_exit_code=$?
    # sha256_output kept for diagnostic use
    export sha256_output
    set -e

    if [[ $sha256sum_exit_code -ne 0 ]]; then
        echo "Unpacked files failed sha256" \
             "checksum verification."
        exit $sha256sum_exit_code
    fi
}

check_sha256sum

if [ -f "$SRC_DIR/${CLI_VERSION}-${HOSTTYPE}" ]; then
    DIST_TARGET=/opt/weka/dist/image
    DIST_TARGET=${DIST_TARGET}/${CLI_VERSION}-${HOSTTYPE}
    cp "$SRC_DIR/${CLI_VERSION}-${HOSTTYPE}" \
       "$AGENT_EXE_PATH"
else
    DIST_TARGET=/opt/weka/dist/cli/${CLI_VERSION}
    cp "$SRC_DIR/${CLI_VERSION}" "$AGENT_EXE_PATH"
fi

chmod a+x "$AGENT_EXE_PATH"
"${AGENT_EXE_PATH}" local install-agent
echo "WekaIO CLI ${CLI_VERSION} is now installed" \
     "and set as active"

mkdir -p /opt/weka/dist/release /opt/weka/dist/image
cp "$SRC_DIR"/*.squashfs /opt/weka/dist/image

for f in "$SRC_DIR/${CLI_VERSION}"-*; do
    bf=$(basename "$f")
    if [ -f "$f" ] && \
       [ ! -f "/opt/weka/dist/image/$bf" ]; then
        cp "$f" /opt/weka/dist/image
    fi
done

for f in "$SRC_DIR"/wekactl*; do
    [ -f "$f" ] && cp "$f" /opt/weka/dist/image/
done

for f in "$SRC_DIR"/resources_generator*; do
    [ -f "$f" ] && cp "$f" /opt/weka/dist/image/
done

current_exists=true
cp "$SRC_DIR/$SPEC_FILE" /opt/weka/dist/release

if ! weka version current > /dev/null 2>&1; then
    weka version set "$CLI_VERSION"
    current_exists=false

    echo "Enabling version..."
    weka version get --set-current "$CLI_VERSION"
else
    echo "Validating version..."
    weka version get "$CLI_VERSION" > /dev/null 2>&1
fi

echo "Preparing version..."
weka version prepare "$CLI_VERSION"

if ! $current_exists; then
    echo "Starting weka..."
    weka local start
fi
