#! /bin/bash
set -e

echo Downloading the MAESTRO dataset \(87 GB\) ...
curl -O https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip

echo Extracting the files ...
unzip -o maestro-v3.0.0-midi.zip


# echo Converting the audio files to FLAC ...
# COUNTER=0
# for f in MAESTRO/*/*.wav; do
#     COUNTER=$((COUNTER + 1))
#     echo -ne "\rConverting ($COUNTER/1184) ..."
#     ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 16000 ${f/\.wav/.flac}
# done

# echo
# echo Preparation complete!
