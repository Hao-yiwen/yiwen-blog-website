#!/bin/bash

# Function to measure and display compression and decompression metrics
measure_compression() {
    local cmd=$1
    local file=$2
    local compressed_file=$3
    local decompress_cmd=$4

    echo "Compressing with $cmd..."

    # Measure compression time
    start_time=$(date +%s.%N)
    eval "$cmd $file" > /dev/null
    end_time=$(date +%s.%N)
    compression_time=$(echo "($end_time - $start_time) * 1000" | bc -l)

    original_size=$(stat -f%z "$file")
    compressed_size=$(stat -f%z "$compressed_file")

    # Calculate compression ratio
    compression_ratio=$(echo "scale=2; $compressed_size / $original_size" | bc)

    echo "Original Size: $original_size bytes"
    echo "Compressed Size: $compressed_size bytes"
    echo "Compression Ratio: $compression_ratio"
    printf "Compression Time: %.2f ms\n" $compression_time

    echo "Decompressing with $decompress_cmd..."

    # Measure decompression time
    start_time=$(date +%s.%N)
    eval "$decompress_cmd $compressed_file" > /dev/null
    end_time=$(date +%s.%N)
    decompression_time=$(echo "($end_time - $start_time) * 1000" | bc -l)

    printf "Decompression Time: %.2f ms\n" $decompression_time
    echo
}

file="main.jsbundle"
gzip_file="main.jsbundle.gz"
zstd_file="main.jsbundle.zst"
seven_zip_file="main_jsbundle.7z"

# Ensure the example file exists
if [ ! -f "$file" ]; then
    echo "File $file not found!"
    exit 1
fi

# Measure gzip
measure_compression "gzip -k -f" "$file" "$gzip_file" "gzip -d -f"

# Measure zstd
measure_compression "zstd -q -f -o $zstd_file" "$file" "$zstd_file" "zstd -d -q -f -o $file"

# Measure 7z
measure_compression "7z a -y -bd $seven_zip_file" "$file" "$seven_zip_file" "7z x -y -bd $seven_zip_file"

# Clean up
rm -f "$gzip_file" "$zstd_file" "$seven_zip_file"