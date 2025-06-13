import json
import sys

def merge_json_dicts(file1, file2, output_filename="combined_output.json"):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged = {}

    all_keys = set(data1.keys()) | set(data2.keys())
    for key in all_keys:
        if key in data1 and key in data2:
            merged[key] = {**data1[key], **data2[key]}
        elif key in data1:
            merged[key] = data1[key]
        else:
            merged[key] = data2[key]

    with open(output_filename, 'w') as out:
        json.dump(merged, out, indent=2)

    print(f"Merged JSON written to '{output_filename}'")

if __name__ == "__main__":
    filename1 = "table_1_compress.json"
    filename2_clip = "table_1_compress_clip.json"
    storing_name = "ablations_final/table_1_compress_final.json"
    merge_json_dicts(file1=filename1,file2=filename2_clip,output_filename=storing_name)
