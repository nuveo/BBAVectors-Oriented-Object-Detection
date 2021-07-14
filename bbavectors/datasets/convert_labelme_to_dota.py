import os
import json
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert labelme annotations to DOTA format.')
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, default=None)
    args = parser.parse_args()
    return args


def proc_annotation(dst_dir, filepath):
    dst_dir = os.path.join(dst_dir, "labelTxt")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    txtpath = os.path.join(
        dst_dir,
        os.path.basename(filepath).rsplit(".", 1)[0] + ".txt"
    )

    with open(filepath, 'r') as fp:
        json_data = json.load(fp)

    with open(txtpath, 'w') as fp:
        for shape in json_data['shapes']:
            label = [shape['label']]
            difficult = [0]
            points = sum(shape['points'], [])
            points = [round(p) for p in points]
            assert len(points) == 8

            data = tuple(points + label + difficult)
            line = "%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %s %d\n" % data
            fp.write(line)


def proc_image(dst_dir, filepath):
    dst_dir = os.path.join(dst_dir, "images")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    shutil.copy(filepath, dst_dir)


if __name__ == "__main__":
    args = parse_args()

    if args.dst is None:
        args.dst = os.path.dirname(args.src)

    for file in os.listdir(args.src):
        filepath = os.path.join(args.src, file)
        if ".json" in file:
            proc_annotation(args.dst, filepath)
        else:
            proc_image(args.dst, filepath)
