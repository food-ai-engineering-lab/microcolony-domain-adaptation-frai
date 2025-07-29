import os
import random


base_dir = '/mnt/projects/bhatta70/test/20x/'
new_dir = '/mnt/projects/bhatta70/train/20x-fewshot/'

base_files = os.listdir(base_dir)
new_files = os.listdir(new_dir)
for file in base_files:
    assert file not in new_files, f"File {file} already exists in new directory"

assert len(os.listdir(new_dir)) == 0, "New directory should be empty"

classes = {}
for filename in os.listdir(base_dir):
    classname = filename[:2].upper()
    if classname not in classes:
        classes[classname] = []
    classes[classname].append(filename)



k=5

for key, sample_list in classes.items():
    random.shuffle(sample_list)
    samples = sample_list[:k]
    for sample in samples:
        fp = os.path.join(base_dir, sample)
        new_fp = os.path.join(new_dir, sample)
        print(f'mv {fp} {new_fp}')
        os.system(f'mv {fp} {new_fp}')
        # remove from base_dir
        os.system(f'rm {fp}')

    print("Class: ", key, "Samples: ", len(samples))