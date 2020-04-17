import json
import os

train_json = 'datasets/train.json'
val_json = 'datasets/val.json'
# test_json = 'datasets/test.json'  # 还没放出来


def generate(json_file, output):
    f2 = open(output + '.txt', 'w')
    with open(json_file, 'r') as f:
        a = json.load(f)
        images = a['images']
        for i, image in enumerate(images):
            if i % 10000 == 9999:
                print(i, '/', len(images))
            image_id = image['image_id']
            class_id = image['class_id']
            path = os.path.join('datasets/%s/%s/%s' % (output, class_id, image_id))
            path = os.path.abspath(path)
            line = path + ' ' + class_id
            f2.writelines(line + '\n')

    f2.close()


generate(train_json, 'train')
generate(val_json, 'val')



