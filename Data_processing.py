# 下载完整的VOC数据集并解压
# !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
# !tar -xvf VOCtrainval_06-Nov-2007.tar

# 只提取其中的三类作为新的数据集

import os
import shutil
from xml.etree import ElementTree as ET

# 配置路径
voc_root = "VOCdevkit/VOC2007"
annot_dir = os.path.join(voc_root, "Annotations")
img_dir = os.path.join(voc_root, "JPEGImages")
output_root = "VOC2007_tiny"

# 要保留的类别
target_classes = {"bird": 1, "cat": 2, "dog": 3}  # 1~3 作为新标签（背景为0）

# 准备输出目录
os.makedirs(os.path.join(output_root, "Annotations"), exist_ok=True)
os.makedirs(os.path.join(output_root, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(output_root, "ImageSets/Main"), exist_ok=True)

# 遍历所有 XML 标注文件
kept_images = set()
for xml_file in os.listdir(annot_dir):
    if not xml_file.endswith('.xml'):
        continue
    tree = ET.parse(os.path.join(annot_dir, xml_file))
    root = tree.getroot()
    # 检查是否包含目标类别
    objects = root.findall('object')
    contains_target = False
    for obj in objects:
        name = obj.find('name').text
        if name in target_classes:
            contains_target = True
            break
    if contains_target:
        # 修改标注文件，只保留目标类别
        new_objs = []
        for obj in objects:
            name = obj.find('name').text
            if name in target_classes:
                # 修改类别名为我们统一的映射（可选，也可以保留原名）
                # 这里我们将类别名改为数字字符串，方便后续解析，但保留原名也行
                # 为保持与原代码的灵活性，保留原名，只需在代码中做映射
                new_objs.append(obj)
            # 删除非目标类别的标注
        # 如果处理后还有目标，重写 XML
        if len(new_objs) > 0:
            # 删除原有 object 节点
            for obj in root.findall('object'):
                root.remove(obj)
            for obj in new_objs:
                root.append(obj)
            # 保存新的标注到子集文件夹
            tree.write(os.path.join(output_root, "Annotations", xml_file))
            kept_images.add(xml_file[:-4])   # 保存图片 id

# 复制对应的图片
for img_id in kept_images:
    src = os.path.join(img_dir, img_id + ".jpg")
    dst = os.path.join(output_root, "JPEGImages", img_id + ".jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)

# 生成 train.txt 文件（全部作为训练集，因为数据量小）
with open(os.path.join(output_root, "ImageSets/Main/train.txt"), 'w') as f:
    for img_id in sorted(kept_images):
        f.write(img_id + "\n")

print(f"抽取完成！共 {len(kept_images)} 张图片，类别：{list(target_classes.keys())}")
