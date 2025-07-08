import cv2
import os
import glob

def create_video_from_images(image_folder, video_name, fps=30):
    """
    从文件夹中的图片序列创建视频。

    :param image_folder: 包含图片的文件夹路径。
    :param video_name: 输出视频文件的名称 (例如 'output.mp4')。
    :param fps: 视频的帧率。
    """
    # 获取所有png图片的文件路径
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))

    if not image_files:
        print(f"在文件夹 '{image_folder}' 中未找到PNG图片。")
        return

    # 读取第一张图片以获取帧的尺寸
    frame = cv2.imread(image_files[0])
    if frame is None:
        print(f"无法读取第一张图片: {image_files[0]}")
        return
    height, width, layers = frame.shape
    size = (width, height)

    # 定义视频编码器并创建VideoWriter对象
    # 'mp4v' 是 .mp4 文件常用的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, size)

    # 逐一读取图片并写入视频
    for filename in image_files:
        img = cv2.imread(filename)
        if img is not None:
            out.write(img)
        else:
            print(f"警告: 跳过无法读取的文件 {filename}")

    # 释放VideoWriter对象
    out.release()
    print(f"视频 '{video_name}' 已成功创建。")

if __name__ == '__main__':
    # 设置图片文件夹路径和输出视频名称
    img_dir = '/hpc2hdd/home/zlin810/TestData/drone/UAV-Flow/trajectory_video/image_ep0_with_arrow'
    output_video = 'test.mp4'
    
    # 调用函数创建视频
    create_video_from_images(img_dir, output_video)