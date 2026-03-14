import os, argparse, asyncio, aiohttp
from minio import Minio
from minio.error import S3Error
from pathlib import Path

"""
用于从MINIO下载数据,根据文件夹组织进行下载。
params:
    local_path: 本地文件夹
    minio_path: minio网络文件夹, 指定minio_path即可(例如： incorrect就下载整个incorrect文件夹)
usage:
    python download_minio_dir.py --minio_path incorrect/WCJSY
"""

async def download_file(minio_client, bucket_name, object_name, local_path):
    try:
        await minio_client.fget_object(bucket_name, object_name, local_path)
        print(f"成功下载文件 {object_name} 到 {local_path}")
    except S3Error as err:
        print(f"下载文件 {object_name} 失败： {err}")

async def batch_download_minio_folder(minio_client, bucket_name, folder_prefix, local_folder):
    try:
        objects = minio_client.list_objects(bucket_name, prefix=folder_prefix, recursive=True)
    except S3Error as err:
        print(f"获取文件夹 {folder_prefix} 下的文件列表失败： {err}")
        return

    tasks = []
    for obj in objects:
        object_name = obj.object_name
        if object_name.startswith(folder_prefix):
            local_path = os.path.join(local_folder, os.path.relpath(object_name, folder_prefix))
            task = asyncio.create_task(download_file(minio_client, bucket_name, object_name, local_path))
            tasks.append(task)

    await asyncio.gather(*tasks)

async def main(args):

    minio_client = Minio(
        "36.133.76.226:19011",
        access_key="minio",
        secret_key="minioihaiking2021",
        secure=False
    )
    bucket_name = "sturgeon"
    folder_prefix = "dataFactory/" + args.minio_path + "/"
    await batch_download_minio_folder(minio_client, bucket_name, folder_prefix, os.path.join(args.local_folder, args.minio_path))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download minio script.')
    parser.add_argument('--local_folder', type=str, default='.', help='local path where instore image files')
    parser.add_argument('--minio_path', type=str, default='incorrect/WCJSY/2024/06',help='minio relative path')
    args = parser.parse_args()
    # 默认下载到当前文件夹，若指定文件夹则自动创建
    Path(args.local_folder).mkdir(exist_ok=True)
    asyncio.run(main(args))

