from modelscope.hub.api import HubApi
import os

def upload_single_file_to_modelscope():
    # 1. Replace with your ModelScope SDK Access Token
    your_access_token = '7a98f476-9ce6-4deb-951d-099ea11c4edb' # Example token, replace if necessary

    # 2. Initialize HubApi and login
    api = HubApi()
    try:
        api.login(your_access_token)
        print("Successfully logged in to ModelScope.")
    except Exception as e:
        print(f"Failed to login to ModelScope: {e}")
        return

    # 3. Define your repository and file details
    repo_id = 'LXX3123/UAV-FLOW'
    
    # --- Specify the single file to upload ---
    local_file_to_upload = '/home/adminroot/lxx/dataset/UAV-FLOW/train-00026-of-00054.parquet'
    # --- Specify the desired path in the repository for this file ---
    # If you want it at the root of the dataset, just use the filename.
    # If it was in a subdirectory in the original structure, replicate that here.
    # For example: 'subfolder/train-00026-of-00054.parquet'
    path_in_repo_for_file = 'train-00026-of-00054.parquet' 
    # --- End of single file specification ---

    if not os.path.isfile(local_file_to_upload):
        print(f"Error: The specified local file does not exist: {local_file_to_upload}")
        return

    commit_message = f'Upload specific file'

    print(f"Attempting to upload '{local_file_to_upload}' to '{repo_id}/{path_in_repo_for_file}'...")
    try:
        api.upload_file(
            path_or_fileobj=local_file_to_upload,
            path_in_repo=path_in_repo_for_file,
            repo_id=repo_id,
            repo_type='dataset', # Must be 'dataset' for datasets
            commit_message=commit_message
            # revision='master' # Optional: Target branch, default is 'master'
        )
        print(f"Successfully uploaded '{local_file_to_upload}' to '{repo_id}/{path_in_repo_for_file}'.")
    except Exception as e:
        print(f"An error occurred during upload of '{local_file_to_upload}': {e}")

    print(f"\nFinished attempting to upload the single file.")
    print(f"You can view your dataset at: https://modelscope.cn/datasets/{repo_id}/files")

if __name__ == '__main__':
    # Before running, ensure you have the modelscope library installed:
    # pip install modelscope
    upload_single_file_to_modelscope()