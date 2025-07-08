from modelscope.hub.api import HubApi
import os

def upload_dataset_to_modelscope():
    # 1. Replace with your ModelScope SDK Access Token
    # You can get your token from: https://modelscope.cn/my/myaccesstoken
    your_access_token = '7a98f476-9ce6-4deb-951d-099ea11c4edb' # Example token, replace if necessary

    # 2. Initialize HubApi and login
    api = HubApi()
    try:
        api.login(your_access_token)
        print("Successfully logged in to ModelScope.")
    except Exception as e:
        print(f"Failed to login to ModelScope: {e}")
        return

    # 3. Define your repository and local folder details
    repo_id = 'LXX3123/UAV-FLOW-lerobot-pass'

    local_dataset_folder = '/home/adminroot/lxx/dataset/uav_flow_lerobot_format/fixed_command/pass/train/uav_flow' # IMPORTANT: Update this path
    
    if not os.path.isdir(local_dataset_folder):
        print(f"Error: The specified local_dataset_folder does not exist or is not a directory: {local_dataset_folder}")
        return

    # This will be the base directory in your ModelScope repository
    # where the contents of local_dataset_folder will be uploaded.
    # For example, if local_dataset_folder contains 'images/pic.jpg',
    # it will be uploaded to 'LXX3123/UAV-FLOW-lerobot-pass/UAV-FLOW-lerobot-pass_data_files/images/pic.jpg'.
    base_path_in_repo = '/' # You can customize this folder name

    # 4. Upload files individually
    print(f"Attempting to upload files from '{local_dataset_folder}' to ModelScope repository '{repo_id}' under '{base_path_in_repo}'...")

    for root, _, files in os.walk(local_dataset_folder):
        if not files:
            # Handle empty subdirectories if necessary, or just skip
            if root == local_dataset_folder and not _: # Check if the root itself is empty of subdirs too
                 print(f"Warning: No files found directly in '{local_dataset_folder}'. Subdirectories will be scanned.")
            elif root != local_dataset_folder:
                 print(f"Info: No files in subdirectory '{root}'. Skipping.")

        files.sort() # Add this line to sort files alphabetically

        for filename in files:
            local_file_path = os.path.join(root, filename)
            
            # Calculate relative path from local_dataset_folder to the current file
            # This preserves the directory structure.
            relative_file_path = os.path.relpath(local_file_path, local_dataset_folder)
            
            # Construct the path in the repository
            # Joins the base_path_in_repo with the relative file path
            repo_file_path = os.path.join(base_path_in_repo, relative_file_path)
            
            # Ensure repository paths use forward slashes, common for web/APIs
            repo_file_path = repo_file_path.replace(os.sep, '/')
            
            commit_message = f'Upload dataset file'

            print(f"Uploading '{local_file_path}' to '{repo_id}/{repo_file_path}'...")
            try:
                api.upload_file(
                    path_or_fileobj=local_file_path,
                    path_in_repo=repo_file_path,
                    repo_id=repo_id,
                    repo_type='dataset', # Must be 'dataset' for datasets
                    commit_message=commit_message
                    # revision='master' # Optional: Target branch, default is 'master'
                )
                print(f"Successfully uploaded '{local_file_path}' to '{repo_id}/{repo_file_path}'.")
            except Exception as e:
                print(f"An error occurred during upload of '{local_file_path}': {e}")
                # Decide if you want to continue with other files or stop
                # For now, it continues

    print(f"\nFinished attempting to upload files from '{local_dataset_folder}'.")
    print(f"You can view your dataset at: https://modelscope.cn/datasets/{repo_id}/files")

if __name__ == '__main__':
    # Before running, ensure you have the modelscope library installed:
    # pip install modelscope
    upload_dataset_to_modelscope()