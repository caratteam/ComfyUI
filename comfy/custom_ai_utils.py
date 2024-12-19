import os
import json

supported_pt_extensions = {'ckpt', 'pt', 'bin', 'pth', 'safetensors', 'pkl', 'sft'}

# 폴더형태로 넣어줘야해서 sub-file이 너무 많아지는 경우를 제외
excluded_dirs = ['diffusers']

def _get_directory_names(path: str) -> list:
    """
    주어진 경로에서 모든 디렉토리명을 가져옵니다.
    """
    try:
        directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        return directories
    except FileNotFoundError:
        raise Exception(f"경로를 찾을 수 없습니다: {path}")
    except Exception as e:
        raise Exception(f"오류 발생: {str(e)}")


def _find_file_in_subdirectories(dirname: str, filename: str, models_path: str, possible_list: list):
    """
    주어진 경로의 모든 하위 폴더에서 특정 파일명을 찾습니다.
    """
    dirpath = os.path.join(models_path, dirname)
    try:
        output_path = None
        for root, dirs, files in os.walk(dirpath):
            filename_lower = filename.lower()
            filename_lower = filename_lower.replace(' ', '')
            files_lower = [file.lower() for file in files]
            if filename_lower in files_lower:
                full_path = os.path.join(root, filename)
                output_path = full_path.split(f'{dirpath}/')[-1]
                if output_path in possible_list:
                    return output_path
                else:
                    output_path = None
        if not output_path:
            return output_path
            
    except Exception as e:
        raise Exception(f"오류 발생: {str(e)}")


def find_file_in_models(filename: str, models_path: str, possible_list: list):
    filename_lower = filename.lower()
    filename_lower = filename_lower.replace(' ', '')
    if (
        "flux" in filename_lower
        and "dev" in filename_lower
        and "fill" not in filename_lower
        and "redux" not in filename_lower
        and "safetensors" in filename_lower
    ):
        output_path = "flux/flux1-dev-fp8.safetensors"
        return output_path

    all_dir_list = _get_directory_names(models_path)
    dir_list = [item for item in all_dir_list if item not in excluded_dirs]

    for dir in dir_list:
        output_path = _find_file_in_subdirectories(dir, filename, models_path, possible_list)
        if output_path:
            return output_path
    if not output_path:
        return None


def nordy_preprocessing(file_path: str):
    try:
        # JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 데이터 수정 (예: 특정 키 추가 또는 변경)
        node_info = data.get('nodes')

        is_change = False
        for node in node_info:
            for key, value in node.items():
                if key == "type":
                    if isinstance(value, str):
                        if "_legacy" in value:
                            origin_name = value.split('_legacy')[0]
                            node[key] = origin_name
                            is_change = True
                elif key == "properties":
                    for k, v in value.items():
                        if k == "Node name for S&R":
                            if "_legacy" in v:
                                origin_name = v.split('_legacy')[0]
                                node[key][k] = origin_name
                                is_change = True

        if is_change:
            # 수정된 데이터 다시 저장
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
        
    except Exception as e:
        print(f"[Nordy preprocessing Error] {e}")


def modify_model_name(invalid_name: str, valid_name: str, file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            workflow = json.load(file)

        node_info = workflow.get('nodes')

        for node in node_info:
            for key, value in node.items():
                if key == "widgets_values":
                    for i, widget in enumerate(value):
                        if isinstance(widget, str):
                            if invalid_name in widget:
                                node[key][i] = valid_name

        # 수정된 데이터 다시 저장
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(workflow, file, indent=4, ensure_ascii=False) 
        
        return workflow

    except Exception as e:
        print(f"[Modify model name Error] {e}")