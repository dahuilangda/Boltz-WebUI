# run_single_prediction.py
import sys
import os
import json
import tempfile
import shutil
import traceback
import yaml
import hashlib
import glob
import csv
import zipfile
import requests
import time
from pathlib import Path

sys.path.append(os.getcwd())
from boltz_wrapper import predict
from config import MSA_SERVER_URL

# MSA 缓存配置
MSA_CACHE_CONFIG = {
    'cache_dir': '/tmp/boltz_msa_cache',
    'enable_cache': True
}

def get_sequence_hash(sequence: str) -> str:
    """计算序列的MD5哈希值作为缓存键"""
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()

def request_msa_from_server(sequence: str, timeout: int = 600) -> dict:
    """
    从 ColabFold MSA 服务器请求多序列比对
    
    Args:
        sequence: 蛋白质序列（FASTA 格式）
        timeout: 请求超时时间（秒）
    
    Returns:
        包含 MSA 结果的字典，如果失败则返回 None
    """
    try:
        print(f"🔍 正在从 MSA 服务器请求多序列比对: {MSA_SERVER_URL}", file=sys.stderr)
        
        # 准备请求数据
        # 确保序列是 FASTA 格式
        if not sequence.startswith('>'):
            sequence = f">query\n{sequence}"
        
        # ColabFold MSA 服务器使用 form data 格式
        payload = {
            "q": sequence,
            "mode": "colabfold"
        }
        
        # 提交搜索任务
        submit_url = f"{MSA_SERVER_URL}/ticket/msa"
        print(f"📤 提交 MSA 搜索任务到: {submit_url}", file=sys.stderr)
        
        response = requests.post(submit_url, data=payload, timeout=30)
        if response.status_code != 200:
            print(f"❌ MSA 任务提交失败: {response.status_code} - {response.text}", file=sys.stderr)
            return None
        
        result = response.json()
        ticket_id = result.get("id")
        if not ticket_id:
            print(f"❌ 未获取到有效的任务 ID: {result}", file=sys.stderr)
            return None
        
        print(f"✅ MSA 任务已提交，任务 ID: {ticket_id}", file=sys.stderr)
        
        # 轮询结果
        result_url = f"{MSA_SERVER_URL}/ticket/{ticket_id}"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                print(f"⏳ 检查 MSA 任务状态...", file=sys.stderr)
                response = requests.get(result_url, timeout=30)
                
                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get("status") == "COMPLETE":
                        print(f"✅ MSA 搜索完成，获取到结果", file=sys.stderr)
                        return result_data
                    elif result_data.get("status") == "ERROR":
                        print(f"❌ MSA 搜索失败: {result_data.get('error', '未知错误')}", file=sys.stderr)
                        return None
                    else:
                        print(f"⏳ MSA 任务状态: {result_data.get('status', 'PENDING')}", file=sys.stderr)
                elif response.status_code == 404:
                    print(f"⏳ 任务尚未完成或不存在", file=sys.stderr)
                else:
                    print(f"⚠️ 检查状态时出现错误: {response.status_code}", file=sys.stderr)
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ 检查状态时网络错误: {e}", file=sys.stderr)
            
            # 等待一段时间再次检查
            time.sleep(10)
        
        print(f"⏰ MSA 搜索超时 ({timeout}秒)", file=sys.stderr)
        return None
        
    except Exception as e:
        print(f"❌ MSA 服务器请求失败: {e}", file=sys.stderr)
        return None

def save_msa_result_to_file(msa_result: dict, output_path: str) -> bool:
    """
    将 MSA 结果保存到文件
    
    Args:
        msa_result: MSA 服务器返回的结果
        output_path: 输出文件路径
    
    Returns:
        是否成功保存
    """
    try:
        # 根据结果格式保存为 A3M 文件
        if 'entries' in msa_result:
            with open(output_path, 'w') as f:
                for entry in msa_result['entries']:
                    name = entry.get('name', 'unknown')
                    sequence = entry.get('sequence', '')
                    if sequence:
                        f.write(f">{name}\n{sequence}\n")
            return True
        else:
            print(f"❌ MSA 结果格式不支持: {msa_result.keys()}", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 保存 MSA 结果失败: {e}", file=sys.stderr)
        return False

def generate_msa_for_sequences(yaml_content: str, temp_dir: str) -> bool:
    """
    为 YAML 中的蛋白质序列生成 MSA
    
    Args:
        yaml_content: YAML 配置内容
        temp_dir: 临时目录
    
    Returns:
        是否成功生成 MSA
    """
    try:
        print(f"🧬 开始为蛋白质序列生成 MSA", file=sys.stderr)
        
        # 解析 YAML 获取蛋白质序列
        yaml_data = yaml.safe_load(yaml_content)
        protein_sequences = {}
        
        for entity in yaml_data.get('sequences', []):
            if entity.get('protein', {}).get('id'):
                protein_id = entity['protein']['id']
                sequence = entity['protein'].get('sequence', '')
                if sequence:
                    protein_sequences[protein_id] = sequence
        
        if not protein_sequences:
            print("❌ 未找到蛋白质序列，跳过 MSA 生成", file=sys.stderr)
            return False
        
        print(f"🔍 找到 {len(protein_sequences)} 个蛋白质序列需要生成 MSA", file=sys.stderr)
        
        # 为每个蛋白质序列生成 MSA
        success_count = 0
        for protein_id, sequence in protein_sequences.items():
            print(f"🧬 正在为蛋白质 {protein_id} 生成 MSA...", file=sys.stderr)
            
            # 检查临时目录中是否已经存在
            output_path = os.path.join(temp_dir, f"{protein_id}_msa.a3m")
            if os.path.exists(output_path):
                print(f"✅ 临时目录中已存在 MSA 文件: {output_path}", file=sys.stderr)
                success_count += 1
                continue
            
            # 检查缓存（统一使用 msa_ 前缀）
            sequence_hash = get_sequence_hash(sequence)
            cache_dir = MSA_CACHE_CONFIG['cache_dir']
            cached_msa_path = os.path.join(cache_dir, f"msa_{sequence_hash}.a3m")
            
            if MSA_CACHE_CONFIG['enable_cache'] and os.path.exists(cached_msa_path):
                print(f"✅ 找到缓存的 MSA 文件: {cached_msa_path}", file=sys.stderr)
                # 复制到临时目录
                shutil.copy2(cached_msa_path, output_path)
                success_count += 1
                continue
            
            # 从服务器请求 MSA
            msa_result = request_msa_from_server(sequence)
            if msa_result:
                # 保存到临时目录
                if save_msa_result_to_file(msa_result, output_path):
                    success_count += 1
                    
                    # 缓存结果（统一使用 msa_ 前缀）
                    if MSA_CACHE_CONFIG['enable_cache']:
                        os.makedirs(cache_dir, exist_ok=True)
                        shutil.copy2(output_path, cached_msa_path)
                        print(f"💾 MSA 结果已缓存: {cached_msa_path}", file=sys.stderr)
                else:
                    print(f"❌ 保存 MSA 文件失败: {protein_id}", file=sys.stderr)
            else:
                print(f"❌ 获取 MSA 失败: {protein_id}", file=sys.stderr)
        
        print(f"✅ MSA 生成完成: {success_count}/{len(protein_sequences)} 个成功", file=sys.stderr)
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 生成 MSA 时出现错误: {e}", file=sys.stderr)
        return False

def cache_msa_files_from_temp_dir(temp_dir: str, yaml_content: str):
    """
    从临时目录中缓存生成的MSA文件
    支持从colabfold server生成的CSV格式MSA文件
    为每个蛋白质组分单独缓存MSA，适用于结构预测和分子设计
    """
    if not MSA_CACHE_CONFIG['enable_cache']:
        return
    
    try:
        # 解析YAML获取蛋白质序列
        yaml_data = yaml.safe_load(yaml_content)
        protein_sequences = {}
        
        # 提取所有蛋白质序列（支持结构预测和分子设计）
        for entity in yaml_data.get('sequences', []):
            if entity.get('protein', {}).get('id'):
                protein_id = entity['protein']['id']
                sequence = entity['protein'].get('sequence', '')
                if sequence:
                    protein_sequences[protein_id] = sequence
        
        if not protein_sequences:
            print("未找到蛋白质序列，跳过MSA缓存", file=sys.stderr)
            return
        
        print(f"需要缓存的蛋白质组分: {list(protein_sequences.keys())}", file=sys.stderr)
        
        # 设置缓存目录
        cache_dir = MSA_CACHE_CONFIG['cache_dir']
        os.makedirs(cache_dir, exist_ok=True)
        
        # 递归搜索临时目录中的MSA文件
        print(f"递归搜索临时目录中的MSA文件: {temp_dir}", file=sys.stderr)
        
        # 为每个蛋白质组分单独查找对应的MSA文件
        protein_msa_map = {}  # protein_id -> [msa_files]
        
        # 搜索所有MSA文件
        all_msa_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.a3m'):
                    file_path = os.path.join(root, file)
                    all_msa_files.append(file_path)
        
        if not all_msa_files:
            print(f"在临时目录中未找到任何MSA文件: {temp_dir}", file=sys.stderr)
            return
        
        print(f"找到 {len(all_msa_files)} 个MSA文件: {[os.path.basename(f) for f in all_msa_files]}", file=sys.stderr)
        
        # 为每个蛋白质组分匹配对应的MSA文件
        for protein_id in protein_sequences.keys():
            protein_msa_map[protein_id] = []
            
            for msa_file in all_msa_files:
                filename = os.path.basename(msa_file)
                
                # 精确匹配：文件名包含protein ID
                if protein_id.lower() in filename.lower():
                    protein_msa_map[protein_id].append(msa_file)
                    continue
                    
                # 索引匹配：如果protein_id是字母，尝试匹配对应的数字索引
                # 例如：protein A -> _0.csv, protein B -> _1.csv
                if len(protein_id) == 1 and protein_id.isalpha():
                    protein_index = ord(protein_id.upper()) - ord('A')
                    if f"_{protein_index}." in filename:
                        protein_msa_map[protein_id].append(msa_file)
                        continue
                
                # 通用匹配：如果只有一个蛋白质组分，使用通用MSA文件
                if len(protein_sequences) == 1 and any(pattern in filename.lower() for pattern in ['msa', '_0.csv', '_0.a3m']):
                    protein_msa_map[protein_id].append(msa_file)
        
        # 处理每个蛋白质组分的MSA文件
        cached_count = 0
        for protein_id, msa_files in protein_msa_map.items():
            if not msa_files:
                print(f"❌ 蛋白质组分 {protein_id} 未找到对应的MSA文件", file=sys.stderr)
                continue
                
            print(f"🔍 处理蛋白质组分 {protein_id} 的 {len(msa_files)} 个MSA文件", file=sys.stderr)
            
            for msa_file in msa_files:
                if cache_single_protein_msa(protein_id, protein_sequences[protein_id], msa_file, cache_dir):
                    cached_count += 1
                    break  # 成功缓存一个就够了
        
        print(f"✅ MSA缓存完成，成功缓存 {cached_count}/{len(protein_sequences)} 个蛋白质组分", file=sys.stderr)
                
    except Exception as e:
        print(f"❌ 缓存MSA文件失败: {e}", file=sys.stderr)

def cache_single_protein_msa(protein_id: str, protein_sequence: str, msa_file: str, cache_dir: str) -> bool:
    """
    为单个蛋白质组分缓存MSA文件
    返回是否成功缓存
    """
    try:
        filename = os.path.basename(msa_file)
        file_ext = os.path.splitext(filename)[1].lower()
        
        print(f"  📂 处理MSA文件: {filename}", file=sys.stderr)
        
        if file_ext == '.csv':
            # 处理CSV格式的MSA文件（来自colabfold server）
            with open(msa_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) >= 2 and 'sequence' in header:
                    sequences = []
                    for row in reader:
                        if len(row) >= 2 and row[1]:
                            sequences.append(row[1])
                    
                    if sequences:
                        # 第一个序列通常是查询序列
                        query_sequence = sequences[0]
                        print(f"    从CSV提取的查询序列: {query_sequence[:50]}...", file=sys.stderr)
                        
                        # 验证序列是否匹配
                        if is_sequence_match(protein_sequence, query_sequence):
                            # 转换CSV格式到A3M格式
                            a3m_content = f">{protein_id}\n{query_sequence}\n"
                            for i, seq in enumerate(sequences[1:], 1):
                                a3m_content += f">seq_{i}\n{seq}\n"
                            
                            # 缓存转换后的A3M文件
                            seq_hash = get_sequence_hash(protein_sequence)
                            cache_path = os.path.join(cache_dir, f"msa_{seq_hash}.a3m")
                            with open(cache_path, 'w') as cache_file:
                                cache_file.write(a3m_content)
                            print(f"    ✅ 成功缓存蛋白质组分 {protein_id} 的MSA (从CSV转换): {cache_path}", file=sys.stderr)
                            print(f"       序列哈希: {seq_hash}", file=sys.stderr)
                            print(f"       MSA序列数: {len(sequences)}", file=sys.stderr)
                            return True
                        else:
                            print(f"    ❌ CSV文件中的查询序列与蛋白质组分 {protein_id} 不匹配", file=sys.stderr)
                            return False
        
        elif file_ext == '.a3m':
            # 处理A3M格式的MSA文件
            with open(msa_file, 'r') as f:
                msa_content = f.read()
            
            # 从MSA内容中提取查询序列（第一个序列）
            lines = msa_content.strip().split('\n')
            if len(lines) >= 2 and lines[0].startswith('>'):
                query_sequence = lines[1]
                
                # 验证序列是否匹配
                if is_sequence_match(protein_sequence, query_sequence):
                    # 缓存MSA文件
                    seq_hash = get_sequence_hash(protein_sequence)
                    cache_path = os.path.join(cache_dir, f"msa_{seq_hash}.a3m")
                    shutil.copy2(msa_file, cache_path)
                    print(f"    ✅ 成功缓存蛋白质组分 {protein_id} 的MSA: {cache_path}", file=sys.stderr)
                    print(f"       序列哈希: {seq_hash}", file=sys.stderr)
                    return True
                else:
                    print(f"    ❌ A3M文件中的查询序列与蛋白质组分 {protein_id} 不匹配", file=sys.stderr)
                    return False
        
        return False
        
    except Exception as e:
        print(f"    ❌ 处理蛋白质组分 {protein_id} 的MSA文件失败 {msa_file}: {e}", file=sys.stderr)
        return False

def is_sequence_match(protein_sequence: str, query_sequence: str) -> bool:
    """
    检查蛋白质序列和查询序列是否匹配
    支持完全匹配、容错匹配和相似度匹配
    """
    # 完全匹配
    if protein_sequence == query_sequence:
        return True
    
    # 容错匹配：去除空格和特殊字符后比较
    clean_protein = protein_sequence.replace('-', '').replace(' ', '').upper()
    clean_query = query_sequence.replace('-', '').replace(' ', '').upper()
    if clean_protein == clean_query:
        return True
    
    # 子序列匹配：查询序列可能是蛋白质序列的一部分
    if clean_query in clean_protein or clean_protein in clean_query:
        # 计算相似度
        similarity = len(set(clean_query) & set(clean_protein)) / max(len(clean_query), len(clean_protein))
        if similarity > 0.8:  # 80%相似度阈值
            return True
    
    return False

def find_results_dir(base_dir: str) -> str:
    result_path = None
    max_depth = -1
    for root, dirs, files in os.walk(base_dir):
        if any(f.endswith((".cif")) for f in files):
            depth = root.count(os.sep)
            if depth > max_depth:
                max_depth = depth
                result_path = root

    if result_path:
        print(f"Found results in directory: {result_path}", file=sys.stderr)
        return result_path

    raise FileNotFoundError(f"Could not find any directory containing result files within the base directory {base_dir}")

def get_cached_a3m_files(yaml_content: str) -> list:
    """
    获取与当前预测任务相关的a3m缓存文件
    返回缓存文件路径列表
    """
    cached_a3m_files = []
    
    if not MSA_CACHE_CONFIG['enable_cache']:
        return cached_a3m_files
    
    try:
        # 解析YAML获取蛋白质序列
        yaml_data = yaml.safe_load(yaml_content)
        protein_sequences = {}
        
        # 提取所有蛋白质序列
        for entity in yaml_data.get('sequences', []):
            if entity.get('protein', {}).get('id'):
                protein_id = entity['protein']['id']
                sequence = entity['protein'].get('sequence', '')
                if sequence:
                    protein_sequences[protein_id] = sequence
        
        if not protein_sequences:
            print("未找到蛋白质序列，跳过a3m文件收集", file=sys.stderr)
            return cached_a3m_files
        
        cache_dir = MSA_CACHE_CONFIG['cache_dir']
        if not os.path.exists(cache_dir):
            return cached_a3m_files
        
        print(f"查找缓存的a3m文件，蛋白质组分: {list(protein_sequences.keys())}", file=sys.stderr)
        
        # 为每个蛋白质序列查找对应的缓存文件
        for protein_id, sequence in protein_sequences.items():
            seq_hash = get_sequence_hash(sequence)
            cache_file_path = os.path.join(cache_dir, f"msa_{seq_hash}.a3m")
            
            if os.path.exists(cache_file_path):
                cached_a3m_files.append({
                    'path': cache_file_path,
                    'protein_id': protein_id,
                    'filename': f"{protein_id}_msa.a3m"
                })
                print(f"找到缓存文件: {protein_id} -> {cache_file_path}", file=sys.stderr)
        
        print(f"总共找到 {len(cached_a3m_files)} 个a3m缓存文件", file=sys.stderr)
        
    except Exception as e:
        print(f"获取a3m缓存文件失败: {e}", file=sys.stderr)
    
    return cached_a3m_files

def create_archive_with_a3m(output_archive_path: str, output_directory_path: str, yaml_content: str):
    """
    创建包含预测结果和a3m缓存文件的zip归档
    """
    try:
        # 获取相关的a3m缓存文件
        cached_a3m_files = get_cached_a3m_files(yaml_content)
        
        # 创建zip文件
        with zipfile.ZipFile(output_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加预测结果文件
            for root, dirs, files in os.walk(output_directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 计算相对路径，保持目录结构
                    arcname = os.path.relpath(file_path, output_directory_path)
                    zipf.write(file_path, arcname)
                    print(f"添加结果文件: {arcname}", file=sys.stderr)
            
            # 添加a3m缓存文件
            if cached_a3m_files:
                # 在zip中创建msa目录
                for a3m_info in cached_a3m_files:
                    cache_file_path = a3m_info['path']
                    filename = a3m_info['filename']
                    # 将a3m文件放在msa子目录中
                    arcname = f"msa/{filename}"
                    zipf.write(cache_file_path, arcname)
                    print(f"添加a3m缓存文件: {arcname}", file=sys.stderr)
                
                print(f"✅ 成功添加 {len(cached_a3m_files)} 个a3m缓存文件到zip归档", file=sys.stderr)
            else:
                print("⚠️ 未找到相关的a3m缓存文件", file=sys.stderr)
        
        print(f"✅ 归档创建完成: {output_archive_path}", file=sys.stderr)
        
    except Exception as e:
        print(f"❌ 创建包含a3m文件的归档失败: {e}", file=sys.stderr)
        # 如果失败，回退到原来的方式
        archive_base_name = output_archive_path.rsplit('.', 1)[0]
        created_archive_path = shutil.make_archive(
            base_name=archive_base_name,
            format='zip',
            root_dir=output_directory_path
        )
        print(f"回退到标准归档方式: {created_archive_path}", file=sys.stderr)

def main():
    """
    Main function to run a single prediction based on arguments provided in a JSON file.
    The JSON file should contain the necessary parameters for the prediction, including:
    - output_archive_path: Path where the output archive will be saved.
    - yaml_content: YAML content as a string that will be written to a temporary file.
    - Other parameters that will be passed to the predict function as command-line arguments.
    """
    if len(sys.argv) != 2:
        print("Usage: python run_single_prediction.py <args_file_path>")
        sys.exit(1)

    args_file_path = sys.argv[1]

    try:
        with open(args_file_path, 'r') as f:
            predict_args = json.load(f)

        output_archive_path = predict_args.pop("output_archive_path")

        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_content = predict_args.pop("yaml_content")

            tmp_yaml_path = os.path.join(temp_dir, 'data.yaml')
            with open(tmp_yaml_path, 'w') as tmp_yaml:
                tmp_yaml.write(yaml_content)

            predict_args['data'] = tmp_yaml_path
            predict_args['out_dir'] = temp_dir
            
            # 在预测之前生成 MSA（如果配置了 MSA 服务器）
            if MSA_SERVER_URL and MSA_SERVER_URL != "":
                print(f"🧬 开始使用 MSA 服务器生成多序列比对: {MSA_SERVER_URL}", file=sys.stderr)
                msa_generated = generate_msa_for_sequences(yaml_content, temp_dir)
                if msa_generated:
                    print(f"✅ MSA 生成成功，将用于结构预测", file=sys.stderr)
                else:
                    print(f"⚠️ MSA 生成失败，将使用默认方法进行预测", file=sys.stderr)
            else:
                print(f"ℹ️ 未配置 MSA 服务器，跳过 MSA 生成", file=sys.stderr)
            
            POSITIONAL_KEYS = ['data']
            
            cmd_positional = []
            cmd_options = []

            for key, value in predict_args.items():
                if key in POSITIONAL_KEYS:
                    cmd_positional.append(str(value))
                else:
                    if value is None:
                        continue
                    if isinstance(value, bool):
                        if value:
                            cmd_options.append(f'--{key}')
                    else:
                        cmd_options.append(f'--{key}')
                        cmd_options.append(str(value))

            cmd_args = cmd_positional + cmd_options

            print(f"DEBUG: Invoking predict with args: {cmd_args}", file=sys.stderr)
            predict.main(args=cmd_args, standalone_mode=False)

            # 在预测完成后立即缓存MSA文件
            cache_msa_files_from_temp_dir(temp_dir, yaml_content)

            output_directory_path = find_results_dir(temp_dir)

            if not os.listdir(output_directory_path):
                raise NotADirectoryError(f"Prediction result directory was found but is empty: {output_directory_path}")

            # 使用新的函数创建包含a3m文件的归档
            create_archive_with_a3m(output_archive_path, output_directory_path, yaml_content)

            if not os.path.exists(output_archive_path):
                raise FileNotFoundError(f"CRITICAL ERROR: Archive not found at {output_archive_path} immediately after creation.")
            
            print(f"DEBUG: Archive successfully created at: {output_archive_path}", file=sys.stderr)

    except Exception as e:
        print(f"Error during prediction subprocess: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()