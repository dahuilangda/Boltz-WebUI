import streamlit as st
import requests
import time
import json
import zipfile
import io
import yaml
import string
import uuid
import py3Dmol
import re
import subprocess
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.Structure import Structure
import math
import os
import pandas as pd
import glob
from datetime import datetime
import tempfile
from streamlit_ketcher import st_ketcher

try:
    import psutil
except ImportError:
    psutil = None

API_URL = "http://127.0.0.1:5000"
TYPE_TO_DISPLAY = {
    'protein': '🧬 蛋白质',
    'ligand': '💊 小分子',
    'dna': '🔗 DNA',
    'rna': '📜 RNA'
}

TYPE_SPECIFIC_INFO = {
    'protein': {
        'placeholder': "例如: MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV...",
        'help': "请输入标准的单字母氨基酸序列。"
    },
    'dna': {
        'placeholder': "例如: GTCGAC... (A, T, C, G)",
        'help': "请输入标准的单字母脱氧核糖核酸序列 (A, T, C, G)。"
    },
    'rna': {
        'placeholder': "例如: GUCGAC... (A, U, C, G)",
        'help': "请输入标准的单字母核糖核酸序列 (A, U, C, G)。"
    }
}

# Designer 相关配置
DESIGNER_CONFIG = {
    'work_dir': '/tmp/boltz_designer',
    'api_token': os.getenv('API_SECRET_TOKEN', 'your_default_api_token'),
    'server_url': API_URL
}


def read_cif_from_string(cif_content: str) -> Structure:
    """Parses a CIF string into a BioPython Structure object."""
    parser = MMCIFParser(QUIET=True)
    file_like = io.StringIO(cif_content)
    structure = parser.get_structure('protein', file_like)
    return structure

def extract_protein_residue_bfactors(structure: Structure):
    """Extracts b-factors for protein/rna/dna residues only."""
    residue_bfactors = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                hetfield = residue.get_id()[0]
                if hetfield.strip() == "":
                    resseq = residue.get_id()[1]
                    chain_id = chain.id
                    atom_bfactors = [atom.get_bfactor() for atom in residue]
                    if atom_bfactors:
                        avg_bfactor = sum(atom_bfactors) / len(atom_bfactors)
                        residue_bfactors[(chain_id, resseq)] = avg_bfactor
    return residue_bfactors

def get_color_from_bfactor(bfactor: float) -> str:
    """Maps a b-factor (pLDDT score) to a specific color."""
    if bfactor >= 90: return '#0053D6'
    elif 70 <= bfactor < 90: return '#65CBF3'
    elif 50 <= bfactor < 70: return '#FFDB13'
    else: return '#FF7D45'

def visualize_structure_py3dmol(
    cif_content: str,
    residue_bfactors: dict,
    protein_style: str = 'cartoon',
    ligand_style: str = 'ball-and-stick',
    spin: bool = False,
    color_scheme: str = 'pLDDT'
) -> str:
    view = py3Dmol.view(width='100%', height=600)
    view.addModel(cif_content, 'cif')

    if color_scheme == 'pLDDT':
        if protein_style == 'cartoon':
            view.setStyle({'model': -1, 'hetflag': False}, {'cartoon': {'color': 'white'}})
            for (chain_id, resseq), avg_bfactor in residue_bfactors.items():
                color = get_color_from_bfactor(avg_bfactor)
                view.setStyle({'chain': chain_id, 'resi': resseq}, {'cartoon': {'color': color}})
        elif protein_style == 'stick':
            view.setStyle({'model': -1, 'hetflag': False}, {'stick': {'colorscheme': 'default'}})
        elif protein_style == 'sphere':
            view.setStyle({'model': -1, 'hetflag': False}, {'sphere': {'colorscheme': 'default'}})

        ligand_color_map = {}
        try:
            parsed_structure = read_cif_from_string(cif_content)
            for atom in parsed_structure.get_atoms():
                residue = atom.get_parent()
                if residue.get_id()[0].strip() != "":
                    serial = atom.get_serial_number()
                    bfactor = atom.get_bfactor()
                    color = get_color_from_bfactor(bfactor)
                    ligand_color_map[serial-1] = color
        except Exception as e:
            print(f"Error parsing CIF for ligand coloring: {e}")
        
        custom_colorscheme = {'prop': 'serial', 'map': ligand_color_map}
        if ligand_color_map:
            if ligand_style == 'ball-and-stick':
                view.setStyle({'hetflag': True}, {'stick': {'colorscheme': custom_colorscheme, 'radius': 0.15}})
                view.addStyle({'hetflag': True}, {'sphere': {'colorscheme': custom_colorscheme, 'scale': 0.25}})
            elif ligand_style == 'space-filling':
                view.setStyle({'hetflag': True}, {'sphere': {'colorscheme': custom_colorscheme}})
            elif ligand_style == 'stick':
                view.setStyle({'hetflag': True}, {'stick': {'colorscheme': custom_colorscheme, 'radius': 0.25}})
            elif ligand_style == 'line':
                view.setStyle({'hetflag': True}, {'line': {'colorscheme': custom_colorscheme}})
        else:
            view.setStyle({'hetflag': True}, {'stick': {'colorscheme': 'default'}})

    else:
        common_scheme_name = 'chain'
        if color_scheme == 'Rainbow':
            common_scheme_name = 'spectrum'
        elif color_scheme == 'Secondary Structure':
            if protein_style == 'cartoon':
                common_scheme_name = 'ssPyMOL'
            else:
                common_scheme_name = 'chain'

        view.setStyle({'hetflag': False}, {protein_style: {'colorscheme': common_scheme_name}})

        if ligand_style == 'ball-and-stick':
            view.setStyle({'hetflag': True}, {'stick': {'colorscheme': common_scheme_name, 'radius': 0.15}})
            view.addStyle({'hetflag': True}, {'sphere': {'colorscheme': common_scheme_name, 'scale': 0.25}})
        elif ligand_style == 'space-filling':
            view.setStyle({'hetflag': True}, {'sphere': {'colorscheme': common_scheme_name}})
        elif ligand_style == 'stick':
            view.setStyle({'hetflag': True}, {'stick': {'colorscheme': common_scheme_name, 'radius': 0.25}})
        elif ligand_style == 'line':
            view.setStyle({'hetflag': True}, {'line': {'colorscheme': common_scheme_name}})

    view.setBackgroundColor('#F0F2F6')
    view.zoomTo()
    if spin:
        view.spin(True)

    return view._make_html()

def export_to_pdb(cif_content: str) -> str:
    """Converts CIF content to PDB string."""
    structure = read_cif_from_string(cif_content)
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_buffer = io.StringIO()
    pdb_io.save(pdb_buffer)
    return pdb_buffer.getvalue()

def submit_job(yaml_content: str, use_msa: bool) -> str:
    """
    提交预测任务到后端 API。
    """
    files = {'yaml_file': ('input.yaml', yaml_content)}
    data = {'use_msa_server': str(use_msa).lower(), 'priority': 'high'}
    headers = {'X-API-Token': os.getenv('API_SECRET_TOKEN', 'your_default_api_token')}
    
    response = requests.post(f"{API_URL}/predict", files=files, data=data, headers=headers)
    response.raise_for_status()
    return response.json()['task_id']

def get_status(task_id: str) -> dict:
    """
    查询指定 task_id 的 Celery 任务状态。
    """
    response = requests.get(f"{API_URL}/status/{task_id}")
    response.raise_for_status()
    return response.json()

def download_and_process_results(task_id: str) -> tuple[dict, bytes]:
    """
    下载并处理任务结果ZIP文件。
    """
    response = requests.get(f"{API_URL}/results/{task_id}", stream=True)
    response.raise_for_status()
    raw_zip_bytes = response.content
    zip_buffer = io.BytesIO(raw_zip_bytes)
    results = {}
    cif_candidate, confidence_candidate, affinity_candidate = None, None, None
    with zipfile.ZipFile(zip_buffer, 'r') as zf:
        all_files = zf.namelist()
        for filename in all_files:
            if filename.endswith((".cif", ".pdb")):
                if "_model_0.cif" in filename or "ranked_0.cif" in filename:
                    cif_candidate = filename
                elif "_unrelaxed_rank_001_alphafold2_ptm.pdb" in filename:
                    cif_candidate = filename
                elif cif_candidate is None:
                    cif_candidate = filename
            if "confidence" in filename and filename.endswith(".json"):
                confidence_candidate = filename
            if "affinity" in filename and filename.endswith(".json"):
                affinity_candidate = filename
        
        if cif_candidate:
            if cif_candidate.endswith(".cif"):
                results['cif'] = zf.read(cif_candidate).decode('utf-8')
            elif cif_candidate.endswith(".pdb"):
                results['cif'] = zf.read(cif_candidate).decode('utf-8')
            
        if confidence_candidate:
            results['confidence'] = json.loads(zf.read(confidence_candidate))
        if affinity_candidate:
            results['affinity'] = json.loads(zf.read(affinity_candidate))
            
    if 'cif' not in results or 'confidence' not in results:
        raise FileNotFoundError("未能从结果文件中找到预测的结构 (.cif/.pdb) 或置信度信息 (.json)。")
    return results, raw_zip_bytes

def generate_yaml_from_state():
    """
    Generates the YAML configuration string based on the current session state.
    """
    if not st.session_state.get('components'):
        return None
        
    sequences_list = []
    chain_letters = string.ascii_uppercase + string.ascii_lowercase + string.digits
    next_letter_idx = 0
    
    for comp in st.session_state.components:
        num_copies = comp.get('num_copies', 1)
        current_ids = []
        for j in range(num_copies):
            if next_letter_idx + j < len(chain_letters):
                current_ids.append(chain_letters[next_letter_idx + j])
            else:
                st.warning(f"警告: 拷贝数过多，链ID可能重复或不足。请减少拷贝数或调整代码。")
                current_ids.append(f"UNK_{j}")
        next_letter_idx += num_copies
        
        component_dict = {'id': current_ids if len(current_ids) > 1 else current_ids[0]}

        if comp['type'] in ['protein', 'dna', 'rna']:
            component_dict['sequence'] = comp['sequence']
            if comp['type'] == 'protein' and comp.get('cyclic', False):
                component_dict['cyclic'] = True
            if comp['type'] == 'protein' and not st.session_state.use_msa_server:
                component_dict['msa'] = 'empty'
        elif comp['type'] == 'ligand':
            # 对于ketcher输入，实际存储的是SMILES，所以统一使用smiles字段
            input_method = comp['input_method']
            if input_method == 'ketcher':
                component_dict['smiles'] = comp['sequence']
            else:
                component_dict[input_method] = comp['sequence']
            
        sequences_list.append({comp['type']: component_dict})
        
    if not sequences_list:
        return None
        
    final_yaml_dict = {'version': 1, 'sequences': sequences_list}
    
    if st.session_state.properties.get('affinity') and st.session_state.properties.get('binder'):
        final_yaml_dict['properties'] = [{'affinity': {'binder': st.session_state.properties['binder']}}]
        
    return yaml.dump(final_yaml_dict, sort_keys=False, indent=2, default_flow_style=False)

# ========== Designer 相关函数 ==========

def create_designer_template_yaml(target_protein_sequence: str, target_chain_id: str = "A") -> str:
    """创建 Designer 的模板 YAML 配置"""
    template_dict = {
        'version': 1,
        'sequences': [
            {
                'protein': {
                    'id': target_chain_id,
                    'sequence': target_protein_sequence,
                    'msa': 'empty'
                }
            }
        ]
    }
    return yaml.dump(template_dict, sort_keys=False, indent=2, default_flow_style=False)

def create_designer_complex_yaml(components: list) -> str:
    """为多组分复合物创建 Designer 的模板 YAML 配置"""
    sequences_list = []
    chain_counter = 0  # 用于自动分配链ID
    
    for comp in components:
        if not comp.get('sequence', '').strip():
            continue  # 跳过空序列的组分
            
        num_copies = comp.get('num_copies', 1)
        
        # 为每个拷贝创建独立的组分
        for copy_idx in range(num_copies):
            # 自动分配链ID (A, B, C, ...)
            chain_id = string.ascii_uppercase[chain_counter] if chain_counter < 26 else f"Z{chain_counter-25}"
            chain_counter += 1
            
            if comp['type'] == 'protein':
                component_dict = {
                    'protein': {
                        'id': chain_id,
                        'sequence': comp['sequence'],
                        'msa': 'empty'
                    }
                }
            elif comp['type'] == 'dna':
                component_dict = {
                    'dna': {
                        'id': chain_id,
                        'sequence': comp['sequence']
                    }
                }
            elif comp['type'] == 'rna':
                component_dict = {
                    'rna': {
                        'id': chain_id,
                        'sequence': comp['sequence']
                    }
                }
            elif comp['type'] == 'ligand':
                input_method = comp.get('input_method', 'smiles')
                # 对于ketcher输入，实际存储的是SMILES，所以统一使用smiles字段
                actual_method = 'smiles' if input_method == 'ketcher' else input_method
                component_dict = {
                    'ligand': {
                        'id': chain_id,
                        actual_method: comp['sequence']
                    }
                }
            else:
                continue  # 跳过未知类型
                
            sequences_list.append(component_dict)
    
    if not sequences_list:
        raise ValueError("没有有效的组分序列")
        
    template_dict = {'version': 1, 'sequences': sequences_list}
    return yaml.dump(template_dict, sort_keys=False, indent=2, default_flow_style=False)

def run_designer_workflow(params: dict, work_dir: str) -> str:
    """运行 Designer 工作流程（真实实现）"""
    try:
        # 创建工作目录
        os.makedirs(work_dir, exist_ok=True)
        
        # 尝试集成实际的 Designer 模块
        designer_script_path = './designer/run_design.py'
        
        if os.path.exists(designer_script_path):
            # 计算设计链ID - 寻找下一个可用的链ID
            target_chain_id = params.get('target_chain_id', 'A')
            available_chains = string.ascii_uppercase
            used_chains = set()
            
            # 从模板YAML中解析已使用的链ID
            try:
                with open(params.get('template_path', ''), 'r') as f:
                    template_data = yaml.safe_load(f)
                    if 'sequences' in template_data:
                        for seq in template_data['sequences']:
                            for seq_type, seq_data in seq.items():
                                if 'id' in seq_data:
                                    used_chains.add(seq_data['id'])
            except Exception as e:
                print(f"Warning: Could not parse template YAML: {e}")
            
            # 找到下一个可用的链ID
            binder_chain_id = None
            for chain in available_chains:
                if chain not in used_chains:
                    binder_chain_id = chain
                    break
            
            if not binder_chain_id:
                binder_chain_id = "Z"  # 备用选项
            
            # 构建运行命令，直接传递参数
            cmd = [
                "python", "run_design.py",
                "--yaml_template", params.get('template_path', ''),
                "--binder_chain", binder_chain_id,  # 动态设计链ID
                "--binder_length", str(params.get('binder_length', 20)),
                "--iterations", str(params.get('generations', 5)),
                "--population_size", str(params.get('population_size', 10)),
                "--num_elites", str(params.get('elite_size', 3)),
                "--mutation_rate", str(params.get('mutation_rate', 0.3)),  # 新增：传递mutation_rate
                "--output_csv", os.path.join(work_dir, f"design_summary_{params.get('task_id', 'unknown')}.csv"),
                "--keep_temp_files"  # 保留临时文件以便下载结构
            ]
            
            # 添加增强功能参数
            if params.get('enable_enhanced', True):
                cmd.extend([
                    "--convergence-window", str(params.get('convergence_window', 5)),
                    "--convergence-threshold", str(params.get('convergence_threshold', 0.001)),
                    "--max-stagnation", str(params.get('max_stagnation', 3)),
                    "--initial-temperature", str(params.get('initial_temperature', 1.0)),
                    "--min-temperature", str(params.get('min_temperature', 0.1))
                ])
            else:
                cmd.append("--disable-enhanced")
            
            # 添加糖肽相关参数
            if params.get('design_type') == 'glycopeptide' and params.get('glycan_type'):
                cmd.extend([
                    "--glycan_ccd", params.get('glycan_type'),
                    "--glycosylation_site", str(params.get('glycosylation_site', 10))
                ])
            
            # 在后台运行设计任务
            # 创建日志文件
            log_file = os.path.join(work_dir, 'design.log')
            
            with open(log_file, 'w') as log:
                log.write(f"设计任务开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write(f"参数: {json.dumps(params, indent=2)}\n")
                log.write(f"命令: {' '.join(cmd)}\n")
                log.write("-" * 50 + "\n")
                
                # 启动异步进程
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd='./designer'
                )
                
                # 将进程ID写入状态文件
                status_file = os.path.join(work_dir, 'status.json')
                status_data = {
                    'task_id': params.get('task_id', 'unknown'),
                    'status': 'running',
                    'process_id': process.pid,
                    'start_time': datetime.now().isoformat(),
                    'params': params
                }
                
                with open(status_file, 'w') as f:
                    json.dump(status_data, f, indent=2)
                
                return "running"
        else:
            # Designer 脚本不存在，返回错误
            print(f"❌ Designer 脚本未找到: {designer_script_path}")
            
            # 创建错误状态文件
            status_file = os.path.join(work_dir, 'status.json')
            status_data = {
                'task_id': params.get('task_id', 'unknown'),
                'status': 'failed',
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'params': params,
                'error': f"Designer script not found at {designer_script_path}"
            }
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            
            return "failed"
            
    except Exception as e:
        print(f"Error in run_designer_workflow: {e}")
        return "failed"


def submit_designer_job(
    template_yaml_content: str,
    design_type: str,
    binder_length: int,
    target_chain_id: str = "A",
    generations: int = 5,
    population_size: int = 10,
    elite_size: int = 3,
    mutation_rate: float = 0.3,
    glycan_type: str = None,
    glycosylation_site: int = None,
    # 增强功能参数
    convergence_window: int = 5,
    convergence_threshold: float = 0.001,
    max_stagnation: int = 3,
    initial_temperature: float = 1.0,
    min_temperature: float = 0.1,
    enable_enhanced: bool = True
) -> dict:
    """提交 Designer 任务"""
    try:
        # 创建临时工作目录
        work_dir = tempfile.mkdtemp(prefix="boltz_designer_")
        template_path = os.path.join(work_dir, "template.yaml")
        
        # 保存模板文件
        with open(template_path, 'w') as f:
            f.write(template_yaml_content)
        
        # 构建设计参数
        design_params = {
            'template_path': template_path,
            'design_type': design_type,
            'binder_length': binder_length,
            'target_chain_id': target_chain_id,
            'generations': generations,
            'population_size': population_size,
            'elite_size': elite_size,
            'mutation_rate': mutation_rate,
            'work_dir': work_dir,
            # 增强功能参数
            'convergence_window': convergence_window,
            'convergence_threshold': convergence_threshold,
            'max_stagnation': max_stagnation,
            'initial_temperature': initial_temperature,
            'min_temperature': min_temperature,
            'enable_enhanced': enable_enhanced
        }
        
        if design_type == 'glycopeptide' and glycan_type:
            design_params['glycan_type'] = glycan_type
            design_params['glycosylation_site'] = glycosylation_site
        
        # 这里调用实际的 Designer 工作流程
        task_id = f"designer_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        design_params['task_id'] = task_id
        
        # 运行设计工作流程
        workflow_status = run_designer_workflow(design_params, work_dir)
        
        return {
            'success': True,
            'task_id': task_id,
            'work_dir': work_dir,
            'params': design_params,
            'initial_status': workflow_status
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_designer_status(task_id: str, work_dir: str = None) -> dict:
    """获取 Designer 任务状态（真实实现）"""
    try:
        # 如果没有提供工作目录，尝试找到它
        if not work_dir:
            # 在临时目录中搜索
            import tempfile
            temp_dir = tempfile.gettempdir()
            
            # 查找匹配的目录
            for item in os.listdir(temp_dir):
                if item.startswith('boltz_designer_'):
                    potential_dir = os.path.join(temp_dir, item)
                    status_file = os.path.join(potential_dir, 'status.json')
                    if os.path.exists(status_file):
                        try:
                            with open(status_file, 'r') as f:
                                status_data = json.load(f)
                                if status_data.get('task_id') == task_id:
                                    work_dir = potential_dir
                                    break
                        except:
                            continue
        
        if not work_dir:
            return {
                'task_id': task_id,
                'state': 'NOT_FOUND',
                'error': '未找到任务工作目录'
            }
        
        # 读取状态文件
        status_file = os.path.join(work_dir, 'status.json')
        
        if not os.path.exists(status_file):
            return {
                'task_id': task_id,
                'state': 'UNKNOWN',
                'error': '状态文件不存在'
            }
        
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        
        current_status = status_data.get('status', 'unknown')
        
        # 检查进程是否还在运行（如果有进程ID）
        process_still_running = False
        
        if current_status == 'running':
            # 检查保存的进程ID是否仍在运行
            if 'process_id' in status_data:
                try:
                    if psutil and psutil.pid_exists(status_data['process_id']):
                        # 进一步验证这个PID确实是我们的run_design.py进程
                        proc = psutil.Process(status_data['process_id'])
                        cmdline = proc.cmdline()
                        if cmdline and 'run_design.py' in ' '.join(cmdline):
                            process_still_running = True
                        else:
                            # PID存在但不是我们的进程，可能被回收重用了
                            process_still_running = False
                except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                    # 进程不存在或无法访问
                    process_still_running = False
            
            # 如果进程已结束，检测完成状态
            if not process_still_running:
                # 检查是否有CSV结果文件存在
                csv_files = []
                try:
                    for filename in os.listdir(work_dir):
                        if filename.startswith('design_summary_') and filename.endswith('.csv'):
                            csv_path = os.path.join(work_dir, filename)
                            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                                csv_files.append(csv_path)
                except Exception:
                    pass
                
                # 检查日志文件是否显示完成
                log_completed = False
                try:
                    log_file = os.path.join(work_dir, 'design.log')
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            if any(indicator in log_content for indicator in [
                                'Design Run Finished', 
                                '设计任务完成', 
                                'Successfully created results package',
                                'Summary CSV successfully saved'
                            ]):
                                log_completed = True
                except Exception:
                    pass
                
                # 检查进度是否显示已完成所有代数
                progress_completed = False
                try:
                    log_file = os.path.join(work_dir, 'design.log')
                    if os.path.exists(log_file):
                        progress_info = parse_design_progress(log_file, status_data.get('params', {}))
                        current_gen = progress_info.get('current_generation', 0)
                        total_gen = progress_info.get('total_generations', 1)
                        best_sequences = progress_info.get('current_best_sequences', [])
                        
                        if current_gen > total_gen and total_gen > 0 and best_sequences:
                            progress_completed = True
                        elif current_gen == total_gen and total_gen > 0 and best_sequences:
                            # 只有在最后一代且有明确完成标记时才认为完成
                            try:
                                log_file = os.path.join(work_dir, 'design.log')
                                if os.path.exists(log_file):
                                    with open(log_file, 'r') as f:
                                        log_content = f.read()
                                        # 只检查非常明确的完成标记
                                        if any(indicator in log_content for indicator in [
                                            'Design Run Finished', 
                                            '设计任务完成', 
                                            'Successfully created results package',
                                            'All generations completed',
                                            f'Finished all {total_gen} generations'
                                        ]):
                                            progress_completed = True
                                        # 或者检查CSV文件的时间戳确认是最近完成的
                                        elif csv_files:
                                            for csv_file in csv_files:
                                                if os.path.exists(csv_file):
                                                    file_age = time.time() - os.path.getmtime(csv_file)
                                                    # 文件必须非常新（10秒内）且序列数量足够才认为刚完成
                                                    if file_age < 10:
                                                        # 额外检查：确保CSV中有足够的数据表明真正完成
                                                        try:
                                                            df = pd.read_csv(csv_file)
                                                            if len(df) >= total_gen:  # 至少有总代数的序列数
                                                                progress_completed = True
                                                                break
                                                        except:
                                                            pass
                            except Exception:
                                # 如果检查失败，不认为完成，继续等待
                                pass
                except Exception:
                    pass
                
                if log_completed or progress_completed:
                    status_data['status'] = 'completed'
                    status_data['end_time'] = datetime.now().isoformat()
                    if csv_files:
                        status_data['csv_files'] = csv_files
                    
                    with open(status_file, 'w') as f:
                        json.dump(status_data, f, indent=2)
                    
                    current_status = 'completed'
        
        # 构建返回状态
        result = {
            'task_id': task_id,
            'state': current_status.upper(),
            'start_time': status_data.get('start_time'),
            'work_dir': work_dir
        }
        
        # 添加进度信息
        if current_status == 'running':
            # 尝试从日志文件解析进度
            log_file = os.path.join(work_dir, 'design.log')
            if os.path.exists(log_file):
                result['progress'] = parse_design_progress(log_file, status_data.get('params', {}))
            else:
                # 如果没有日志文件，提供基础进度信息
                result['progress'] = {
                    'current_generation': 1,
                    'total_generations': status_data.get('params', {}).get('generations', 5),
                    'estimated_progress': 0.1,
                    'best_score': 0.0,
                    'status_message': '任务正在启动...',
                    'pending_tasks': 0,
                    'completed_tasks': 0,
                    'current_status': 'initializing'
                }
        elif current_status == 'completed':
            # 任务完成时也尝试获取最终进度
            log_file = os.path.join(work_dir, 'design.log')
            if os.path.exists(log_file):
                final_progress = parse_design_progress(log_file, status_data.get('params', {}))
                result['progress'] = final_progress
                result['progress']['estimated_progress'] = 1.0
                result['progress']['status_message'] = '设计任务已完成'
        
        # 添加结果摘要（如果已完成）
        if current_status == 'completed' and 'results_summary' in status_data:
            result['results_summary'] = status_data['results_summary']
        
        return result
        
    except Exception as e:
        return {
            'task_id': task_id,
            'state': 'ERROR',
            'error': str(e)
        }


def parse_design_progress(log_file: str, params: dict) -> dict:
    """从日志文件解析设计进度，并从CSV文件读取最佳序列"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        current_generation = 0
        total_generations = params.get('generations', 5)
        best_score = 0.0
        current_status = "initializing"
        pending_tasks = 0
        completed_tasks = 0
        current_best_sequences = []  # 从CSV文件读取的当前最佳序列列表
        
        # 分析日志内容
        for line in lines:
            line = line.strip()
            
            # 检测任务状态
            if 'Polling task' in line and 'PENDING' in line:
                pending_tasks += 1
                current_status = "waiting_for_prediction"
            elif 'Polling task' in line and 'SUCCESS' in line:
                completed_tasks += 1
                current_status = "processing_results"
            elif 'Generation' in line or 'generation' in line or '代演化' in line:
                # 提取世代信息 - 匹配多种格式
                gen_matches = re.findall(r'(?:Generation|第)\s*(\d+)', line, re.IGNORECASE)
                if gen_matches:
                    current_generation = max(current_generation, int(gen_matches[-1]))
                    current_status = "evolving"
                    
                # 匹配中文格式 "正在运行第 X/Y 代演化"
                gen_match = re.search(r'第\s*(\d+)/(\d+)\s*代演化', line)
                if gen_match:
                    current_generation = int(gen_match.group(1))
                    total_generations = int(gen_match.group(2))
                    current_status = "evolving"
                    
            elif 'Completed generation' in line or '完成第' in line or 'Generation.*complete' in line:
                # 确认某代已完成
                gen_matches = re.findall(r'(\d+)', line)
                if gen_matches:
                    current_generation = max(current_generation, int(gen_matches[-1]))
                    current_status = "evolving"
            
            # 提取评分信息 - 优化匹配模式
            if any(keyword in line.lower() for keyword in ['best score', '最佳评分', 'best:', 'top score', 'highest score']):
                # 匹配各种数值格式：整数、小数、科学记数法
                score_matches = re.findall(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)', line)
                if score_matches:
                    try:
                        # 取最后一个匹配的数值作为评分
                        candidate_score = float(score_matches[-1])
                        # 合理性检查：评分通常在0-1之间，但也可能更大
                        if 0 <= candidate_score <= 10:  # 扩大合理范围
                            best_score = max(best_score, candidate_score)
                    except ValueError:
                        pass
                        
            # 匹配其他可能的评分格式
            score_patterns = [
                r'score[:\s]+(\d+\.?\d*)',  # "score: 0.85"
                r'评分[:\s]+(\d+\.?\d*)',    # "评分: 0.85"
                r'fitness[:\s]+(\d+\.?\d*)', # "fitness: 0.85"
                r'ipTM[:\s]+(\d+\.?\d*)',   # "ipTM: 0.85"
                r'pLDDT[:\s]+(\d+\.?\d*)'   # "pLDDT: 85.5"
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    try:
                        candidate_score = float(matches[-1])
                        # 对pLDDT分数特殊处理（通常0-100），转换为0-1
                        if 'plddt' in pattern.lower() and candidate_score > 1:
                            candidate_score = candidate_score / 100.0
                        if 0 <= candidate_score <= 1:
                            best_score = max(best_score, candidate_score)
                    except ValueError:
                        pass
        
        # 从CSV文件读取当前最佳序列
        work_dir = os.path.dirname(log_file)
        csv_file_path = None
        csv_debug_info = {'work_dir': work_dir, 'files_found': [], 'error': None}
        
        # 尝试找到CSV文件
        try:
            files_in_dir = os.listdir(work_dir)
            csv_debug_info['files_found'] = [f for f in files_in_dir if f.endswith('.csv')]
            
            for filename in files_in_dir:
                if filename.startswith('design_summary_') and filename.endswith('.csv'):
                    csv_file_path = os.path.join(work_dir, filename)
                    break
        except Exception as e:
            csv_debug_info['error'] = f"List dir error: {str(e)}"
        
        if csv_file_path and os.path.exists(csv_file_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file_path)
                csv_debug_info['csv_rows'] = len(df)
                csv_debug_info['csv_columns'] = list(df.columns)
                
                # 只有当CSV文件有数据且不为空时，才使用CSV中的评分
                if len(df) > 0 and not df.empty:
                    # 检查是否有有效的评分数据
                    valid_scores = df['composite_score'].dropna()
                    if len(valid_scores) > 0:
                        csv_best_score = valid_scores.max()
                        # 只有当CSV评分合理时才使用（避免使用默认的0或异常值）
                        if csv_best_score > 0 and csv_best_score <= 1.0:
                            # 如果当前代数与CSV数据匹配，优先使用CSV评分
                            csv_generations = df['generation'].dropna() if 'generation' in df.columns else []
                            if len(csv_generations) > 0 and current_generation > 0:
                                max_csv_generation = int(csv_generations.max())
                                # 只有当CSV中的代数与当前代数接近时才使用CSV评分
                                if abs(max_csv_generation - current_generation) <= 1:
                                    best_score = csv_best_score
                            elif current_generation == 0:  # 初始状态，可以使用CSV数据
                                best_score = csv_best_score
                
                # 读取前5个最佳序列
                for idx, row in df.head(5).iterrows():
                    sequence = row.get('sequence', '')
                    score = float(row.get('composite_score', '0'))
                    generation = int(row.get('generation', current_generation))
                    iptm = float(row.get('iptm', '0'))
                    plddt = float(row.get('binder_avg_plddt', '0'))
                    
                    if sequence and len(sequence) >= 8:  # 验证序列有效性
                        current_best_sequences.append({
                            'sequence': sequence,
                            'score': score,
                            'generation': generation,
                            'iptm': iptm,
                            'plddt': plddt
                        })
                        
            except Exception as e:
                # CSV读取失败，使用默认值
                csv_debug_info['read_error'] = str(e)
        else:
            # 没有CSV文件时，将评分重置为0
            best_score = 0.0
        
        # 计算进度
        if total_generations > 0:
            progress_ratio = min(current_generation / total_generations, 1.0)
        else:
            progress_ratio = 0.0
        
        # 根据任务状态调整进度显示
        if current_status == "waiting_for_prediction" and pending_tasks > 0:
            status_msg = f"等待结构预测完成 ({completed_tasks}/{pending_tasks + completed_tasks} 个任务已完成)"
        elif current_status == "evolving":
            if current_generation > 0:
                status_msg = f"第 {current_generation}/{total_generations} 代演化"
            else:
                status_msg = "初始化演化算法"
        elif current_status == "processing_results":
            status_msg = "处理预测结果"
        else:
            status_msg = "初始化中"
        
        return {
            'current_generation': current_generation,
            'total_generations': total_generations,
            'best_score': best_score,
            'estimated_progress': progress_ratio,
            'status_message': status_msg,
            'pending_tasks': pending_tasks,
            'completed_tasks': completed_tasks,
            'current_status': current_status,
            'current_best_sequences': current_best_sequences,  # 从CSV读取
            'debug_info': {  # 添加调试信息
                'sequences_found': len(current_best_sequences),
                'log_lines_processed': len(lines),
                'generation_detected': current_generation > 0,
                'status_detected': current_status,
                'best_score_found': best_score > 0,
                'csv_file_found': csv_file_path is not None,
                'csv_file_path': csv_file_path,
                'csv_debug': csv_debug_info
            }
        }
        
    except Exception as e:
        total_gens = params.get('generations', 5)
        return {
            'current_generation': 0,
            'total_generations': total_gens,
            'best_score': 0.0,
            'estimated_progress': 0.0,
            'status_message': "初始化中",
            'pending_tasks': 0,
            'completed_tasks': 0,
            'current_status': 'initializing',
            'current_best_sequences': [],
            'debug_info': {
                'sequences_found': 0,
                'log_lines_processed': 0,
                'generation_detected': False,
                'status_detected': 'error',
                'error_message': str(e)
            },
            'error': str(e)
        }

def load_designer_results(task_id: str, work_dir: str) -> dict:
    """加载 Designer 结果（真实实现）"""
    try:
        # 查找可能的结果文件
        result_files = {
            'summary_csv': None,
            'best_sequences_json': None,
            'evolution_log': None
        }
        
        # 扫描工作目录和常见的结果目录
        search_dirs = [
            work_dir,
            os.path.join(work_dir, 'results'),
            '/tmp/boltz_designer',
            './designer/temp_design_*',
            f'./designer/temp_design_run_{task_id.split("_")[-1][:10]}*' if '_' in task_id else None
        ]
        
        # 移除 None 值
        search_dirs = [d for d in search_dirs if d is not None]
        
        found_results = []
        
        for search_dir in search_dirs:
            if '*' in search_dir:
                # 使用 glob 匹配模式
                import glob
                matching_dirs = glob.glob(search_dir)
                search_dirs.extend(matching_dirs)
                continue
                
            if not os.path.exists(search_dir):
                continue
                
            try:
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # 查找 CSV 汇总文件
                        if ('design_summary_' in file or 'design_run_summary' in file) and file.endswith('.csv'):
                            result_files['summary_csv'] = file_path
                            
                        # 查找最佳序列 JSON 文件
                        elif 'best_sequences' in file and file.endswith('.json'):
                            result_files['best_sequences_json'] = file_path
                            
                        # 查找演化日志文件
                        elif 'evolution' in file or 'log' in file:
                            result_files['evolution_log'] = file_path
                            
            except Exception as e:
                print(f"Error scanning directory {search_dir}: {e}")
                continue
        
        # 尝试从找到的文件中读取结果
        best_sequences = []
        evolution_history = {
            'generations': [],
            'best_scores': [],
            'avg_scores': []
        }
        
        # 读取 CSV 汇总文件
        if result_files['summary_csv'] and os.path.exists(result_files['summary_csv']):
            try:
                import pandas as pd
                df = pd.read_csv(result_files['summary_csv'])
                
                # 转换 DataFrame 为序列列表
                for idx, row in df.iterrows():
                    seq_data = {
                        'sequence': row.get('sequence', f'UNKNOWN_SEQ_{idx}'),
                        'score': float(row.get('composite_score', 0.0)) if pd.notna(row.get('composite_score')) else 0.0,
                        'iptm': float(row.get('iptm', 0.0)) if pd.notna(row.get('iptm')) else 0.0,
                        'plddt': float(row.get('binder_avg_plddt', 0.0)) if pd.notna(row.get('binder_avg_plddt')) else 0.0,
                        'generation': int(row.get('generation', 1)) if pd.notna(row.get('generation')) else 1,
                        'results_path': row.get('results_path', '') if pd.notna(row.get('results_path')) else ''
                    }
                    best_sequences.append(seq_data)
                    
                # 按评分排序
                best_sequences.sort(key=lambda x: x['score'], reverse=True)
                
                # 从数据中提取演化历史
                if len(best_sequences) > 0:
                    generations = sorted(list(set([seq['generation'] for seq in best_sequences])))
                    best_scores = []
                    avg_scores = []
                    
                    for gen in generations:
                        gen_scores = [seq['score'] for seq in best_sequences if seq['generation'] == gen]
                        if gen_scores:
                            best_scores.append(max(gen_scores))
                            avg_scores.append(sum(gen_scores) / len(gen_scores))
                        else:
                            best_scores.append(0.0)
                            avg_scores.append(0.0)
                    
                    evolution_history = {
                        'generations': generations,
                        'best_scores': best_scores,
                        'avg_scores': avg_scores
                    }
                
                print(f"✅ 成功从 {result_files['summary_csv']} 读取了 {len(best_sequences)} 个序列")
                
            except Exception as e:
                print(f"Error reading CSV file {result_files['summary_csv']}: {e}")
        
        # 读取 JSON 文件（如果存在）
        if result_files['best_sequences_json'] and os.path.exists(result_files['best_sequences_json']):
            try:
                with open(result_files['best_sequences_json'], 'r') as f:
                    json_data = json.load(f)
                    if 'best_sequences' in json_data:
                        best_sequences = json_data['best_sequences']
                    if 'evolution_history' in json_data:
                        evolution_history = json_data['evolution_history']
                        
                print(f"✅ 成功从 {result_files['best_sequences_json']} 读取了 JSON 数据")
                        
            except Exception as e:
                print(f"Error reading JSON file {result_files['best_sequences_json']}: {e}")
        
        # 如果没有找到真实数据，返回错误信息
        if not best_sequences:
            print(f"❌ 未找到真实设计结果文件。搜索的目录: {search_dirs}")
            print(f"📁 查找的文件类型: CSV汇总文件、JSON结果文件、演化日志")
            raise FileNotFoundError(f"No design results found in searched directories: {search_dirs}")
        
        return {
            'best_sequences': best_sequences,
            'evolution_history': evolution_history,
            'result_files': result_files,
            'search_info': {
                'searched_dirs': search_dirs,
                'found_files': {k: v for k, v in result_files.items() if v is not None}
            }
        }
        
    except Exception as e:
        print(f"Error in load_designer_results: {e}")
        # 返回错误信息而不是模拟数据
        raise Exception(f"Failed to load designer results: {str(e)}")

def validate_inputs(components):
    """验证用户输入是否完整且有效。"""
    if not components:
        return False, "请至少添加一个组分。"
    
    # 检查所有组分是否都有有效序列
    valid_components = 0
    for i, comp in enumerate(components):
        sequence = comp.get('sequence', '').strip()
        if not sequence:
            display_name = TYPE_TO_DISPLAY.get(comp.get('type', 'Unknown'), 'Unknown')
            return False, f"错误: 组分 {i+1} ({display_name}) 的序列不能为空。"
        
        # 验证小分子SMILES格式（ketcher也会生成SMILES）
        if comp.get('type') == 'ligand' and comp.get('input_method') in ['smiles', 'ketcher']:
            if sequence and not all(c in string.printable for c in sequence):
                return False, f"错误: 组分 {i+1} (小分子) 的 SMILES 字符串包含非法字符。"
        
        valid_components += 1
    
    # 至少需要一个有效组分（可以是任何类型，包括单独的小分子）
    if valid_components == 0:
        return False, "请至少输入一个有效的组分序列。"
            
    # 亲和力预测验证（只有在启用时才检查）
    if st.session_state.properties.get('affinity'):
        has_ligand_component_with_sequence = any(comp['type'] == 'ligand' and comp.get('sequence', '').strip() for comp in components)
        if not has_ligand_component_with_sequence:
            return False, "已选择计算亲和力，但未提供任何小分子序列。"
        if not st.session_state.properties.get('binder'):
            return False, "已选择计算亲和力，但未选择结合体（Binder）链ID。"
            
    return True, ""

def validate_designer_inputs(designer_components):
    """验证Designer输入是否完整且有效。"""
    if not designer_components:
        return False, "请至少添加一个组分。"
    
    # 检查是否至少有一个目标组分（蛋白质、DNA、RNA或小分子）
    # 支持两种设计模式：
    # 1. 正向设计：给定蛋白质/DNA/RNA，设计结合肽
    # 2. 反向设计：给定小分子，设计结合蛋白
    target_bio_components = [comp for comp in designer_components if comp['type'] in ['protein', 'dna', 'rna'] and comp.get('sequence', '').strip()]
    target_ligand_components = [comp for comp in designer_components if comp['type'] == 'ligand' and comp.get('sequence', '').strip()]
    
    # 至少需要一种目标组分
    if not target_bio_components and not target_ligand_components:
        return False, "请至少添加一个包含序列的蛋白质、DNA、RNA或小分子组分作为设计目标。"
    
    for i, comp in enumerate(designer_components):
        if comp.get('sequence', '').strip():  # 只验证非空序列
            comp_type = comp.get('type')
            sequence = comp.get('sequence', '').strip()
            
            if comp_type == 'protein':
                # 验证蛋白质序列只包含标准氨基酸字符
                valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                if not all(c.upper() in valid_aa for c in sequence):
                    return False, f"错误: 组分 {i+1} (蛋白质) 包含非标准氨基酸字符。请使用标准20种氨基酸。"
            
            elif comp_type == 'dna':
                # 验证DNA序列只包含ATGC
                valid_dna = set('ATGC')
                if not all(c.upper() in valid_dna for c in sequence):
                    return False, f"错误: 组分 {i+1} (DNA) 包含非法核苷酸。请只使用A、T、G、C。"
            
            elif comp_type == 'rna':
                # 验证RNA序列只包含AUGC
                valid_rna = set('AUGC')
                if not all(c.upper() in valid_rna for c in sequence):
                    return False, f"错误: 组分 {i+1} (RNA) 包含非法核苷酸。请只使用A、U、G、C。"
            
            elif comp_type == 'ligand' and comp.get('input_method') in ['smiles', 'ketcher']:
                # 验证SMILES字符串（ketcher也会生成SMILES）
                if not all(c in string.printable for c in sequence):
                    return False, f"错误: 组分 {i+1} (小分子) 的 SMILES 字符串包含非法字符。"
    
    return True, ""

# ========== Streamlit 应用程序 ==========

st.set_page_config(layout="centered", page_title="Boltz-WebUI", page_icon="🧬")

# 初始化 session state
if 'components' not in st.session_state: st.session_state.components = []
if 'task_id' not in st.session_state: st.session_state.task_id = None
if 'results' not in st.session_state: st.session_state.results = None
if 'raw_zip' not in st.session_state: st.session_state.raw_zip = None
if 'error' not in st.session_state: st.session_state.error = None
if 'properties' not in st.session_state: st.session_state.properties = {'affinity': False, 'binder': None}
if 'use_msa_server' not in st.session_state: st.session_state.use_msa_server = False

# Designer 相关 session state
if 'designer_task_id' not in st.session_state: st.session_state.designer_task_id = None
if 'designer_work_dir' not in st.session_state: st.session_state.designer_work_dir = None
if 'designer_results' not in st.session_state: st.session_state.designer_results = None
if 'designer_error' not in st.session_state: st.session_state.designer_error = None
if 'designer_config' not in st.session_state: st.session_state.designer_config = {}

if not st.session_state.components:
    st.session_state.components.append({
        'id': str(uuid.uuid4()), 'type': 'protein', 'num_copies': 1, 'sequence': '', 'input_method': 'smiles', 'cyclic': False
    })

# CSS 样式
st.markdown(f"""
<style>
    .stApp {{
        background-color: #FFFFFF;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }}
    div.block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1024px;
    }}
    h1 {{
        color: #0053D6;
        text-align: left;
        margin-bottom: 0.5rem;
    }}
    h3 {{
        color: #555555;
        text-align: left;
    }}
    h2, h3, h4 {{
        color: #333333;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }}
    .stButton>button {{
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: 500;
    }}
    .stButton>button[kind="primary"] {{
        background-color: #007bff;
        color: white;
        border: none;
    }}
    .stButton>button[kind="primary"]:hover {{
        background-color: #0056b3;
    }}
    .stButton>button[kind="secondary"] {{
        background-color: #f0f2f6;
        color: #333333;
        border: 1px solid #ddd;
    }}
    .stButton>button[kind="secondary"]:hover {{
        background-color: #e0e0e0;
    }}
    .stButton>button[data-testid="baseButton-secondary"] {{
        border: none !important;
        background-color: transparent !important;
        color: #888 !important;
        padding: 0 !important;
        font-size: 1.2rem;
    }}
    .stButton>button[data-testid="baseButton-secondary"]:hover {{
        color: #ff4b4b !important;
    }}
    div[data-testid="stCheckbox"] {{
        display: flex;
        align-items: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }}
    .stExpander {{
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }}
    .stExpander>div>div[data-testid="stExpanderToggleIcon"] {{
        font-size: 1.5rem;
    }}
    .stCode {{
        background-color: #f8f8f8;
        border-left: 5px solid #007bff;
        padding: 10px;
        border-radius: 5px;
    }}
    .stAlert {{
        border-radius: 8px;
    }}
    hr {{
        border-top: 1px solid #eee;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    
    .loader {{
      border: 6px solid #f3f3f3;
      border-top: 6px solid #007bff;
      border-radius: 50%;
      width: 50px;
      height: 50px;
      animation: spin 1.5s linear infinite;
      margin: 20px auto;
    }}

    @keyframes spin {{
      0% {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}
    
    /* 简洁标签页样式 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: transparent;
        padding: 0;
        border-radius: 0;
        margin-bottom: 1.5rem;
        box-shadow: none;
        border-bottom: 2px solid #f1f5f9;
        justify-content: flex-start;
        width: auto;
        max-width: 300px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        background: transparent;
        border-radius: 0;
        color: #64748b;
        font-weight: 500;
        font-size: 14px;
        border: none;
        padding: 0 16px;
        transition: all 0.2s ease;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: auto;
        border-bottom: 2px solid transparent;
    }}
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
        color: #374151;
        background: #f8fafc;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: transparent !important;
        color: #1e293b !important;
        border-bottom: 2px solid #3b82f6 !important;
        font-weight: 600 !important;
    }}
    
    /* 移除所有图标和额外装饰 */
    .stTabs [data-baseweb="tab"]::before,
    .stTabs [data-baseweb="tab"]::after {{
        display: none;
    }}
</style>
""", unsafe_allow_html=True)

st.title("🧬 Boltz-WebUI")
st.markdown("蛋白质-分子复合物结构预测与设计平台")

# 创建标签页
tab1, tab2 = st.tabs(["结构预测", "分子设计"])

# ========== 结构预测标签页 ==========
with tab1:
    st.markdown("### 🔬 分子复合物结构预测")
    st.markdown("输入您的生物分子序列，获得高精度的3D结构预测结果。")
    
    is_running = (
        st.session_state.task_id is not None and st.session_state.results is None and st.session_state.error is None
    )

    with st.expander("🔧 **步骤 1: 配置您的预测任务**", expanded=not is_running and not st.session_state.results):
        st.markdown("填写以下信息，定义您希望预测的生物大分子和小分子组分。")
        id_to_delete = None
        
        for i, component in enumerate(st.session_state.components[:]):
            st.markdown(f"---")
            st.subheader(f"组分 {i+1}", anchor=False)
            
            cols_type_copies = st.columns([3, 1])
            type_options = list(TYPE_TO_DISPLAY.keys())
            current_type = component.get('type', 'protein')
            current_type_index = type_options.index(current_type)
            
            selected_type = cols_type_copies[0].selectbox(
                "选择组分类型", options=type_options, format_func=lambda x: TYPE_TO_DISPLAY[x],
                key=f"type_{component['id']}", index=current_type_index, disabled=is_running,
                help="选择此组分是蛋白质、DNA、RNA 还是小分子。"
            )

            if selected_type != current_type:
                st.session_state.components[i]['sequence'] = ''
                st.session_state.components[i]['type'] = selected_type
                # Reset cyclic for new type if changed from protein
                if selected_type != 'protein':
                    st.session_state.components[i]['cyclic'] = False
                st.rerun()

            st.session_state.components[i]['num_copies'] = cols_type_copies[1].number_input(
                "拷贝数", min_value=1, max_value=20, step=1, key=f"copies_{component['id']}",
                value=component.get('num_copies', 1), disabled=is_running,
                help="此组分的拷贝数。可设置为2（二聚体）、3（三聚体）等。每个拷贝将分配独立的链ID。"
            )

            if selected_type == 'ligand':
                method_options = ["smiles", "ccd", "ketcher"]
                current_method_index = method_options.index(component.get('input_method', 'smiles'))
                
                st.session_state.components[i]['input_method'] = st.radio(
                    "小分子输入方式", method_options, key=f"ligand_type_{component['id']}",
                    index=current_method_index, disabled=is_running, horizontal=True,
                    help="选择通过SMILES字符串、PDB CCD代码或分子编辑器输入小分子。"
                )
                
                num_copies = component.get('num_copies', 1)
                
                if st.session_state.components[i]['input_method'] == 'smiles':
                    st.session_state.components[i]['sequence'] = st.text_input(
                        f"SMILES 字符串 ({'单分子' if num_copies == 1 else f'{num_copies}个分子'})",
                        value=component.get('sequence', ''),
                        placeholder="例如: CC(=O)NC1=CC=C(C=C1)O",
                        key=f"seq_{component['id']}",
                        disabled=is_running,
                        help="输入SMILES（简化分子线性输入系统）字符串来描述分子结构。"
                    )
                
                elif st.session_state.components[i]['input_method'] == 'ccd':
                    st.session_state.components[i]['sequence'] = st.text_input(
                        f"CCD 代码 ({'单分子' if num_copies == 1 else f'{num_copies}个分子'})",
                        value=component.get('sequence', ''),
                        placeholder="例如: HEM, NAD, ATP",
                        key=f"seq_{component['id']}",
                        disabled=is_running,
                        help="输入标准化合物组件字典（CCD）中的三字母或多字母代码。"
                    )
                
                else:  # ketcher
                    initial_smiles = st.session_state.components[i].get('sequence', '')
                    
                    st.info("🎨 在下方 **Ketcher 编辑器** 中绘制分子，或直接粘贴 SMILES 字符串。**编辑完成后，请点击编辑器内部的 'Apply' 按钮，SMILES 字符串将自动更新。**", icon="💡")
                    
                    ketcher_current_smiles = st_ketcher(
                        value=initial_smiles,
                        key=f"ketcher_{component['id']}",
                        height=400
                    )
                    
                    # 更加严格的SMILES更新逻辑
                    if ketcher_current_smiles is not None:
                        # 清理空白字符
                        ketcher_current_smiles = ketcher_current_smiles.strip()
                        if ketcher_current_smiles != initial_smiles:
                            st.session_state.components[i]['sequence'] = ketcher_current_smiles
                            if ketcher_current_smiles:
                                st.toast("✅ SMILES 字符串已成功更新！", icon="🧪")
                            else:
                                st.toast("📝 SMILES 字符串已清空", icon="🗑️")
                        
                    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem'>", unsafe_allow_html=True)
                    st.caption("✨ Ketcher 生成的 SMILES 字符串:")
                    current_smiles_in_state = st.session_state.components[i].get('sequence', '')
                    if current_smiles_in_state:
                        st.code(current_smiles_in_state, language='smiles')
                        
                        # 显示 SMILES 基本信息
                        smiles_length = len(current_smiles_in_state)
                        atom_count = current_smiles_in_state.count('C') + current_smiles_in_state.count('N') + \
                                   current_smiles_in_state.count('O') + current_smiles_in_state.count('S')
                        st.caption(f"📊 长度: {smiles_length} 字符 | 主要原子数: ~{atom_count}")
                        
                        # 基本格式验证
                        if not all(c in string.printable for c in current_smiles_in_state):
                            st.warning("⚠️ SMILES 字符串包含非打印字符，可能导致预测失败。", icon="⚠️")
                        elif len(current_smiles_in_state.strip()) == 0:
                            st.warning("⚠️ SMILES 字符串为空。", icon="⚠️")
                        else:
                            st.success("SMILES 格式看起来正常", icon="✅")
                    else:
                        st.info("👆 请开始绘制或粘贴，SMILES 将会显示在这里。")
            else:
                placeholder_text = TYPE_SPECIFIC_INFO.get(selected_type, {}).get('placeholder', '')
                help_text = TYPE_SPECIFIC_INFO.get(selected_type, {}).get('help', '')
                
                # 生成友好的中文标签
                num_copies = component.get('num_copies', 1)
                if selected_type == 'protein':
                    label = f"蛋白质序列 ({'单体' if num_copies == 1 else f'{num_copies}聚体'})"
                elif selected_type == 'dna':
                    label = f"DNA序列 ({'单链' if num_copies == 1 else f'{num_copies}链'})"
                elif selected_type == 'rna':
                    label = f"RNA序列 ({'单链' if num_copies == 1 else f'{num_copies}链'})"
                else:
                    label = f"输入 {selected_type.capitalize()} 序列"
                
                st.session_state.components[i]['sequence'] = st.text_area(
                    label, 
                    height=120, key=f"seq_{component['id']}",
                    value=component.get('sequence', ''),
                    placeholder=placeholder_text,
                    help=help_text,
                    disabled=is_running
                )
                
                # Add cyclic peptide option for protein type
                if selected_type == 'protein':
                    st.session_state.components[i]['cyclic'] = st.checkbox(
                        "环肽 (Cyclic Peptide)",
                        value=st.session_state.components[i].get('cyclic', False),
                        key=f"cyclic_{component['id']}",
                        help="勾选此项表示该蛋白质序列是一个环状肽。对于环肽，模型将尝试生成闭合的环状结构。",
                        disabled=is_running
                    )
            
            delete_col, _ = st.columns([10, 1])
            with delete_col:
                if len(st.session_state.components) > 1:
                    if st.button("🗑️ 删除此组分", key=f"del_{component['id']}", help="从任务中移除此组分", disabled=is_running):
                        id_to_delete = component['id']
        
        if id_to_delete:
            st.session_state.components = [c for c in st.session_state.components if c.get('id') != id_to_delete]
            st.rerun()

        st.markdown("---")
        st.button("➕ 添加新组分", on_click=lambda: st.session_state.components.append({'id': str(uuid.uuid4()), 'type': 'protein', 'num_copies': 1, 'sequence': '', 'input_method': 'smiles', 'cyclic': False}), disabled=is_running, use_container_width=True)

        st.subheader("全局与高级设置", anchor=False)

        st.session_state.use_msa_server = st.checkbox(
            "启用 MSA 序列搜索 (推荐用于蛋白质)",
            value=st.session_state.get('use_msa_server', False),
            help="勾选此项将使用外部服务器为蛋白质序列生成多序列比对(MSA)。这可以显著提升对新颖蛋白质的预测精度，但会增加任务耗时。",
            disabled=is_running
        )
        
        has_ligand_component = any(comp['type'] == 'ligand' for comp in st.session_state.components)
        if has_ligand_component:
            st.session_state.properties['affinity'] = st.checkbox(
                "🔬 计算结合亲和力 (Affinity)",
                value=st.session_state.properties.get('affinity', False),
                disabled=is_running,
                help="勾选后，模型将尝试预测小分子与大分子组分之间的结合亲和力。请确保至少输入了一个小分子组分。"
            )
            if st.session_state.properties['affinity']:
                chain_letter_idx = 0
                valid_ligand_chains = []
                for comp in st.session_state.components:
                    if comp.get('sequence', '').strip():
                        num_copies = comp.get('num_copies', 1)
                        if comp['type'] == 'ligand':
                            for j in range(num_copies):
                                if (chain_letter_idx + j) < len(string.ascii_uppercase):
                                    chain_id = string.ascii_uppercase[(chain_letter_idx + j)]
                                    valid_ligand_chains.append(chain_id)
                                else:
                                    # 超出了可用的链ID范围
                                    chain_id = f"L{j}"
                                    valid_ligand_chains.append(chain_id)
                        chain_letter_idx += num_copies
                
                if valid_ligand_chains:
                    current_binder = st.session_state.properties.get('binder')
                    try:
                        binder_index = valid_ligand_chains.index(current_binder)
                    except ValueError:
                        binder_index = 0 if valid_ligand_chains else -1
                    
                    if binder_index != -1:
                        st.session_state.properties['binder'] = st.selectbox(
                            "选择作为结合体(Binder)的小分子链 ID",
                            options=valid_ligand_chains,
                            index=binder_index,
                            help="被选中的小分子链将被视为与其余所有链形成复合物的结合伙伴。预测结果将围绕此结合事件进行评估。",
                            disabled=is_running
                        )
                    else:
                        st.session_state.properties['binder'] = None
                        st.warning("请为至少一个小分子组分输入序列(SMILES/CCD)以选择结合体。", icon="⚠️")
                else:
                    st.session_state.properties['binder'] = None
                    st.warning("请为至少一个小分子组分输入序列(SMILES/CCD)以选择结合体。", icon="⚠️")
        else:
            if 'properties' in st.session_state:
                st.session_state.properties['affinity'] = False
                st.session_state.properties['binder'] = None

    is_valid, validation_message = validate_inputs(st.session_state.components)
    yaml_preview = generate_yaml_from_state() if is_valid else None

    # 添加 YAML 预览功能，帮助用户调试
    if yaml_preview and is_valid:
        with st.expander("📋 **预览生成的 YAML 配置**", expanded=False):
            st.markdown("以下是根据您的输入生成的 YAML 配置文件，将被发送给 Boltz 模型进行预测：")
            st.code(yaml_preview, language='yaml')
            
            # 特别提示 ketcher 转换
            has_ketcher = any(comp.get('type') == 'ligand' and comp.get('input_method') == 'ketcher' 
                            for comp in st.session_state.components)
            if has_ketcher:
                st.info("💡 **注意**: Ketcher 绘制的分子已自动转换为 `smiles` 字段，这是 Boltz 模型要求的格式。", icon="🔄")

    if st.button("🚀 提交预测任务", type="primary", disabled=(not is_valid or is_running), use_container_width=True):
        st.session_state.task_id = None
        st.session_state.results = None
        st.session_state.raw_zip = None
        st.session_state.error = None
        
        with st.spinner("⏳ 正在提交任务，请稍候..."):
            try:
                task_id = submit_job(
                    yaml_content=yaml_preview,
                    use_msa=st.session_state.use_msa_server
                )
                st.session_state.task_id = task_id
                st.toast(f"🎉 任务已成功提交！任务ID: {task_id}", icon="✅")
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ **任务提交失败：无法连接到API服务器或服务器返回错误**。请检查后端服务是否运行正常。详情: {e}")
                st.session_state.error = {"error_message": str(e), "type": "API Connection Error"}
            except Exception as e:
                st.error(f"❌ **任务提交失败：发生未知错误**。详情: {e}")
                st.session_state.error = {"error_message": str(e), "type": "Client Error"}

    if not is_valid and not is_running:
        st.error(f"⚠️ **无法提交**: {validation_message}")

    if st.session_state.task_id and not st.session_state.results:
        st.divider()
        st.header("✨ **步骤 2: 查看预测结果**", anchor=False)
        if not st.session_state.error:
            spinner_and_status_placeholder = st.empty()
            
            while True:
                try:
                    status_data = get_status(st.session_state.task_id)
                    current_state = status_data.get('state', 'UNKNOWN')
                    
                    with spinner_and_status_placeholder.container():
                        if current_state == 'SUCCESS':
                            st.success("🎉 任务成功完成！正在下载并渲染结果...")
                            try:
                                processed_results, raw_zip_bytes = download_and_process_results(st.session_state.task_id)
                                st.session_state.results = processed_results
                                st.session_state.raw_zip = raw_zip_bytes
                                st.toast("✅ 结果已成功加载！", icon="🎊")
                                st.rerun()
                                break 
                            except (FileNotFoundError, json.JSONDecodeError) as e:
                                st.session_state.error = {"error_message": f"处理结果文件失败：{e}", "type": "Result File Error"}
                                st.error(f"❌ **结果文件处理失败**：{e}")
                                break
                            except requests.exceptions.RequestException as e:
                                st.session_state.error = {"error_message": f"下载结果文件失败：{e}", "type": "Download Error"}
                                st.error(f"❌ **下载结果文件失败**：{e}")
                                break
                        elif current_state == 'FAILURE':
                            st.session_state.error = status_data.get('info', {})
                            error_message = st.session_state.error.get('exc_message', '未知错误')
                            st.error(f"❌ **任务失败**：{error_message}")
                            
                            # 显示调试信息
                            with st.expander("🔍 **调试信息**", expanded=False):
                                st.markdown("**任务ID：**")
                                st.code(st.session_state.task_id)
                                
                                st.markdown("**提交的 YAML 配置：**")
                                if yaml_preview:
                                    st.code(yaml_preview, language='yaml')
                                
                                st.markdown("**完整错误信息：**")
                                st.json(st.session_state.error)
                                
                                # 特别检查是否是 ketcher 相关问题
                                has_ketcher = any(comp.get('type') == 'ligand' and comp.get('input_method') == 'ketcher' 
                                                for comp in st.session_state.components)
                                if has_ketcher:
                                    st.markdown("**Ketcher 组分信息：**")
                                    ketcher_components = [comp for comp in st.session_state.components 
                                                        if comp.get('type') == 'ligand' and comp.get('input_method') == 'ketcher']
                                    for idx, comp in enumerate(ketcher_components):
                                        st.markdown(f"- 组分 {idx+1}: `{comp.get('sequence', 'empty')}`")
                            break
                        elif current_state == 'PENDING':
                            st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
                            st.info("🕒 任务正在排队中，请耐心等待...")
                        elif current_state == 'STARTED' or current_state == 'PROGRESS':
                            info_message = status_data.get('info', {}).get('message', f"当前状态: **{current_state}**")
                            st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
                            st.info(f"🔬 **任务正在运行**：{info_message} (页面将每 10 秒自动刷新)", icon="⏳")
                        else:
                            st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
                            st.warning(f"❓ 任务状态未知或正在初始化... (当前状态: **{current_state}**)")

                    time.sleep(10)
                except requests.exceptions.RequestException as e:
                    spinner_and_status_placeholder.error(f"🚨 **无法获取任务状态：API连接失败**。请检查后端服务是否运行正常。详情: {e}")
                    st.session_state.error = {"error_message": str(e), "type": "API Connection Error"}
                    break
                except Exception as e:
                    spinner_and_status_placeholder.error(f"🚨 **获取任务状态时发生未知错误**。详情: {e}")
                    st.session_state.error = {"error_message": str(e), "type": "Client Error"}
                    break

    if st.session_state.error:
        st.error("ℹ️ 任务执行失败，详细信息如下：")
        st.json(st.session_state.error)
        if st.button("🔄 重置并重新开始", type="secondary"):
            for key in ['task_id', 'results', 'raw_zip', 'error', 'components', 'properties', 'use_msa_server']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if st.session_state.results:
        st.divider()
        st.header("✅ **步骤 2: 预测结果展示**", anchor=False)

        cif_data = st.session_state.results.get('cif', '')
        confidence_data = st.session_state.results.get('confidence', {})
        affinity_data = st.session_state.results.get('affinity', {})

        col_vis, col_data = st.columns([3, 2])

        with col_vis:
            st.subheader("📊 3D 结构可视化", anchor=False)
            if cif_data:
                try:
                    structure = read_cif_from_string(cif_data)
                    protein_bfactors = extract_protein_residue_bfactors(structure)
                    
                    with st.expander("⚙️ **视图设置**", expanded=True):
                        row1_col1, row1_col2 = st.columns(2)
                        with row1_col1:
                            st.selectbox("大分子样式", ['cartoon', 'stick', 'sphere'], key='protein_style_vis', help="选择蛋白质、DNA、RNA 等大分子的渲染样式。", index=0)
                        with row1_col2:
                            st.selectbox(
                                "着色方案",
                                ['pLDDT', 'Chain', 'Rainbow', 'Secondary Structure'],
                                key='color_scheme_vis',
                                help="""
                                选择整个复合物的着色方式：
                                - **pLDDT**: 根据预测置信度着色 (默认)，蓝色表示高置信，橙色表示低置信。
                                - **Chain**: 按不同的分子链着色。
                                - **Rainbow**: 从N端到C端按彩虹色渐变。
                                - **Secondary Structure**: 根据分子的二级结构（如螺旋、折叠）着色。
                                """,
                                index=0
                            )
                        row2_col1, row2_col2 = st.columns(2)
                        with row2_col1:
                            st.selectbox("配体样式", ['ball-and-stick', 'space-filling', 'stick', 'line'], key='ligand_style_vis', help="选择小分子的渲染样式。", index=0)
                        with row2_col2:
                            st.checkbox("🔄 旋转模型", key='spin_model_vis', value=False, help="勾选后，模型将自动围绕Z轴旋转。")
                    
                    view_html = visualize_structure_py3dmol(
                        cif_content=cif_data,
                        residue_bfactors=protein_bfactors,
                        protein_style=st.session_state.protein_style_vis,
                        ligand_style=st.session_state.ligand_style_vis,
                        spin=st.session_state.spin_model_vis,
                        color_scheme=st.session_state.color_scheme_vis
                    )
                    st.components.v1.html(view_html, height=600, scrolling=False)
                except Exception as e:
                    st.error("加载 3D 结构时发生错误。请尝试刷新页面或检查输入数据。", icon="❌")
                    st.exception(e)
            else:
                st.warning("未能从结果中提取出有效的分子结构文件 (.cif/.pdb)，无法显示 3D 结构。", icon="⚠️")

        with col_data:
            st.subheader("📈 预测质量与亲和力评估", anchor=False)
            st.markdown("---")

            st.markdown("<b>pLDDT 置信度图例</b>", unsafe_allow_html=True)
            st.markdown("""
            <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 5px; margin-bottom: 25px;'>
                <div style='display: flex; align-items: center;'><div style='width: 15px; height: 15px; background-color: #0053D6; border-radius: 3px; margin-right: 5px;'></div><span><b>极高</b> (> 90)</span></div>
                <div style='display: flex; align-items: center;'><div style='width: 15px; height: 15px; background-color: #65CBF3; border-radius: 3px; margin-right: 5px;'></div><span><b>高</b> (70-90)</span></div>
                <div style='display: flex; align-items: center;'><div style='width: 15px; height: 15px; background-color: #FFDB13; border-radius: 3px; margin-right: 5px;'></div><span><b>中等</b> (50-70)</span></div>
                <div style='display: flex; align-items: center;'><div style='width: 15px; height: 15px; background-color: #FF7D45; border-radius: 3px; margin-right: 5px;'></div><span><b>低</b> (&lt; 50)</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<b>全局结构质量指标</b>", unsafe_allow_html=True)
            cols_metrics = st.columns(2)
            cols_metrics[0].metric(
                "平均 pLDDT",
                f"{confidence_data.get('complex_plddt', 0):.2f}",
                help="预测的局部距离差异检验 (pLDDT) 是一个 0-100 范围内的单残基置信度得分，代表模型对局部结构预测的信心。这是整个复合物所有残基的平均 pLDDT 分数。值越高越好。"
            )
            cols_metrics[1].metric(
                "pTM",
                f"{confidence_data.get('ptm', 0):.4f}",
                help="预测的模板建模评分 (pTM) 是一个 0-1 范围内的分数，用于衡量预测结构与真实结构在全局拓扑结构上的相似性。pTM > 0.5 通常表示预测了正确的折叠方式。值越高越好。"
            )
            cols_metrics[0].metric(
                "ipTM",
                f"{confidence_data.get('iptm', 0):.4f}",
                help="界面预测模板建模评分 (ipTM) 是专门用于评估链间相互作用界面准确性的指标 (0-1)。ipTM > 0.85 通常表明对复合物的相互作用方式有很高的置信度。值越高越好。"
            )
            cols_metrics[1].metric(
                "PAE (Å)",
                f"{confidence_data.get('complex_pde', 0):.2f}",
                help="预测的对齐误差 (PAE) 表示残基对之间的预期位置误差（单位为埃 Å）。较低的值表示对不同结构域和链的相对位置和方向有更高的信心。这里显示的是整个复合物的平均误差。值越低越好。"
            )
            
            if affinity_data and st.session_state.properties.get('affinity'):
                st.markdown("<br><b>亲和力预测指标</b>", unsafe_allow_html=True)
                log_ic50_in_uM = affinity_data.get("affinity_pred_value")
                if log_ic50_in_uM is not None:
                    ic50_uM = math.pow(10, log_ic50_in_uM)
                    if ic50_uM > 1000:
                        display_ic50 = f"{ic50_uM/1000:.3f} mM"
                    elif ic50_uM > 1000000:
                         display_ic50 = f"{ic50_uM/1000000:.3f} M"
                    else:
                        display_ic50 = f"{ic50_uM:.3f} µM"
                    
                    pIC50 = 6 - log_ic50_in_uM
                    delta_g = -1.364 * pIC50
                    
                    st.metric("预测 IC50", display_ic50, help="预测的半数抑制浓度 (IC50) 是指结合体（Binder）抑制其靶标 50% 所需的浓度。它是衡量效力的常用指标，数值越低表示预测的亲和力越强。")
                    affinity_cols = st.columns(2)
                    affinity_cols[0].metric("预测 pIC50", f"{pIC50:.3f}", help="pIC50 是 IC50 值的负对数 (pIC50 = -log10(IC50 in M))。这个标度更便于比较，数值越高表示预测的亲和力越强。")
                    affinity_cols[1].metric("结合自由能 (ΔG)", f"{delta_g:.3f} kcal/mol", help="预测的吉布斯自由能 (ΔG) 反映了结合事件的自发性，由 pIC50 计算得出。负值越大，表明结合作用越强、越有利。")
                binder_prob = affinity_data.get("affinity_probability_binary")
                if binder_prob is not None:
                    st.metric("结合概率", f"{binder_prob:.2%}", help="模型预测结合体与其余组分形成稳定复合物的概率。百分比越高，表明模型对这是一个真实的结合事件越有信心。")
            else:
                st.info("💡 如需亲和力预测结果，请在步骤1中勾选 **计算结合亲和力 (Affinity)** 选项。", icon="ℹ️")

            st.markdown("---")
            st.markdown("<b>📥 下载结果文件</b>", unsafe_allow_html=True)
            if st.session_state.get("raw_zip"):
                st.download_button(
                    label="📥 下载所有结果 (ZIP)",
                    data=st.session_state.raw_zip,
                    file_name=f"boltz_results_{st.session_state.task_id}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                    help="下载包含所有预测结果（CIF、JSON指标等）的原始ZIP文件。"
                )
            
            cols_download = st.columns(2)
            with cols_download[0]:
                if cif_data:
                    st.download_button("📥 下载 PDB", export_to_pdb(cif_data), "predicted_structure.pdb", "chemical/x-pdb", use_container_width=True, help="下载预测结构的PDB格式文件。")
            with cols_download[1]:
                 if cif_data:
                    st.download_button("📄 下载 CIF", cif_data, "predicted_structure.cif", "chemical/x-cif", use_container_width=True, help="下载预测结构的CIF格式文件。")
            
            all_json_data = {"confidence": confidence_data, "affinity": affinity_data}
            st.download_button(
                label="📦 下载指标数据 (JSON)",
                data=json.dumps(all_json_data, indent=2),
                file_name="prediction_metrics.json",
                mime="application/json",
                use_container_width=True,
                help="下载包含pLDDT、pTM、ipTM、PAE以及亲和力预测结果的JSON文件。"
            )

# ========== 分子设计标签页 ==========
with tab2:
    st.markdown("### 🧪 从头分子设计")
    st.markdown("使用演化算法设计分子结合体，优化其与目标复合物的结合亲和力。")
    
    designer_is_running = (
        st.session_state.designer_task_id is not None and 
        st.session_state.designer_results is None and 
        st.session_state.designer_error is None
    )
    
    with st.expander("🎯 **步骤 1: 设置设计目标**", expanded=not designer_is_running and not st.session_state.designer_results):
        st.markdown("配置您的分子设计任务参数。")
        
        # 初始化 Designer 组分状态
        if 'designer_components' not in st.session_state:
            st.session_state.designer_components = [
                {'id': str(uuid.uuid4()), 'type': 'protein', 'sequence': '', 'num_copies': 1}
            ]
        
        # 组分管理
        designer_id_to_delete = None
        for i, component in enumerate(st.session_state.designer_components[:]):
            st.markdown(f"---")
            st.subheader(f"组分 {i+1}", anchor=False)
            
            cols_comp = st.columns([3, 1, 1])
            
            # 组分类型选择
            with cols_comp[0]:
                comp_type_options = ['protein', 'dna', 'rna', 'ligand']
                current_type = component.get('type', 'protein')
                current_type_index = comp_type_options.index(current_type) if current_type in comp_type_options else 0
                
                component['type'] = st.selectbox(
                    "组分类型",
                    options=comp_type_options,
                    format_func=lambda x: {
                        "protein": "🧬 蛋白质/肽链",
                        "dna": "🧬 DNA",
                        "rna": "🧬 RNA", 
                        "ligand": "💊 辅酶/小分子"
                    }[x],
                    key=f"designer_type_{component['id']}",
                    index=current_type_index,
                    disabled=designer_is_running,
                    help="选择此组分的分子类型：蛋白质、DNA、RNA或小分子配体。"
                )
            
            # 拷贝数设置
            with cols_comp[1]:
                component['num_copies'] = st.number_input(
                    "拷贝数",
                    min_value=1,
                    max_value=10,
                    value=component.get('num_copies', 1),
                    step=1,
                    key=f"designer_copies_{component['id']}",
                    disabled=designer_is_running,
                    help="此组分的拷贝数。可设置为2（二聚体）、3（三聚体）等。每个拷贝将分配独立的链ID。"
                )
            
            # 删除按钮
            with cols_comp[2]:
                if len(st.session_state.designer_components) > 1:
                    if st.button("🗑️", key=f"designer_del_{component['id']}", help="删除此组分", disabled=designer_is_running):
                        designer_id_to_delete = component['id']
            
            # 显示预计分配的链ID
            num_copies = component.get('num_copies', 1)
            if num_copies > 1:
                st.caption(f"💡 此组分将创建 {num_copies} 个拷贝，自动分配链ID")
            
            # 序列输入
            if component['type'] == 'protein':
                component['sequence'] = st.text_area(
                    f"蛋白质序列 ({'单体' if num_copies == 1 else f'{num_copies}聚体'})",
                    height=100,
                    value=component.get('sequence', ''),
                    placeholder="例如: MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV...",
                    key=f"designer_seq_{component['id']}",
                    disabled=designer_is_running,
                    help="输入此蛋白质链的完整氨基酸序列。"
                )
            elif component['type'] == 'dna':
                component['sequence'] = st.text_area(
                    f"DNA序列 ({'单链' if num_copies == 1 else f'{num_copies}链'})",
                    height=100,
                    value=component.get('sequence', ''),
                    placeholder="例如: ATGCGTAAGGGATCCGCATGC...",
                    key=f"designer_seq_{component['id']}",
                    disabled=designer_is_running,
                    help="输入DNA核苷酸序列（A、T、G、C）。"
                )
            elif component['type'] == 'rna':
                component['sequence'] = st.text_area(
                    f"RNA序列 ({'单链' if num_copies == 1 else f'{num_copies}链'})",
                    height=100,
                    value=component.get('sequence', ''),
                    placeholder="例如: AUGCGUAAGGAUCCGCAUGC...",
                    key=f"designer_seq_{component['id']}",
                    disabled=designer_is_running,
                    help="输入RNA核苷酸序列（A、U、G、C）。"
                )
            else:  # ligand
                component['input_method'] = st.radio(
                    "小分子输入方式",
                    ["smiles", "ccd", "ketcher"],
                    key=f"designer_method_{component['id']}",
                    horizontal=True,
                    disabled=designer_is_running,
                    help="选择通过SMILES字符串、PDB CCD代码或分子编辑器输入小分子。"
                )
                
                if component.get('input_method', 'smiles') == 'smiles':
                    component['sequence'] = st.text_input(
                        f"SMILES 字符串 ({'单分子' if num_copies == 1 else f'{num_copies}个分子'})",
                        value=component.get('sequence', ''),
                        placeholder="例如: CC(=O)NC1=CC=C(C=C1)O",
                        key=f"designer_seq_{component['id']}",
                        disabled=designer_is_running
                    )
                elif component.get('input_method', 'smiles') == 'ccd':
                    component['sequence'] = st.text_input(
                        f"CCD 代码 ({'单分子' if num_copies == 1 else f'{num_copies}个分子'})",
                        value=component.get('sequence', ''),
                        placeholder="例如: HEM, NAD, ATP",
                        key=f"designer_seq_{component['id']}",
                        disabled=designer_is_running
                    )
                else:  # ketcher
                    current_smiles = component.get('sequence', '')
                    smiles_from_ketcher = st_ketcher(
                        value=current_smiles,
                        key=f"designer_ketcher_{component['id']}",
                        height=400
                    )
                    
                    if smiles_from_ketcher is not None and smiles_from_ketcher != current_smiles:
                        st.session_state.designer_components[i]['sequence'] = smiles_from_ketcher
                        if smiles_from_ketcher:
                            st.toast("✅ SMILES 字符串已成功更新！", icon="🧪")
                        st.rerun()
                    
                    # 显示当前SMILES
                    current_smiles_display = st.session_state.designer_components[i].get('sequence', '')
                    if current_smiles_display:
                        st.caption("✨ 当前 SMILES 字符串:")
                        st.code(current_smiles_display, language='smiles')
                    else:
                        st.info("👆 请开始绘制或粘贴，SMILES 将会显示在这里。")
        
        # 删除组分
        if designer_id_to_delete:
            st.session_state.designer_components = [c for c in st.session_state.designer_components if c['id'] != designer_id_to_delete]
            st.rerun()
        
        # 添加组分按钮
        if st.button("➕ 添加新组分", disabled=designer_is_running, help="添加新的蛋白质、DNA/RNA或小分子组分"):
            st.session_state.designer_components.append({
                'id': str(uuid.uuid4()),
                'type': 'protein',
                'sequence': '',
                'num_copies': 1
            })
            st.rerun()
        
        # 设计目标
        st.subheader("设计目标", anchor=False)
        
        # 分别检查生物大分子和小分子目标
        target_bio_chains = [comp for comp in st.session_state.designer_components if comp['type'] in ['protein', 'dna', 'rna'] and comp.get('sequence', '').strip()]
        target_ligand_chains = [comp for comp in st.session_state.designer_components if comp['type'] == 'ligand' and comp.get('sequence', '').strip()]
        
        # 自动计算目标链ID和结合肽链ID
        if target_bio_chains or target_ligand_chains:
            # 计算总链数以确定结合肽的链ID
            total_chains = 0
            for comp in st.session_state.designer_components:
                if comp.get('sequence', '').strip():
                    total_chains += comp.get('num_copies', 1)
            
            # 结合肽链ID自动为下一个可用链ID
            binder_chain_id = string.ascii_uppercase[total_chains] if total_chains < 26 else f"Z{total_chains-25}"
            target_chain_id = 'A'  # 默认目标为第一个链
            
            # 显示设计模式和目标类型
            if target_bio_chains and target_ligand_chains:
                # 混合模式：既有生物大分子又有小分子
                bio_types = []
                for comp in target_bio_chains:
                    comp_type_name = {"protein": "蛋白质", "dna": "DNA", "rna": "RNA"}[comp['type']]
                    copies = comp.get('num_copies', 1)
                    if copies > 1:
                        bio_types.append(f"{comp_type_name}({copies}聚体)")
                    else:
                        bio_types.append(comp_type_name)
                
                ligand_count = sum(comp.get('num_copies', 1) for comp in target_ligand_chains)
                ligand_desc = f"{ligand_count}个小分子" if ligand_count > 1 else "小分子"
                
                bio_desc = "、".join(bio_types)
                target_desc = f"{bio_desc} 和 {ligand_desc}"
                st.info(f"💡 **混合设计模式**: 针对 **{target_desc}** 复合物设计结合肽，将作为链 **{binder_chain_id}** 形成复合物。", icon="🔗")
                
            elif target_bio_chains:
                # 正向设计：给定生物大分子，设计结合肽
                target_types = []
                for comp in target_bio_chains:
                    comp_type_name = {"protein": "蛋白质", "dna": "DNA", "rna": "RNA"}[comp['type']]
                    copies = comp.get('num_copies', 1)
                    if copies > 1:
                        target_types.append(f"{comp_type_name}({copies}聚体)")
                    else:
                        target_types.append(comp_type_name)
                
                target_desc = "、".join(target_types)
                st.info(f"💡 **正向设计模式**: 针对 **{target_desc}** 设计结合肽，将作为链 **{binder_chain_id}** 形成复合物。", icon="🧬")
                
            else:
                # 反向设计：给定小分子，设计结合蛋白
                ligand_count = sum(comp.get('num_copies', 1) for comp in target_ligand_chains)
                ligand_desc = f"{ligand_count}个小分子" if ligand_count > 1 else "小分子"
                st.info(f"💡 **反向设计模式**: 针对 **{ligand_desc}** 设计结合蛋白质，将作为链 **{binder_chain_id}** 形成复合物。", icon="�")
        else:
            target_chain_id = 'A'
            binder_chain_id = 'B'
            # 只有当用户确实有组分但没有目标组分时才显示警告
            has_any_components = any(comp.get('sequence', '').strip() for comp in st.session_state.designer_components)
            if has_any_components:
                st.warning("请至少添加一个蛋白质、DNA、RNA或小分子组分作为设计目标。", icon="⚠️")
            else:
                st.info("💡 请添加目标复合物组分以开始分子设计。支持正向设计（给定蛋白质设计结合肽）和反向设计（给定小分子设计结合蛋白）。", icon="ℹ️")
        
        # 设计类型选择
        st.subheader("设计参数", anchor=False)
        col1, col2 = st.columns(2)
        
        with col1:
            design_type = st.selectbox(
                "设计类型",
                options=["peptide", "glycopeptide"],
                format_func=lambda x: "🧬 多肽设计" if x == "peptide" else "🍯 糖肽设计",
                help="选择是设计普通多肽还是含有糖基修饰的糖肽。",
                disabled=designer_is_running
            )
        
        with col2:
            binder_length = st.number_input(
                "结合肽长度",
                min_value=5,
                max_value=50,
                value=20,
                step=1,
                help="设计的结合肽的氨基酸残基数量。",
                disabled=designer_is_running
            )
        
        # 演化算法参数
        st.subheader("演化算法参数", anchor=False)
        
        # 优化模式选择 (新增)
        st.subheader("🚀 优化模式选择", anchor=False)
        optimization_mode = st.selectbox(
            "选择优化策略",
            options=["balanced", "stable", "aggressive", "conservative", "custom"],
            format_func=lambda x: {
                "balanced": "⚖️ 平衡模式 (推荐)",
                "stable": "🎯 平稳优化",
                "aggressive": "🔥 激进探索", 
                "conservative": "🛡️ 保守设计",
                "custom": "⚙️ 自定义配置"
            }[x],
            index=0,
            help="选择预设的优化策略或自定义配置。不同策略适用于不同的设计场景。",
            disabled=designer_is_running
        )
        
        # 显示模式说明
        mode_descriptions = {
            "balanced": "⚖️ **平衡模式**: 综合考虑探索性和收敛性，适用于大多数设计任务。",
            "stable": "🎯 **平稳优化**: 稳定收敛，减少分数波动，适用于需要可重复结果的场景。",
            "aggressive": "🔥 **激进探索**: 快速突破局部最优，适用于初始分数较低或需要大幅改进的场景。",
            "conservative": "🛡️ **保守设计**: 小步优化，适用于已有较好序列或对稳定性要求高的场景。",
            "custom": "⚙️ **自定义配置**: 手动调整所有参数，适用于高级用户。"
        }
        st.info(mode_descriptions[optimization_mode])
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            generations = st.number_input(
                "演化代数",
                min_value=2,
                max_value=20,
                value=8,
                step=1,
                help="演化算法的迭代次数。更多代数通常产生更好的结果，但需要更长时间。",
                disabled=designer_is_running
            )
        
        with col4:
            population_size = st.number_input(
                "种群大小",
                min_value=2,
                max_value=50,
                value=12,
                step=1,
                help="每一代中的候选序列数量。",
                disabled=designer_is_running
            )
        
        with col5:
            elite_size = st.number_input(
                "精英保留数",
                min_value=2,
                max_value=min(10, population_size//2),
                value=max(2, min(5, population_size//3)),
                step=1,
                help="每一代中保留的最优个体数量。",
                disabled=designer_is_running
            )
        
        col6, col7 = st.columns(2)
        with col6:
            mutation_rate = st.slider(
                "突变率",
                min_value=0.1,
                max_value=0.8,
                value=0.3,
                step=0.05,
                help="每一代中发生突变的概率。",
                disabled=designer_is_running
            )
        
        # 高级参数配置
        if optimization_mode == "custom":
            st.subheader("🔧 高级参数配置", anchor=False)
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            
            with col_adv1:
                convergence_window = st.number_input(
                    "收敛窗口",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="收敛检测的滑动窗口大小。较小值更敏感。",
                    disabled=designer_is_running
                )
                
                convergence_threshold = st.number_input(
                    "收敛阈值",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.001,
                    format="%.4f",
                    help="收敛检测的分数方差阈值。较小值更严格。",
                    disabled=designer_is_running
                )
            
            with col_adv2:
                max_stagnation = st.number_input(
                    "最大停滞周期",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="触发早停的最大停滞周期数。较小值更激进。",
                    disabled=designer_is_running
                )
                
                initial_temperature = st.number_input(
                    "初始温度",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    help="自适应突变的初始温度。较高值更探索性。",
                    disabled=designer_is_running
                )
            
            with col_adv3:
                min_temperature = st.number_input(
                    "最小温度",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    help="自适应突变的最小温度。较高值保持更多随机性。",
                    disabled=designer_is_running
                )
                
                enable_enhanced = st.checkbox(
                    "启用增强功能",
                    value=True,
                    help="启用自适应突变、Pareto优化等增强功能。",
                    disabled=designer_is_running
                )
        else:
            # 预设模式的参数映射
            preset_params = {
                "balanced": {
                    "convergence_window": 5,
                    "convergence_threshold": 0.001,
                    "max_stagnation": 3,
                    "initial_temperature": 1.0,
                    "min_temperature": 0.1,
                    "enable_enhanced": True
                },
                "stable": {
                    "convergence_window": 5,
                    "convergence_threshold": 0.001,
                    "max_stagnation": 3,
                    "initial_temperature": 1.0,
                    "min_temperature": 0.1,
                    "enable_enhanced": True
                },
                "aggressive": {
                    "convergence_window": 3,
                    "convergence_threshold": 0.002,
                    "max_stagnation": 2,
                    "initial_temperature": 2.0,
                    "min_temperature": 0.2,
                    "enable_enhanced": True
                },
                "conservative": {
                    "convergence_window": 6,
                    "convergence_threshold": 0.0005,
                    "max_stagnation": 5,
                    "initial_temperature": 0.5,
                    "min_temperature": 0.05,
                    "enable_enhanced": True
                }
            }
            
            params = preset_params[optimization_mode]
            convergence_window = params["convergence_window"]
            convergence_threshold = params["convergence_threshold"]
            max_stagnation = params["max_stagnation"]
            initial_temperature = params["initial_temperature"]
            min_temperature = params["min_temperature"]
            enable_enhanced = params["enable_enhanced"]
        
        # 糖肽特有参数
        if design_type == "glycopeptide":
            with col7:
                # 糖基类型选项和描述
                glycan_options = {
                    "NAG": "N-乙酰葡糖胺 (N-acetylglucosamine) - 最常见的N-连接糖基化起始糖",
                    "MAN": "甘露糖 (Mannose) - 常见的高甘露糖型糖链组分",
                    "GAL": "半乳糖 (Galactose) - 复合型糖链的末端糖",
                    "FUC": "岩藻糖 (Fucose) - 分支糖链，增加分子多样性",
                    "NAN": "神经氨酸 (Neuraminic acid/Sialic acid) - 带负电荷的末端糖",
                    "GLC": "葡萄糖 (Glucose) - 基础单糖，能量代谢相关",
                    "XYL": "木糖 (Xylose) - 植物糖蛋白常见糖基",
                    "GALNAC": "N-乙酰半乳糖胺 (N-acetylgalactosamine) - O-连接糖基化起始糖",
                    "GLCA": "葡萄糖醛酸 (Glucuronic acid) - 带负电荷，参与解毒代谢"
                }
                
                glycan_type = st.selectbox(
                    "糖基类型",
                    options=list(glycan_options.keys()),
                    format_func=lambda x: f"{glycan_options[x].split(' (')[0]} ({x})",
                    index=0,  # 默认选择 NAG
                    help="选择要在糖肽中使用的糖基类型。不同糖基具有不同的化学性质和生物学功能。",
                    disabled=designer_is_running
                )
                
                # 显示选中糖基的详细信息
                st.info(f"**{glycan_type}**: {glycan_options[glycan_type]}", icon="🍯")
            
            # 糖基化位点参数
            glycosylation_site = st.number_input(
                "糖基化位点",
                min_value=1,
                max_value=binder_length,
                value=min(10, binder_length),  # 默认位点10，但不超过肽长度
                step=1,
                help=f"肽链上用于连接糖基的氨基酸位置 (1-{binder_length})。",
                disabled=designer_is_running
            )
        else:
            glycan_type = None
            glycosylation_site = None
    
    # 验证输入
    designer_is_valid, validation_message = validate_designer_inputs(st.session_state.designer_components)
    
    # 添加糖肽参数验证
    if design_type == "glycopeptide":
        if not glycan_type:
            designer_is_valid = False
            validation_message = "糖肽设计模式需要选择糖基类型。"
        elif not glycosylation_site or glycosylation_site < 1 or glycosylation_site > binder_length:
            designer_is_valid = False
            validation_message = f"糖基化位点必须在 1 到 {binder_length} 范围内。"
    
    # 提交设计任务
    if st.button("🚀 开始分子设计", type="primary", disabled=(not designer_is_valid or designer_is_running), use_container_width=True):
        st.session_state.designer_task_id = None
        st.session_state.designer_results = None
        st.session_state.designer_error = None
        
        with st.spinner("⏳ 正在启动设计任务，请稍候..."):
            try:
                # 创建复合物模板 YAML
                template_yaml = create_designer_complex_yaml(st.session_state.designer_components)
                
                # 提交设计任务
                result = submit_designer_job(
                    template_yaml_content=template_yaml,
                    design_type=design_type,
                    binder_length=binder_length,
                    target_chain_id=target_chain_id,
                    generations=generations,
                    population_size=population_size,
                    elite_size=elite_size,
                    mutation_rate=mutation_rate,
                    glycan_type=glycan_type,
                    glycosylation_site=glycosylation_site,
                    # 增强功能参数
                    convergence_window=convergence_window,
                    convergence_threshold=convergence_threshold,
                    max_stagnation=max_stagnation,
                    initial_temperature=initial_temperature,
                    min_temperature=min_temperature,
                    enable_enhanced=enable_enhanced
                )
                
                if result['success']:
                    st.session_state.designer_task_id = result['task_id']
                    st.session_state.designer_work_dir = result['work_dir']
                    st.session_state.designer_config = result['params']
                    st.toast(f"🎉 设计任务已成功启动！任务ID: {result['task_id']}", icon="✅")
                    st.rerun()
                else:
                    st.error(f"❌ **任务启动失败**：{result['error']}")
                    st.session_state.designer_error = {"error_message": result['error'], "type": "Task Start Error"}
                    
            except Exception as e:
                st.error(f"❌ **任务启动失败：发生未知错误**。详情: {e}")
                st.session_state.designer_error = {"error_message": str(e), "type": "Client Error"}
    
    if not designer_is_valid and not designer_is_running:
        # 只有当用户确实有输入内容时才显示验证错误
        has_user_input = any(comp.get('sequence', '').strip() for comp in st.session_state.designer_components)
        if has_user_input:
            st.error(f"⚠️ **无法启动设计**: {validation_message}")
    
    # 显示设计进度和结果
    if st.session_state.designer_task_id and not st.session_state.designer_results:
        st.divider()
        st.header("🔄 **步骤 2: 设计进度监控**", anchor=False)
        
        if not st.session_state.designer_error:
            # 检查任务状态并处理错误
            try:
                work_dir = st.session_state.get('designer_work_dir', None)
                status_data = get_designer_status(st.session_state.designer_task_id, work_dir)
                
                # 验证状态数据
                if not status_data or 'state' not in status_data:
                    st.error("❌ 无法获取任务状态信息，任务可能已失败或被中断")
                    st.session_state.designer_error = {"error_message": "无法获取任务状态", "type": "Status Error"}
                elif status_data.get('error'):
                    st.error(f"❌ 任务执行错误: {status_data['error']}")
                    st.session_state.designer_error = {"error_message": status_data['error'], "type": "Task Error"}
                else:
                    # 状态检查成功，显示进度
                    current_state = status_data['state']
                    
                    if current_state in ['COMPLETED', 'SUCCESS']:
                        st.success("🎉 设计任务已完成！正在加载结果...")
                        try:
                            work_dir = st.session_state.get('designer_work_dir', '/tmp')
                            results = load_designer_results(st.session_state.designer_task_id, work_dir)
                            st.session_state.designer_results = results
                            st.toast("✅ 设计任务已完成！", icon="🎊")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 加载结果时发生错误: {e}")
                            st.session_state.designer_error = {"error_message": str(e), "type": "Result Loading Error"}
                    
                    elif current_state in ['ERROR', 'FAILED', 'CANCELLED']:
                        error_msg = status_data.get('error', status_data.get('message', '任务失败，原因未知'))
                        st.error(f"❌ 设计任务失败: {error_msg}")
                        st.session_state.designer_error = {"error_message": error_msg, "type": "Task Error"}
                    
                    elif current_state == 'RUNNING':
                        progress = status_data.get('progress', {})
                        progress_value = min(1.0, max(0.0, progress.get('estimated_progress', 0.0)))
                        status_msg = progress.get('status_message', '设计进行中')
                        current_status = progress.get('current_status', 'unknown')
                        
                        if current_status == 'waiting_for_prediction':
                            pending = progress.get('pending_tasks', 0)
                            completed = progress.get('completed_tasks', 0)
                            total_tasks = pending + completed
                            if total_tasks > 0:
                                task_progress = completed / total_tasks
                                st.progress(task_progress, text=f"结构预测进度: {completed}/{total_tasks} 任务完成")
                            else:
                                st.progress(0.1, text="正在初始化结构预测任务...")
                            st.info(f"🔬 {status_msg}")
                        
                        elif current_status == 'evolving':
                            current_gen = progress.get('current_generation', 0)
                            total_gen = progress.get('total_generations', 1)
                            best_score = progress.get('best_score', 0.0)
                            debug_info = progress.get('debug_info', {})
                            
                            # 检查是否已完成所有代数且有结果
                            current_best_sequences = progress.get('current_best_sequences', [])
                            
                            # 更严格的完成检测逻辑：
                            # 1. 当前代数严格大于总代数（完全结束）
                            # 2. 或者当前代数等于总代数且有明确的完成证据
                            # 3. 或者run_design.py进程已经结束且有结果文件
                            task_completed = False
                            
                            if current_gen > total_gen and total_gen > 0:
                                task_completed = True
                            elif current_gen == total_gen and total_gen > 0 and current_best_sequences:
                                # 最后一代的情况，需要非常严格的验证
                                try:
                                    work_dir = st.session_state.get('designer_work_dir', '/tmp')
                                    log_file = os.path.join(work_dir, 'design.log')
                                    if os.path.exists(log_file):
                                        with open(log_file, 'r') as f:
                                            log_content = f.read()
                                            # 检查是否有明确的最终完成标记
                                            if any(indicator in log_content for indicator in [
                                                'Design Run Finished', 
                                                '设计任务完成', 
                                                'Successfully created results package',
                                                'All generations completed',
                                                f'Finished all {total_gen} generations'
                                            ]):
                                                task_completed = True
                                            # 或者检查CSV文件是否最近被更新且包含足够数据
                                            else:
                                                csv_files = [f for f in os.listdir(work_dir) 
                                                           if f.startswith('design_summary_') and f.endswith('.csv')]
                                                for csv_file in csv_files:
                                                    csv_path = os.path.join(work_dir, csv_file)
                                                    if os.path.exists(csv_path):
                                                        file_age = time.time() - os.path.getmtime(csv_path)
                                                        if file_age < 15:  # 15秒内修改过
                                                            # 额外验证：检查CSV中的代数数据
                                                            try:
                                                                df = pd.read_csv(csv_path)
                                                                if len(df) > 0:
                                                                    max_gen_in_csv = df['generation'].max() if 'generation' in df.columns else 0
                                                                    # 确保CSV中确实包含了最后一代的数据
                                                                    if max_gen_in_csv >= total_gen:
                                                                        task_completed = True
                                                                        break
                                                            except:
                                                                pass
                                except Exception:
                                    # 如果检查失败，不认为完成
                                    pass
                            
                            # 额外检查：特定的run_design.py 进程是否还在运行
                            if not task_completed:
                                try:
                                    # 检查保存的进程ID是否仍在运行
                                    work_dir = st.session_state.get('designer_work_dir', '/tmp')
                                    status_file_path = os.path.join(work_dir, 'status.json')
                                    design_process_running = False
                                    
                                    if os.path.exists(status_file_path):
                                        with open(status_file_path, 'r') as f:
                                            status_info = json.load(f)
                                            saved_pid = status_info.get('process_id')
                                            
                                            if saved_pid and psutil:
                                                try:
                                                    if psutil.pid_exists(saved_pid):
                                                        # 进一步验证这个PID确实是我们的run_design.py进程
                                                        proc = psutil.Process(saved_pid)
                                                        cmdline = proc.cmdline()
                                                        if cmdline and 'run_design.py' in ' '.join(cmdline):
                                                            design_process_running = True
                                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                                    design_process_running = False
                                    
                                    # 如果run_design.py进程已经结束，且有结果文件，可能任务已完成
                                    if not design_process_running and current_best_sequences:
                                        # 检查是否有最近的结果文件
                                        csv_files = [f for f in os.listdir(work_dir) 
                                                   if f.startswith('design_summary_') and f.endswith('.csv')]
                                        for csv_file in csv_files:
                                            csv_path = os.path.join(work_dir, csv_file)
                                            if os.path.exists(csv_path):
                                                file_age = time.time() - os.path.getmtime(csv_path)
                                                if file_age < 30:  # 30秒内修改过
                                                    task_completed = True
                                                    break
                                except Exception:
                                    # 如果检查失败，继续使用原有的检测逻辑
                                    pass
                            
                            if task_completed:
                                st.success("🎉 设计任务已完成！正在加载最终结果...")
                                try:
                                    work_dir = st.session_state.get('designer_work_dir', '/tmp')
                                    results = load_designer_results(st.session_state.designer_task_id, work_dir)
                                    st.session_state.designer_results = results
                                    st.toast("✅ 设计任务已完成！", icon="🎊")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ 加载结果时发生错误: {e}")
                                    st.session_state.designer_error = {"error_message": str(e), "type": "Result Loading Error"}
                            else:
                                if current_gen > 0:
                                    # 正常的进度条显示逻辑
                                    gen_progress = min(current_gen / total_gen, 1.0)
                                    
                                    st.progress(gen_progress, text=f"演化进度: 第 {current_gen}/{total_gen} 代 | 当前最佳评分: {best_score:.3f}")
                                    
                                    # 如果当前代数等于总代数，显示最后一代进行中的状态
                                    if current_gen == total_gen:
                                        st.info("🧬 正在完成最后一代演化，请稍候...")
                                else:
                                    st.progress(0.0, text="准备开始演化...")
                                
                                st.info(f"🧬 {status_msg}")
                                
                                # # 可选的调试信息展示
                                # if debug_info and st.checkbox("显示调试信息", key="show_debug_evolving"):
                                #     with st.expander("🔧 调试信息", expanded=False):
                                #         st.json(debug_info)
                                
                                # 显示当前最佳序列
                                if current_best_sequences:
                                    with st.expander(f"🏆 当前最佳序列 (第 {current_gen} 代)", expanded=True):
                                        for i, seq_info in enumerate(current_best_sequences[:3]):
                                            rank = i + 1
                                            score = seq_info.get('score', 0)
                                            sequence = seq_info.get('sequence', '')
                                            iptm = seq_info.get('iptm', 0)
                                            plddt = seq_info.get('plddt', 0)
                                            generation = current_gen
                                            
                                            if not sequence or len(sequence) < 8:
                                                continue
                                            
                                            if score >= 0.8:
                                                score_color = "🟢"
                                            elif score >= 0.7:
                                                score_color = "🟡"
                                            elif score >= 0.6:
                                                score_color = "🟠"
                                            else:
                                                score_color = "🔴"
                                            
                                            st.markdown(f"**#{rank}** {score_color} 综合评分: {score:.3f} | ipTM: {iptm:.3f} | pLDDT: {plddt:.1f} | 代数: {generation}")
                                            st.code(sequence, language="text")
                                        
                                        if len(current_best_sequences) > 3:
                                            st.caption(f"还有 {len(current_best_sequences) - 3} 个候选序列...")
                                else:
                                    st.caption("💡 当前代最佳序列将在演化过程中显示")
                        
                        elif current_status == 'processing_results':
                            st.progress(0.9, text="正在处理和分析结果...")
                            st.info(f"⚙️ {status_msg}")
                        
                        else:
                            st.progress(progress_value, text=f"设计进度: {int(progress_value * 100)}%")
                            st.info(f"🔄 {status_msg}")
                        
                        # 显示刷新倒计时
                        countdown_placeholder = st.empty()
                        for remaining in range(10, 0, -1):
                            countdown_placeholder.caption(f"🔄 将在 {remaining} 秒后自动刷新...")
                            time.sleep(1)
                        
                        st.rerun()
                    
                    else:
                        # 处理其他状态，包括可能的"未明确标记为完成但实际已完成"的情况
                        # 检查是否有完成的迹象
                        progress = status_data.get('progress', {})
                        current_gen = progress.get('current_generation', 0)
                        total_gen = progress.get('total_generations', 1)
                        csv_sequences = progress.get('current_best_sequences', [])
                        
                        # 最严格的完成检测：
                        # 1. 代数严格超过总代数（完全结束）
                        # 2. 或者代数等于总代数且有明确完成证据
                        # 3. 或者run_design.py进程已经结束且有结果文件
                        task_likely_completed = False
                        
                        if current_gen > total_gen and total_gen > 0:
                            task_likely_completed = True
                        elif current_gen == total_gen and total_gen > 0 and csv_sequences:
                            # 最后一代的情况，需要非常严格的验证
                            try:
                                work_dir = st.session_state.get('designer_work_dir', '/tmp')
                                log_file = os.path.join(work_dir, 'design.log')
                                if os.path.exists(log_file):
                                    with open(log_file, 'r') as f:
                                        log_content = f.read()
                                        # 检查明确的完成标记
                                        if any(indicator in log_content for indicator in [
                                            'Design Run Finished', 
                                            '设计任务完成', 
                                            'Successfully created results package',
                                            'All generations completed',
                                            f'Finished all {total_gen} generations'
                                        ]):
                                            task_likely_completed = True
                                        else:
                                            # 检查CSV文件的新鲜度和数据完整性
                                            csv_files = [f for f in os.listdir(work_dir) 
                                                       if f.startswith('design_summary_') and f.endswith('.csv')]
                                            for csv_file in csv_files:
                                                csv_path = os.path.join(work_dir, csv_file)
                                                if os.path.exists(csv_path):
                                                    file_age = time.time() - os.path.getmtime(csv_path)
                                                    if file_age < 15:  # 15秒内修改过
                                                        # 验证CSV数据的完整性
                                                        try:
                                                            df = pd.read_csv(csv_path)
                                                            if len(df) > 0:
                                                                max_gen_in_csv = df['generation'].max() if 'generation' in df.columns else 0
                                                                if max_gen_in_csv >= total_gen:
                                                                    task_likely_completed = True
                                                                    break
                                                        except:
                                                            pass
                            except Exception:
                                # 检查失败时，不认为完成
                                pass
                        
                        # 额外检查：特定的run_design.py 进程是否还在运行
                        if not task_likely_completed:
                            try:
                                work_dir = st.session_state.get('designer_work_dir', '/tmp')
                                status_file_path = os.path.join(work_dir, 'status.json')
                                design_process_running = False
                                
                                if os.path.exists(status_file_path):
                                    with open(status_file_path, 'r') as f:
                                        status_info = json.load(f)
                                        saved_pid = status_info.get('process_id')
                                        
                                        if saved_pid and psutil:
                                            try:
                                                if psutil.pid_exists(saved_pid):
                                                    # 进一步验证这个PID确实是我们的run_design.py进程
                                                    proc = psutil.Process(saved_pid)
                                                    cmdline = proc.cmdline()
                                                    if cmdline and 'run_design.py' in ' '.join(cmdline):
                                                        design_process_running = True
                                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                                design_process_running = False
                                
                                # 如果run_design.py进程已经结束，且有结果文件，可能任务已完成
                                if not design_process_running and csv_sequences:
                                    # 检查是否有最近的结果文件
                                    csv_files = [f for f in os.listdir(work_dir) 
                                               if f.startswith('design_summary_') and f.endswith('.csv')]
                                    for csv_file in csv_files:
                                        csv_path = os.path.join(work_dir, csv_file)
                                        if os.path.exists(csv_path):
                                            file_age = time.time() - os.path.getmtime(csv_path)
                                            if file_age < 30:  # 30秒内修改过
                                                task_likely_completed = True
                                                break
                            except Exception:
                                # 如果检查失败，继续使用原有的检测逻辑
                                pass
                        
                        if task_likely_completed:
                            st.success("🎉 设计任务已完成！正在加载结果...")
                            try:
                                work_dir = st.session_state.get('designer_work_dir', '/tmp')
                                results = load_designer_results(st.session_state.designer_task_id, work_dir)
                                st.session_state.designer_results = results
                                st.toast("✅ 设计任务已完成！", icon="🎊")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ 加载结果时发生错误: {e}")
                                st.session_state.designer_error = {"error_message": str(e), "type": "Result Loading Error"}
                        else:
                            st.info(f"🕒 设计任务状态: {current_state}，正在检查完成状态...")
                            # 给用户更多信息
                            if current_gen > 0 and total_gen > 0:
                                st.caption(f"当前进度: 第 {current_gen}/{total_gen} 代")
                            if csv_sequences:
                                st.caption(f"已找到 {len(csv_sequences)} 个候选序列")
                            time.sleep(3)
                            st.rerun()
                        
            except Exception as e:
                st.error(f"❌ 获取任务状态时发生错误: {e}")
                st.session_state.designer_error = {"error_message": str(e), "type": "Status Check Error"}

        # 显示设计结果
    if st.session_state.designer_results:
        st.divider()
        st.header("🏆 **步骤 2: 设计结果展示**", anchor=False)
        
        results = st.session_state.designer_results
        best_sequences = results['best_sequences']
        evolution_history = results['evolution_history']
        
        # 结果统计摘要
        st.subheader("📊 设计统计摘要", anchor=False)
        
        # 应用阈值过滤
        score_threshold = 0.6
        high_quality_sequences = [seq for seq in best_sequences if seq.get('score', 0) >= score_threshold]
        top_sequences = high_quality_sequences[:10]  # Top 10
        
        col_stats = st.columns(4)
        col_stats[0].metric("总设计数", len(best_sequences))
        col_stats[1].metric("高质量设计", len(high_quality_sequences), help=f"评分 ≥ {score_threshold}")
        col_stats[2].metric("Top 10 选中", len(top_sequences))
        if best_sequences:
            col_stats[3].metric("最高评分", f"{max(seq.get('score', 0) for seq in best_sequences):.3f}")
        
        # 设置阈值控制
        with st.expander("🎛️ **结果过滤设置**", expanded=False):
            st.markdown("调整以下参数来筛选和显示设计结果：")
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                custom_threshold = st.slider(
                    "评分阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=score_threshold,
                    step=0.05,
                    help="只显示评分高于此阈值的设计"
                )
                
            with col_filter2:
                max_display = st.slider(
                    "最大显示数量",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=1,
                    help="最多显示多少个符合条件的设计"
                )
            
            # 重新过滤
            if custom_threshold != score_threshold:
                high_quality_sequences = [seq for seq in best_sequences if seq.get('score', 0) >= custom_threshold]
                top_sequences = high_quality_sequences[:max_display]
                
                # 更新统计
                col_stats[1].metric("高质量设计", len(high_quality_sequences), help=f"评分 ≥ {custom_threshold}")
                col_stats[2].metric(f"Top {max_display} 选中", len(top_sequences))
        
        # 最佳设计序列
        st.subheader("🥇 最佳设计序列", anchor=False)
        
        if not top_sequences:
            st.warning(f"😔 没有找到评分高于 {custom_threshold} 的设计序列。请尝试降低阈值或检查设计参数。")
        else:
            st.success(f"🎉 找到 {len(top_sequences)} 个高质量设计序列！")
            
            for i, seq_data in enumerate(top_sequences):
                rank = i + 1
                score = seq_data.get('score', 0)
                
                # 评分颜色编码
                if score >= 0.8:
                    score_color = "🟢"  # 绿色 - 优秀
                elif score >= 0.7:
                    score_color = "🟡"  # 黄色 - 良好
                elif score >= 0.6:
                    score_color = "🟠"  # 橙色 - 可接受
                else:
                    score_color = "🔴"  # 红色 - 较低
                
                with st.expander(
                    f"**第 {rank} 名** {score_color} 评分: {score:.3f}", 
                    expanded=(i < 3)  # 默认展开前3个
                ):
                    st.code(seq_data['sequence'], language="text")
                    
                    col_metrics = st.columns(4)
                    col_metrics[0].metric("综合评分", f"{score:.3f}")
                    col_metrics[1].metric("ipTM", f"{seq_data.get('iptm', 0):.3f}")
                    col_metrics[2].metric("pLDDT", f"{seq_data.get('plddt', 0):.3f}")
                    col_metrics[3].metric("发现代数", seq_data.get('generation', 'N/A'))
                    
                    # 下载结构文件
                    results_path = seq_data.get('results_path', '')
                    if results_path and os.path.exists(results_path):
                        # 查找CIF文件
                        cif_files = [f for f in os.listdir(results_path) if f.endswith('.cif')]
                        if cif_files:
                            # 优先选择rank_1的文件，否则选择第一个
                            cif_file = next((f for f in cif_files if 'rank_1' in f), cif_files[0])
                            cif_path = os.path.join(results_path, cif_file)
                            
                            try:
                                with open(cif_path, 'r') as f:
                                    cif_data = f.read()
                                
                                col_download = st.columns(2)
                                with col_download[0]:
                                    st.download_button(
                                        label="📄 下载 CIF",
                                        data=cif_data,
                                        file_name=f"rank_{rank}_designed_structure.cif",
                                        mime="chemical/x-cif",
                                        use_container_width=True,
                                        key=f"download_cif_{i}",
                                        help="下载该设计序列的3D结构文件 (CIF格式)"
                                    )
                                
                                with col_download[1]:
                                    # 转换为PDB格式并下载
                                    try:
                                        pdb_data = export_to_pdb(cif_data)
                                        st.download_button(
                                            label="📥 下载 PDB",
                                            data=pdb_data,
                                            file_name=f"rank_{rank}_designed_structure.pdb",
                                            mime="chemical/x-pdb",
                                            use_container_width=True,
                                            key=f"download_pdb_{i}",
                                            help="下载该设计序列的3D结构文件 (PDB格式)"
                                        )
                                    except Exception as e:
                                        st.caption(f"PDB转换失败: {str(e)}")
                                        
                            except Exception as e:
                                st.caption(f"⚠️ 结构文件读取失败: {str(e)}")
                        else:
                            st.caption("⚠️ 未找到结构文件")
                    else:
                        st.caption("⚠️ 结构文件路径不可用")
        
        # 演化历史图表
        st.subheader("📈 演化历史", anchor=False)
        
        # 创建演化曲线数据
        chart_data = pd.DataFrame({
            '代数': evolution_history.get('generations', []),
            '最佳评分': evolution_history.get('best_scores', []),
            '平均评分': evolution_history.get('avg_scores', [])
        })
        
        if not chart_data.empty:
            st.line_chart(chart_data.set_index('代数'))
        else:
            st.info("暂无演化历史数据可显示。")
        
        # 下载设计结果
        st.markdown("<b>📥 下载设计结果</b>", unsafe_allow_html=True)
        
        col_download = st.columns(2)
        
        # 1. CSV 下载
        with col_download[0]:
            if top_sequences:
                sequences_csv = pd.DataFrame(top_sequences)
                sequences_csv_str = sequences_csv.to_csv(index=False)
                
                st.download_button(
                    label="📊 Top序列 (CSV)",
                    data=sequences_csv_str,
                    file_name=f"top_designed_sequences_{st.session_state.designer_task_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help=f"下载前 {len(top_sequences)} 个高质量设计序列"
                )
            else:
                st.button("📊 CSV下载", disabled=True, help="无符合条件的序列")
        
        # 2. JSON 下载
        with col_download[1]:
            results_json = json.dumps({
                'summary': {
                    'total_sequences': len(best_sequences),
                    'high_quality_sequences': len(high_quality_sequences),
                    'threshold_applied': custom_threshold,
                    'top_selected': len(top_sequences)
                },
                'top_sequences': top_sequences,
                'evolution_history': evolution_history
            }, indent=2)
            
            st.download_button(
                label="📦 完整结果 (JSON)",
                data=results_json,
                file_name=f"design_results_{st.session_state.designer_task_id}.json",
                mime="application/json",
                use_container_width=True,
                help="下载包含演化历史的完整结果数据"
            )
    
    # 显示错误信息
    if st.session_state.designer_error:
        st.error("ℹ️ 设计任务执行失败，详细信息如下：")
        st.json(st.session_state.designer_error)
        if st.button("🔄 重置设计器", type="secondary"):
            for key in ['designer_task_id', 'designer_results', 'designer_error', 'designer_config']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
