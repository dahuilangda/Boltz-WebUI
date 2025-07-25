# /Boltz-WebUI/designer/design_utils.py

"""
design_utils.py

该模块提供了蛋白质和糖肽设计所需的核心数据结构和辅助工具函数。
功能包括：
- 定义氨基酸、单糖和糖基化位点的常量。
- 提供基于BLOSUM62矩阵的氨基酸替换分数。
- 实现序列生成、突变和结构预测结果解析的功能。

所有函数均为无状态的，不依赖于外部类实例。
"""

import random
import os
import json
import logging
import numpy as np
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# --- 初始化日志记录器 ---
# 该模块的日志记录器将继承主入口点的配置
logger = logging.getLogger(__name__)

# --- 核心常量 ---

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"

# --- 糖化学常量 ---
MONOSACCHARIDES = {
    # 最常见的N-连接糖基化起始糖
    'NAG': {'atom': 'C1', 'type': ['N-linked', 'O-linked'], 'name': 'N-乙酰葡糖胺', 'eng_name': 'N-acetylglucosamine'},
    
    # 常见的高甘露糖型糖链组分
    'MAN': {'atom': 'C1', 'type': ['N-linked'], 'name': '甘露糖', 'eng_name': 'Mannose'},
    
    # 复合型糖链的末端糖
    'GAL': {'atom': 'C1', 'type': ['N-linked', 'O-linked'], 'name': '半乳糖', 'eng_name': 'Galactose'},
    
    # 分支糖链，增加分子多样性
    'FUC': {'atom': 'C1', 'type': ['N-linked', 'O-linked'], 'name': '岩藻糖', 'eng_name': 'Fucose'},
    
    # 带负电荷的末端糖（神经氨酸/唾液酸）
    'NAN': {'atom': 'C2', 'type': ['O-linked'], 'name': '神经氨酸', 'eng_name': 'Neuraminic acid'},
    
    # 额外的常用糖基
    'GLC': {'atom': 'C1', 'type': ['N-linked', 'O-linked'], 'name': '葡萄糖', 'eng_name': 'Glucose'},
    'XYL': {'atom': 'C1', 'type': ['N-linked'], 'name': '木糖', 'eng_name': 'Xylose'},
    'GLCNAC': {'atom': 'C1', 'type': ['N-linked', 'O-linked'], 'name': 'N-乙酰葡糖胺', 'eng_name': 'N-acetylglucosamine'},
    'GALNAC': {'atom': 'C1', 'type': ['O-linked'], 'name': 'N-乙酰半乳糖胺', 'eng_name': 'N-acetylgalactosamine'},
    'GLCA': {'atom': 'C1', 'type': ['O-linked'], 'name': '葡萄糖醛酸', 'eng_name': 'Glucuronic acid'},
    
    # 历史兼容性保留（SIA是旧的神经氨酸代号）
    'SIA': {'atom': 'C2', 'type': ['O-linked'], 'name': '唾液酸', 'eng_name': 'Sialic acid'},
}

GLYCOSYLATION_SITES = {
    # N-连接糖基化：发生在天冬酰胺(N)上，通常在Asn-X-Ser/Thr基序中
    'N-linked': {
        'N': 'ND2'  # 天冬酰胺的侧链胺基氮原子
    },
    # O-连接糖基化：发生在丝氨酸(S)或苏氨酸(T)的羟基上
    'O-linked': {
        'S': 'OG',    # 丝氨酸的羟基氧原子
        'T': 'OG1',   # 苏氨酸的羟基氧原子
        'Y': 'OH'     # 酪氨酸的酚羟基（较少见但存在）
    },
    # C-连接糖基化：较少见，发生在色氨酸上
    'C-linked': {
        'W': 'CD1'    # 色氨酸吲哚环的C2位
    }
}

# BLOSUM62 替换矩阵
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4}
}

def get_valid_residues_for_glycan(glycan_ccd: str) -> list:
    """根据给定的聚糖CCD代码，返回其兼容的氨基酸残基列表。"""
    if not glycan_ccd or glycan_ccd not in MONOSACCHARIDES:
        raise KeyError(f"Glycan CCD '{glycan_ccd}' is not defined in MONOSACCHARIDES.")
    allowed_types = MONOSACCHARIDES[glycan_ccd]['type']
    valid_residues = []
    if 'N-linked' in allowed_types:
        valid_residues.extend(GLYCOSYLATION_SITES['N-linked'].keys())
    if 'O-linked' in allowed_types:
        valid_residues.extend(GLYCOSYLATION_SITES['O-linked'].keys())
    if 'C-linked' in allowed_types:
        valid_residues.extend(GLYCOSYLATION_SITES['C-linked'].keys())
    if not valid_residues:
        raise ValueError(f"No valid glycosylation types found for glycan '{glycan_ccd}'.")
    return list(set(valid_residues))


def generate_random_sequence(length: int, glycosylation_site: int = None, glycan_ccd: str = None) -> str:
    """生成一个随机的氨基酸序列。"""
    seq = list("".join(random.choice(AMINO_ACIDS) for _ in range(length)))
    if glycosylation_site is not None:
        if 0 <= glycosylation_site < length:
            if glycan_ccd:
                valid_residues = get_valid_residues_for_glycan(glycan_ccd)
            else:
                # 默认包含所有类型的糖基化位点
                valid_residues = (list(GLYCOSYLATION_SITES['N-linked'].keys()) + 
                                list(GLYCOSYLATION_SITES['O-linked'].keys()) + 
                                list(GLYCOSYLATION_SITES['C-linked'].keys()))
            seq[glycosylation_site] = random.choice(valid_residues)
        else:
            raise ValueError("glycosylation_site index is out of bounds for the given sequence length.")
    return "".join(seq)


def mutate_sequence(
    sequence: str,
    mutation_rate: float = 0.1,
    plddt_scores: list = None,
    temperature: float = 1.0,
    glycosylation_site: int = None,
    glycan_ccd: str = None,
    position_selection_temp: float = 1.0
) -> str:
    """
    对序列进行点突变，突变过程受pLDDT和BLOSUM62矩阵指导。

    - **突变位置选择**: 优先选择pLDDT分数较低（即模型预测的低置信度）的区域。
      `position_selection_temp` 参数用于调节该选择压力：
        - temp=1.0: 标准权重。
        - temp>1.0: 降低pLDDT的影响，位置选择更趋于随机，增强探索性。
        - temp<1.0: 增强pLDDT的影响，位置选择更集中于低分区域，增强利用性。
    - **氨基酸替换选择**: 通过带温度的Softmax函数和BLOSUM62矩阵进行加权。
    - **糖基化位点保护**: 指定的糖基化位点将被保护，确保其残基始终与指定的聚糖类型兼容。

    Args:
        sequence (str): 原始氨基酸序列。
        mutation_rate (float): 序列中要突变的残基比例。
        plddt_scores (list, optional): 与序列对应的pLDDT分数列表。
        temperature (float): Softmax函数的温度因子，用于氨基酸替换选择。
        glycosylation_site (int, optional): 要保护的糖基化位点（0-based索引）。
        glycan_ccd (str, optional): 聚糖的CCD代码，用于验证糖基化位点。
        position_selection_temp (float): 用于调节pLDDT指导位置选择的温度因子。

    Returns:
        str: 突变后的新序列。
    """
    new_sequence = list(sequence)
    num_mutations = max(1, int(len(sequence) * mutation_rate))

    # --- 步骤 1: 选择突变位置 (pLDDT指导) ---
    available_indices = list(range(len(sequence)))
    if glycosylation_site is not None and glycosylation_site in available_indices:
        available_indices.remove(glycosylation_site)

    if not available_indices:
        logger.warning("No available positions to mutate after excluding the glycosylation site. Returning original sequence.")
        return sequence

    positions_to_mutate = []
    if plddt_scores and len(plddt_scores) == len(sequence):
        # 根据 (100 - pLDDT) 对位置进行加权，优先选择低置信度区域
        # 应用 position_selection_temp 来调整选择压力
        safe_temp = max(position_selection_temp, 1e-6) # 避免除以零
        weights = [(100.0 - plddt_scores[i]) / safe_temp for i in available_indices]
        
        total_weight = sum(weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
            k = min(num_mutations, len(available_indices))
            positions_to_mutate = np.random.choice(available_indices, size=k, replace=False, p=probabilities)
        else:
            # 如果所有pLDDT都为100或权重因其他原因失效，则随机选择
            logger.debug("All pLDDT scores are high; falling back to random position selection.")
            positions_to_mutate = random.sample(available_indices, k=min(num_mutations, len(available_indices)))
    else:
        if plddt_scores:
            logger.warning("pLDDT scores length mismatch or not provided. Falling back to random position selection.")
        # 如果没有提供pLDDT，则随机选择突变位置
        positions_to_mutate = random.sample(available_indices, k=min(num_mutations, len(available_indices)))

    # --- 步骤 2: 选择替换的氨基酸 (BLOSUM62) ---
    for pos in positions_to_mutate:
        original_aa = new_sequence[pos]
        substitution_scores = BLOSUM62.get(original_aa, {})
        possible_aas = [aa for aa in AMINO_ACIDS if aa != original_aa]
        if not possible_aas: continue

        scores = [substitution_scores.get(aa, 0) for aa in possible_aas]
        scores_array = np.array(scores) / temperature
        probabilities = np.exp(scores_array - np.max(scores_array)) # Softmax
        probabilities /= np.sum(probabilities)

        new_aa = np.random.choice(possible_aas, p=probabilities)
        new_sequence[pos] = new_aa
        
    # --- 步骤 3: 确保糖基化位点对于指定的聚糖仍然有效 ---
    if glycosylation_site is not None and glycan_ccd:
        valid_residues = get_valid_residues_for_glycan(glycan_ccd)
        if new_sequence[glycosylation_site] not in valid_residues:
            logger.debug(
                f"Correcting glycosylation site {glycosylation_site} from "
                f"'{new_sequence[glycosylation_site]}' to a valid residue for '{glycan_ccd}'."
            )
            new_sequence[glycosylation_site] = random.choice(valid_residues)

    return "".join(new_sequence)


def parse_confidence_metrics(results_path: str, binder_chain_id: str) -> dict:
    """从Boltz预测输出文件中解析关键的置信度指标（如ipTM, pLDDT）。"""
    metrics = {
        'iptm': 0.0, 'ptm': 0.0, 'complex_plddt': 0.0,
        'binder_avg_plddt': 0.0, 'plddts': []
    }
    # --- Parse confidence.json file ---
    try:
        json_path = next((os.path.join(results_path, f) for f in os.listdir(results_path) if f.startswith('confidence_') and f.endswith('.json')), None)
        if json_path:
            with open(json_path, 'r') as f: data = json.load(f)
            metrics.update({
                'ptm': data.get('ptm', 0.0),
                'complex_plddt': data.get('complex_plddt', 0.0)
            })
            pair_iptm = data.get('pair_chains_iptm', {})
            chain_ids = list(pair_iptm.keys())
            if len(chain_ids) > 1:
                c1, c2 = chain_ids[0], chain_ids[1]
                metrics['iptm'] = pair_iptm.get(c1, {}).get(c2, 0.0)
            if metrics['iptm'] == 0.0:
                metrics['iptm'] = data.get('iptm', 0.0)
    except Exception as e:
        logger.warning(f"Could not parse confidence metrics from JSON in {results_path}. Error: {e}")

    # --- Parse pLDDT scores from CIF file ---
    try:
        cif_files = [f for f in os.listdir(results_path) if f.endswith('.cif')]
        if cif_files:
            rank_1_cif = next((f for f in cif_files if 'rank_1' in f), cif_files[0])
            cif_path = os.path.join(results_path, rank_1_cif)
            with open(cif_path, 'r') as f: lines = f.readlines()

            header, atom_lines, in_loop = [], [], False
            for line in lines:
                s_line = line.strip()
                if s_line.startswith('_atom_site.'):
                    header.append(s_line)
                    in_loop = True
                elif in_loop and (s_line.startswith('ATOM') or s_line.startswith('HETATM')):
                    atom_lines.append(s_line)
                elif in_loop and not s_line:
                    in_loop = False

            if header and atom_lines:
                h_map = {name: i for i, name in enumerate(header)}
                chain_col = h_map.get('_atom_site.label_asym_id') or h_map.get('_atom_site.auth_asym_id')
                res_col, bfactor_col = h_map.get('_atom_site.label_seq_id'), h_map.get('_atom_site.B_iso_or_equiv')

                if all(c is not None for c in [chain_col, res_col, bfactor_col]):
                    plddts, last_res_id = [], None
                    for atom_line in atom_lines:
                        fields = atom_line.split()
                        if len(fields) > max(chain_col, res_col, bfactor_col):
                            chain_id = fields[chain_col]
                            res_id = fields[res_col]
                            if chain_id == binder_chain_id and res_id != last_res_id:
                                try:
                                    plddts.append(float(fields[bfactor_col]))
                                    last_res_id = res_id
                                except (ValueError, IndexError):
                                    continue
                    if plddts:
                        metrics['plddts'] = plddts
                        metrics['binder_avg_plddt'] = np.mean(plddts)
    except Exception as e:
        logger.warning(f"Error parsing pLDDTs from CIF file in {results_path}. Error: {e}")

    return metrics


# === 增强版功能类 ===

class AdvancedMutationEngine:
    """高级自适应突变引擎"""
    
    def __init__(self):
        self.motif_patterns = defaultdict(float)
        self.position_preferences = defaultdict(lambda: defaultdict(float))
        self.energy_landscape = {}
        self.mutation_history = defaultdict(list)
        self.strategy_success_rates = defaultdict(float)
        
        # 突变策略权重
        self.strategy_weights = {
            'conservative': 0.3,
            'aggressive': 0.2,
            'motif_guided': 0.2,
            'energy_guided': 0.2,
            'diversity_driven': 0.1
        }
        
        # 预定义有益motif
        self.beneficial_motifs = {
            'RGD': 0.8, 'YIGSR': 0.7, 'PHSRN': 0.6, 'DGEA': 0.6,
            'KRG': 0.5, 'KRSR': 0.6, 'GP': 0.4, 'WW': 0.4, 'FF': 0.4
        }
    
    def learn_from_sequence(self, sequence: str, score: float):
        """从序列结果中学习模式"""
        if score < 0.5:
            return
            
        # 学习位置偏好
        for i, aa in enumerate(sequence):
            self.position_preferences[i][aa] += score
        
        # 学习motif模式
        for length in range(2, min(6, len(sequence) + 1)):
            for i in range(len(sequence) - length + 1):
                motif = sequence[i:i+length]
                self.motif_patterns[motif] += score
        
        # 更新能量景观
        energy = -(score)  # 转换为能量值
        self.energy_landscape[sequence] = energy
    
    def select_mutation_strategy(self) -> str:
        """根据成功率选择突变策略"""
        # 调整权重
        total_success = sum(self.strategy_success_rates.values())
        if total_success > 0:
            adjusted_weights = {}
            for strategy, base_weight in self.strategy_weights.items():
                success_boost = self.strategy_success_rates.get(strategy, 0.0) / total_success
                adjusted_weights[strategy] = base_weight * (1.0 + success_boost)
        else:
            adjusted_weights = self.strategy_weights
        
        # 随机选择
        total_weight = sum(adjusted_weights.values())
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for strategy, weight in adjusted_weights.items():
            cumulative += weight
            if r <= cumulative:
                return strategy
        
        return 'conservative'
    
    def conservative_mutation(self, sequence: str, plddt_scores: List[float] = None, 
                            num_mutations: int = None) -> str:
        """保守突变：偏向BLOSUM62高分替换"""
        if num_mutations is None:
            num_mutations = max(1, len(sequence) // 8)
        
        new_sequence = list(sequence)
        
        # 选择突变位置
        if plddt_scores and len(plddt_scores) == len(sequence):
            weights = [(100 - score) ** 2 for score in plddt_scores]
            total_weight = sum(weights)
            if total_weight > 0:
                positions = []
                for _ in range(num_mutations):
                    r = random.random() * total_weight
                    cumsum = 0
                    for i, w in enumerate(weights):
                        cumsum += w
                        if r <= cumsum:
                            positions.append(i)
                            break
                positions = list(set(positions))
            else:
                positions = random.sample(range(len(sequence)), num_mutations)
        else:
            positions = random.sample(range(len(sequence)), num_mutations)
        
        for pos in positions:
            original_aa = sequence[pos]
            candidates = [(aa, score) for aa, score in BLOSUM62.get(original_aa, {}).items()
                         if aa != original_aa and score > 0]
            
            if candidates:
                weights = [score for _, score in candidates]
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                r = random.random()
                cumsum = 0
                chosen_aa = candidates[0][0]
                for (aa, _), prob in zip(candidates, probs):
                    cumsum += prob
                    if r <= cumsum:
                        chosen_aa = aa
                        break
                new_sequence[pos] = chosen_aa
            else:
                # 使用位置偏好
                if pos in self.position_preferences:
                    pos_prefs = self.position_preferences[pos]
                    if pos_prefs:
                        best_aa = max(pos_prefs.items(), key=lambda x: x[1])[0]
                        if best_aa != original_aa:
                            new_sequence[pos] = best_aa
        
        return ''.join(new_sequence)
    
    def aggressive_mutation(self, sequence: str, num_mutations: int = None) -> str:
        """激进突变：大范围探索"""
        if num_mutations is None:
            num_mutations = max(2, len(sequence) // 4)
        
        new_sequence = list(sequence)
        positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))
        
        for pos in positions:
            current_aa = sequence[pos]
            new_aa = random.choice([aa for aa in AMINO_ACIDS if aa != current_aa])
            new_sequence[pos] = new_aa
        
        return ''.join(new_sequence)
    
    def motif_guided_mutation(self, sequence: str) -> str:
        """motif导引突变"""
        new_sequence = list(sequence)
        
        # 选择有益motif
        all_motifs = {**self.beneficial_motifs, **dict(self.motif_patterns)}
        beneficial_motifs = [motif for motif, score in all_motifs.items() if score >= 0.3]
        
        if beneficial_motifs:
            motif = random.choice(beneficial_motifs)
            if len(motif) <= len(sequence):
                start_pos = random.randint(0, len(sequence) - len(motif))
                for i, aa in enumerate(motif):
                    new_sequence[start_pos + i] = aa
        
        # 额外保守突变
        remaining_pos = [i for i in range(len(sequence)) if new_sequence[i] == sequence[i]]
        if remaining_pos:
            num_additional = min(2, len(remaining_pos))
            additional_positions = random.sample(remaining_pos, num_additional)
            
            for pos in additional_positions:
                original_aa = sequence[pos]
                candidates = [(aa, score) for aa, score in BLOSUM62.get(original_aa, {}).items()
                             if score > 0 and aa != original_aa]
                if candidates:
                    chosen_aa = random.choice(candidates)[0]
                    new_sequence[pos] = chosen_aa
        
        return ''.join(new_sequence)
    
    def energy_guided_mutation(self, sequence: str, temperature: float = 1.0) -> str:
        """能量导引突变"""
        new_sequence = list(sequence)
        num_mutations = max(1, len(sequence) // 6)
        
        # 获取低保守性位置
        variable_positions = []
        for i in range(len(sequence)):
            if i in self.position_preferences:
                total_weight = sum(self.position_preferences[i].values())
                current_weight = self.position_preferences[i].get(sequence[i], 0)
                conservation = current_weight / total_weight if total_weight > 0 else 0
                if conservation < 0.3:
                    variable_positions.append(i)
        
        if len(variable_positions) >= num_mutations:
            positions = random.sample(variable_positions, num_mutations)
        else:
            positions = variable_positions + random.sample(
                [i for i in range(len(sequence)) if i not in variable_positions],
                min(num_mutations - len(variable_positions), len(sequence) - len(variable_positions))
            )
        
        for pos in positions:
            current_aa = sequence[pos]
            if pos in self.position_preferences:
                pos_prefs = self.position_preferences[pos]
                candidates = [(aa, weight) for aa, weight in pos_prefs.items() if aa != current_aa]
                if candidates:
                    # 温度调整
                    adj_candidates = [(aa, weight ** (1.0 / temperature)) for aa, weight in candidates]
                    total_weight = sum(weight for _, weight in adj_candidates)
                    if total_weight > 0:
                        r = random.random() * total_weight
                        cumsum = 0
                        for aa, weight in adj_candidates:
                            cumsum += weight
                            if r <= cumsum:
                                new_sequence[pos] = aa
                                break
        
        return ''.join(new_sequence)
    
    def diversity_driven_mutation(self, sequence: str, elite_sequences: List[str]) -> str:
        """多样性驱动突变"""
        if not elite_sequences:
            return self.aggressive_mutation(sequence)
        
        new_sequence = list(sequence)
        
        # 计算相似性
        similarities = []
        for elite_seq in elite_sequences:
            if len(elite_seq) == len(sequence):
                sim = sum(a == b for a, b in zip(sequence, elite_seq)) / len(sequence)
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.5
        
        # 根据相似性调整突变强度
        if avg_similarity > 0.8:
            num_mutations = max(3, len(sequence) // 3)
        else:
            num_mutations = max(1, len(sequence) // 6)
        
        # 选择差异小的位置进行突变
        position_differences = []
        for i in range(len(sequence)):
            diff_count = sum(1 for elite_seq in elite_sequences 
                           if len(elite_seq) > i and elite_seq[i] != sequence[i])
            position_differences.append((i, diff_count))
        
        position_differences.sort(key=lambda x: x[1])
        positions_to_mutate = [pos for pos, _ in position_differences[:num_mutations]]
        
        for pos in positions_to_mutate:
            current_aa = sequence[pos]
            # 选择在精英群体中该位置出现频率低的氨基酸
            elite_aas_at_pos = [elite_seq[pos] for elite_seq in elite_sequences 
                               if len(elite_seq) > pos]
            aa_counts = Counter(elite_aas_at_pos)
            
            rare_aas = [aa for aa in AMINO_ACIDS 
                       if aa != current_aa and aa_counts.get(aa, 0) <= 1]
            
            if rare_aas:
                new_sequence[pos] = random.choice(rare_aas)
            else:
                new_sequence[pos] = random.choice([aa for aa in AMINO_ACIDS if aa != current_aa])
        
        return ''.join(new_sequence)
    
    def adaptive_mutate(self, sequence: str, parent_metrics: Dict = None, 
                       elite_sequences: List[str] = None, 
                       temperature: float = 1.0) -> Tuple[str, str]:
        """执行自适应突变"""
        strategy = self.select_mutation_strategy()
        
        if strategy == 'conservative':
            plddt_scores = parent_metrics.get('plddts', []) if parent_metrics else []
            mutated_sequence = self.conservative_mutation(sequence, plddt_scores)
        elif strategy == 'aggressive':
            mutated_sequence = self.aggressive_mutation(sequence)
        elif strategy == 'motif_guided':
            mutated_sequence = self.motif_guided_mutation(sequence)
        elif strategy == 'energy_guided':
            mutated_sequence = self.energy_guided_mutation(sequence, temperature)
        elif strategy == 'diversity_driven':
            mutated_sequence = self.diversity_driven_mutation(sequence, elite_sequences or [])
        else:
            mutated_sequence = self.conservative_mutation(sequence)
        
        return mutated_sequence, strategy
    
    def update_strategy_success(self, strategy: str, improvement: float):
        """更新策略成功率"""
        self.strategy_success_rates[strategy] = (
            0.9 * self.strategy_success_rates[strategy] + 0.1 * max(0, improvement)
        )


class ParetoOptimizer:
    """Pareto多目标优化器"""
    
    def __init__(self):
        self.pareto_front = []
    
    def dominates(self, solution1: Dict, solution2: Dict) -> bool:
        """检查solution1是否支配solution2"""
        iptm1, plddt1 = solution1.get('iptm', 0), solution1.get('binder_avg_plddt', 0)
        iptm2, plddt2 = solution2.get('iptm', 0), solution2.get('binder_avg_plddt', 0)
        
        return (iptm1 >= iptm2 and plddt1 >= plddt2) and (iptm1 > iptm2 or plddt1 > plddt2)
    
    def update_pareto_front(self, new_solutions: List[Dict]):
        """更新Pareto前沿"""
        all_solutions = self.pareto_front + new_solutions
        pareto_front = []
        
        for solution in all_solutions:
            is_dominated = False
            for other in all_solutions:
                if other != solution and self.dominates(other, solution):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(solution)
        
        self.pareto_front = pareto_front
    
    def get_diverse_elites(self, num_elites: int) -> List[Dict]:
        """从Pareto前沿选择多样化精英"""
        if not self.pareto_front:
            return []
        
        if len(self.pareto_front) <= num_elites:
            return self.pareto_front
        
        # 简单的多样性选择：基于目标值分布
        solutions = self.pareto_front.copy()
        
        # 计算拥挤距离
        for solution in solutions:
            solution['crowding_distance'] = 0
        
        for obj_func in ['iptm', 'binder_avg_plddt']:
            solutions.sort(key=lambda x: x.get(obj_func, 0))
            
            if len(solutions) > 2:
                solutions[0]['crowding_distance'] = float('inf')
                solutions[-1]['crowding_distance'] = float('inf')
                
                obj_range = solutions[-1].get(obj_func, 0) - solutions[0].get(obj_func, 0)
                if obj_range > 0:
                    for i in range(1, len(solutions) - 1):
                        distance = (solutions[i+1].get(obj_func, 0) - solutions[i-1].get(obj_func, 0)) / obj_range
                        solutions[i]['crowding_distance'] += distance
        
        solutions.sort(key=lambda x: x.get('crowding_distance', 0), reverse=True)
        return solutions[:num_elites]


def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
    """计算两个序列的相似性"""
    if len(seq1) != len(seq2):
        return 0.0
    
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


def extract_sequence_features(sequence: str) -> Dict[str, float]:
    """提取序列的生化特征"""
    features = {}
    
    # 氨基酸组成
    aa_counts = Counter(sequence)
    total_length = len(sequence)
    
    # 疏水性氨基酸比例
    hydrophobic = set('AILMFPWYV')
    features['hydrophobic_ratio'] = sum(aa_counts[aa] for aa in hydrophobic if aa in aa_counts) / total_length
    
    # 极性氨基酸比例
    polar = set('NQST')
    features['polar_ratio'] = sum(aa_counts[aa] for aa in polar if aa in aa_counts) / total_length
    
    # 带电氨基酸比例
    charged = set('DEKRHC')
    features['charged_ratio'] = sum(aa_counts[aa] for aa in charged if aa in aa_counts) / total_length
    
    # 芳香性氨基酸比例
    aromatic = set('FWY')
    features['aromatic_ratio'] = sum(aa_counts[aa] for aa in aromatic if aa in aa_counts) / total_length
    
    # 小氨基酸比例
    small = set('AGS')
    features['small_ratio'] = sum(aa_counts[aa] for aa in small if aa in aa_counts) / total_length
    
    # Pro含量（结构柔性）
    features['pro_ratio'] = aa_counts.get('P', 0) / total_length
    
    # Gly含量（结构柔性）
    features['gly_ratio'] = aa_counts.get('G', 0) / total_length
    
    # Cys含量（二硫键潜力）
    features['cys_ratio'] = aa_counts.get('C', 0) / total_length
    
    return features


def analyze_population_diversity(sequences: List[str]) -> Dict[str, float]:
    """分析群体多样性"""
    if len(sequences) < 2:
        return {'avg_pairwise_similarity': 0.0, 'position_entropy': 0.0}
    
    # 平均成对相似性
    similarities = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sim = calculate_sequence_similarity(sequences[i], sequences[j])
            similarities.append(sim)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    # 位置熵
    position_entropies = []
    seq_length = len(sequences[0]) if sequences else 0
    
    for pos in range(seq_length):
        aa_counts = Counter(seq[pos] for seq in sequences if len(seq) > pos)
        total = sum(aa_counts.values())
        
        if total > 0:
            entropy = -sum((count / total) * math.log2(count / total) 
                          for count in aa_counts.values() if count > 0)
            position_entropies.append(entropy)
    
    avg_entropy = sum(position_entropies) / len(position_entropies) if position_entropies else 0.0
    
    return {
        'avg_pairwise_similarity': avg_similarity,
        'position_entropy': avg_entropy,
        'num_unique_sequences': len(set(sequences)),
        'diversity_index': 1.0 - avg_similarity  # 多样性指数
    }