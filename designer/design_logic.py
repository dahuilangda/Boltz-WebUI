# /Boltz-WebUI/designer/design_logic.py

"""
design_logic.py

该模块定义了`Designer`类，它是蛋白质和糖肽设计过程的核心控制器。
它实现了基于演化策略的梯度自由优化循环：
1.  **生成 (Generation)**: 基于精英群体创建新一代候选序列。
2.  **评估 (Evaluation)**: 并行提交候选者到Boltz-WebUI API进行结构预测和打分。
3.  **选择 (Selection)**: 根据一个复合评分函数（结合了ipTM和pLDDT）选择新的精英群体。

该类管理工作目录、与API客户端的交互以及整个设计流程的状态。
新增了自适应超参数调整机制，以应对过早收敛问题。
"""

import os
import yaml
import shutil
import time
import csv
import random
import copy
import logging
import concurrent.futures

# 本地模块导入
from api_client import BoltzApiClient
from design_utils import (
    generate_random_sequence,
    mutate_sequence,
    parse_confidence_metrics,
    MONOSACCHARIDES,
    GLYCOSYLATION_SITES,
    get_valid_residues_for_glycan,
    AdvancedMutationEngine,
    ParetoOptimizer,
    calculate_sequence_similarity,
    extract_sequence_features,
    analyze_population_diversity
)

logger = logging.getLogger(__name__)


class Designer:
    """
    管理蛋白质和糖肽的多谱系、梯度自由设计优化循环。
    集成了自适应机制以动态调整探索强度，防止过早收敛。
    """
    def __init__(self, base_yaml_path: str, client: BoltzApiClient, use_msa_server: bool = False):
        """初始化Designer实例。
        
        Args:
            base_yaml_path: 基础YAML配置文件路径
            client: BoltzApiClient实例
            use_msa_server: 当序列找不到MSA缓存时是否使用MSA服务器
        """
        self.base_yaml_path = base_yaml_path
        self.client = client
        self.use_msa_server = use_msa_server
        with open(base_yaml_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.work_dir = f"temp_design_run_{int(time.time())}"
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.evaluated_sequences = set()
        
        # --- 初始化自适应超参数 ---
        self.hparams = {
            'mutation_rate': 0.1,         # 初始突变率
            'pos_select_temp': 1.0,       # 初始pLDDT位置选择温度
        }
        # 定义超参数的边界和调整因子，以提供专业且可控的调整
        self.hparam_configs = {
            'mutation_rate': {'base': 0.1, 'max': 0.5, 'decay': 0.95, 'increase': 1.2},
            'pos_select_temp': {'base': 1.0, 'max': 10.0, 'decay': 0.9, 'increase': 1.5},
        }
        
        # === 增强版功能集成 ===
        self.mutation_engine = AdvancedMutationEngine()
        self.pareto_optimizer = ParetoOptimizer()
        self.enable_enhanced_features = True  # 控制是否使用增强功能
        
        # 收敛检测参数
        self.convergence_window = 5
        self.convergence_threshold = 0.001
        self.score_history = []
        self.stagnation_counter = 0
        self.max_stagnation = 3
        
        # 温度和自适应参数
        self.temperature = 1.0
        self.min_temperature = 0.1
        self.temperature_decay = 0.95
        
        logger.info(f"Temporary working directory created at: {os.path.abspath(self.work_dir)}")
        if self.enable_enhanced_features:
            logger.info("Enhanced features enabled: adaptive mutations, Pareto optimization, convergence detection")

    def _evaluate_one_candidate(self, candidate_info: tuple) -> tuple:
        """提交单个候选者进行评估，轮询结果，并解析指标。"""
        if len(candidate_info) >= 5:
            generation_index, sequence, binder_chain_id, design_params, strategy_used = candidate_info
        else:
            generation_index, sequence, binder_chain_id, design_params = candidate_info
            strategy_used = 'unknown'
            
        logger.info(f"[Gen {generation_index}] Evaluating new unique candidate: {sequence[:20]}... (Strategy: {strategy_used})")
        
        try:
            candidate_yaml_path = self._create_candidate_yaml(sequence, binder_chain_id, design_params)
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to create YAML for sequence '{sequence}'. Skipping. Reason: {e}")
            return (sequence, None, None)

        task_id = self.client.submit_job(candidate_yaml_path, use_msa_server=self.use_msa_server)
        if not task_id:
            return (sequence, None, None)

        status = self.client.poll_status(task_id)
        if not status or status.get('state') != 'SUCCESS':
            logger.warning(f"Job {task_id} for sequence '{sequence}' failed or did not complete successfully. Status: {status.get('state')}")
            return (sequence, None, None)

        results_path = self.client.download_and_unzip_results(task_id, self.work_dir)
        if not results_path:
            return (sequence, None, None)

        metrics = parse_confidence_metrics(results_path, binder_chain_id)
        metrics['mutation_strategy'] = strategy_used  # 添加策略信息
        iptm_score = metrics.get('iptm', 0.0)
        plddt_score = metrics.get('binder_avg_plddt', 0.0)
        logger.info(f"Candidate '{sequence[:20]}...' evaluated. ipTM: {iptm_score:.4f}, pLDDT: {plddt_score:.2f}")
        return (sequence, metrics, results_path)
    
    def _write_summary_csv(self, all_results: list, output_csv_path: str, keep_temp_files: bool):
        """将所有已评估候选者的摘要写入CSV文件，按复合分数排名。"""
        if not all_results:
            logger.warning("No results were generated to save to CSV.")
            return
        all_results.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
        header = ['rank', 'generation', 'sequence', 'composite_score', 'iptm', 'binder_avg_plddt', 'ptm', 'complex_plddt']
        if keep_temp_files:
            header.append('results_path')
        logger.info(f"Writing {len(all_results)} total results to {os.path.abspath(output_csv_path)}...")
        try:
            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for i, result_data in enumerate(all_results):
                    result_data['rank'] = i + 1
                    for key in ['composite_score', 'iptm', 'ptm', 'complex_plddt', 'binder_avg_plddt']:
                        if key in result_data and isinstance(result_data[key], float):
                            result_data[key] = f"{result_data[key]:.4f}"
                    row_to_write = {key: result_data.get(key, 'N/A') for key in header}
                    writer.writerow(row_to_write)
            logger.info(f"Summary CSV successfully saved to {output_csv_path}")
        except IOError as e:
            logger.error(f"Could not write to CSV file at {output_csv_path}. Reason: {e}")

    def _update_realtime_csv(self, all_results: list, output_csv_path: str, keep_temp_files: bool):
        """实时更新CSV文件，在每一代结束后调用，只显示当前最佳的前10个结果。"""
        if not all_results:
            return
        
        # 按复合分数排序，只保留前10个
        all_results.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
        top_results = all_results[:10]
        
        header = ['rank', 'generation', 'sequence', 'composite_score', 'iptm', 'binder_avg_plddt', 'ptm', 'complex_plddt']
        if keep_temp_files:
            header.append('results_path')
            
        try:
            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for i, result_data in enumerate(top_results):
                    result_data_copy = result_data.copy()  # 避免修改原始数据
                    result_data_copy['rank'] = i + 1
                    for key in ['composite_score', 'iptm', 'ptm', 'complex_plddt', 'binder_avg_plddt']:
                        if key in result_data_copy and isinstance(result_data_copy[key], float):
                            result_data_copy[key] = f"{result_data_copy[key]:.4f}"
                    row_to_write = {key: result_data_copy.get(key, 'N/A') for key in header}
                    writer.writerow(row_to_write)
            logger.info(f"Real-time CSV updated with top {len(top_results)} results")
        except IOError as e:
            logger.error(f"Could not write to real-time CSV file at {output_csv_path}. Reason: {e}")
            
    def _update_adaptive_hparams(self, attempts: int, population_size: int):
        """根据种群多样性指标（生成尝试次数）动态调整超参数。"""
        if population_size == 0: return

        avg_attempts = attempts / population_size
        DIVERSITY_PRESSURE_THRESHOLD = 7.5  # 当平均尝试次数超过此阈值，认为多样性不足

        if avg_attempts > DIVERSITY_PRESSURE_THRESHOLD:
            # --- 探索压力不足，增强探索 ---
            old_rate = self.hparams['mutation_rate']
            old_temp = self.hparams['pos_select_temp']
            
            # 提高突变率
            cfg = self.hparam_configs['mutation_rate']
            self.hparams['mutation_rate'] = min(old_rate * cfg['increase'], cfg['max'])
            
            # 提高pLDDT选择温度，降低其影响
            cfg_temp = self.hparam_configs['pos_select_temp']
            self.hparams['pos_select_temp'] = min(old_temp * cfg_temp['increase'], cfg_temp['max'])
            
            logger.warning(
                f"Low diversity detected (avg attempts: {avg_attempts:.1f}). Increasing exploration pressure. "
                f"Mutation rate: {old_rate:.3f} -> {self.hparams['mutation_rate']:.3f}, "
                f"pLDDT Temp: {old_temp:.2f} -> {self.hparams['pos_select_temp']:.2f}."
            )
        else:
            # --- 多样性充足，逐渐恢复基准值，加强利用 ---
            is_decaying = False
            # 衰减突变率
            cfg = self.hparam_configs['mutation_rate']
            if self.hparams['mutation_rate'] > cfg['base']:
                self.hparams['mutation_rate'] = max(self.hparams['mutation_rate'] * cfg['decay'], cfg['base'])
                is_decaying = True

            # 衰减pLDDT选择温度
            cfg_temp = self.hparam_configs['pos_select_temp']
            if self.hparams['pos_select_temp'] > cfg_temp['base']:
                self.hparams['pos_select_temp'] = max(self.hparams['pos_select_temp'] * cfg_temp['decay'], cfg_temp['base'])
                is_decaying = True
            
            if is_decaying:
                 logger.info(
                    f"Sufficient diversity (avg attempts: {avg_attempts:.1f}). "
                    f"Decaying exploration pressure towards baseline. "
                    f"Current mutation rate: {self.hparams['mutation_rate']:.3f}, "
                    f"pLDDT Temp: {self.hparams['pos_select_temp']:.2f}."
                )

    def run(self, iterations: int, **kwargs):
        """
        执行使用演化策略的主要并行设计循环。
        **kwargs 包含了所有其他的运行参数。
        """
        # 从 kwargs 中解包参数，提供默认值
        population_size = kwargs.get('population_size', 8)
        num_elites = kwargs.get('num_elites', 2)
        mutation_rate = kwargs.get('mutation_rate', 0.3)
        binder_chain_id = kwargs['binder_chain_id']
        binder_length = kwargs['binder_length']
        initial_binder_sequence = kwargs.get('initial_binder_sequence')
        glycan_ccd = kwargs.get('glycan_ccd')
        glycan_chain_id = kwargs.get('glycan_chain_id', 'C')
        glycosylation_site = kwargs.get('glycosylation_site')
        output_csv_path = kwargs.get('output_csv_path', f"design_summary_{int(time.time())}.csv")
        keep_temp_files = kwargs.get('keep_temp_files', False)
        weight_iptm = kwargs.get('weight_iptm', 0.7)
        weight_plddt = kwargs.get('weight_plddt', 0.3)

        logger.info("--- Starting Multi-Lineage Design Run with Adaptive Hyperparameters ---")
        logger.info(f"Scoring weights -> ipTM: {weight_iptm}, pLDDT: {weight_plddt}")
        logger.info(f"Mutation rate: {mutation_rate}")
        if num_elites >= population_size:
            raise ValueError("`num_elites` must be less than `population_size`.")
        
        design_params = {
            'is_glycopeptide': bool(glycan_ccd), 'glycan_ccd': glycan_ccd,
            'glycan_chain_id': glycan_chain_id, 'glycosylation_site': glycosylation_site
        }

        def calculate_composite_score(metrics: dict) -> float:
            iptm = metrics.get('iptm', 0.0)
            avg_plddt_normalized = metrics.get('binder_avg_plddt', 0.0) / 100.0
            return (weight_iptm * iptm) + (weight_plddt * avg_plddt_normalized)

        elite_population = []
        all_results_data = []

        if initial_binder_sequence and initial_binder_sequence not in self.evaluated_sequences:
            self.evaluated_sequences.add(initial_binder_sequence)
            _, metrics, res_path = self._evaluate_one_candidate((0, initial_binder_sequence, binder_chain_id, design_params))
            if metrics:
                metrics['composite_score'] = calculate_composite_score(metrics)
                entry = {'generation': 0, 'sequence': initial_binder_sequence, **metrics}
                if keep_temp_files and res_path: entry['results_path'] = os.path.abspath(res_path)
                all_results_data.append(entry)
                elite_population = [{'sequence': initial_binder_sequence, 'metrics': metrics}]

        with concurrent.futures.ThreadPoolExecutor(max_workers=population_size) as executor:
            for i in range(iterations):
                start_time = time.time()
                logger.info(f"--- Generation {i+1}/{iterations} ---")

                # --- 1. 候选者生成 ---
                candidates_to_evaluate = []
                max_attempts = population_size * 25
                attempts = 0

                while len(candidates_to_evaluate) < population_size and attempts < max_attempts:
                    new_seq = None
                    strategy_used = None
                    
                    if not elite_population:
                        if not candidates_to_evaluate:
                            logger.info(f"Seeding generation with {population_size} new random sequences...")
                        new_seq = generate_random_sequence(binder_length, glycosylation_site, design_params.get('glycan_ccd'))
                        strategy_used = 'random'
                    else:
                        if not candidates_to_evaluate:
                            logger.info(f"Evolving {len(elite_population)} elites to create {population_size} new candidates...")
                        
                        if self.enable_enhanced_features:
                            # 使用增强的自适应突变
                            parent = random.choice(elite_population)
                            elite_sequences = [e['sequence'] for e in elite_population]
                            
                            new_seq, strategy_used = self.mutation_engine.adaptive_mutate(
                                parent['sequence'],
                                parent_metrics=parent['metrics'],
                                elite_sequences=elite_sequences,
                                temperature=self.temperature
                            )
                        else:
                            # 使用原始突变方法
                            parent = random.choice(elite_population)
                            new_seq = mutate_sequence(
                                parent['sequence'],
                                mutation_rate=mutation_rate,
                                plddt_scores=parent['metrics'].get('plddts', []),
                                glycosylation_site=glycosylation_site,
                                glycan_ccd=design_params.get('glycan_ccd'),
                                position_selection_temp=self.hparams['pos_select_temp']
                            )
                            strategy_used = 'traditional'
                    
                    if new_seq and new_seq not in self.evaluated_sequences:
                        self.evaluated_sequences.add(new_seq)
                        candidate_info = (i + 1, new_seq, binder_chain_id, design_params)
                        candidate_info = candidate_info + (strategy_used,)  # 添加策略信息
                        candidates_to_evaluate.append(candidate_info)
                    attempts += 1
                
                if not candidates_to_evaluate:
                    logger.critical(f"Could not generate any new unique candidates after {max_attempts} attempts. This may indicate extreme premature convergence. Stopping run.")
                    break

                # --- 2. 评估 ---
                results = list(executor.map(self._evaluate_one_candidate, candidates_to_evaluate))
                valid_results = [res for res in results if res[1] is not None]

                if not valid_results and not elite_population:
                     logger.critical("First generation failed completely. Stopping run.")
                     break
                
                # 增强功能：学习序列模式
                if self.enable_enhanced_features:
                    for seq, metrics, res_path in valid_results:
                        composite_score = calculate_composite_score(metrics)
                        self.mutation_engine.learn_from_sequence(seq, composite_score)
                
                # 处理结果
                current_generation_results = []
                for seq, metrics, res_path in valid_results:
                    metrics['composite_score'] = calculate_composite_score(metrics)
                    entry = {'generation': i + 1, 'sequence': seq, **metrics}
                    if keep_temp_files and res_path: entry['results_path'] = os.path.abspath(res_path)
                    all_results_data.append(entry)
                    current_generation_results.append(entry)
                
                # === 增强版精英选择 (Pareto优化) ===
                if self.enable_enhanced_features and current_generation_results:
                    # 更新Pareto前沿
                    self.pareto_optimizer.update_pareto_front(current_generation_results)
                    
                    # 从Pareto前沿和传统排序中选择精英
                    pareto_elites = self.pareto_optimizer.get_diverse_elites(max(1, num_elites // 2))
                    
                    # 按复合分数排序选择剩余精英
                    all_results_data.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
                    traditional_elites = []
                    pareto_sequences = {e.get('sequence') for e in pareto_elites}
                    
                    for result in all_results_data:
                        if len(traditional_elites) >= num_elites - len(pareto_elites):
                            break
                        if result.get('sequence') not in pareto_sequences:
                            traditional_elites.append(result)
                    
                    # 合并精英群体
                    combined_elites = pareto_elites + traditional_elites
                    
                    # 添加Pareto标记
                    pareto_sequences_set = {e.get('sequence') for e in pareto_elites}
                    for result in all_results_data:
                        result['is_pareto_optimal'] = result.get('sequence') in pareto_sequences_set
                    
                    new_elites = []
                    for result in combined_elites[:num_elites]:
                        elite_entry = {
                            'sequence': result.get('sequence'),
                            'metrics': result
                        }
                        new_elites.append(elite_entry)
                    
                    elite_population = new_elites
                else:
                    # 传统精英选择
                    all_results_data.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
                    new_elites = []
                    seen_elite_seqs = set()
                    for result in all_results_data:
                        if len(new_elites) >= num_elites: break
                        sequence = result.get('sequence')
                        if sequence not in seen_elite_seqs:
                            elite_entry = {
                                'sequence': sequence,
                                'metrics': result 
                            }
                            new_elites.append(elite_entry)
                            seen_elite_seqs.add(sequence)
                    elite_population = new_elites
                
                # --- 3. 动态调整超参数 ---
                self._update_adaptive_hparams(attempts, len(candidates_to_evaluate))
                
                # === 增强版功能：收敛检测和温度调整 ===
                if self.enable_enhanced_features and elite_population:
                    current_best_score = elite_population[0]['metrics'].get('composite_score', 0.0)
                    self.score_history.append(current_best_score)
                    
                    # 检测收敛
                    if len(self.score_history) >= self.convergence_window:
                        recent_scores = self.score_history[-self.convergence_window:]
                        score_variance = max(recent_scores) - min(recent_scores)
                        
                        if score_variance < self.convergence_threshold:
                            self.stagnation_counter += 1
                            logger.info(f"Convergence detected: score variance {score_variance:.4f} < threshold {self.convergence_threshold}")
                            
                            if self.stagnation_counter >= self.max_stagnation:
                                logger.info(f"Early stopping triggered after {self.stagnation_counter} stagnation cycles")
                                break
                        else:
                            self.stagnation_counter = 0
                    
                    # 温度调整
                    if self.stagnation_counter > 0:
                        # 增加温度促进探索
                        self.temperature = min(10.0, self.temperature * 1.1)
                        logger.debug(f"Increasing temperature to {self.temperature:.2f} due to stagnation")
                    else:
                        # 逐渐降低温度
                        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
                        logger.debug(f"Cooling temperature to {self.temperature:.2f}")
                    
                    # 更新突变策略成功率
                    if len(current_generation_results) > 0:
                        best_current_score = max(r.get('composite_score', 0.0) for r in current_generation_results)
                        if len(self.score_history) > 1:
                            previous_best = self.score_history[-2]
                            improvement = best_current_score - previous_best
                            
                            # 更新策略成功率
                            for result in current_generation_results:
                                strategy = result.get('mutation_strategy')
                                if strategy and strategy != 'random':
                                    self.mutation_engine.update_strategy_success(strategy, improvement)
                    
                    # 多样性分析
                    elite_sequences = [e['sequence'] for e in elite_population]
                    diversity_metrics = analyze_population_diversity(elite_sequences)
                    logger.debug(f"Population diversity - similarity: {diversity_metrics.get('avg_pairwise_similarity', 0):.3f}, "
                               f"entropy: {diversity_metrics.get('position_entropy', 0):.3f}")
                
                # --- 4. 实时更新CSV文件 ---
                if all_results_data:
                    self._update_realtime_csv(all_results_data, output_csv_path, keep_temp_files)
                
                # 日志记录
                if elite_population:
                    best_elite = elite_population[0]
                    logger.info(
                        f"Generation {i+1} complete. Best score so far: {best_elite['metrics'].get('composite_score', 0.0):.4f} "
                        f"(ipTM: {best_elite['metrics'].get('iptm', 0.0):.4f}, pLDDT: {best_elite['metrics'].get('binder_avg_plddt', 0.0):.2f})"
                    )
                
                end_time = time.time()
                logger.info(f"--- Generation finished in {end_time - start_time:.2f} seconds ---")

        # --- 最终处理 ---
        logger.info("\n--- Design Run Finished ---")
        if all_results_data:
            final_best_entry = max(all_results_data, key=lambda r: r['composite_score'])
            logger.info(f"Final best sequence: {final_best_entry['sequence']}")
            logger.info(f"Final best composite score: {final_best_entry['composite_score']:.4f} "
                        f"(ipTM: {final_best_entry.get('iptm', 0.0):.4f}, "
                        f"pLDDT: {final_best_entry.get('binder_avg_plddt', 0.0):.2f})")
            
            # 应用阈值过滤和Top10选择
            filtered_results = self._filter_and_select_top_results(all_results_data)
            logger.info(f"Filtered results: {len(filtered_results)} sequences above threshold")
            
        self._write_summary_csv(all_results_data, output_csv_path, keep_temp_files)
        
        # 生成结果ZIP包（如果有符合条件的结果）
        if all_results_data:
            zip_path = self._generate_results_zip(filtered_results, output_csv_path, keep_temp_files)
            if zip_path:
                logger.info(f"Results ZIP package generated: {zip_path}")
        
        if not keep_temp_files:
            logger.info(f"Cleaning up temporary directory: {self.work_dir}")
            shutil.rmtree(self.work_dir)
        else:
            logger.info(f"Temporary files are kept in: {os.path.abspath(self.work_dir)}")
    
    def _filter_and_select_top_results(self, all_results_data: list, score_threshold: float = 0.6, max_results: int = 10) -> list:
        """
        过滤并选择最佳结果。
        
        Args:
            all_results_data: 所有设计结果
            score_threshold: 最低分数阈值
            max_results: 最大返回结果数量
            
        Returns:
            符合条件的Top结果列表
        """
        # 按复合分数排序
        sorted_results = sorted(all_results_data, key=lambda x: x.get('composite_score', 0.0), reverse=True)
        
        # 应用阈值过滤
        filtered_results = [r for r in sorted_results if r.get('composite_score', 0.0) >= score_threshold]
        
        # 取Top N
        top_results = filtered_results[:max_results]
        
        logger.info(f"Applied filters: score >= {score_threshold}, top {max_results}")
        logger.info(f"Results summary: {len(sorted_results)} total -> {len(filtered_results)} above threshold -> {len(top_results)} selected")
        
        return top_results
    
    def _generate_results_zip(self, filtered_results: list, output_csv_path: str, keep_temp_files: bool) -> str:
        """
        生成包含CIF文件和摘要的ZIP压缩包。
        
        Args:
            filtered_results: 过滤后的结果列表
            output_csv_path: CSV文件路径
            keep_temp_files: 是否保留临时文件
            
        Returns:
            ZIP文件路径
        """
        if not filtered_results:
            logger.warning("No results to package - skipping ZIP generation")
            return None
            
        import zipfile
        
        zip_filename = output_csv_path.replace('.csv', '_results.zip')
        
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加摘要CSV文件
                if os.path.exists(output_csv_path):
                    zipf.write(output_csv_path, 'design_summary.csv')
                
                # 添加Top结果的CIF文件
                for i, result in enumerate(filtered_results, 1):
                    results_path = result.get('results_path')
                    if results_path and os.path.exists(results_path):
                        # 查找CIF文件
                        for file in os.listdir(results_path):
                            if file.endswith('.cif'):
                                cif_path = os.path.join(results_path, file)
                                # 重命名为有意义的名称
                                new_name = f"rank_{i:02d}_score_{result.get('composite_score', 0):.3f}_{result.get('sequence', 'unknown')[:10]}.cif"
                                zipf.write(cif_path, f"structures/{new_name}")
                                break
                
                # 添加结果摘要JSON
                results_json = {
                    'summary': {
                        'total_designs': len(filtered_results),
                        'best_score': filtered_results[0].get('composite_score', 0) if filtered_results else 0,
                        'threshold_applied': 0.6,
                        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'top_results': filtered_results
                }
                
                import json
                zipf.writestr('results_summary.json', json.dumps(results_json, indent=2))
                
            logger.info(f"Successfully created results package: {zip_filename}")
            return zip_filename
            
        except Exception as e:
            logger.error(f"Failed to create results ZIP: {e}")
            return None

    def _create_candidate_yaml(self, sequence: str, chain_id: str, design_params: dict) -> str:
        """为候选序列动态创建Boltz YAML配置文件。"""
        config = copy.deepcopy(self.base_config)
        if 'sequences' not in config or not isinstance(config.get('sequences'), list):
            config['sequences'] = []

        found_chain = False
        for i, seq_block in enumerate(config['sequences']):
            if 'protein' in seq_block and seq_block.get('protein', {}).get('id') == chain_id:
                config['sequences'][i]['protein']['sequence'] = sequence
                found_chain = True
                break
        
        if not found_chain:
            config['sequences'].append({'protein': {'id': chain_id, 'sequence': sequence, 'msa': 'empty'}})

        if design_params.get('is_glycopeptide'):
            site_idx = design_params['glycosylation_site']
            glycan_ccd = design_params['glycan_ccd']
            residue_at_site = sequence[site_idx]
            
            valid_residues = get_valid_residues_for_glycan(glycan_ccd)
            if residue_at_site not in valid_residues:
                raise ValueError(
                    f"Residue '{residue_at_site}' at site {site_idx + 1} is not a valid attachment point "
                    f"for the '{glycan_ccd}' glycan (requires one of: {valid_residues})."
                )

            attachment_atom = None
            if residue_at_site in GLYCOSYLATION_SITES['N-linked']:
                attachment_atom = GLYCOSYLATION_SITES['N-linked'][residue_at_site]
            elif residue_at_site in GLYCOSYLATION_SITES['O-linked']:
                attachment_atom = GLYCOSYLATION_SITES['O-linked'][residue_at_site]

            if not attachment_atom:
                raise ValueError(f"Could not find attachment atom for residue '{residue_at_site}'.")

            config['sequences'].append({'ligand': {'id': design_params['glycan_chain_id'], 'ccd': glycan_ccd}})

            if 'constraints' not in config or not isinstance(config.get('constraints'), list):
                config['constraints'] = []
            
            config['constraints'].append({
                'bond': {
                    'atom1': [chain_id, site_idx + 1, attachment_atom],
                    'atom2': [design_params['glycan_chain_id'], 1, MONOSACCHARIDES[glycan_ccd]['atom']]
                }
            })

        yaml_path = os.path.join(self.work_dir, f"candidate_{os.getpid()}_{hash(sequence)}.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        return yaml_path